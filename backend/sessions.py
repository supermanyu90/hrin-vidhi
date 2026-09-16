"""In-memory session store.

Sessions tie the six pipeline stages into one borrower conversation: the voice
note, the photographed documents, the money math, the compliance report and
(from M6) the letter.

**Nothing here is written to disk.** That is the §1 privacy rule made
structural rather than promised: a session lives in this process's memory,
expires on a TTL, and takes its contents with it. Uploaded image bytes are
never stored at all — they are parsed in the request that carried them and
dropped, so even a session dump cannot leak a borrower's loan paper.

Single-process by design. Running uvicorn with multiple workers would give
each worker its own store and scatter a borrower's session across them; for a
demo that is the right trade, and `--workers 1` is what `main.py` runs.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field

from backend.config import get_settings
from backend.schemas import Language, SessionState

log = logging.getLogger(__name__)

#: Hard cap. A runaway client cannot exhaust memory; the oldest session is
#: evicted first. Far above anything a demo will reach.
MAX_SESSIONS = 500


@dataclass(slots=True)
class _Entry:
    state: SessionState
    last_touched: float = field(default_factory=time.monotonic)


class SessionStore:
    """TTL-bounded session storage. Async-safe within one process."""

    def __init__(self, ttl_seconds: int, max_sessions: int = MAX_SESSIONS) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds
        self._max = max_sessions

    async def create(self, language: Language) -> SessionState:
        # `token_urlsafe` rather than uuid4: a session id is a bearer token for
        # a borrower's loan documents, so it should not be guessable.
        session_id = secrets.token_urlsafe(16)
        state = SessionState(session_id=session_id, language=language)
        async with self._lock:
            self._drop_expired_locked()
            self._make_room_locked()
            self._entries[session_id] = _Entry(state=state)
        log.info("Session %s opened (language=%s)", session_id[:8], language.value)
        return state

    async def get(self, session_id: str) -> SessionState | None:
        async with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._entries[session_id]
                log.info("Session %s expired", session_id[:8])
                return None
            entry.last_touched = time.monotonic()
            return entry.state

    async def update(self, state: SessionState) -> None:
        """Persist a mutated state back into the store."""
        async with self._lock:
            entry = self._entries.get(state.session_id)
            if entry is None:
                return
            entry.state = state
            entry.last_touched = time.monotonic()

    async def delete(self, session_id: str) -> bool:
        """Borrower-initiated erase. Returns True if something was removed."""
        async with self._lock:
            existed = self._entries.pop(session_id, None) is not None
        if existed:
            log.info("Session %s deleted at the user's request", session_id[:8])
        return existed

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            # Expired entries are dropped (they are already gone, logically),
            # but the cap is NOT enforced here: `stats` is a read, and a read
            # must never discard a borrower's live session.
            self._drop_expired_locked()
            return {"active_sessions": len(self._entries), "ttl_seconds": self._ttl}

    # -- internals ---------------------------------------------------------

    def _is_expired(self, entry: _Entry) -> bool:
        return (time.monotonic() - entry.last_touched) > self._ttl

    def _drop_expired_locked(self) -> None:
        """Remove sessions past their TTL. Safe to call from a read."""
        expired = [sid for sid, e in self._entries.items() if self._is_expired(e)]
        for session_id in expired:
            del self._entries[session_id]
        if expired:
            log.info("Evicted %d expired session(s)", len(expired))

    def _make_room_locked(self) -> None:
        """Free a slot for one new session by dropping the least-recently-used.

        Only ever called from `create`. Keeping it separate from expiry is the
        point: this one discards *live* data, so nothing that merely reads the
        store may invoke it.
        """
        while len(self._entries) >= self._max:
            oldest = min(self._entries.items(), key=lambda kv: kv[1].last_touched)[0]
            del self._entries[oldest]
            log.warning("Session cap reached; evicted oldest session %s", oldest[:8])


_store: SessionStore | None = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore(ttl_seconds=get_settings().session_ttl_seconds)
    return _store


def reset_store() -> None:
    """Tests need a clean store between cases."""
    global _store
    _store = None
