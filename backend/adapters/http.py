"""One HTTP client, reused across requests.

Each adapter used to open `httpx.AsyncClient` per call and close it again, so
every Sarvam request paid for a fresh TCP connection and a TLS handshake.
Measured against the live API: 727ms per call with a new client, 567ms
reusing one. A voice note makes two of those calls, speech recognition then
translation, so the borrower waited about a third of a second for nothing.

Two things this has to get right that a plain module-level client does not.

An `AsyncClient` binds to the event loop that first used it. Tests call
`asyncio.run` repeatedly, and each run builds a new loop — a client held from
a previous one raises on use. So the loop is tracked and the client rebuilt
when it changes.

And the pool must be modest. A serverless instance handles few concurrent
requests; a large pool would hold connections open against the provider's
limits for no gain.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

_client: Any = None
_loop: asyncio.AbstractEventLoop | None = None

#: Small on purpose — see the module docstring.
_LIMITS = dict(max_connections=10, max_keepalive_connections=5)


async def shared_client():
    """The process-wide client, rebuilt if the event loop has changed."""
    global _client, _loop

    import httpx

    running = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _loop is not running:
        if _client is not None and not _client.is_closed:
            # Belongs to a loop that is gone; closing it there is not possible.
            log.debug("Rebuilding the shared HTTP client for a new event loop")
        _client = httpx.AsyncClient(limits=httpx.Limits(**_LIMITS))
        _loop = running
    return _client


async def aclose_shared_client() -> None:
    """Release the pool. Called on application shutdown."""
    global _client, _loop
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client, _loop = None, None
