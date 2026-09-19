"""LLM completion.

The LLM is the *least* load-bearing adapter in this system, by design. Every
borrower-facing artefact — the rights script, the grievance letter — is built
from a deterministic template populated by the rules engine, and the LLM only
smooths the prose. That is why `MockLLM` returning templated text is an honest
mock rather than a fake: in DEMO_MODE the pipeline produces the same factual
content it would with a key set, just less fluently worded.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from backend.adapters.base import LLM, AdapterError, require_sdk
from backend.config import Settings

log = logging.getLogger(__name__)

#: Callers tag prompts with `TASK: <name>` on the first line so the mock can
#: dispatch. Real adapters ignore the tag — it reads as ordinary instruction.
_TASK_RE = re.compile(r"^\s*TASK:\s*([a-z_]+)\s*$", re.MULTILINE)


def extract_task(prompt: str) -> str | None:
    match = _TASK_RE.search(prompt)
    return match.group(1) if match else None


class MockLLM(LLM):
    """Deterministic, offline stand-in.

    For every task the pipeline defines, the caller passes a pre-rendered
    deterministic draft under a `DRAFT:` marker. The mock returns that draft
    unchanged — the LLM's only job in the real path is to tighten it, so
    returning it verbatim is a faithful no-op rather than invented content.
    """

    provider = "mock"

    DRAFT_MARKER = "DRAFT:"

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        effort: Literal["low", "medium", "high"] = "medium",
    ) -> str:
        task = extract_task(prompt)
        draft = self._extract_draft(prompt)
        if draft is not None:
            log.debug("MockLLM returning caller draft verbatim (task=%s)", task)
            return draft
        log.warning(
            "MockLLM got no DRAFT block (task=%s); returning an explicit placeholder "
            "rather than inventing text",
            task,
        )
        return (
            "[This section could not be drafted offline. Run with ANTHROPIC_API_KEY set, "
            "or fill it in by hand before sending.]"
        )

    def _extract_draft(self, prompt: str) -> str | None:
        index = prompt.find(self.DRAFT_MARKER)
        if index == -1:
            return None
        return prompt[index + len(self.DRAFT_MARKER) :].strip() or None




class AnthropicLLM(LLM):
    """Claude completion. Requires ANTHROPIC_API_KEY.

    Streams because grievance letters can run long, and a non-streaming call at
    a high `max_tokens` risks an SDK HTTP timeout.
    """

    provider = "anthropic"

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise AdapterError(self.provider, "ANTHROPIC_API_KEY is not set", recoverable=False)
        require_sdk(self.provider, "anthropic", "anthropic")
        self._key = settings.anthropic_api_key
        self._model = settings.anthropic_model

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        effort: Literal["low", "medium", "high"] = "medium",
    ) -> str:
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AdapterError(
                self.provider, "`anthropic` package is not installed", recoverable=False
            ) from exc

        try:
            client = AsyncAnthropic(api_key=self._key)
            async with client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                system=system or "",
                output_config={"effort": effort},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = await stream.get_final_message()
        except Exception as exc:
            raise AdapterError(self.provider, f"completion failed: {exc}") from exc

        if message.stop_reason == "refusal":
            raise AdapterError(self.provider, "model declined this request")

        text = "".join(block.text for block in message.content if block.type == "text")
        if not text.strip():
            raise AdapterError(self.provider, "model returned empty text")
        return text.strip()
