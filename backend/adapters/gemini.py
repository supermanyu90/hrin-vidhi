"""Google Gemini: an alternative to the Anthropic adapters.

Covers the same two capabilities Anthropic does — vision document extraction
(F2) and prose tightening (F6, and the RAG answer) — so a deployment can run
on Google alone, on Anthropic alone, or on one for each.

Nothing else in the system changes. That is the whole point of the adapter
layer: the rules engine, the money math and the letter templates never learn
which provider answered.

SDK shape verified against ai.google.dev (September 2026):
    from google import genai
    client.interactions.create(model=, input=, response_format=, system_instruction=)
Structured output is `response_format={"type": "text",
"mime_type": "application/json", "schema": <json schema>}`, and the text comes
back on `interaction.output_text`.

The SDK is called through `asyncio.to_thread` rather than an async client: the
synchronous surface is the one the documentation pins down, and blocking the
event loop on a vision call would stall every other request on a single
worker.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Literal

from backend.adapters.base import (
    LLM,
    AdapterError,
    DocumentParser,
    require_sdk,
    with_timeout,
)
from backend.adapters.docparser import VISION_SYSTEM_PROMPT
from backend.config import Settings
from backend.schemas import DocumentExtraction, ParsedDocument

log = logging.getLogger(__name__)


def _client(api_key: str):
    """Build a Gemini client, turning a missing package into an AdapterError."""
    try:
        from google import genai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AdapterError(
            "gemini",
            "the `google-genai` package is not installed (pip install google-genai)",
            recoverable=False,
        ) from exc
    return genai.Client(api_key=api_key)


def _output_text(interaction) -> str:
    """Pull the text out, tolerating an SDK that renames the accessor."""
    text = getattr(interaction, "output_text", None)
    if text:
        return text
    # Older/newer surfaces have used `.text`; fall back before giving up so a
    # minor SDK bump degrades to the mock rather than crashing the request.
    text = getattr(interaction, "text", None)
    if text:
        return text
    raise AdapterError("gemini", "no text found on the model response")




class GeminiDocumentParser(DocumentParser):
    """Gemini vision extraction. Requires GOOGLE_API_KEY.

    Uses the same `VISION_SYSTEM_PROMPT` as the Anthropic parser, so the two
    providers are held to an identical contract — extract only what is
    present, never guess, quote threats verbatim — and the fixtures stay valid
    for both.
    """

    provider = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.google_api_key:
            raise AdapterError(self.provider, "GOOGLE_API_KEY is not set", recoverable=False)
        require_sdk(self.provider, "google.genai", "google-genai")
        self._key = settings.google_api_key
        self._model = settings.google_model
        self._timeout = settings.provider_timeout_seconds

    async def parse(
        self,
        image: bytes,
        *,
        mime_type: str = "image/jpeg",
        hint: str | None = None,
    ) -> ParsedDocument:
        # Gemini caps an inline request at 20MB; our upload limit is 12MB, so
        # base64 expansion (~4/3) still leaves headroom.
        client = _client(self._key)
        user_text = (
            "Extract every fact this document actually shows. Leave anything you "
            "cannot read as null.\n"
            f"Document hint from the user: {hint or 'none given'}."
        )

        def call():
            return client.interactions.create(
                model=self._model,
                system_instruction=VISION_SYSTEM_PROMPT,
                input=[
                    {"type": "text", "text": user_text},
                    {
                        "type": "image",
                        "data": base64.b64encode(image).decode("utf-8"),
                        "mime_type": mime_type,
                    },
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": DocumentExtraction.model_json_schema(),
                },
            )

        try:
            interaction = await with_timeout(
                asyncio.to_thread(call), self._timeout, self.provider, "vision extraction"
            )
            raw = _output_text(interaction)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(self.provider, f"vision extraction failed: {exc}") from exc

        try:
            extraction = DocumentExtraction.model_validate_json(raw)
        except Exception as exc:
            # Schema-constrained output should make this impossible, but a
            # malformed parse must not reach the money math.
            raise AdapterError(self.provider, f"model returned off-schema JSON: {exc}") from exc

        return ParsedDocument.from_extraction(extraction, provider=self.provider)


class GeminiLLM(LLM):
    """Gemini completion. Requires GOOGLE_API_KEY.

    Only ever asked to tighten prose over a deterministic draft, and its output
    is verified before use — see `grievance/drafter._verify`, which discards an
    edit that invents a section number or drops a citation, whichever provider
    produced it.
    """

    provider = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.google_api_key:
            raise AdapterError(self.provider, "GOOGLE_API_KEY is not set", recoverable=False)
        require_sdk(self.provider, "google.genai", "google-genai")
        self._key = settings.google_api_key
        self._model = settings.google_model
        self._timeout = settings.provider_timeout_seconds

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        effort: Literal["low", "medium", "high"] = "medium",
    ) -> str:
        # `effort` is Anthropic's dial and has no verified Gemini equivalent in
        # the documented `generation_config`, so it is accepted and ignored
        # here rather than mapped onto a knob that may not exist.
        client = _client(self._key)

        def call():
            return client.interactions.create(
                model=self._model,
                system_instruction=system or None,
                input=prompt,
                generation_config={"max_output_tokens": max_tokens},
            )

        try:
            interaction = await with_timeout(
                asyncio.to_thread(call), self._timeout, self.provider, "completion"
            )
            text = _output_text(interaction)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(self.provider, f"completion failed: {exc}") from exc

        if not text.strip():
            raise AdapterError(self.provider, "model returned empty text")
        return text.strip()


def extraction_schema_json() -> str:
    """The schema handed to the model, for debugging a rejected request."""
    return json.dumps(DocumentExtraction.model_json_schema(), indent=2)
