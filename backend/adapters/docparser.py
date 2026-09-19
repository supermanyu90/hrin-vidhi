"""Document parsing: loan papers and recovery notices -> typed facts.

The hard rule for every implementation, mock or real: **extract only what is
present**. A field the image does not support stays `None`. A parser that
guesses a principal or invents a registration number would poison the money
math and the grievance letter downstream, which are the two things a judge
will check.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from backend.adapters.base import AdapterError, DocumentParser, require_sdk
from backend.config import FIXTURES_DIR, Settings
from backend.schemas import DocumentExtraction, ParsedDocument

log = logging.getLogger(__name__)

PARSE_FIXTURE_DIR = FIXTURES_DIR / "parses"

# The extraction contract, shared by every real vision backend. Kept as one
# string so the mock fixtures and the real prompt cannot drift apart.
VISION_SYSTEM_PROMPT = """\
You extract structured facts from photographs of Indian loan documents and debt
recovery notices. You are a transcription tool, not an analyst.

Absolute rules:
- Output ONLY valid JSON matching the schema given in the user message.
- Use null for any field the image does not clearly show. NEVER guess, infer,
  average, or fill a field from typical values.
- Quote threatening language verbatim in `quote` fields. Do not paraphrase,
  soften, or translate it.
- `sender_is_court` is true ONLY if the document carries a court name AND a
  case/cause number AND a judicial seal or registry stamp. A document merely
  titled "Court Notice", "Legal Notice", or "Final Notice" is not a court
  document; in that case set sender_is_court to false.
- `nbfc_registration_status`: "verified" only if an RBI Certificate of
  Registration number is printed AND legible; "unverified" if some registration
  text appears but the number is unclear; "absent" if no registration appears.
- `rate_type`: "flat" if the document says flat / "on full amount" / interest
  computed on the original principal; "reducing" if it says reducing balance or
  diminishing balance; otherwise "unknown".
- Record anything you could not read in `unreadable_regions`, and note where a
  figure came from in `source_notes`.
- Do not add commentary, legal opinion, or fields outside the schema.
"""


@lru_cache(maxsize=8)
def _load_parse_fixture(name: str) -> dict:
    path = PARSE_FIXTURE_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if not k.startswith("_")}


class MockDocumentParser(DocumentParser):
    """Serves the seeded parse for whichever document the demo uploaded.

    Selection is by `hint` when the UI supplies one, else by a cheap heuristic
    on the uploaded bytes' size — deliberately crude, because in DEMO_MODE the
    judge picks the document from a tray, not the parser.
    """

    provider = "mock"

    LOAN_FIXTURE = "loan_paper_sahyadri"
    NOTICE_FIXTURE = "notice_recovery_agency"

    async def parse(
        self,
        image: bytes,
        *,
        mime_type: str = "image/jpeg",
        hint: str | None = None,
    ) -> ParsedDocument:
        name = self._select_fixture(image, hint)
        try:
            payload = _load_parse_fixture(name)
        except FileNotFoundError as exc:
            raise AdapterError(self.provider, f"missing parse fixture {name!r}") from exc
        return ParsedDocument.model_validate(payload)

    def _select_fixture(self, image: bytes, hint: str | None) -> str:
        if hint:
            token = hint.strip().lower()
            if any(w in token for w in ("notice", "harass", "recovery", "threat", "court")):
                return self.NOTICE_FIXTURE
            if any(w in token for w in ("loan", "sanction", "agreement", "paper", "kfs")):
                return self.LOAN_FIXTURE
            log.info("Unrecognised parse hint %r, defaulting to loan paper", hint)
        return self.LOAN_FIXTURE




class AnthropicDocumentParser(DocumentParser):
    """Claude vision extraction via structured outputs. Requires ANTHROPIC_API_KEY.

    `DocumentExtraction` is passed as the output format, so the API constrains
    the response to our schema rather than us asking for JSON and hoping. There
    is deliberately no `temperature`: current Claude models reject sampling
    parameters, and the schema constraint is what buys us determinism of shape.
    """

    provider = "anthropic"

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise AdapterError(self.provider, "ANTHROPIC_API_KEY is not set", recoverable=False)
        require_sdk(self.provider, "anthropic", "anthropic")
        self._key = settings.anthropic_api_key
        self._model = settings.anthropic_model
        self._effort = settings.anthropic_effort

    async def parse(
        self,
        image: bytes,
        *,
        mime_type: str = "image/jpeg",
        hint: str | None = None,
    ) -> ParsedDocument:
        import base64

        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AdapterError(
                self.provider, "`anthropic` package is not installed", recoverable=False
            ) from exc

        user_text = (
            "Extract every fact this document actually shows. Leave anything you "
            "cannot read as null.\n"
            f"Document hint from the user: {hint or 'none given'}."
        )
        try:
            client = AsyncAnthropic(api_key=self._key)
            response = await client.messages.parse(
                model=self._model,
                max_tokens=8192,
                system=VISION_SYSTEM_PROMPT,
                output_format=DocumentExtraction,
                output_config={"effort": self._effort},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64.b64encode(image).decode(),
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise AdapterError(self.provider, f"vision extraction failed: {exc}") from exc

        # A refusal is a successful HTTP call with no usable content — check it
        # before touching `parsed_output`, or this raises an opaque AttributeError.
        if response.stop_reason == "refusal":
            raise AdapterError(self.provider, "model declined to process this image")
        if response.stop_reason == "max_tokens":
            raise AdapterError(self.provider, "extraction truncated; raise max_tokens")

        extraction = response.parsed_output
        if extraction is None:
            raise AdapterError(self.provider, "model returned no parsed output")
        return ParsedDocument.from_extraction(extraction, provider=self.provider)
