"""Adapter selection — the single place that decides mock vs real.

Resolution order for each capability:

  1. DEMO_MODE=true          -> always the mock. No exceptions, no network.
  2. `<name>_provider=<x>`   -> that provider, and it is an error if its key
                                is missing. An explicit choice fails loudly.
  3. `<name>_provider=auto`  -> the best provider whose credentials are present,
                                falling back to the mock when none are.

Construction failures in (3) degrade to the mock and log a warning, so a
half-configured `.env` still demos. Failures in (2) do not — if you named a
provider, you want to know it did not load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from backend.adapters.base import (
    LLM,
    Adapter,
    AdapterError,
    DocumentParser,
    SpeechToText,
    TextToSpeech,
    Translator,
)
from backend.adapters.docparser import AnthropicDocumentParser, MockDocumentParser
from backend.adapters.gemini import GeminiDocumentParser, GeminiLLM
from backend.adapters.llm import AnthropicLLM, MockLLM
from backend.adapters.stt import BhashiniSpeechToText, MockSpeechToText, SarvamSpeechToText
from backend.adapters.translate import MockTranslator, SarvamTranslator
from backend.adapters.tts import GTTSTextToSpeech, MockTextToSpeech, SarvamTextToSpeech
from backend.config import Settings, get_settings

log = logging.getLogger(__name__)


#: How one capability ended up where it is. Reported at /health and shown in
#: the interface, because "is this real inference or a fixture?" is the first
#: question anyone watching a demo actually has.
LIVE = "live"          # a real provider is answering
FALLBACK = "fallback"  # wanted a provider, none was available
FORCED = "forced"      # DEMO_MODE=true: offline on purpose


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    """What is answering for one capability, and why."""

    provider: str
    state: str
    detail: str

    @property
    def is_live(self) -> bool:
        return self.state == LIVE


@dataclass(frozen=True, slots=True)
class AdapterSet:
    """Everything the pipeline needs, resolved once at startup."""

    stt: SpeechToText
    translate: Translator
    tts: TextToSpeech
    docparser: DocumentParser
    llm: LLM
    status: dict[str, CapabilityStatus] = field(default_factory=dict)

    def describe(self) -> dict[str, str]:
        return {
            "stt": self.stt.provider,
            "translate": self.translate.provider,
            "tts": self.tts.provider,
            "docparser": self.docparser.provider,
            "llm": self.llm.provider,
        }

    @property
    def all_mock(self) -> bool:
        return all(a.is_mock for a in (self.stt, self.translate, self.tts, self.docparser, self.llm))

    @property
    def live_count(self) -> int:
        return sum(1 for s in self.status.values() if s.is_live)

    @property
    def mode(self) -> str:
        """One word for the whole application, for the badge in the interface.

        `genai` the moment any real provider is answering — a demo that is
        reading a photographed document with a vision model is not in fallback
        just because its text-to-speech is local.
        """
        if self.live_count:
            return "genai"
        return "forced" if any(s.state == FORCED for s in self.status.values()) else "fallback"


def _resolve(
    name: str,
    choice: str,
    settings: Settings,
    builders: dict[str, callable],
    mock_factory: callable,
    auto_order: list[str],
    warnings: list[str],
    status: dict[str, CapabilityStatus],
) -> Adapter:
    """Pick the implementation and record, in `status`, how it was picked."""
    # A blank choice is an unset one. Settings already normalises this; the
    # repeat is deliberate, because the failure mode is the whole site.
    choice = (choice or "").strip().lower() or "auto"

    def record(adapter: Adapter, state: str, detail: str) -> Adapter:
        status[name] = CapabilityStatus(provider=adapter.provider, state=state, detail=detail)
        return adapter

    if settings.demo_mode:
        return record(mock_factory(), FORCED, "DEMO_MODE is on: staying offline on purpose")

    if choice == "mock":
        return record(mock_factory(), FORCED, f"{name.upper()}_PROVIDER=mock was set explicitly")

    if choice != "auto":
        builder = builders.get(choice)
        if builder is None:
            raise AdapterError(
                name, f"unknown provider {choice!r}; known: {sorted(builders)}", recoverable=False
            )
        # Explicit choice: let the error surface rather than silently mocking.
        return record(builder(settings), LIVE, f"{choice} was named explicitly")

    # Why each candidate was passed over, so the fallback can say something
    # more useful than "no key" when the real cause was a missing package.
    refused: list[str] = []

    for candidate in auto_order:
        builder = builders.get(candidate)
        if builder is None:
            continue
        try:
            adapter = builder(settings)
        except AdapterError as exc:
            # AdapterError already names the provider, so no prefix here.
            refused.append(str(exc))
            continue
        except Exception as exc:  # noqa: BLE001
            message = f"{name}: {candidate} failed to initialise ({exc}); falling back"
            log.warning(message)
            warnings.append(message)
            continue
        log.info("%s adapter -> %s", name, candidate)
        return record(adapter, LIVE, f"{candidate} key found")

    log.info("%s adapter -> mock (no credentials found)", name)
    detail = "; ".join(refused) if refused else f"no provider available ({', '.join(auto_order)})"
    return record(mock_factory(), FALLBACK, detail)


def build_adapters(settings: Settings | None = None) -> tuple[AdapterSet, list[str]]:
    """Construct the adapter set. Returns (adapters, warnings-for-/health)."""
    settings = settings or get_settings()
    warnings: list[str] = []
    status: dict[str, CapabilityStatus] = {}

    adapters = AdapterSet(
        stt=_resolve(
            "stt",
            settings.stt_provider,
            settings,
            {"sarvam": SarvamSpeechToText, "bhashini": BhashiniSpeechToText},
            MockSpeechToText,
            ["sarvam", "bhashini"],
            warnings,
            status,
        ),
        translate=_resolve(
            "translate",
            settings.translate_provider,
            settings,
            {"sarvam": SarvamTranslator},
            MockTranslator,
            ["sarvam"],
            warnings,
            status,
        ),
        tts=_resolve(
            "tts",
            settings.tts_provider,
            settings,
            {"sarvam": SarvamTextToSpeech, "gtts": lambda _s: GTTSTextToSpeech()},
            MockTextToSpeech,
            ["sarvam"],  # gtts needs network but no key, so never chosen automatically
            warnings,
            status,
        ),
        docparser=_resolve(
            "docparser",
            settings.docparser_provider,
            settings,
            {"anthropic": AnthropicDocumentParser, "gemini": GeminiDocumentParser},
            MockDocumentParser,
            # Anthropic first only because it was implemented first; either is
            # a complete answer, and naming a provider explicitly overrides this.
            ["anthropic", "gemini"],
            warnings,
            status,
        ),
        llm=_resolve(
            "llm",
            settings.llm_provider,
            settings,
            {"anthropic": AnthropicLLM, "gemini": GeminiLLM},
            MockLLM,
            ["anthropic", "gemini"],
            warnings,
            status,
        ),
        status=status,
    )

    if settings.demo_mode and not adapters.all_mock:  # pragma: no cover - defensive
        raise AdapterError("registry", "DEMO_MODE is on but a real adapter was selected")

    return adapters, warnings


@lru_cache(maxsize=1)
def get_adapters() -> AdapterSet:
    adapters, _ = build_adapters()
    return adapters


def reset_adapter_cache() -> None:
    get_adapters.cache_clear()
