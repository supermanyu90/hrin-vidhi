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
from dataclasses import dataclass
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
from backend.adapters.llm import AnthropicLLM, MockLLM
from backend.adapters.stt import BhashiniSpeechToText, MockSpeechToText, SarvamSpeechToText
from backend.adapters.translate import MockTranslator, SarvamTranslator
from backend.adapters.tts import GTTSTextToSpeech, MockTextToSpeech, SarvamTextToSpeech
from backend.config import Settings, get_settings

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AdapterSet:
    """Everything the pipeline needs, resolved once at startup."""

    stt: SpeechToText
    translate: Translator
    tts: TextToSpeech
    docparser: DocumentParser
    llm: LLM

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


def _resolve(
    name: str,
    choice: str,
    settings: Settings,
    builders: dict[str, callable],
    mock_factory: callable,
    auto_order: list[str],
    warnings: list[str],
) -> Adapter:
    choice = choice.strip().lower()

    if settings.demo_mode:
        return mock_factory()

    if choice == "mock":
        return mock_factory()

    if choice != "auto":
        builder = builders.get(choice)
        if builder is None:
            raise AdapterError(
                name, f"unknown provider {choice!r}; known: {sorted(builders)}", recoverable=False
            )
        # Explicit choice: let the error surface rather than silently mocking.
        return builder(settings)

    for candidate in auto_order:
        builder = builders.get(candidate)
        if builder is None:
            continue
        try:
            adapter = builder(settings)
        except AdapterError:
            continue  # credentials absent; try the next
        except Exception as exc:  # noqa: BLE001
            message = f"{name}: {candidate} failed to initialise ({exc}); falling back"
            log.warning(message)
            warnings.append(message)
            continue
        log.info("%s adapter -> %s", name, candidate)
        return adapter

    log.info("%s adapter -> mock (no credentials found)", name)
    return mock_factory()


def build_adapters(settings: Settings | None = None) -> tuple[AdapterSet, list[str]]:
    """Construct the adapter set. Returns (adapters, warnings-for-/health)."""
    settings = settings or get_settings()
    warnings: list[str] = []

    adapters = AdapterSet(
        stt=_resolve(
            "stt",
            settings.stt_provider,
            settings,
            {"sarvam": SarvamSpeechToText, "bhashini": BhashiniSpeechToText},
            MockSpeechToText,
            ["sarvam", "bhashini"],
            warnings,
        ),
        translate=_resolve(
            "translate",
            settings.translate_provider,
            settings,
            {"sarvam": SarvamTranslator},
            MockTranslator,
            ["sarvam"],
            warnings,
        ),
        tts=_resolve(
            "tts",
            settings.tts_provider,
            settings,
            {"sarvam": SarvamTextToSpeech, "gtts": lambda _s: GTTSTextToSpeech()},
            MockTextToSpeech,
            ["sarvam"],  # gtts needs network but no key, so never chosen automatically
            warnings,
        ),
        docparser=_resolve(
            "docparser",
            settings.docparser_provider,
            settings,
            {"anthropic": AnthropicDocumentParser},
            MockDocumentParser,
            ["anthropic"],
            warnings,
        ),
        llm=_resolve(
            "llm",
            settings.llm_provider,
            settings,
            {"anthropic": AnthropicLLM},
            MockLLM,
            ["anthropic"],
            warnings,
        ),
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
