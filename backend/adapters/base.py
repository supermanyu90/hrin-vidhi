"""Adapter contracts.

Each of the five external capabilities (speech-to-text, translation,
text-to-speech, document parsing, LLM completion) is an abstract base class
with at least two implementations: a `Mock*` backed by fixtures, and a `Real*`
backed by a provider SDK or HTTP API.

These ABCs are deliberately free of FastAPI, pydantic-settings and project
imports beyond `schemas`, so the whole `adapters/` package can be lifted into
another project unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from backend.schemas import (
    Language,
    ParsedDocument,
    SpeechAudio,
    Transcript,
    TranslationResult,
)


class AdapterError(RuntimeError):
    """Raised when a real adapter cannot fulfil a call.

    The pipeline catches this and falls back to the mock so a flaky venue
    network degrades the demo's fidelity rather than breaking it.
    """

    def __init__(self, adapter: str, message: str, *, recoverable: bool = True) -> None:
        super().__init__(f"[{adapter}] {message}")
        self.adapter = adapter
        self.recoverable = recoverable


class Adapter(ABC):  # noqa: B024 - see below
    """Common surface so the health route can introspect what is wired up.

    Deliberately has no abstract methods: it is a marker base carrying
    `provider` and `describe()`. The five capability ABCs below add the
    abstract methods; this exists so the registry can treat any adapter
    uniformly when reporting what is live.
    """

    #: Short identifier reported by /health, e.g. "mock" or "sarvam".
    provider: str = "unknown"

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def describe(self) -> str:
        return self.provider


class SpeechToText(Adapter):
    """Audio bytes in a declared language -> verbatim native transcript."""

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        language: Language,
        *,
        mime_type: str = "audio/webm",
    ) -> Transcript:
        """Return a Transcript whose `english_text` may be empty.

        Filling `english_text` is the Translate adapter's job; an STT provider
        that happens to return a translation may populate it directly.
        """


class Translator(Adapter):
    @abstractmethod
    async def translate(
        self,
        text: str,
        source: Language,
        target: Language,
    ) -> TranslationResult:
        ...


class TextToSpeech(Adapter):
    @abstractmethod
    async def synthesize(self, text: str, language: Language) -> SpeechAudio:
        ...


class DocumentParser(Adapter):
    """Image bytes -> typed LoanFacts / NoticeFacts. Must never guess a value."""

    @abstractmethod
    async def parse(
        self,
        image: bytes,
        *,
        mime_type: str = "image/jpeg",
        hint: str | None = None,
    ) -> ParsedDocument:
        """`hint` lets the UI say "this is a notice" to skip classification."""


class LLM(Adapter):
    """Text completion. Used only to tighten prose over a deterministic skeleton.

    Note the absence of a `temperature` knob: current Claude models reject
    sampling parameters outright, so depth/cost is expressed as `effort` and
    determinism comes from the templates, not from the decoder.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        effort: Literal["low", "medium", "high"] = "medium",
    ) -> str:
        ...


def require_sdk(provider: str, module: str, package: str) -> None:
    """Fail construction when a provider's SDK is absent.

    The registry reports a constructed adapter as `live`, and the interface
    shows that to the user. An adapter that constructs without its SDK would
    claim to be live and then fall back on the first real call — the badge
    would be lying. Checking the import here keeps the reported state true.
    """
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise AdapterError(
            provider,
            f"the `{package}` package is not installed (pip install {package})",
            recoverable=False,
        )
