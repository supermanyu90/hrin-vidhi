"""Runtime configuration. The only place environment variables are read.

DEMO_MODE forces every adapter to its offline fallback. It defaults to *false*:
the application reaches for a real provider whenever a key is present, and
falls back only when one is not.

A fresh clone still demos with no keys and no network — but that guarantee now
comes from the fallback itself rather than from the flag. Each `*_PROVIDER` is
`auto`, which tries the real providers in order and resolves to the mock when
none can be constructed. Turning DEMO_MODE on is therefore an explicit choice
("stay offline even though a key is present"), not the default path.

Whichever way each capability resolved is reported at `/health` and shown in
the interface, so nobody has to guess whether they are watching real inference
or a fixture.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
CORPUS_DIR = BACKEND_DIR / "corpus"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

VERSION = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- the switch ---------------------------------------------------------
    demo_mode: bool = Field(
        default=False,
        description=(
            "Force every adapter to its offline fallback, ignoring any key that is "
            "present. Off by default: absence of a key already produces the fallback."
        ),
    )

    # -- server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    # -- provider keys (all optional; absence forces the mock) --------------
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"
    #: Depth/cost dial for the Anthropic adapters. `temperature` is rejected on
    #: Opus 5, so effort is the knob we expose instead.
    anthropic_effort: str = "medium"
    # Google — an alternative to Anthropic for vision and prose.
    google_api_key: str | None = None
    #: gemini-3.5-flash, not the newer 3.8: both extract this corpus correctly,
    #: but 3.8-flash allows only 20 requests a day on the free tier, which a
    #: single rehearsal exhausts. 3.5-flash is verified at 22s for a document
    #: page and honours the response schema, which gemini-2.5-flash does not —
    #: 2.5 returns its own field names and every extraction fails validation.
    google_model: str = "gemini-3.5-flash"

    bhashini_api_key: str | None = None
    bhashini_user_id: str | None = None
    sarvam_api_key: str | None = None

    # -- adapter selection --------------------------------------------------
    # "auto" picks mock in DEMO_MODE, else the best provider whose key is set.
    stt_provider: str = "auto"
    translate_provider: str = "auto"
    tts_provider: str = "auto"
    docparser_provider: str = "auto"
    llm_provider: str = "auto"

    #: Wall-clock ceiling on any single provider call.
    #:
    #: Provider SDKs retry internally with backoff. A transient 503 from Google
    #: was observed turning one document upload into three retries over 110
    #: seconds and still climbing — the borrower just watches a spinner, and on
    #: Vercel the function is killed at 30s with nothing to show. Past this
    #: ceiling we give up on the provider and fall back, which the interface
    #: then reports honestly.
    provider_timeout_seconds: float = Field(default=40.0, gt=0)

    # -- privacy ------------------------------------------------------------
    persist_uploads: bool = Field(
        default=False,
        description="When false, uploaded images are held in memory for the request only.",
    )
    session_ttl_seconds: int = 3600

    # -- retrieval ----------------------------------------------------------
    rag_min_score: float = Field(
        default=0.18,
        description="Below this cosine/BM25 score we say 'couldn't confirm' rather than cite.",
    )
    rag_min_coverage: float = Field(
        default=0.34,
        description=(
            "Minimum share of the question's distinct terms a chunk must contain. Score "
            "alone is not enough: one rare term matching ('night' in an off-topic question) "
            "can score highly while answering nothing. Both gates must pass to cite."
        ),
    )
    rag_relative_floor: float = Field(
        default=0.55,
        description=(
            "A chunk scoring below this fraction of the best hit is dropped. Keeps a "
            "borrower-facing answer to the material that actually answers the question "
            "instead of padding it with weakly-related clauses."
        ),
    )
    rag_relative_coverage: float = Field(
        default=0.6,
        description=(
            "A chunk answering far less of the question than the best hit is dropped. "
            "The score floor alone is not enough: a concept covering every term of the "
            "question and one sharing a single word can land within a few hundredths of "
            "each other, and the weaker one then pads the answer with material the "
            "borrower did not ask about."
        ),
    )
    rag_top_k: int = 4
    embedding_model: str = "all-MiniLM-L6-v2"

    @model_validator(mode="before")
    @classmethod
    def _blank_means_unset(cls, values):
        """Treat a blank environment variable as absent, not as a value.

        A dashboard is a text box: adding DEMO_MODE and leaving it empty is a
        normal thing to do, and pydantic then refuses to parse "" as a boolean
        and raises during import. That does not degrade the site — it takes the
        whole function down before a single route is registered, and the
        deployment log shows a validation error rather than anything about
        configuration.

        A blank means "not configured" for EVERY setting, including the text
        ones. An earlier version of this guard exempted strings, on the
        reasoning that "" is a legitimate hostname or key — and a blank
        STT_PROVIDER then reached the registry as a provider named empty
        string, which took production down with:

            AdapterError: [stt] unknown provider ''; known: ['bhashini', 'sarvam']

        There is no setting here where an empty string is the intended value:
        a blank key is absent (None is already the default and both are
        falsy), a blank provider is `auto`, a blank model is the default
        model. Naming a provider that does not exist still fails loudly, which
        is the point of that error — leaving the box empty is not naming one.
        """
        if not isinstance(values, dict):
            return values

        return {
            key: value
            for key, value in values.items()
            if not (isinstance(value, str) and not value.strip() and key.lower() in cls.model_fields)
        }

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_google(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_bhashini(self) -> bool:
        return bool(self.bhashini_api_key and self.bhashini_user_id)

    @property
    def has_sarvam(self) -> bool:
        return bool(self.sarvam_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Tests flip env vars between cases; this makes the next read pick them up."""
    get_settings.cache_clear()
