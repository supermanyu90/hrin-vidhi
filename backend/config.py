"""Runtime configuration. The only place environment variables are read.

DEMO_MODE is the load-bearing switch: when true, every adapter resolves to its
mock implementation and the whole pipeline runs with no keys and no network.
It defaults to *true* on purpose — a fresh clone must demo before it can fail.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
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
        default=True,
        description="True => all adapters are mocks. The demo path. Never needs a key.",
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
    rag_top_k: int = 4
    embedding_model: str = "all-MiniLM-L6-v2"

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

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
