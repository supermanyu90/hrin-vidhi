"""Keeps the suite hermetic.

`Settings` reads `.env`, which is exactly right for running the application and
exactly wrong for running its tests. The moment a real key landed in `.env`,
the suite started constructing real adapters and calling Google: runtime went
from 10 seconds to 113, five tests failed on configuration they never set, and
an ordinary `pytest -q` was spending money.

Environment variables take precedence over `.env` in pydantic-settings, so
blanking them here masks whatever the developer has configured. Tests that
want a provider still pass it explicitly — `Settings(anthropic_api_key="x")` —
and those init arguments outrank both.

The rule this enforces: a test run means the same thing on a laptop with every
key configured and on a CI box with none.
"""

from __future__ import annotations

import contextlib
import os

import pytest

#: Credentials: blanked, so no provider can be constructed.
_MASKED_KEYS = (
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "SARVAM_API_KEY",
    "BHASHINI_API_KEY",
    "BHASHINI_USER_ID",
)

#: Provider selection: reset to the default rather than blanked. An empty
#: string is not a valid choice — the registry rejects it — so masking these
#: with "" swapped one kind of .env leakage for another.
_MASKED_CHOICES = (
    "STT_PROVIDER",
    "TRANSLATE_PROVIDER",
    "TTS_PROVIDER",
    "DOCPARSER_PROVIDER",
    "LLM_PROVIDER",
)


@pytest.fixture(autouse=True, scope="session")
def _hermetic_environment() -> None:
    """Blank provider configuration for the whole session, before any import."""
    # "" rather than unset: an empty value still beats the .env entry, where
    # deleting the variable would let .env show through again.
    for name in _MASKED_KEYS:
        os.environ[name] = ""
    for name in _MASKED_CHOICES:
        os.environ[name] = "auto"
    os.environ["DEMO_MODE"] = "false"

    # Any Settings built during collection must not survive with .env values.
    from backend.config import reset_settings_cache

    reset_settings_cache()


@pytest.fixture(autouse=True)
def _no_accidental_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test reaches for a provider SDK.

    A hang is the worst way to learn this: the suite retried a 503 with backoff
    for nearly two minutes before anyone could see what was wrong.
    """

    def _refuse(*args, **kwargs):  # pragma: no cover - only runs on a mistake
        raise AssertionError(
            "A test tried to construct a real provider client. Tests must use the "
            "offline implementations or stub the SDK — see tests/conftest.py."
        )

    for module, attr in (("anthropic", "AsyncAnthropic"), ("google.genai", "Client")):
        # raising=False covers an SDK that is not installed in this environment.
        with contextlib.suppress(ImportError, AttributeError, ModuleNotFoundError):
            monkeypatch.setattr(f"{module}.{attr}", _refuse, raising=False)
