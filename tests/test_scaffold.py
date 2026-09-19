"""Milestone 1: the scaffold holds up and DEMO_MODE is genuinely offline."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.adapters.base import AdapterError
from backend.adapters.registry import build_adapters
from backend.config import Settings
from backend.privacy import redact
from backend.schemas import (
    DocumentKind,
    Language,
    LoanFacts,
    NBFCRegistrationStatus,
    ParsedDocument,
    RateType,
)


@pytest.fixture
def demo_settings() -> Settings:
    return Settings(demo_mode=True, anthropic_api_key=None, sarvam_api_key=None)


@pytest.fixture
def client() -> TestClient:
    from backend.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Health / wiring
# ---------------------------------------------------------------------------


def test_health_reports_every_capability_and_how_it_resolved(client: TestClient) -> None:
    """With no keys the app runs, and says plainly that it is on fallbacks."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["adapters"]) == {"stt", "translate", "tts", "docparser", "llm"}
    assert all(v == "mock" for v in body["adapters"].values()), body["adapters"]

    # The point of the mode field: a fallback must be visible, not silent.
    assert body["mode"] == "fallback"
    assert body["live_capabilities"] == 0
    assert set(body["capabilities"]) == set(body["adapters"])
    assert all(c["state"] == "fallback" for c in body["capabilities"].values())
    # And the reason has to be actionable, not just "mock".
    assert all(c["detail"] for c in body["capabilities"].values())


def test_demo_mode_is_off_unless_asked_for() -> None:
    """It is an explicit offline switch now, not the default path."""
    assert Settings().demo_mode is False


def test_demo_mode_forces_fallback_even_with_a_key() -> None:
    adapters, _ = build_adapters(Settings(demo_mode=True, sarvam_api_key="x"))
    assert adapters.mode == "forced"
    assert all(s.state == "forced" for s in adapters.status.values())
    assert adapters.all_mock


def test_a_usable_key_puts_the_app_in_genai_mode() -> None:
    adapters, _ = build_adapters(Settings(demo_mode=False, sarvam_api_key="x"))
    assert adapters.mode == "genai"
    assert adapters.status["stt"].state == "live"
    # Unconfigured capabilities still fall back, and say so.
    assert adapters.status["docparser"].state == "fallback"


def test_a_key_without_its_sdk_does_not_claim_to_be_live() -> None:
    """Otherwise the badge says GenAI and the first request quietly falls back."""
    adapters, _ = build_adapters(Settings(demo_mode=False, anthropic_api_key="x"))
    assert adapters.status["docparser"].state == "fallback"
    assert "not installed" in adapters.status["docparser"].detail


def test_frontend_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# Adapter resolution
# ---------------------------------------------------------------------------


def test_demo_mode_ignores_present_keys(demo_settings: Settings) -> None:
    """A stray key in the environment must not pull a real adapter into a demo."""
    settings = demo_settings.model_copy(
        update={"anthropic_api_key": "sk-ant-not-real", "sarvam_api_key": "also-not-real"}
    )
    adapters, _ = build_adapters(settings)
    assert adapters.all_mock


def test_auto_falls_back_to_mock_without_keys() -> None:
    settings = Settings(demo_mode=False, anthropic_api_key=None, sarvam_api_key=None)
    adapters, _ = build_adapters(settings)
    assert adapters.all_mock


def test_explicit_provider_without_key_fails_loudly() -> None:
    """Naming a provider is a claim it is configured; silently mocking would hide a bug."""
    settings = Settings(demo_mode=False, llm_provider="anthropic", anthropic_api_key=None)
    with pytest.raises(AdapterError, match="ANTHROPIC_API_KEY"):
        build_adapters(settings)


def test_unknown_provider_name_is_rejected() -> None:
    settings = Settings(demo_mode=False, stt_provider="whisper")
    with pytest.raises(AdapterError, match="unknown provider"):
        build_adapters(settings)


def test_mock_can_be_forced_outside_demo_mode() -> None:
    settings = Settings(demo_mode=False, tts_provider="mock", sarvam_api_key="x")
    adapters, _ = build_adapters(settings)
    assert adapters.tts.is_mock


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", list(Language))
def test_stt_fixture_exists_for_every_language(demo_settings: Settings, language: Language) -> None:
    adapters, _ = build_adapters(demo_settings)
    transcript = asyncio.run(adapters.stt.transcribe(b"", language))
    assert transcript.language is language
    assert transcript.native_text.strip()
    assert transcript.english_text.strip()
    # Non-English fixtures must not be the English text copied over.
    if language is not Language.ENGLISH:
        assert transcript.native_text != transcript.english_text


def test_translate_resolves_the_seeded_story(demo_settings: Settings) -> None:
    adapters, _ = build_adapters(demo_settings)
    transcript = asyncio.run(adapters.stt.transcribe(b"", Language.MARATHI))
    result = asyncio.run(
        adapters.translate.translate(transcript.native_text, Language.MARATHI, Language.ENGLISH)
    )
    assert result.text == transcript.english_text


def test_translate_passes_unknown_text_through_unchanged(demo_settings: Settings) -> None:
    """The mock must never fabricate a translation it does not have."""
    adapters, _ = build_adapters(demo_settings)
    result = asyncio.run(
        adapters.translate.translate("कुछ अनजान वाक्य", Language.HINDI, Language.ENGLISH)
    )
    assert result.text == "कुछ अनजान वाक्य"


def test_tts_either_speaks_or_says_it_cannot(demo_settings: Settings) -> None:
    """Platform-dependent by design, and honest either way.

    Where an offline voice exists (macOS `say`) the mock returns real speech.
    Where it does not — every Linux host, so every serverless deployment — it
    returns the script with NO audio rather than a silent file, and the client
    speaks it with the Web Speech API instead.
    """
    adapters, _ = build_adapters(demo_settings)
    audio = asyncio.run(adapters.tts.synthesize("यह एक जाँच है।", Language.HINDI))

    assert audio.text
    assert audio.provider.startswith("mock")

    if audio.audio_base64:
        raw = base64.b64decode(audio.audio_base64)
        assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "not a valid WAV container"
        assert audio.provider in {"mock-fixture", "mock-say"}
    else:
        assert audio.provider == "mock-unavailable"


def test_the_mock_never_returns_a_silent_file_pretending_to_be_speech() -> None:
    """A silent WAV looks like a voice note to the UI and is ~6MB of base64.

    Returning nothing is both smaller and more truthful.
    """
    from backend.adapters import tts as tts_module

    assert not hasattr(tts_module, "_silent_wav")
    source = (Path(tts_module.__file__)).read_text(encoding="utf-8")
    assert "mock-silent" not in source


def test_llm_mock_never_invents_content(demo_settings: Settings) -> None:
    adapters, _ = build_adapters(demo_settings)
    passthrough = asyncio.run(
        adapters.llm.complete("TASK: rights_script\nDRAFT:\nThey cannot call.")
    )
    assert passthrough == "They cannot call."

    placeholder = asyncio.run(adapters.llm.complete("TASK: rights_script\nNo draft here."))
    assert "could not be drafted offline" in placeholder


# ---------------------------------------------------------------------------
# F2 acceptance: the seeded loan paper parses as specified
# ---------------------------------------------------------------------------


def test_loan_paper_fixture_meets_f2_acceptance(demo_settings: Settings) -> None:
    adapters, _ = build_adapters(demo_settings)
    parsed = asyncio.run(adapters.docparser.parse(b"", hint="loan paper"))
    assert parsed.kind is DocumentKind.LOAN_PAPER
    facts = parsed.loan_facts
    assert facts is not None
    assert facts.rate_type is RateType.FLAT
    assert facts.quoted_rate == 12.0
    assert facts.principal == 80_000.0
    assert facts.tenure_months == 24
    assert facts.nbfc_registration_status is NBFCRegistrationStatus.ABSENT


def test_notice_fixture_carries_the_f3_trigger_facts(demo_settings: Settings) -> None:
    """F3 needs an out-of-hours contact and a family threat to fire two rules."""
    adapters, _ = build_adapters(demo_settings)
    parsed = asyncio.run(adapters.docparser.parse(b"", hint="harassment notice"))
    assert parsed.kind is DocumentKind.NOTICE
    notice = parsed.notice_facts
    assert notice is not None
    assert notice.sender_is_court is False
    assert any(c.time_24h == "06:30" for c in notice.contact_times_mentioned)
    assert any(t.category.value == "family_contact" for t in notice.threats_detected)


# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------


def test_unknown_fields_are_rejected() -> None:
    """extra='forbid' is what makes a drifting adapter fail loudly."""
    with pytest.raises(ValueError):
        ParsedDocument.model_validate(
            {"kind": "loan_paper", "provider": "mock", "surprise_field": 1}
        )


def test_loan_facts_default_to_unknown_not_zero() -> None:
    """A field we could not read must be None — never a plausible-looking default."""
    facts = LoanFacts()
    assert facts.principal is None
    assert facts.quoted_rate is None
    assert facts.rate_type is RateType.UNKNOWN
    assert facts.nbfc_registration_status is NBFCRegistrationStatus.ABSENT


def test_contact_event_rejects_a_malformed_time() -> None:
    from backend.schemas import ContactEvent

    ContactEvent(time_24h="06:30")
    with pytest.raises(ValueError):
        ContactEvent(time_24h="6:30am")


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, must_not_contain",
    [
        ("call me on 9876543210 today", "9876543210"),
        ("+91 9876543210 rang at 6am", "9876543210"),
        ("loan account 123456789012345 overdue", "123456789012345"),
        ("aadhaar 1234 5678 9012 attached", "1234 5678 9012"),
        ("pan ABCDE1234F on file", "ABCDE1234F"),
        ("write to ramesh@example.com", "ramesh@example.com"),
    ],
)
def test_redaction_removes_identifiers(raw: str, must_not_contain: str) -> None:
    assert must_not_contain not in redact(raw)


def test_redaction_keeps_the_surrounding_text() -> None:
    assert "overdue" in redact("loan account 123456789012345 overdue")


def test_redaction_leaves_ordinary_numbers_alone() -> None:
    """Loan amounts and rates must survive — they are the whole analysis."""
    assert redact("principal 80000 at 12 percent over 24 months") == (
        "principal 80000 at 12 percent over 24 months"
    )


# ---------------------------------------------------------------------------
# Provider choice — Anthropic and Google are interchangeable
# ---------------------------------------------------------------------------


def test_google_key_alone_activates_vision_and_prose(sdks_present) -> None:
    """The whole point of the adapter layer: one key, no code change."""
    settings = Settings(demo_mode=False, google_api_key="g-x")
    adapters, _ = build_adapters(settings)
    assert adapters.docparser.provider == "gemini"
    assert adapters.llm.provider == "gemini"


@pytest.fixture
def sdks_present(monkeypatch):
    """Pretend both provider SDKs are installed.

    `require_sdk` deliberately refuses to construct an adapter whose package is
    missing, so the reported state cannot claim `live` and then fall back. These
    tests are about which provider gets *selected*, which is independent of what
    happens to be installed in this environment.
    """
    import backend.adapters.docparser as dp
    import backend.adapters.gemini as gm
    import backend.adapters.llm as lm

    for module in (dp, gm, lm):
        monkeypatch.setattr(module, "require_sdk", lambda *a, **k: None)


def test_either_provider_alone_is_a_complete_answer(sdks_present) -> None:
    for key, expected in (("anthropic_api_key", "anthropic"), ("google_api_key", "gemini")):
        adapters, _ = build_adapters(Settings(demo_mode=False, **{key: "x"}))
        assert adapters.docparser.provider == expected
        assert adapters.llm.provider == expected


def test_a_named_provider_beats_auto_selection(sdks_present) -> None:
    """With both keys set, the operator still decides."""
    settings = Settings(
        demo_mode=False,
        anthropic_api_key="a-x",
        google_api_key="g-x",
        docparser_provider="gemini",
        llm_provider="gemini",
    )
    adapters, _ = build_adapters(settings)
    assert adapters.docparser.provider == "gemini"
    assert adapters.llm.provider == "gemini"


def test_providers_can_be_mixed(sdks_present) -> None:
    """Gemini for vision, Claude for prose, or the other way round."""
    settings = Settings(
        demo_mode=False,
        anthropic_api_key="a-x",
        google_api_key="g-x",
        docparser_provider="gemini",
        llm_provider="anthropic",
    )
    adapters, _ = build_adapters(settings)
    assert adapters.docparser.provider == "gemini"
    assert adapters.llm.provider == "anthropic"


def test_gemini_without_its_key_fails_loudly() -> None:
    settings = Settings(demo_mode=False, docparser_provider="gemini", google_api_key=None)
    with pytest.raises(AdapterError, match="GOOGLE_API_KEY"):
        build_adapters(settings)


def test_demo_mode_ignores_a_google_key_too() -> None:
    adapters, _ = build_adapters(Settings(demo_mode=True, google_api_key="g-x"))
    assert adapters.all_mock


def test_both_vision_providers_are_held_to_the_same_contract() -> None:
    """A shared prompt is what keeps one fixture valid for both providers."""
    import inspect

    from backend.adapters.docparser import VISION_SYSTEM_PROMPT
    from backend.adapters.gemini import GeminiDocumentParser

    source = inspect.getsource(GeminiDocumentParser)
    assert "VISION_SYSTEM_PROMPT" in source, "Gemini must not carry its own prompt"
    assert "NEVER guess" in VISION_SYSTEM_PROMPT


def test_gemini_reports_a_missing_package_as_recoverable_config() -> None:
    """`google-genai` is optional, so its absence degrades to the mock."""
    from backend.adapters.gemini import _client

    try:
        import google.genai  # noqa: F401
    except ImportError:
        with pytest.raises(AdapterError, match="google-genai"):
            _client("g-x")
    else:
        pytest.skip("google-genai is installed in this environment")
