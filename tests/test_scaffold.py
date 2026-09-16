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


def test_health_reports_all_mocks_in_demo_mode(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["demo_mode"] is True
    assert set(body["adapters"]) == {"stt", "translate", "tts", "docparser", "llm"}
    assert all(v == "mock" for v in body["adapters"].values()), body["adapters"]


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
    passthrough = asyncio.run(adapters.llm.complete("TASK: rights_script\nDRAFT:\nThey cannot call."))
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
