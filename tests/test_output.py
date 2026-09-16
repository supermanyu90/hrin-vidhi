"""F4 — the spoken rights explainer. F6 — the grievance letter.

The load-bearing tests are the ones about what these outputs may *not* contain:
the letter cannot assert a violation the report did not make, and the LLM pass
cannot smuggle a section number past the verifier.
"""

from __future__ import annotations

import asyncio
import base64
import re

import pytest

from backend.adapters.base import LLM, AdapterError
from backend.adapters.docparser import MockDocumentParser
from backend.adapters.llm import MockLLM
from backend.adapters.tts import MockTextToSpeech
from backend.analysis.rules import RuleContext, evaluate
from backend.explainer.script import (
    build_script,
    disclaimer_for,
    explain_rights,
    load_phrasebook,
    select_messages,
)
from backend.finance import analyse_debt
from backend.grievance.drafter import (
    _verify,
    build_letter_text,
    build_summary,
    draft_grievance,
    enumerate_placeholders,
)
from backend.schemas import (
    ComplianceReport,
    ContactEvent,
    DetectedThreat,
    Language,
    NoticeFacts,
    NoticeType,
    ThreatCategory,
)

LANGUAGES = list(Language)


@pytest.fixture(scope="module")
def facts():
    loan = asyncio.run(MockDocumentParser().parse(b"", hint="loan paper")).loan_facts
    notice = asyncio.run(MockDocumentParser().parse(b"", hint="notice")).notice_facts
    debt = analyse_debt(
        loan.principal,
        loan.quoted_rate,
        loan.tenure_months,
        rate_type=loan.rate_type,
        processing_fee=loan.processing_fee or 0.0,
        other_fees_total=sum(f.amount or 0.0 for f in loan.other_fees),
        include_schedules=False,
    )
    report = evaluate(RuleContext(loan=loan, notice=notice, debt=debt))
    return loan, notice, debt, report


# ---------------------------------------------------------------------------
# F4 — phrasebook integrity
# ---------------------------------------------------------------------------


def test_every_message_exists_in_all_six_languages() -> None:
    book = load_phrasebook()
    assert len(book) >= 13
    expected = {lang.value for lang in LANGUAGES}
    for key, entry in book.items():
        assert set(entry) == expected, f"{key} is missing {expected ^ set(entry)}"
        for language, text in entry.items():
            assert text.strip(), f"{key}/{language} is empty"


def test_non_english_messages_are_in_their_own_script() -> None:
    """Catches an untranslated English string pasted into a vernacular slot."""
    book = load_phrasebook()
    for key, entry in book.items():
        for language, text in entry.items():
            if language == "en":
                continue
            stripped = re.sub(r"\{\w+\}", "", text)
            latin = [c for c in stripped if "a" <= c.lower() <= "z"]
            assert not latin, f"{key}/{language} contains Latin letters: {latin[:8]}"


def test_placeholders_match_across_languages() -> None:
    """A missing {extra} in one language would drop a figure from the script."""
    book = load_phrasebook()
    for key, entry in book.items():
        english = set(re.findall(r"\{(\w+)\}", entry["en"]))
        for language, text in entry.items():
            assert set(re.findall(r"\{(\w+)\}", text)) == english, f"{key}/{language}"


def test_every_mapped_rule_has_a_phrasebook_entry() -> None:
    from backend.explainer.script import MESSAGE_ORDER, RULE_TO_MESSAGE

    book = load_phrasebook()
    for _, message in RULE_TO_MESSAGE:
        assert message in book, message
        assert message in MESSAGE_ORDER, f"{message} would never be spoken"


# ---------------------------------------------------------------------------
# F4 — the script
# ---------------------------------------------------------------------------


def test_f4_acceptance_voice_note_in_the_selected_language(facts) -> None:
    """Spec F4: the F3 report yields an inline audio file in the selected language."""
    _, _, debt, report = facts
    explanation = asyncio.run(
        explain_rights(report, Language.MARATHI, debt, tts=MockTextToSpeech())
    )
    assert explanation.language is Language.MARATHI
    assert explanation.voice_note is not None
    raw = base64.b64decode(explanation.voice_note.audio_base64)
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    assert len(raw) > 1000


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_script_is_produced_in_every_language(facts, language: Language) -> None:
    _, _, debt, report = facts
    script, points = build_script(report, language, debt)
    assert script.strip()
    assert points
    if language is not Language.ENGLISH:
        assert not re.search(r"[a-z]{4,}", script.replace("percent", "")), language


def test_the_script_says_the_things_the_spec_asks_for(facts) -> None:
    """The register the spec names: contact hours, and agent-is-not-a-court."""
    _, _, debt, report = facts
    script, _ = build_script(report, Language.ENGLISH, debt)
    assert "before eight in the morning or after seven at night" in script
    assert "A recovery agent is not a court" in script
    assert "not a crime" in script


def test_the_script_quotes_only_computed_figures(facts) -> None:
    _, _, debt, report = facts
    script, _ = build_script(report, Language.ENGLISH, debt)
    assert f"{debt.quoted_rate:g} percent" in script
    assert f"{debt.all_in_apr:.0f} percent" in script
    assert "14,169" in script  # the computed hidden-cost gap


def test_no_money_sentence_when_the_maths_did_not_run(facts) -> None:
    """Rather than speak a figure we do not have, the sentence is dropped."""
    _, _, _, report = facts
    script, _ = build_script(report, Language.ENGLISH, debt=None)
    assert "percent" not in script
    assert script.strip()


def test_only_fired_rules_are_spoken() -> None:
    """A right the borrower's facts did not raise is not read out to them."""
    notice = NoticeFacts(contact_times_mentioned=[ContactEvent(time_24h="06:30")])
    report = evaluate(RuleContext(notice=notice))
    assert select_messages(report) == ["recovery_hours"]

    script, _ = build_script(report, Language.ENGLISH)
    assert "before eight in the morning" in script
    assert "not a court" not in script


def test_a_clean_report_says_so_without_claiming_all_is_well() -> None:
    script, points = build_script(ComplianceReport(overall_summary=""), Language.ENGLISH)
    assert "did not find a clear problem" in script
    assert "does not mean everything is fine" in script
    assert points == []


def test_the_disclaimer_is_spoken_and_returned(facts) -> None:
    """§9: the disclaimer is both spoken and shown, in the borrower's language."""
    _, _, debt, report = facts
    for language in LANGUAGES:
        explanation = asyncio.run(
            explain_rights(report, language, debt, tts=MockTextToSpeech())
        )
        assert explanation.disclaimer_native.strip()
        assert explanation.disclaimer_english.strip()
        # The spoken text is script + disclaimer.
        assert explanation.disclaimer_native in explanation.voice_note.text


def test_the_disclaimer_matches_the_shared_phrasebook() -> None:
    from backend.adapters.translate import PHRASEBOOK

    for language in LANGUAGES:
        assert disclaimer_for(language) == PHRASEBOOK["DISCLAIMER"][language]


def test_a_failing_tts_still_returns_the_script(facts) -> None:
    from backend.adapters.base import TextToSpeech

    class BrokenTTS(TextToSpeech):
        provider = "broken"

        async def synthesize(self, text, language):
            raise AdapterError("broken", "no voice available")

    _, _, debt, report = facts
    explanation = asyncio.run(explain_rights(report, Language.HINDI, debt, tts=BrokenTTS()))
    assert explanation.voice_note is None
    assert explanation.script_native.strip()


# ---------------------------------------------------------------------------
# F6 — the letter
# ---------------------------------------------------------------------------


def test_f6_acceptance_letter_names_violations_with_citations_and_both_addressees(facts) -> None:
    """Spec F6: violations with citations, Nodal Officer + RB-IOS addressing,
    and no placeholder left unlabelled."""
    loan, notice, debt, report = facts
    letter = asyncio.run(draft_grievance(report, loan, notice, debt, Language.MARATHI, llm=MockLLM()))

    for violation in report.violations:
        assert violation.title in letter.body_english, violation.rule_id
        for citation in violation.citations:
            assert citation.citation in letter.body_english

    roles = " ".join(a.role for a in letter.addressees)
    assert "Nodal Officer" in roles
    assert "Ombudsman" in roles
    assert "cms.rbi.org.in" in letter.body_english

    # Every placeholder is a self-describing instruction, not a bare [NAME].
    assert letter.placeholders
    for placeholder in letter.placeholders:
        assert len(placeholder) > 8, placeholder
        assert placeholder.isupper() or " " in placeholder


def test_the_two_acceptance_violations_reach_the_letter() -> None:
    """The F3 fixture — a 06:30 call and a family threat — named in the letter."""
    notice = NoticeFacts(
        sender="Some Recovery Agency",
        notice_type=NoticeType.RECOVERY,
        contact_times_mentioned=[ContactEvent(time_24h="06:30", channel="call")],
        threats_detected=[
            DetectedThreat(category=ThreatCategory.FAMILY_CONTACT, quote="We will tell your family.")
        ],
    )
    report = evaluate(RuleContext(notice=notice))
    letter = asyncio.run(draft_grievance(report, notice=notice, llm=MockLLM()))

    assert set(letter.violations_cited) == {"R-001-recovery-hours", "R-002-threat-family-contact"}
    assert "06:30" in letter.body_english
    assert "We will tell your family." in letter.body_english
    assert letter.citations


def test_the_letter_asserts_nothing_the_report_did_not(facts) -> None:
    """The structural guarantee: violations_cited can never exceed the report."""
    loan, notice, debt, report = facts
    letter = asyncio.run(draft_grievance(report, loan, notice, debt, llm=MockLLM()))
    assert set(letter.violations_cited) <= {v.rule_id for v in report.violations}


def test_an_empty_report_produces_a_letter_with_no_accusations() -> None:
    letter = asyncio.run(draft_grievance(ComplianceReport(overall_summary=""), llm=MockLLM()))
    assert letter.violations_cited == []
    assert "WHY I SAY THIS IS NOT PERMITTED" not in letter.body_english
    assert "WHAT I ASK YOU TO DO" in letter.body_english


def test_threat_quotes_are_reproduced_verbatim_in_the_letter(facts) -> None:
    _, notice, _, report = facts
    letter = asyncio.run(draft_grievance(report, notice=notice, llm=MockLLM()))
    for threat in notice.threats_detected:
        assert threat.quote in letter.body_english


def test_identical_reliefs_are_not_repeated(facts) -> None:
    """Three threat violations share one relief; asking a regulator for the
    same thing three times reads as careless."""
    loan, notice, debt, report = facts
    body = build_letter_text(report, loan, notice, debt)
    section = body[body.index("WHAT I ASK YOU TO DO"):body.index("IF I DO NOT HEAR")]
    asks = [re.sub(r"^\s*\d+\.\d+\s*", "", x).strip() for x in section.splitlines() if re.match(r"\s*\d+\.\d+", x)]
    assert asks
    assert len(asks) == len(set(asks)), "duplicate relief in the letter"


def test_flags_are_put_to_the_lender_not_the_borrower(facts) -> None:
    """`flag.action` is borrower-facing advice; echoing it at the lender
    ('verify this lender on RBI's list') would be nonsense."""
    loan, notice, debt, report = facts
    body = build_letter_text(report, loan, notice, debt)
    assert "What I ask you to state:" in body
    for flag in report.flags:
        assert flag.action not in body, flag.flag_id
        if flag.lender_ask:
            assert flag.lender_ask in body, flag.flag_id


def test_flags_are_framed_as_questions_not_accusations(facts) -> None:
    loan, notice, debt, report = facts
    body = build_letter_text(report, loan, notice, debt)
    assert "I do not allege wrongdoing on these points" in body


def test_the_letter_admits_its_references_are_summaries(facts) -> None:
    loan, notice, debt, report = facts
    body = build_letter_text(report, loan, notice, debt)
    assert "plain-language summaries" in body
    assert "I am not a lawyer" in body


def test_the_chronology_only_contains_extracted_facts(facts) -> None:
    loan, notice, debt, report = facts
    body = build_letter_text(report, loan, notice, debt)
    chronology = body[body.index("2. WHAT HAPPENED"):body.index("3. WHY I SAY")]
    for event in notice.contact_times_mentioned:
        assert event.time_24h in chronology


def test_the_borrower_name_is_filled_in_when_supplied(facts) -> None:
    loan, notice, debt, report = facts
    letter = asyncio.run(
        draft_grievance(report, loan, notice, debt, borrower_name="Ramesh Kumbhar", llm=MockLLM())
    )
    assert "Ramesh Kumbhar" in letter.body_english
    assert "[YOUR FULL NAME]" not in letter.placeholders


def test_the_filename_is_derived_from_the_lender(facts) -> None:
    loan, notice, debt, report = facts
    letter = asyncio.run(draft_grievance(report, loan, notice, debt, llm=MockLLM()))
    assert letter.filename_suggestion.endswith(".txt")
    assert "sahyadri" in letter.filename_suggestion


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_vernacular_summary_is_produced_in_every_language(facts, language: Language) -> None:
    _, _, debt, report = facts
    summary = build_summary(report, language, debt)
    assert summary.strip()
    if language is not Language.ENGLISH:
        assert not re.search(r"[a-z]{4,}", summary.replace("percent", "")), language


# ---------------------------------------------------------------------------
# F6 — the LLM guardrail
# ---------------------------------------------------------------------------


def test_placeholder_enumeration_is_ordered_and_deduplicated() -> None:
    text = "[FIRST THING] then [SECOND THING] then [FIRST THING] again"
    assert enumerate_placeholders(text) == ["[FIRST THING]", "[SECOND THING]"]


@pytest.mark.parametrize(
    "edited, reason",
    [
        ("", "empty"),
        ("Under Section 420 you must pay.", "section"),
        ("short", "short"),
    ],
)
def test_the_verifier_rejects_a_bad_edit(edited: str, reason: str) -> None:
    draft = (
        "1. ABOUT ME\n   [YOUR FULL NAME]\n"
        "        Reference relied on: RBI Fair Practices Code — recovery hours\n" * 20
    )
    assert _verify(draft, edited) is not None, reason


def test_the_verifier_accepts_a_genuine_tightening() -> None:
    draft = (
        "1. ABOUT ME AND THIS LOAN\n"
        "   1.1 I am a borrower of your company and I am writing to you today.\n"
        "   1.2 My account is [YOUR LOAN ACCOUNT NUMBER].\n"
        "        Reference relied on: RBI Fair Practices Code — recovery hours\n"
    )
    edited = (
        "1. ABOUT ME AND THIS LOAN\n"
        "   1.1 I am a borrower of your company and I write to you today.\n"
        "   1.2 My account is [YOUR LOAN ACCOUNT NUMBER].\n"
        "        Reference relied on: RBI Fair Practices Code — recovery hours\n"
    )
    assert _verify(draft, edited) is None


def test_an_llm_that_invents_a_section_is_discarded(facts) -> None:
    """The guardrail that matters most: a fabricated citation must never
    reach a letter addressed to a regulator."""

    class FabricatingLLM(LLM):
        provider = "fabricator"

        async def complete(self, prompt, *, system=None, max_tokens=2048, effort="medium"):
            draft = prompt.split("DRAFT:", 1)[1]
            return draft.replace(
                "Sir / Madam,",
                "Sir / Madam,\n\nThis is an offence under Section 420 of the Indian Penal Code.",
            )

    loan, notice, debt, report = facts
    letter = asyncio.run(draft_grievance(report, loan, notice, debt, llm=FabricatingLLM()))
    assert "Section 420" not in letter.body_english
    assert "Indian Penal Code" not in letter.body_english


def test_an_llm_that_drops_a_citation_is_discarded(facts) -> None:
    class CitationDroppingLLM(LLM):
        provider = "dropper"

        async def complete(self, prompt, *, system=None, max_tokens=2048, effort="medium"):
            draft = prompt.split("DRAFT:", 1)[1]
            return "\n".join(
                line for line in draft.splitlines() if "Reference relied on" not in line
            )

    loan, notice, debt, report = facts
    letter = asyncio.run(draft_grievance(report, loan, notice, debt, llm=CitationDroppingLLM()))
    assert "Reference relied on" in letter.body_english


def test_a_failing_llm_still_returns_the_deterministic_letter(facts) -> None:
    class BrokenLLM(LLM):
        provider = "broken"

        async def complete(self, prompt, *, system=None, max_tokens=2048, effort="medium"):
            raise AdapterError("broken", "provider is down")

    loan, notice, debt, report = facts
    letter = asyncio.run(draft_grievance(report, loan, notice, debt, llm=BrokenLLM()))
    assert letter.body_english.strip()
    assert letter.violations_cited


def test_the_summary_does_not_repeat_itself(facts) -> None:
    """The intro and the follow-up steps once shared an opening sentence, so
    the borrower was told twice that a letter had been written for them."""
    import re as _re

    _, _, debt, report = facts
    for language in LANGUAGES:
        summary = build_summary(report, language, debt)
        sentences = [s.strip() for s in _re.split(r"(?<=[.।!?]) ", summary) if s.strip()]
        duplicates = {s for s in sentences if sentences.count(s) > 1}
        assert not duplicates, f"{language.value}: {duplicates}"


def test_the_summary_covers_what_the_borrower_must_do(facts) -> None:
    _, _, debt, report = facts
    summary = build_summary(report, Language.ENGLISH, debt)
    assert "square brackets" in summary          # fill in the blanks
    assert "grievance officer" in summary        # where to send it first
    assert "thirty days" in summary              # when to escalate
    assert "Ombudsman" in summary
