"""F3 — corpus integrity, retrieval, deterministic rules, and grounded Q&A.

The load-bearing tests here are the ones asserting what the system *refuses*
to say: no citation means no assertion, and low retrieval confidence means
"couldn't confirm" rather than a plausible-sounding invention.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from backend.adapters.docparser import MockDocumentParser
from backend.adapters.llm import MockLLM
from backend.analysis import rules as rules_module
from backend.analysis.citations import resolve
from backend.analysis.corpus_store import load_chunks
from backend.analysis.rag import COULD_NOT_CONFIRM, answer_question
from backend.analysis.retriever import BM25Retriever, get_retriever, tokenize
from backend.analysis.rules import RuleContext, evaluate
from backend.finance import analyse_debt
from backend.schemas import (
    ContactEvent,
    DetectedThreat,
    Fee,
    LoanFacts,
    NBFCRegistrationStatus,
    NoticeFacts,
    NoticeType,
    RateType,
    Severity,
    ThreatCategory,
)


@pytest.fixture(scope="module")
def loan() -> LoanFacts:
    parsed = asyncio.run(MockDocumentParser().parse(b"", hint="loan paper"))
    assert parsed.loan_facts is not None
    return parsed.loan_facts


@pytest.fixture(scope="module")
def notice() -> NoticeFacts:
    parsed = asyncio.run(MockDocumentParser().parse(b"", hint="notice"))
    assert parsed.notice_facts is not None
    return parsed.notice_facts


@pytest.fixture(scope="module")
def full_report(loan: LoanFacts, notice: NoticeFacts):
    debt = analyse_debt(
        loan.principal,
        loan.quoted_rate,
        loan.tenure_months,
        rate_type=loan.rate_type,
        processing_fee=loan.processing_fee or 0.0,
        other_fees_total=sum(f.amount or 0.0 for f in loan.other_fees),
        include_schedules=False,
    )
    return evaluate(RuleContext(loan=loan, notice=notice, debt=debt))


# ---------------------------------------------------------------------------
# Corpus integrity
# ---------------------------------------------------------------------------


def test_corpus_loads() -> None:
    assert len(load_chunks()) >= 25


def test_chunk_ids_are_unique() -> None:
    ids = [c.chunk_id for c in load_chunks()]
    assert len(ids) == len(set(ids))


def test_every_chunk_is_marked_a_summary() -> None:
    """Seeded content is paraphrase, never statutory text. The UI says so."""
    assert all(c.is_summary for c in load_chunks())


def test_no_chunk_claims_to_be_lawyer_reviewed() -> None:
    """`verified_by` is the hook for vetted content; nothing may preset it."""
    assert all(c.verified_by is None for c in load_chunks())


def test_every_chunk_carries_source_and_citation() -> None:
    for chunk in load_chunks():
        assert chunk.source.strip(), chunk.chunk_id
        assert chunk.citation.strip(), chunk.chunk_id
        assert len(chunk.text) > 80, chunk.chunk_id


def test_all_five_required_sources_are_seeded() -> None:
    """Spec §5 names five bodies of reference material; all must be present."""
    blob = " ".join(f"{c.source} {c.citation}" for c in load_chunks()).lower()
    for required in (
        "digital lending",
        "fair practices",
        "consumer protection act, 2019",
        "negotiable instruments act",
        "integrated ombudsman",
    ):
        assert required in blob, required


def test_section_numbers_appear_only_where_we_claim_certainty() -> None:
    """Guards §4's 'never fabricate a statute or section number'.

    Any chunk asserting a bare 'Section N' must be one of the few we vouch
    for. A new chunk that invents one fails here rather than in a letter.
    """
    import re

    allowed = {
        "ni-138-what-it-covers",
        "ni-138-notice-and-time",
        "ni-138-what-is-not-a-138-case",
        "cpa-unfair-trade-practice",
        "cpa-consumer-rights",
    }
    pattern = re.compile(r"\bSection\s+\d", re.IGNORECASE)
    offenders = [
        c.chunk_id
        for c in load_chunks()
        if c.chunk_id not in allowed and pattern.search(f"{c.citation} {c.text}")
    ]
    assert not offenders, f"unvouched section numbers in {offenders}"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_tokenizer_drops_stopwords_and_single_characters() -> None:
    assert tokenize("The agent called me at 6 a.m.") == ["agent", "call"]


def test_tokenizer_stems_so_inflections_match() -> None:
    """Without this, a borrower asking about 'calls' misses text saying 'call'."""
    for pair in [("calls", "call"), ("called", "call"), ("charges", "charge"),
                 ("threatened", "threaten"), ("agencies", "agency")]:
        assert tokenize(pair[0]) == tokenize(pair[1]), pair


def test_stemmer_handles_short_words_without_mangling_them() -> None:
    """Over-eager stemming would collapse unrelated short words together."""
    assert tokenize("fees") == ["fee"]
    assert tokenize("does") == []  # stopword


@pytest.mark.parametrize(
    "query, expected_chunk",
    [
        ("agent called me at 6 in the morning", "rbi-fpc-recovery-hours"),
        ("they will tell my wife's family about the loan", "rbi-fpc-third-party-contact"),
        ("is a court notice from a recovery agency genuine", "ni-138-court-notice-vs-private-letter"),
        ("what is the real interest on a flat rate loan", "rbi-dl-apr-disclosure"),
        ("how do I complain to the RBI ombudsman", "rbios-what-it-can-address"),
        ("can they arrest me for not paying", "ni-138-no-arrest-for-debt"),
        ("cheque bounce notice how many days", "ni-138-notice-and-time"),
        ("they did not tell me about the processing fee", "rbi-dl-no-hidden-charges"),
    ],
)
def test_retrieval_finds_the_right_chunk(query: str, expected_chunk: str) -> None:
    hits = get_retriever().search(query, 3)
    assert hits, query
    assert expected_chunk in {h.chunk.chunk_id for h in hits}, (
        query,
        [h.chunk.chunk_id for h in hits],
    )


def test_scores_are_normalised_and_descending() -> None:
    hits = get_retriever().search("recovery agent called at night", 5)
    assert hits
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_off_topic_query_retrieves_nothing_usable() -> None:
    hits = get_retriever().search("what is the capital of France", 4)
    assert not hits or max(h.score for h in hits) < 0.3


def test_empty_corpus_retrieves_nothing_rather_than_crashing() -> None:
    assert BM25Retriever(()).search("anything at all", 4) == []


# ---------------------------------------------------------------------------
# Citations — the "no citation, no assertion" guarantee
# ---------------------------------------------------------------------------


def test_resolve_reports_unknown_chunk_ids() -> None:
    citations, missing = resolve(["rbi-fpc-recovery-hours", "does-not-exist"])
    assert [c.chunk_id for c in citations] == ["rbi-fpc-recovery-hours"]
    assert missing == ["does-not-exist"]


def test_every_rule_citation_resolves(full_report) -> None:
    """If a rule cites a chunk that does not exist, it is silently demoted to
    `unconfirmed` and vanishes from the letter. This test is what stops that
    happening quietly after a corpus edit."""
    assert full_report.unconfirmed == []


def test_a_rule_citing_a_missing_chunk_is_demoted_not_asserted(monkeypatch) -> None:
    """The structural guarantee, exercised directly."""

    def broken_rule(ctx: RuleContext) -> list[rules_module.Finding]:
        return [
            rules_module.Finding(
                rule_id="R-999-test",
                title="Invented violation",
                severity=Severity.CRITICAL,
                offending_fact="fact",
                explanation="explanation",
                chunk_ids=["chunk-that-does-not-exist"],
                relief_sought="relief",
            )
        ]

    monkeypatch.setattr(rules_module, "ALL_RULES", (broken_rule,))
    report = evaluate(RuleContext())
    assert report.violations == []
    assert report.flags == []
    assert len(report.unconfirmed) == 1
    assert "chunk-that-does-not-exist" in report.unconfirmed[0]


def test_report_citations_are_deduplicated(full_report) -> None:
    ids = [c.chunk_id for c in full_report.citations]
    assert len(ids) == len(set(ids))


def test_every_violation_carries_at_least_one_citation(full_report) -> None:
    for violation in full_report.violations:
        assert violation.citations, violation.rule_id
        assert violation.relief_sought.strip(), violation.rule_id


def test_every_flag_carries_an_action(full_report) -> None:
    for flag in full_report.flags:
        assert flag.citations, flag.flag_id
        assert flag.action.strip(), flag.flag_id


# ---------------------------------------------------------------------------
# F3 acceptance
# ---------------------------------------------------------------------------


def test_f3_acceptance_early_call_and_family_threat() -> None:
    """Spec F3: an agent calling at 06:30 who threatened the family produces
    two violations, each with a Fair Practices Code citation."""
    notice = NoticeFacts(
        sender="Some Recovery Agency",
        notice_type=NoticeType.RECOVERY,
        contact_times_mentioned=[ContactEvent(time_24h="06:30", channel="call")],
        threats_detected=[
            DetectedThreat(
                category=ThreatCategory.FAMILY_CONTACT,
                quote="We will inform your wife's family.",
            )
        ],
    )
    report = evaluate(RuleContext(notice=notice))

    rule_ids = {v.rule_id for v in report.violations}
    assert "R-001-recovery-hours" in rule_ids
    assert "R-002-threat-family-contact" in rule_ids
    assert len(report.violations) == 2

    for violation in report.violations:
        sources = " ".join(c.source for c in violation.citations).lower()
        assert "fair practices code" in sources, violation.rule_id


def test_the_seeded_story_produces_both_violations_and_flags(full_report) -> None:
    assert len(full_report.violations) >= 5
    assert len(full_report.flags) >= 3
    assert full_report.overall_summary
    assert "not a legal finding" in full_report.overall_summary


# ---------------------------------------------------------------------------
# Individual rules — each must fire, and each must stay silent when it should
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("time_24h", ["06:30", "07:59", "19:00", "21:45", "23:30", "00:15"])
def test_contacts_outside_the_window_are_violations(time_24h: str) -> None:
    notice = NoticeFacts(contact_times_mentioned=[ContactEvent(time_24h=time_24h)])
    report = evaluate(RuleContext(notice=notice))
    assert any(v.rule_id == "R-001-recovery-hours" for v in report.violations), time_24h


@pytest.mark.parametrize("time_24h", ["08:00", "12:00", "18:59"])
def test_contacts_inside_the_window_are_not_violations(time_24h: str) -> None:
    notice = NoticeFacts(contact_times_mentioned=[ContactEvent(time_24h=time_24h)])
    report = evaluate(RuleContext(notice=notice))
    assert not any(v.rule_id == "R-001-recovery-hours" for v in report.violations), time_24h


def test_multiple_out_of_hours_contacts_collapse_into_one_violation() -> None:
    """One rule, one finding — listing the same breach twice would inflate a letter."""
    notice = NoticeFacts(
        contact_times_mentioned=[
            ContactEvent(time_24h="06:30", on_date=date(2026, 9, 14)),
            ContactEvent(time_24h="21:45", on_date=date(2026, 9, 12)),
        ]
    )
    report = evaluate(RuleContext(notice=notice))
    hours = [v for v in report.violations if v.rule_id == "R-001-recovery-hours"]
    assert len(hours) == 1
    assert "06:30" in hours[0].offending_fact and "21:45" in hours[0].offending_fact


def test_arrest_and_violence_threats_are_critical() -> None:
    for category in (ThreatCategory.ARREST_CLAIM, ThreatCategory.VIOLENCE):
        notice = NoticeFacts(
            threats_detected=[DetectedThreat(category=category, quote="threatening words")]
        )
        report = evaluate(RuleContext(notice=notice))
        assert report.violations
        assert report.violations[0].severity is Severity.CRITICAL, category


def test_threat_quotes_are_reproduced_verbatim() -> None:
    """The offending words go into a regulator's letter; paraphrase would be a liability."""
    quote = "Our men will visit your village and announce your default."
    notice = NoticeFacts(
        threats_detected=[DetectedThreat(category=ThreatCategory.PUBLIC_SHAMING, quote=quote)]
    )
    report = evaluate(RuleContext(notice=notice))
    assert any(quote in v.offending_fact for v in report.violations)


def test_fake_court_notice_fires_only_when_disproved() -> None:
    base = dict(notice_type=NoticeType.COURT_NOTICE_CLAIM, sender="Vasuli Associates")

    disproved = evaluate(RuleContext(notice=NoticeFacts(**base, sender_is_court=False)))
    assert any(v.rule_id == "R-003-court-notice-misrepresentation" for v in disproved.violations)

    # `None` means the parser could not tell — an unproven misrepresentation
    # must not be asserted.
    unknown = evaluate(RuleContext(notice=NoticeFacts(**base, sender_is_court=None)))
    assert not any(v.rule_id == "R-003-court-notice-misrepresentation" for v in unknown.violations)

    genuine = evaluate(RuleContext(notice=NoticeFacts(**base, sender_is_court=True)))
    assert not any(v.rule_id == "R-003-court-notice-misrepresentation" for v in genuine.violations)


def test_undisclosed_fees_are_a_violation_but_disclosed_ones_are_not() -> None:
    hidden = LoanFacts(
        other_fees=[Fee(label="Documentation charges", amount=750, disclosed_in_sanction_letter=False)]
    )
    assert any(
        v.rule_id == "R-004-undisclosed-charges"
        for v in evaluate(RuleContext(loan=hidden)).violations
    )

    disclosed = LoanFacts(
        other_fees=[Fee(label="Documentation charges", amount=750, disclosed_in_sanction_letter=True)]
    )
    assert not any(
        v.rule_id == "R-004-undisclosed-charges"
        for v in evaluate(RuleContext(loan=disclosed)).violations
    )

    # Unknown disclosure status must not become an accusation.
    unknown = LoanFacts(other_fees=[Fee(label="Documentation charges", amount=750)])
    assert not any(
        v.rule_id == "R-004-undisclosed-charges"
        for v in evaluate(RuleContext(loan=unknown)).violations
    )


def test_apr_gap_is_a_flag_not_a_violation(loan: LoanFacts) -> None:
    """Spec F3 lists the rate illusion as a flag; asserting it as a breach
    would overstate what the documents prove."""
    debt = analyse_debt(80_000, 12.0, 24, include_schedules=False)
    report = evaluate(RuleContext(loan=loan, debt=debt))
    assert any(f.flag_id == "F-001-apr-understated" for f in report.flags)
    assert not any(v.rule_id == "F-001-apr-understated" for v in report.violations)


def test_an_honest_reducing_quote_raises_no_apr_flag() -> None:
    honest = LoanFacts(
        principal=80_000, quoted_rate=12.0, rate_type=RateType.REDUCING, tenure_months=24
    )
    debt = analyse_debt(80_000, 12.0, 24, rate_type=RateType.REDUCING, include_schedules=False)
    report = evaluate(RuleContext(loan=honest, debt=debt))
    assert not any(f.flag_id == "F-001-apr-understated" for f in report.flags)


def test_verified_nbfc_registration_raises_no_flag() -> None:
    verified = LoanFacts(
        lender_name="Some Finserv",
        nbfc_registration_no="N-13.01234",
        nbfc_registration_status=NBFCRegistrationStatus.VERIFIED,
    )
    report = evaluate(RuleContext(loan=verified))
    assert not any(f.flag_id == "F-003-nbfc-registration-unverified" for f in report.flags)


def test_absent_nbfc_registration_flags_and_says_to_check_rbis_list() -> None:
    report = evaluate(RuleContext(loan=LoanFacts(lender_name="Sahyadri Finserv")))
    flags = [f for f in report.flags if f.flag_id == "F-003-nbfc-registration-unverified"]
    assert flags
    assert "rbi" in flags[0].action.lower()


def test_empty_context_produces_a_clean_but_honest_report() -> None:
    report = evaluate(RuleContext())
    assert report.violations == []
    assert report.flags == []
    assert "not a clean bill of health" in report.overall_summary


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


def test_grounded_answer_cites_what_it_used() -> None:
    answer = asyncio.run(answer_question("Can they call me at 6 in the morning?"))
    assert answer.grounded
    assert answer.citations
    assert "rbi-fpc-recovery-hours" in {c.chunk_id for c in answer.citations}
    for citation in answer.citations:
        assert citation.retrieval_score is not None


def test_answer_text_is_drawn_from_the_retrieved_chunks() -> None:
    """Nothing in the answer may be invented: every sentence traces to a quote."""
    answer = asyncio.run(answer_question("Can a recovery agent have me arrested?"))
    assert answer.grounded
    assert any(c.quote[:60] in answer.answer for c in answer.citations)


def test_a_narrow_question_cites_only_the_rule_that_answers_it() -> None:
    """The relative gate: marginal chunks must not pad a borrower-facing answer."""
    answer = asyncio.run(answer_question("Can they call me at 6 in the morning?"))
    assert answer.grounded
    assert [c.chunk_id for c in answer.citations] == ["rbi-fpc-recovery-hours"]


def test_a_broad_question_may_legitimately_cite_several_chunks() -> None:
    """The gate is relative, not a cap — a question with several on-topic rules
    should still get all of them."""
    answer = asyncio.run(answer_question("How do I complain to the RBI ombudsman?"))
    assert answer.grounded
    assert len(answer.citations) >= 3
    assert all(c.chunk_id.startswith("rbios-") for c in answer.citations)


def test_low_confidence_says_so_instead_of_guessing() -> None:
    answer = asyncio.run(answer_question("What is the capital of France?"))
    assert not answer.grounded
    assert answer.answer == COULD_NOT_CONFIRM
    assert answer.citations == []


def test_ungrounded_answer_names_no_statute() -> None:
    """The failure mode this whole design exists to prevent."""
    answer = asyncio.run(answer_question("Who won the cricket match last night?"))
    lowered = answer.answer.lower()
    assert "section" not in lowered
    assert "act" not in lowered.replace("contact", "")


def test_the_mock_llm_returns_the_grounded_draft_unchanged() -> None:
    plain = asyncio.run(answer_question("Can they call me at 6 in the morning?"))
    polished = asyncio.run(
        answer_question("Can they call me at 6 in the morning?", llm=MockLLM())
    )
    assert polished.answer == plain.answer
    assert polished.grounded


def test_a_broken_llm_falls_back_to_the_deterministic_draft() -> None:
    from backend.adapters.base import LLM, AdapterError

    class BrokenLLM(LLM):
        provider = "broken"

        async def complete(self, prompt, *, system=None, max_tokens=2048, effort="medium"):
            raise AdapterError("broken", "provider is down")

    answer = asyncio.run(answer_question("Can they call me at 6 in the morning?", llm=BrokenLLM()))
    assert answer.grounded
    assert answer.citations
    assert "8:00" in answer.answer or "8 " in answer.answer


def test_answers_carry_the_summary_caveat() -> None:
    answer = asyncio.run(answer_question("What is a key fact statement?"))
    assert "summaries" in answer.answer.lower() or "checked" in answer.answer.lower()
