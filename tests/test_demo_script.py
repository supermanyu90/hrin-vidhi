"""The demo script makes specific factual claims. This keeps them true.

`docs/DEMO.md` quotes figures, Marathi button labels and test counts. A judge
reads those off a screen while the presenter says them aloud, so a number that
drifts is worse than a missing one.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from backend.adapters.docparser import MockDocumentParser
from backend.adapters.llm import MockLLM
from backend.analysis.corpus_store import load_chunks
from backend.analysis.rules import RuleContext, evaluate
from backend.config import FRONTEND_DIR, PROJECT_ROOT
from backend.finance import analyse_debt
from backend.grievance.drafter import draft_grievance
from backend.schemas import Language

DEMO = (PROJECT_ROOT / "docs" / "DEMO.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def story():
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
    return loan, notice, debt, evaluate(RuleContext(loan=loan, notice=notice, debt=debt))


def test_the_demo_file_exists_and_is_substantial() -> None:
    assert len(DEMO) > 4000
    for heading in ("Before you present", "The story", "The script",
                    "If something goes wrong", "Questions judges ask"):
        assert heading in DEMO, heading


def test_the_quoted_money_figures_are_what_the_system_produces(story) -> None:
    loan, _, debt, _ = story
    assert "₹80,000" in DEMO and loan.principal == 80_000
    assert "28.94%" in DEMO and f"{debt.all_in_apr:.2f}%" == "28.94%"
    assert "21.57%" in DEMO and f"{debt.effective_apr:.2f}%" == "21.57%"
    assert "₹14,169" in DEMO and f"{debt.hidden_cost_gap:,.0f}" == "14,169"
    assert "₹5,350" in DEMO
    assert (loan.processing_fee or 0) + sum(f.amount or 0 for f in loan.other_fees) == 5_350


WORDS = {"Five": 5, "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10}


def test_the_quoted_violation_count_is_right(story) -> None:
    _, _, _, report = story
    match = re.search(r"\b(\w+) violations\b", DEMO)
    assert match, "DEMO.md no longer states a violation count"
    word = match.group(1)
    claimed = WORDS.get(word, int(word) if word.isdigit() else None)
    assert claimed == len(report.violations), f"DEMO.md says {word}, engine finds {len(report.violations)}"


def test_the_quoted_placeholder_count_is_right(story) -> None:
    loan, notice, debt, report = story
    letter = asyncio.run(draft_grievance(report, loan, notice, debt, Language.MARATHI, llm=MockLLM()))
    assert "Ten blanks" in DEMO
    assert len(letter.placeholders) == 10


def test_the_quoted_corpus_size_is_right() -> None:
    assert '"chunk_count": 31' in DEMO
    assert len(load_chunks()) == 31


def test_the_quoted_test_count_matches_the_suite() -> None:
    """Asks pytest itself how many tests there are, rather than guessing.

    The presenter offers `pytest -q` as a recovery move in front of judges, so
    the number they say out loud has to be the number that appears.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(PROJECT_ROOT / "tests")],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    assert match, result.stdout[-500:]
    actual = int(match.group(1))

    claimed = int(re.search(r"(\d+) tests, ~", DEMO).group(1))
    assert claimed == actual, f"DEMO.md claims {claimed} tests; pytest collects {actual}"

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{actual} tests" in readme, f"README.md does not mention {actual} tests"


def test_every_marathi_label_the_script_tells_you_to_click_exists() -> None:
    """The presenter reads these off the page while pointing at a button."""
    i18n = (FRONTEND_DIR / "i18n.js").read_text(encoding="utf-8")
    for label in re.findall(r"\*\*([^\*]*[ऀ-ॿ][^\*]*)\*\*", DEMO):
        for part in re.split(r"\s*(?:→|·)\s*", label):
            part = part.strip()
            if any("ऀ" <= ch <= "ॿ" for ch in part):
                assert part in i18n, f"{part!r} is in DEMO.md but not in the UI"


def test_the_disclaimer_promise_holds() -> None:
    """The script closes on the disclaimer; it must really be everywhere."""
    from backend.adapters.translate import PHRASEBOOK
    from backend.explainer.script import disclaimer_for

    assert "not legal advice" in DEMO
    for language in Language:
        assert disclaimer_for(language) == PHRASEBOOK["DISCLAIMER"][language]


def test_the_stated_limitations_are_still_true() -> None:
    """If one of these gets implemented, the honesty section must be updated."""
    assert "Auto-detect of spoken language** is not implemented" in DEMO
    routes = (PROJECT_ROOT / "backend" / "routes" / "intake.py").read_text()
    assert "auto" not in routes.lower().split("language is declared")[0][-400:]

    assert "WAV, not MP3" in DEMO
    assert all(c.verified_by is None for c in load_chunks()), (
        "a chunk is now lawyer-reviewed; update DEMO.md"
    )


def test_the_recovery_commands_are_real() -> None:
    assert "curl -s localhost:8000/health" in DEMO
    assert "curl -s localhost:8000/corpus/status" in DEMO
    assert "pytest -q" in DEMO
    assert "DEMO_MODE=true python -m backend.main" in DEMO
