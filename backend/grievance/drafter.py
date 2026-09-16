"""F6 — assembling the grievance letter.

Deterministic skeleton first, LLM second and only to tighten prose. The
drafter can only say what the `ComplianceReport` already says: it walks
`report.violations` and `report.flags` and nothing else, so a finding the
rules engine demoted to `unconfirmed` (because its citation did not resolve)
cannot reach the letter.

The LLM pass is **verified, not trusted**. Its output is checked for dropped
citations and for invented section numbers, and any failure falls back to the
deterministic draft. See `_verify`.
"""

from __future__ import annotations

import logging
import re

from backend.adapters.base import LLM, AdapterError
from backend.explainer.script import build_script, disclaimer_for, phrase
from backend.grievance.templates import (
    BORROWER_BLOCK,
    CITATION_CAVEAT,
    DRAFTER_SYSTEM_PROMPT,
    ENCLOSURES,
    ESCALATION_NOTE,
    SIGN_OFF,
    SUBJECT_TEMPLATE,
    nodal_officer,
    rbi_ombudsman,
)
from backend.schemas import (
    Citation,
    ComplianceReport,
    DebtAnalysis,
    GrievanceLetter,
    Language,
    LoanFacts,
    NoticeFacts,
)

log = logging.getLogger(__name__)

PLACEHOLDER_RE = re.compile(r"\[([A-Z][^\[\]]{3,})\]")
SECTION_RE = re.compile(r"\b(?:Section|Sec\.|Article|Rule|Clause)\s+\d+[A-Za-z0-9()\-]*", re.I)


def _money(amount: float | None) -> str:
    if amount is None:
        return "[AMOUNT — FILL IN FROM YOUR PAPERS]"
    # Indian grouping: 1,23,456 rather than 123,456.
    digits = f"{int(round(amount))}"
    whole = digits
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts) + "," + tail
    return f"Rs {whole}"


def _cite(citation: Citation) -> str:
    return f"{citation.source} — {citation.citation}"


def build_letter_text(
    report: ComplianceReport,
    loan: LoanFacts | None,
    notice: NoticeFacts | None,
    debt: DebtAnalysis | None,
) -> str:
    """The deterministic letter. Every sentence traces to an input."""
    lender = loan.lender_name if loan else None
    descriptor = ""
    if report.violations:
        descriptor = " — recovery conduct and disclosure of charges"

    parts: list[str] = [
        BORROWER_BLOCK,
        "",
        "To,",
        nodal_officer(lender).address_block,
        "",
        "Subject: " + SUBJECT_TEMPLATE.format(descriptor=descriptor),
        "",
        "Sir / Madam,",
        "",
        "1. ABOUT ME AND THIS LOAN",
    ]

    # -- 1. the loan itself -------------------------------------------------
    intro = ["   1.1 I am a borrower of your company."]
    if loan:
        bits = []
        if loan.principal is not None:
            bits.append(f"a loan of {_money(loan.principal)}")
        if loan.loan_purpose:
            bits.append(f"taken for {loan.loan_purpose}")
        if loan.sanction_date:
            bits.append(f"sanctioned on {loan.sanction_date.isoformat()}")
        if bits:
            intro.append("       I took " + ", ".join(bits) + ".")
        if loan.quoted_rate is not None and loan.tenure_months:
            rate_type = {"flat": "flat", "reducing": "reducing balance"}.get(
                loan.rate_type.value, "a basis not stated on the document"
            )
            intro.append(
                f"       The rate quoted to me was {loan.quoted_rate:g}% per annum "
                f"({rate_type}), over {loan.tenure_months} months."
            )
        if loan.emi is not None:
            intro.append(f"       The instalment is {_money(loan.emi)} a month.")
    intro.append(
        "   1.2 My loan account number is [YOUR LOAN ACCOUNT NUMBER, AS PRINTED ON YOUR PAPERS]."
    )
    parts += intro

    # -- 2. what happened ---------------------------------------------------
    parts += ["", "2. WHAT HAPPENED"]
    chronology = _chronology(notice, debt)
    if chronology:
        parts += chronology
    else:
        parts.append(
            "   2.1 [DESCRIBE, IN YOUR OWN WORDS AND IN ORDER OF DATE, WHAT HAPPENED — "
            "when they called, what was said, and who said it]"
        )

    # -- 3. the violations, each with its citation --------------------------
    if report.violations:
        parts += ["", "3. WHY I SAY THIS IS NOT PERMITTED"]
        for index, violation in enumerate(report.violations, start=1):
            parts += [
                "",
                f"   3.{index} {violation.title}",
                f"        What happened: {violation.offending_fact}",
                f"        Why this is a problem: {violation.explanation}",
            ]
            for citation in violation.citations:
                parts.append(f"        Reference relied on: {_cite(citation)}")
        parts += ["", "   " + CITATION_CAVEAT]

    # -- 4. the flags, as questions rather than accusations -----------------
    if report.flags:
        parts += [
            "",
            "4. POINTS ON WHICH I ASK YOU TO SATISFY ME",
            "",
            "   These are matters I cannot confirm from the papers I hold, and on which I ask",
            "   you for a written answer. I do not allege wrongdoing on these points.",
        ]
        for index, flag in enumerate(report.flags, start=1):
            parts += ["", f"   4.{index} {flag.title}", f"        {flag.detail}"]
            # `flag.action` is written for the borrower ("verify this lender on
            # RBI's list"); putting that to the lender would be nonsense. The
            # rules supply a separate lender-facing question.
            ask = flag.lender_ask or (
                "Please respond to this point in writing."
            )
            parts.append(f"        What I ask you to state: {ask}")

    # -- 5. relief ----------------------------------------------------------
    section = 5 if report.flags else (4 if report.violations else 3)
    parts += ["", f"{section}. WHAT I ASK YOU TO DO"]
    # Several violations can share a relief — three separate threats all ask the
    # lender to stop and confirm in writing. Asking a regulator for the same
    # thing three times reads as careless, so identical asks collapse to one.
    reliefs: list[str] = []
    for violation in report.violations:
        relief = violation.relief_sought.strip()
        if relief and relief not in reliefs:
            reliefs.append(relief)
    if reliefs:
        for index, relief in enumerate(reliefs, start=1):
            parts.append(f"   {section}.{index} {relief}")
        offset = len(reliefs)
    else:
        offset = 0
    parts += [
        f"   {section}.{offset + 1} Acknowledge this complaint in writing, with a complaint "
        "reference number, and tell me the name of the officer handling it.",
        f"   {section}.{offset + 2} Send me a full statement of my account showing every "
        "amount charged and every amount received.",
    ]

    # -- 6. escalation ------------------------------------------------------
    parts += ["", f"{section + 1}. IF I DO NOT HEAR FROM YOU", "", "   " + ESCALATION_NOTE]
    parts += ["", SIGN_OFF, "", ENCLOSURES]

    return "\n".join(parts)


def _chronology(notice: NoticeFacts | None, debt: DebtAnalysis | None) -> list[str]:
    """Dated facts, in order, drawn only from what was extracted."""
    lines: list[str] = []
    counter = 1

    if notice is not None:
        events = sorted(
            notice.contact_times_mentioned,
            key=lambda e: (e.on_date.isoformat() if e.on_date else "", e.time_24h),
        )
        for event in events:
            when = f"on {event.on_date.isoformat()} " if event.on_date else ""
            who = f" by {event.caller}" if event.caller else ""
            lines.append(
                f"   2.{counter} I was contacted {when}at {event.time_24h}"
                f"{who} ({event.channel})."
            )
            counter += 1

        for threat in notice.threats_detected:
            lines.append(
                f'   2.{counter} I was told, in terms: "{threat.quote}"'
            )
            counter += 1

        if notice.notice_type.value == "court_notice_claim" and notice.sender_is_court is False:
            sender = notice.sender or "a private recovery agency"
            lines.append(
                f"   2.{counter} I received a document headed as a court notice, issued by "
                f"{sender}. It carries no court name, case number or court seal."
            )
            counter += 1

        if notice.demand_amount is not None:
            deadline = (
                f" within {notice.deadline_days} days"
                if notice.deadline_days is not None
                else ""
            )
            lines.append(
                f"   2.{counter} The amount demanded is {_money(notice.demand_amount)}{deadline}."
            )
            counter += 1

    if debt is not None and debt.hidden_cost_gap > 1:
        lines.append(
            f"   2.{counter} On working through the instalments, the rate quoted to me as "
            f"{debt.quoted_rate:g}% amounts to an effective {debt.effective_apr:.2f}% per annum "
            f"({debt.all_in_apr:.2f}% once charges are included). Over the full term that is "
            f"about {_money(debt.hidden_cost_gap)} more than the same loan would have cost at "
            f"a genuine {debt.quoted_rate:g}%."
        )
        counter += 1

    if lines:
        lines.append(
            f"   2.{counter} [ADD ANYTHING ELSE THAT HAPPENED, WITH DATES, IN YOUR OWN WORDS]"
        )
    return lines


def enumerate_placeholders(text: str) -> list[str]:
    """Every blank the borrower must fill, in the order they appear."""
    seen: list[str] = []
    for match in PLACEHOLDER_RE.finditer(text):
        label = f"[{match.group(1)}]"
        if label not in seen:
            seen.append(label)
    return seen


def build_summary(
    report: ComplianceReport,
    language: Language,
    debt: DebtAnalysis | None,
) -> str:
    """A plain-language account of what the borrower is about to send.

    Reuses the F4 rights sentences, so the letter's summary and the spoken
    voice note say the same things in the same words.
    """
    _, points = build_script(report, language, debt)
    lines = [phrase("letter_summary_intro", language)]
    lines += points[:-1] if points else []       # the trailing "what next" is replaced below
    lines.append(phrase("letter_summary_relief", language))
    lines.append(phrase("letter_summary_keep_copy", language))
    # Not `what_next`: it opens with "I have written a letter for you", which
    # the summary has already said. This carries only the follow-up steps.
    lines.append(phrase("letter_summary_escalate", language))
    return " ".join(line for line in lines if line)


async def draft_grievance(
    report: ComplianceReport,
    loan: LoanFacts | None = None,
    notice: NoticeFacts | None = None,
    debt: DebtAnalysis | None = None,
    language: Language = Language.HINDI,
    borrower_name: str | None = None,
    llm: LLM | None = None,
) -> GrievanceLetter:
    """Build the letter. The LLM may only tighten what is already written."""
    body = build_letter_text(report, loan, notice, debt)

    if borrower_name:
        body = body.replace("[YOUR FULL NAME]", borrower_name)

    if llm is not None:
        body = await _polish(llm, body)

    lender = loan.lender_name if loan else None
    descriptor = " — recovery conduct and disclosure of charges" if report.violations else ""

    return GrievanceLetter(
        subject=SUBJECT_TEMPLATE.format(descriptor=descriptor),
        body_english=body,
        summary_native=build_summary(report, language, debt),
        summary_language=language,
        addressees=[nodal_officer(lender), rbi_ombudsman()],
        violations_cited=[v.rule_id for v in report.violations],
        citations=list(report.citations),
        placeholders=enumerate_placeholders(body),
        escalation_note=rbi_ombudsman().note or "",
        disclaimer=disclaimer_for(Language.ENGLISH),
        filename_suggestion=_filename(lender),
    )


def _filename(lender: str | None) -> str:
    if not lender:
        return "grievance-letter.txt"
    slug = re.sub(r"[^a-z0-9]+", "-", lender.lower()).strip("-")[:40]
    return f"grievance-letter-{slug}.txt" if slug else "grievance-letter.txt"


async def _polish(llm: LLM, draft: str) -> str:
    """Let the LLM tighten the prose, then verify it did not do anything else."""
    prompt = f"TASK: grievance_letter\n\nDRAFT:\n{draft}"
    try:
        edited = await llm.complete(
            prompt, system=DRAFTER_SYSTEM_PROMPT, max_tokens=8192, effort="low"
        )
    except AdapterError as exc:
        log.warning("Letter polish failed (%s); sending the deterministic draft", exc)
        return draft

    problem = _verify(draft, edited)
    if problem:
        log.warning("Rejected the edited letter (%s); sending the deterministic draft", problem)
        return draft
    return edited


def _verify(draft: str, edited: str) -> str | None:
    """Return a reason to reject the edit, or None to accept it.

    An editor that drops a citation, loses a placeholder, or introduces a
    section number is not editing — it is changing what the borrower asserts
    to a regulator. That gets discarded rather than shipped.
    """
    if not edited.strip():
        return "empty output"

    invented = set(SECTION_RE.findall(edited)) - set(SECTION_RE.findall(draft))
    if invented:
        return f"introduced section references not in the draft: {sorted(invented)}"

    for line in draft.splitlines():
        marker = line.strip()
        if marker.startswith("Reference relied on:") and marker not in edited:
            return "dropped a citation"

    missing = set(enumerate_placeholders(draft)) - set(enumerate_placeholders(edited))
    if missing:
        return f"dropped placeholders: {sorted(missing)}"

    # A letter that has lost most of its length has lost content, not verbosity.
    if len(edited) < len(draft) * 0.6:
        return "output is far shorter than the draft"

    return None
