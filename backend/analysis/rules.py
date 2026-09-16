"""F3 — the deterministic layer of the Fair-Lending & Anti-Harassment Shield.

Every rule is a pure function from extracted facts to findings. No LLM, no
retrieval, no I/O: given the same `LoanFacts` / `NoticeFacts` it returns the
same result, which is what makes the output auditable in front of a judge and
safe to put in a letter addressed to a regulator.

Two kinds of finding, and the distinction is deliberate:

  * `Violation` — the extracted facts, on their face, breach a specific rule we
    can cite. Assertable in the grievance letter.
  * `Flag` — something the borrower should check or that needs confirmation we
    cannot do from a photograph. Never asserted as a breach.

Rules declare their supporting corpus chunks by id. `evaluate()` resolves them,
and any rule whose citations do not resolve is demoted into `unconfirmed`
rather than asserted — see `analysis/citations.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from backend.analysis.citations import dedupe, resolve
from backend.schemas import (
    ComplianceReport,
    DebtAnalysis,
    Flag,
    LoanFacts,
    NBFCRegistrationStatus,
    NoticeFacts,
    NoticeType,
    RateType,
    Severity,
    ThreatCategory,
    Violation,
)

# Permitted recovery-contact window, from the Fair Practices Code.
CONTACT_WINDOW_START = 8 * 60      # 08:00
CONTACT_WINDOW_END = 19 * 60       # 19:00

#: An effective APR this many percentage points above the quoted rate, with no
#: reducing-balance disclosure, is treated as material. Chosen so that a flat
#: quote (which roughly doubles the true rate) always trips it, while ordinary
#: rounding and fee noise do not.
MATERIAL_APR_GAP_PP = 3.0

#: Penalty above this, expressed per month, is flagged as needing scrutiny.
HIGH_PENALTY_PER_MONTH = 2.0


@dataclass(frozen=True, slots=True)
class Finding:
    """A rule hit, before citations are resolved.

    Rules return these rather than `Violation`/`Flag` directly so that the
    engine — not the rule — decides whether the evidence is citable.
    """

    rule_id: str
    title: str
    severity: Severity
    offending_fact: str
    explanation: str
    chunk_ids: list[str]
    relief_sought: str = ""
    action: str = ""
    lender_ask: str = ""
    is_flag: bool = False


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything the rules are allowed to look at."""

    loan: LoanFacts | None = None
    notice: NoticeFacts | None = None
    debt: DebtAnalysis | None = None
    #: Facts the borrower stated in their voice note, in English. Rules use
    #: this only to corroborate document evidence, never as the sole basis for
    #: a violation — we cannot cite a transcript to a regulator.
    transcript_english: str = ""
    extra_notes: list[str] = field(default_factory=list)


def _minutes(time_24h: str) -> int:
    hour, minute = time_24h.split(":")
    return int(hour) * 60 + int(minute)


# ---------------------------------------------------------------------------
# R-001 — recovery contact outside permitted hours
# ---------------------------------------------------------------------------


def rule_recovery_hours(ctx: RuleContext) -> list[Finding]:
    if ctx.notice is None:
        return []

    offending = [
        event
        for event in ctx.notice.contact_times_mentioned
        if not (CONTACT_WINDOW_START <= _minutes(event.time_24h) < CONTACT_WINDOW_END)
    ]
    if not offending:
        return []

    described = "; ".join(
        f"{e.time_24h}"
        + (f" on {e.on_date.isoformat()}" if e.on_date else "")
        + (f" by {e.caller}" if e.caller else "")
        + f" ({e.channel})"
        for e in offending
    )
    plural = "contacts" if len(offending) > 1 else "contact"
    return [
        Finding(
            rule_id="R-001-recovery-hours",
            title="Recovery contact outside the permitted hours of 8 a.m. to 7 p.m.",
            severity=Severity.HIGH,
            offending_fact=f"Recorded recovery {plural}: {described}.",
            explanation=(
                "A lender and its recovery agents may only contact a borrower between "
                "8:00 in the morning and 7:00 in the evening. Each contact listed above "
                "falls outside that window."
            ),
            chunk_ids=["rbi-fpc-recovery-hours", "rbi-fpc-no-harassment"],
            relief_sought=(
                "Direct the lender and its recovery agents to cease all contact outside "
                "08:00-19:00, and confirm in writing the steps taken against the agents "
                "responsible."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# R-002 — harassment, threats and shaming
# ---------------------------------------------------------------------------

# Each threat category maps to the chunks that actually support a complaint
# about it. Categories absent from this map produce no violation.
_THREAT_SUPPORT: dict[ThreatCategory, tuple[str, str]] = {
    ThreatCategory.FAMILY_CONTACT: (
        "threatening to disclose the debt to the borrower's family or in-laws",
        "rbi-fpc-third-party-contact",
    ),
    ThreatCategory.PUBLIC_SHAMING: (
        "threatening to expose the debt publicly or shame the borrower in their community",
        "rbi-fpc-third-party-contact",
    ),
    ThreatCategory.VIOLENCE: (
        "threatening physical harm",
        "rbi-fpc-no-harassment",
    ),
    ThreatCategory.OBSCENE_LANGUAGE: (
        "using abusive or obscene language",
        "rbi-fpc-no-harassment",
    ),
    ThreatCategory.JOB_LOSS: (
        "threatening to contact the borrower's employer",
        "rbi-fpc-third-party-contact",
    ),
    ThreatCategory.PROPERTY_SEIZURE: (
        "threatening to seize property without the notice and process the agreement requires",
        "rbi-fpc-possession-due-process",
    ),
    ThreatCategory.ARREST_CLAIM: (
        "claiming the borrower will be arrested for non-payment",
        "ni-138-no-arrest-for-debt",
    ),
}


def rule_harassment(ctx: RuleContext) -> list[Finding]:
    if ctx.notice is None or not ctx.notice.threats_detected:
        return []

    findings: list[Finding] = []
    for category in sorted({t.category for t in ctx.notice.threats_detected}, key=lambda c: c.value):
        support = _THREAT_SUPPORT.get(category)
        if support is None:
            continue
        description, chunk_id = support
        quotes = [t.quote for t in ctx.notice.threats_detected if t.category is category]
        quoted = " ".join(f'"{q}"' for q in quotes)

        severity = (
            Severity.CRITICAL
            if category in (ThreatCategory.VIOLENCE, ThreatCategory.ARREST_CLAIM)
            else Severity.HIGH
        )
        findings.append(
            Finding(
                rule_id=f"R-002-threat-{category.value.replace('_', '-')}",
                title=f"Recovery threat: {description}",
                severity=severity,
                offending_fact=f"The notice states: {quoted}",
                explanation=(
                    "Recovery must not involve intimidation, threats or humiliation, and the "
                    "debt must not be disclosed to third parties in order to pressure the "
                    f"borrower. The quoted language amounts to {description}."
                ),
                chunk_ids=[chunk_id, "rbi-fpc-no-harassment"],
                relief_sought=(
                    "Direct the lender to stop this conduct immediately, withdraw the threat "
                    "in writing, and confirm that no third party has been or will be contacted "
                    "about this debt."
                ),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# R-003 — a private agency presenting its letter as court process
# ---------------------------------------------------------------------------


def rule_fake_court_notice(ctx: RuleContext) -> list[Finding]:
    notice = ctx.notice
    if notice is None:
        return []
    if notice.notice_type is not NoticeType.COURT_NOTICE_CLAIM:
        return []
    # Only assert this where the parse positively established it is NOT a
    # court document. `None` means the parser could not tell, and an unproven
    # misrepresentation is not something we will put in a letter.
    if notice.sender_is_court is not False:
        return []

    sender = notice.sender or "the sender"
    return [
        Finding(
            rule_id="R-003-court-notice-misrepresentation",
            title="A private recovery agency's letter presented as a court notice",
            severity=Severity.CRITICAL,
            offending_fact=(
                f"The document is headed as a court notice but was issued by {sender}, "
                "and carries no court name, case number or court seal."
            ),
            explanation=(
                "Court process issues from a court, under its case number and seal, and only "
                "after a case has been filed. A letter on a recovery agency's letterhead is "
                "not a court notice whatever is printed at the top of it. Presenting a private "
                "demand as though it were court process misleads the borrower about their "
                "legal position."
            ),
            chunk_ids=["ni-138-court-notice-vs-private-letter", "rbi-fpc-no-harassment"],
            relief_sought=(
                "Direct the lender to withdraw the document, confirm in writing that no court "
                "proceeding exists against the borrower, and state what action has been taken "
                "against the agency that issued it."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# R-004 — charges that were never disclosed
# ---------------------------------------------------------------------------


def rule_undisclosed_charges(ctx: RuleContext) -> list[Finding]:
    loan = ctx.loan
    if loan is None:
        return []

    undisclosed: list[str] = []
    for fee in loan.other_fees:
        if fee.disclosed_in_sanction_letter is False:
            amount = f" of Rs {fee.amount:,.0f}" if fee.amount is not None else ""
            undisclosed.append(f"{fee.label}{amount}")

    # A processing fee that the parser noted as absent from the sanction letter
    # body counts too. The note is the parser's own observation, not inference.
    fee_note = any(
        "processing fee" in note.lower() and "not in the sanction letter" in note.lower()
        for note in loan.source_notes
    )
    if loan.processing_fee and fee_note:
        undisclosed.insert(0, f"processing fee of Rs {loan.processing_fee:,.0f}")

    if not undisclosed:
        return []

    return [
        Finding(
            rule_id="R-004-undisclosed-charges",
            title="Charges levied that were not disclosed in the sanction letter",
            severity=Severity.HIGH,
            offending_fact="Charges not disclosed up front: " + "; ".join(undisclosed) + ".",
            explanation=(
                "Every fee and charge payable by the borrower has to be disclosed up front, in "
                "the Key Fact Statement and the sanction letter. A charge that surfaces only on "
                "the disbursal voucher, or that is silently deducted from the amount paid out, "
                "has not been properly disclosed."
            ),
            chunk_ids=["rbi-dl-no-hidden-charges", "rbi-dl-kfs-required", "cpa-unfair-trade-practice"],
            relief_sought=(
                "Refund the charges that were not disclosed before sanction, and provide an "
                "itemised statement of every amount deducted from the disbursed sum."
            ),
        )
    ]


def rule_missing_kfs(ctx: RuleContext) -> list[Finding]:
    loan = ctx.loan
    if loan is None or loan.kfs_present is not False:
        return []
    return [
        Finding(
            rule_id="R-005-no-key-fact-statement",
            title="No Key Fact Statement was provided",
            severity=Severity.MEDIUM,
            offending_fact="No Key Fact Statement was found in the documents provided.",
            explanation=(
                "A borrower must be given a Key Fact Statement before committing to the loan, "
                "in a language they understand, setting out the all-inclusive annual rate, the "
                "instalment, every fee, the penalty for late payment and the grievance "
                "officer's details. Whether this requirement reaches an offline loan of this "
                "kind is a question for a lawyer, but its absence should be raised."
            ),
            chunk_ids=["rbi-dl-kfs-required"],
            relief_sought=(
                "Provide the Key Fact Statement for this loan, or confirm in writing that none "
                "was issued."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# R-006 — the rate illusion (a flag, not an assertion)
# ---------------------------------------------------------------------------


def rule_apr_gap(ctx: RuleContext) -> list[Finding]:
    debt, loan = ctx.debt, ctx.loan
    if debt is None:
        return []

    gap = debt.effective_apr - debt.quoted_rate
    if gap < MATERIAL_APR_GAP_PP:
        return []
    # If the lender said plainly that the rate is reducing-balance, the
    # borrower was not misled about the method and there is nothing to flag.
    if loan is not None and loan.rate_type is RateType.REDUCING:
        return []

    disclosed_apr = loan is not None and loan.apr_disclosed is True
    detail = (
        f"The loan is quoted at {debt.quoted_rate:g}% but the instalments imply a true "
        f"reducing-balance rate of {debt.effective_apr:.2f}% per annum "
        f"({debt.all_in_apr:.2f}% once fees are counted). Over {debt.tenure_months} months "
        f"that is about Rs {debt.hidden_cost_gap:,.0f} more than the same loan at a genuine "
        f"{debt.quoted_rate:g}%."
    )
    if not disclosed_apr:
        detail += " No all-inclusive annual percentage rate appears in the documents provided."

    return [
        Finding(
            rule_id="F-001-apr-understated",
            title="The advertised rate is far below the true annual cost",
            severity=Severity.HIGH,
            offending_fact=detail,
            explanation=(
                "Interest quoted flat is charged on the full original amount for the whole "
                "term, ignoring every instalment already repaid, so the real annual cost is "
                "close to double the quoted figure. Lenders are expected to disclose an "
                "all-inclusive annual percentage rate rather than only a headline number."
            ),
            chunk_ids=["rbi-dl-apr-disclosure", "cpa-unfair-trade-practice"],
            action=(
                "Ask the lender in writing for the all-inclusive APR and the full repayment "
                "schedule, and compare it with the figures above."
            ),
            lender_ask=(
                "State the all-inclusive annual percentage rate for this loan, and provide the "
                "full repayment schedule showing how each instalment is split between interest "
                "and principal."
            ),
            is_flag=True,
        )
    ]


def rule_penalty_rate(ctx: RuleContext) -> list[Finding]:
    loan = ctx.loan
    if loan is None or loan.penalty_rate is None:
        return []
    if loan.penalty_rate_basis != "per_month" or loan.penalty_rate < HIGH_PENALTY_PER_MONTH:
        return []

    annualised = loan.penalty_rate * 12
    return [
        Finding(
            rule_id="F-002-penalty-rate-high",
            title="The late-payment penalty is very high",
            severity=Severity.MEDIUM,
            offending_fact=(
                f"Penalty of {loan.penalty_rate:g}% per month on overdue amounts, which is "
                f"about {annualised:g}% a year on the amount in arrears."
            ),
            explanation=(
                "A penalty for late payment must be disclosed in advance and must be "
                "reasonable and proportionate to the default. It is treated as a penal charge "
                "rather than extra interest, and should not be added to the principal so as to "
                "earn further interest."
            ),
            chunk_ids=["rbi-dl-penal-charges-reasonable"],
            action=(
                "Ask the lender to confirm how the penalty has been applied to your account, "
                "and whether it has been added to the principal."
            ),
            lender_ask=(
                "State how the late-payment penalty has been applied to my account to date, and "
                "confirm whether any part of it has been added to the principal."
            ),
            is_flag=True,
        )
    ]


def rule_nbfc_registration(ctx: RuleContext) -> list[Finding]:
    loan = ctx.loan
    if loan is None:
        return []
    if loan.nbfc_registration_status is NBFCRegistrationStatus.VERIFIED:
        return []

    lender = loan.lender_name or "The lender"
    if loan.nbfc_registration_status is NBFCRegistrationStatus.ABSENT:
        fact = f"{lender} shows no RBI Certificate of Registration number on the documents provided."
    else:
        fact = (
            f"{lender} shows registration details that could not be read clearly "
            f"({loan.nbfc_registration_no or 'number not legible'})."
        )

    return [
        Finding(
            rule_id="F-003-nbfc-registration-unverified",
            title="The lender's RBI registration could not be verified",
            severity=Severity.HIGH,
            offending_fact=fact,
            explanation=(
                "A company carrying on NBFC business generally must hold a Certificate of "
                "Registration from the Reserve Bank of India, and RBI publishes the list of "
                "registered NBFCs. This tool cannot check that list, so the borrower must. "
                "If the lender is not registered, the RBI Ombudsman route may not be open and "
                "the complaint should go to the consumer forum or the police instead."
            ),
            chunk_ids=["rbi-fpc-nbfc-registration", "rbios-limits-and-exclusions"],
            action=(
                "Verify this lender on RBI's official list of registered NBFCs at rbi.org.in "
                "before filing, and note down what you find."
            ),
            lender_ask=(
                "State your RBI Certificate of Registration number, and the name under which "
                "your company is entered on the Reserve Bank's list of registered NBFCs."
            ),
            is_flag=True,
        )
    ]


def rule_claims_to_verify(ctx: RuleContext) -> list[Finding]:
    notice = ctx.notice
    if notice is None or not notice.claims_to_verify:
        return []
    return [
        Finding(
            rule_id="F-004-claims-to-verify",
            title="The notice makes claims that could not be confirmed",
            severity=Severity.MEDIUM,
            offending_fact="; ".join(notice.claims_to_verify),
            explanation=(
                "These statements appear in the document but cannot be confirmed from the "
                "document itself. Each should be checked before it is relied on or accepted."
            ),
            chunk_ids=["ni-138-court-notice-vs-private-letter"],
            action=(
                "Ask the sender, in writing, to produce the case number, court and order that "
                "each claim depends on."
            ),
            lender_ask=(
                "Produce the case number, the court, and the order on which each of these "
                "claims depends, or withdraw them in writing."
            ),
            is_flag=True,
        )
    ]


def rule_short_deadline(ctx: RuleContext) -> list[Finding]:
    notice = ctx.notice
    if notice is None or notice.deadline_days is None or notice.deadline_days >= 15:
        return []
    if notice.notice_type not in (NoticeType.COURT_NOTICE_CLAIM, NoticeType.CHEQUE_BOUNCE,
                                  NoticeType.LEGAL_DEMAND):
        return []
    return [
        Finding(
            rule_id="F-005-deadline-shorter-than-statutory",
            title="The deadline demanded is shorter than the law allows in a cheque case",
            severity=Severity.MEDIUM,
            offending_fact=(
                f"The notice demands payment within {notice.deadline_days} day"
                f"{'s' if notice.deadline_days != 1 else ''}."
            ),
            explanation=(
                "Where a demand follows a dishonoured cheque, the law gives the drawer fifteen "
                "days from receiving the notice to pay before any offence arises. A demand for "
                "payment in fewer days does not shorten that period. Whether this notice is "
                "such a demand at all is a question for a lawyer."
            ),
            chunk_ids=["ni-138-notice-and-time", "ni-138-what-is-not-a-138-case"],
            action=(
                "Do not be rushed by the stated deadline. Take the notice to a lawyer or a "
                "legal aid clinic and check what period actually applies."
            ),
            lender_ask=(
                "State the basis on which the deadline in that notice was calculated, and "
                "confirm what period you say actually applies."
            ),
            is_flag=True,
        )
    ]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

ALL_RULES: tuple[Callable[[RuleContext], list[Finding]], ...] = (
    rule_recovery_hours,
    rule_harassment,
    rule_fake_court_notice,
    rule_undisclosed_charges,
    rule_missing_kfs,
    rule_apr_gap,
    rule_penalty_rate,
    rule_nbfc_registration,
    rule_claims_to_verify,
    rule_short_deadline,
)

_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.INFO: 3,
}


def evaluate(ctx: RuleContext) -> ComplianceReport:
    """Run every rule and assemble the report.

    A finding whose citations do not all resolve is moved to `unconfirmed`
    rather than asserted — that is the guarantee the grievance drafter relies on.
    """
    violations: list[Violation] = []
    flags: list[Flag] = []
    unconfirmed: list[str] = []

    for rule in ALL_RULES:
        for finding in rule(ctx):
            citations, missing = resolve(finding.chunk_ids)
            if missing or not citations:
                unconfirmed.append(
                    f"{finding.title} — detected, but not asserted because the supporting "
                    f"corpus reference(s) {', '.join(missing) or '(none declared)'} could not "
                    f"be found. Have this checked manually."
                )
                continue

            if finding.is_flag:
                flags.append(
                    Flag(
                        flag_id=finding.rule_id,
                        title=finding.title,
                        severity=finding.severity,
                        detail=f"{finding.offending_fact} {finding.explanation}".strip(),
                        action=finding.action,
                        lender_ask=finding.lender_ask,
                        citations=citations,
                    )
                )
            else:
                violations.append(
                    Violation(
                        rule_id=finding.rule_id,
                        title=finding.title,
                        severity=finding.severity,
                        offending_fact=finding.offending_fact,
                        explanation=finding.explanation,
                        citations=citations,
                        relief_sought=finding.relief_sought,
                    )
                )

    violations.sort(key=lambda v: (_SEVERITY_ORDER[v.severity], v.rule_id))
    flags.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.flag_id))

    all_citations = dedupe(
        [c for v in violations for c in v.citations] + [c for f in flags for c in f.citations]
    )

    return ComplianceReport(
        violations=violations,
        flags=flags,
        citations=all_citations,
        overall_summary=summarise(violations, flags, ctx),
        unconfirmed=unconfirmed + list(ctx.extra_notes),
    )


def summarise(violations: list[Violation], flags: list[Flag], ctx: RuleContext) -> str:
    """One plain-English paragraph. Deterministic — no model involved."""
    if not violations and not flags:
        return (
            "Nothing in the documents provided breaches the rules this tool checks. That is "
            "not a clean bill of health: it only means no problem was found in what was "
            "readable here."
        )

    parts: list[str] = []
    if violations:
        worst = violations[0].severity.value
        parts.append(
            f"{len(violations)} apparent breach{'es' if len(violations) != 1 else ''} of "
            f"lending and recovery rules {'were' if len(violations) != 1 else 'was'} found, "
            f"the most serious rated {worst}."
        )
    else:
        parts.append("No outright breach was found in the documents provided.")

    if flags:
        parts.append(
            f"{len(flags)} further point{'s' if len(flags) != 1 else ''} "
            f"need{'' if len(flags) != 1 else 's'} checking."
        )

    if ctx.debt is not None and ctx.debt.hidden_cost_gap > 1:
        parts.append(
            f"On the money itself, this loan costs about Rs {ctx.debt.hidden_cost_gap:,.0f} "
            f"more than the same loan at the {ctx.debt.quoted_rate:g}% that was advertised."
        )

    parts.append(
        "This is information and a draft, not a legal finding; have it reviewed before filing."
    )
    return " ".join(parts)
