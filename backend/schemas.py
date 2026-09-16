"""Typed contracts for every stage of the Hrin Vidhi pipeline.

Every module in `backend/` consumes and produces the models defined here and
nothing else. That is what lets any single stage be tested, mocked, or demoed
in isolation (see README §Architecture).

Conventions used throughout:
  * Money is `float` in rupees (INR). Paise precision is not meaningful for the
    advisory numbers we produce, and Decimal would leak into the JSON contract.
  * Rates are *percentages* (12.0 means 12%), never fractions. Any field holding
    a fraction is named `..._fraction` and marked as such.
  * Anything the parser could not read is `None`. A parser may never guess.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Base for every schema: reject unknown keys so a typo'd adapter fails loudly."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------


class Language(str, Enum):
    """Languages the borrower may speak. ISO-639-1 where one exists, else -3."""

    HINDI = "hi"
    MARATHI = "mr"
    TAMIL = "ta"
    TELUGU = "te"
    BHOJPURI = "bho"
    ENGLISH = "en"


LANGUAGE_NAMES: dict[Language, str] = {
    Language.HINDI: "हिन्दी",
    Language.MARATHI: "मराठी",
    Language.TAMIL: "தமிழ்",
    Language.TELUGU: "తెలుగు",
    Language.BHOJPURI: "भोजपुरी",
    Language.ENGLISH: "English",
}


# ---------------------------------------------------------------------------
# Stage 1 & 2 — speech-to-text and translation
# ---------------------------------------------------------------------------


class Transcript(StrictModel):
    """A borrower utterance in their own language, plus an English working copy.

    Both are retained: the native text is what we show and speak back to the
    borrower, the English text is what the rules engine and LLM reason over.
    """

    language: Language
    native_text: str = Field(description="Verbatim transcript in the spoken language.")
    english_text: str = Field(description="English working translation.")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="STT confidence if the provider reports one."
    )
    duration_seconds: float | None = Field(default=None, ge=0.0)
    provider: str = Field(description="Adapter that produced this, e.g. 'mock' or 'sarvam'.")


class TranslationResult(StrictModel):
    text: str
    source_language: Language
    target_language: Language
    provider: str


class SpeechAudio(StrictModel):
    """A synthesised voice note, returned to the UI for inline playback."""

    language: Language
    text: str = Field(description="Script that was spoken, for on-screen display.")
    audio_base64: str | None = Field(
        default=None, description="Base64 mp3/wav payload. None when only a URL is available."
    )
    audio_url: str | None = Field(default=None, description="Served path, e.g. /media/fixture.mp3")
    mime_type: str = "audio/mpeg"
    duration_seconds: float | None = Field(default=None, ge=0.0)
    provider: str


# ---------------------------------------------------------------------------
# Stage 3 — document / notice parsing
# ---------------------------------------------------------------------------


class RateType(str, Enum):
    FLAT = "flat"
    REDUCING = "reducing"
    UNKNOWN = "unknown"


class NBFCRegistrationStatus(str, Enum):
    """What we can say about the lender's RBI registration from the paper alone.

    VERIFIED never means "we checked RBI's live register" — it means a
    registration number was present and matched our offline seed list. The
    borrower is always told to confirm on RBI's official list.
    """

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    ABSENT = "absent"


class Fee(StrictModel):
    label: str = Field(description="As printed on the document, e.g. 'documentation charges'.")
    amount: float | None = Field(default=None, ge=0.0, description="Rupees, if a figure is given.")
    rate_percent: float | None = Field(
        default=None, ge=0.0, description="Used when the fee is quoted as a percentage."
    )
    disclosed_in_sanction_letter: bool | None = Field(
        default=None,
        description="None when the document does not let us tell. Drives the disclosure rule.",
    )


class LoanFacts(StrictModel):
    """Structured reading of a loan paper / sanction letter. Unknowns stay None."""

    principal: float | None = Field(default=None, ge=0.0)
    quoted_rate: float | None = Field(default=None, ge=0.0, description="Percent per annum.")
    rate_type: RateType = RateType.UNKNOWN
    tenure_months: int | None = Field(default=None, ge=1)
    processing_fee: float | None = Field(default=None, ge=0.0, description="Rupees.")
    other_fees: list[Fee] = Field(default_factory=list)
    penalty_rate: float | None = Field(
        default=None, ge=0.0, description="Percent. Per-month unless penalty_rate_basis says otherwise."
    )
    penalty_rate_basis: Literal["per_month", "per_annum", "one_time"] | None = None
    emi: float | None = Field(default=None, ge=0.0)
    lender_name: str | None = None
    nbfc_registration_no: str | None = None
    nbfc_registration_status: NBFCRegistrationStatus = NBFCRegistrationStatus.ABSENT
    loan_purpose: str | None = Field(default=None, description="e.g. 'two-wheeler', 'smartphone'.")
    sanction_date: date | None = None
    kfs_present: bool | None = Field(
        default=None, description="Was a Key Fact Statement attached? None = cannot tell."
    )
    apr_disclosed: bool | None = Field(default=None, description="Was an all-in APR printed anywhere?")
    source_notes: list[str] = Field(
        default_factory=list, description="Parser remarks, e.g. 'penalty clause illegible'."
    )


class ThreatCategory(str, Enum):
    FAMILY_CONTACT = "family_contact"
    PUBLIC_SHAMING = "public_shaming"
    VIOLENCE = "violence"
    PROPERTY_SEIZURE = "property_seizure"
    ARREST_CLAIM = "arrest_claim"
    JOB_LOSS = "job_loss"
    OBSCENE_LANGUAGE = "obscene_language"


class DetectedThreat(StrictModel):
    category: ThreatCategory
    quote: str = Field(description="The offending phrase, as extracted. Never paraphrased.")
    source: Literal["document", "voice_note"] = "document"


class ContactEvent(StrictModel):
    """A recovery contact the borrower reported or the notice records."""

    time_24h: str = Field(
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="Local time, 24-hour 'HH:MM'."
    )
    on_date: date | None = None
    channel: Literal["call", "visit", "sms", "whatsapp", "unknown"] = "unknown"
    caller: str | None = Field(default=None, description="Agent or agency name if stated.")
    source: Literal["document", "voice_note"] = "document"


class NoticeType(str, Enum):
    RECOVERY = "recovery"
    COURT_NOTICE_CLAIM = "court_notice_claim"
    CHEQUE_BOUNCE = "cheque_bounce"
    LEGAL_DEMAND = "legal_demand"
    UNKNOWN = "unknown"


class NoticeFacts(StrictModel):
    """Structured reading of a recovery notice or harassment message."""

    sender: str | None = None
    sender_is_court: bool | None = Field(
        default=None,
        description="True only if the document bears a genuine court identifier. None = cannot tell.",
    )
    sender_is_recovery_agent: bool | None = None
    notice_type: NoticeType = NoticeType.UNKNOWN
    demand_amount: float | None = Field(default=None, ge=0.0)
    deadline_days: int | None = Field(default=None, ge=0)
    issued_on: date | None = None
    threats_detected: list[DetectedThreat] = Field(default_factory=list)
    contact_times_mentioned: list[ContactEvent] = Field(default_factory=list)
    claims_to_verify: list[str] = Field(
        default_factory=list,
        description="Assertions the notice makes that we cannot confirm, e.g. 'arrest warrant issued'.",
    )
    source_notes: list[str] = Field(default_factory=list)


class DocumentKind(str, Enum):
    LOAN_PAPER = "loan_paper"
    NOTICE = "notice"
    UNKNOWN = "unknown"


class DocumentExtraction(StrictModel):
    """Exactly what a vision model is asked to return — no provider metadata.

    Kept separate from `ParsedDocument` so this can be handed to the Anthropic
    structured-outputs API as a JSON schema without asking the model to invent
    fields only the adapter knows (like which provider ran).
    """

    kind: DocumentKind
    loan_facts: LoanFacts | None = None
    notice_facts: NoticeFacts | None = None
    parse_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    unreadable_regions: list[str] = Field(default_factory=list)


class ParsedDocument(StrictModel):
    """What POST /intake/document returns: whichever facts the image supported."""

    kind: DocumentKind
    loan_facts: LoanFacts | None = None
    notice_facts: NoticeFacts | None = None
    provider: str
    parse_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    unreadable_regions: list[str] = Field(default_factory=list)

    @classmethod
    def from_extraction(cls, extraction: DocumentExtraction, provider: str) -> ParsedDocument:
        return cls(**extraction.model_dump(), provider=provider)


# ---------------------------------------------------------------------------
# Stage 4a — money math
# ---------------------------------------------------------------------------


class SchedulePoint(StrictModel):
    """One month of an amortisation curve. Month 0 is disbursal."""

    month: int = Field(ge=0)
    outstanding: float = Field(description="Balance still owed after this month's payment.")
    interest_paid_this_month: float
    principal_paid_this_month: float
    cumulative_interest: float
    cumulative_paid: float = Field(description="Total rupees out of pocket, fees included.")


class RateSolution(StrictModel):
    """Output of the effective-rate root-finder."""

    monthly_rate: float = Field(description="Fraction per month, e.g. 0.0165 for 1.65%/month.")
    nominal_apr: float = Field(description="Percent: monthly_rate * 12 * 100.")
    effective_annual_rate: float = Field(
        description="Percent: ((1+monthly_rate)^12 - 1) * 100. Compounding included."
    )
    iterations: int = Field(ge=0)
    converged: bool
    method: Literal["newton", "bisection", "closed_form", "degenerate"] = "bisection"
    residual: float = Field(
        default=0.0, description="|PV(EMI at solved rate) - principal|, for auditability."
    )


class DebtAnalysis(StrictModel):
    """The credibility core: everything F5 plots and F6 cites.

    `flat_*` describes the schedule as the lender quoted it. `effective_*`
    describes the same cash flows expressed as a reducing-balance rate — the
    number that is comparable to a bank's advertised rate.
    """

    principal: float = Field(ge=0.0)
    quoted_rate: float = Field(ge=0.0, description="Percent per annum, as quoted.")
    rate_type: RateType
    tenure_months: int = Field(ge=1)

    emi: float = Field(description="Monthly instalment implied by the quote.")
    total_payable: float = Field(description="EMI * tenure, excluding fees.")
    total_interest: float

    processing_fee: float = Field(default=0.0, ge=0.0)
    other_fees_total: float = Field(default=0.0, ge=0.0)
    all_in_outflow: float = Field(description="total_payable + every fee.")

    rate_solution: RateSolution
    effective_apr: float = Field(description="Percent. Nominal APR on the EMI stream, no fees.")
    effective_annual_rate: float = Field(description="Percent. Compounded, no fees.")
    all_in_apr: float = Field(
        description="Percent. Nominal APR once fees are treated as a reduction in net disbursal."
    )

    honest_emi: float = Field(
        description="EMI a genuine reducing-balance loan at the SAME quoted percentage "
        "would have charged. The advertised-vs-actual baseline."
    )
    honest_total: float = Field(description="honest_emi * tenure. No fees.")

    hidden_cost_gap: float = Field(
        description="Rupees. all_in_outflow minus what the loan would cost at the quoted rate "
        "read as a true reducing-balance rate. The advertised-vs-actual number."
    )
    hidden_rate_gap: float = Field(description="Percentage points: all_in_apr - quoted_rate.")

    flat_schedule: list[SchedulePoint] = Field(
        default_factory=list, description="The lender's own booking: level interest on principal."
    )
    reducing_schedule: list[SchedulePoint] = Field(
        default_factory=list,
        description="The same EMIs amortised at the solved effective rate. Same total as "
        "flat_schedule by construction — only the interest/principal split differs.",
    )
    honest_schedule: list[SchedulePoint] = Field(
        default_factory=list,
        description="A real reducing-balance loan at the quoted rate. The gap between this "
        "curve and flat_schedule IS hidden_cost_gap, which is what F5 plots.",
    )

    warnings: list[str] = Field(
        default_factory=list, description="Non-convergence, degenerate inputs, assumptions made."
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Stated plainly so a banker in the room can check them."
    )


class DebtTrapRequest(StrictModel):
    """Body of POST /calc/debt-trap — drives the F5 visualizer."""

    principal: float = Field(gt=0.0, le=100_000_000)
    quoted_rate: float = Field(ge=0.0, le=200.0, description="Percent per annum.")
    tenure_months: int = Field(ge=1, le=600)
    rate_type: RateType = RateType.FLAT
    processing_fee: float = Field(default=0.0, ge=0.0)
    other_fees_total: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# Stage 4b — compliance
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    INFO = "info"


class Citation(StrictModel):
    """A pointer into the seeded corpus. Never free text invented by an LLM."""

    chunk_id: str = Field(description="Stable id of the corpus chunk, e.g. 'rbi-fpc-recovery-hours'.")
    title: str
    source: str = Field(description="Human-readable provenance, e.g. 'RBI Fair Practices Code'.")
    citation: str = Field(description="Clause/section reference as recorded in the corpus.")
    quote: str = Field(description="The retrieved summary text. Marked as a summary, not statute.")
    url: str | None = None
    is_summary: bool = Field(
        default=True, description="Always true for seed content; a lawyer may later set it false."
    )
    retrieval_score: float | None = Field(default=None, ge=0.0, le=1.0)


class Violation(StrictModel):
    """A deterministic rule hit. Every field is traceable to an input fact."""

    rule_id: str = Field(description="Stable id, e.g. 'R-001-recovery-hours'.")
    title: str
    severity: Severity
    offending_fact: str = Field(description="The specific extracted fact that triggered the rule.")
    explanation: str = Field(description="Why this is a problem, in plain English.")
    citations: list[Citation] = Field(default_factory=list)
    relief_sought: str = Field(description="What the grievance letter should demand for this.")


class Flag(StrictModel):
    """Something the borrower should check — short of an assertable violation."""

    flag_id: str
    title: str
    severity: Severity
    detail: str
    action: str = Field(description="What the borrower should do, e.g. 'verify on RBI's NBFC list'.")
    lender_ask: str = Field(
        default="",
        description=(
            "The same point put to the lender as a question, for the grievance letter. "
            "Separate from `action` because the two audiences differ: telling a lender to "
            "'verify this lender on RBI's list' is nonsense. Empty means the letter asks "
            "for a written response in general terms instead."
        ),
    )
    citations: list[Citation] = Field(default_factory=list)


class ComplianceReport(StrictModel):
    violations: list[Violation] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    citations: list[Citation] = Field(
        default_factory=list, description="Union of every citation referenced above, de-duplicated."
    )
    overall_summary: str
    unconfirmed: list[str] = Field(
        default_factory=list,
        description="Checks we could not ground in the corpus. Surfaced, never silently dropped.",
    )
    generated_at: datetime = Field(default_factory=_utcnow)


class RagAnswer(StrictModel):
    """Answer to a free-form borrower question, grounded in retrieved chunks."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = Field(
        description="False when retrieval confidence was too low; `answer` then says so."
    )


# ---------------------------------------------------------------------------
# Stage 5 — rights explainer
# ---------------------------------------------------------------------------

DISCLAIMER_EN = (
    "This is information and a draft, not legal advice. Please check your lender on RBI's "
    "official list of registered NBFCs, and show this to a lawyer or a free legal aid clinic "
    "before you file."
)


class RightsExplanation(StrictModel):
    language: Language
    script_native: str = Field(description="What gets spoken, in the borrower's language.")
    script_english: str
    disclaimer_native: str = Field(default="")
    disclaimer_english: str = DISCLAIMER_EN
    voice_note: SpeechAudio | None = None
    key_points: list[str] = Field(
        default_factory=list, description="Bullet form of the script, for on-screen display."
    )


# ---------------------------------------------------------------------------
# Stage 6 — grievance letter
# ---------------------------------------------------------------------------


class Addressee(StrictModel):
    role: str = Field(description="e.g. 'Nodal / Grievance Redressal Officer'.")
    organisation: str
    address_block: str = Field(description="Multi-line address, placeholders clearly labelled.")
    note: str | None = None


class GrievanceLetter(StrictModel):
    subject: str
    body_english: str = Field(description="The full formal letter, ready to print or paste.")
    summary_native: str = Field(description="Plain-language 'here is what you are sending'.")
    summary_language: Language
    addressees: list[Addressee] = Field(default_factory=list)
    violations_cited: list[str] = Field(
        default_factory=list, description="rule_ids asserted. Never longer than the report's list."
    )
    citations: list[Citation] = Field(default_factory=list)
    placeholders: list[str] = Field(
        default_factory=list,
        description="Every [BRACKETED] blank the borrower must fill, enumerated for the UI.",
    )
    escalation_note: str = Field(
        default="", description="When and how to take this to the RBI Ombudsman (RB-IOS)."
    )
    disclaimer: str = DISCLAIMER_EN
    filename_suggestion: str = "grievance-letter.txt"


class GrievanceDraftRequest(StrictModel):
    report: ComplianceReport
    loan_facts: LoanFacts | None = None
    notice_facts: NoticeFacts | None = None
    debt_analysis: DebtAnalysis | None = None
    language: Language = Language.HINDI
    borrower_name: str | None = None


# ---------------------------------------------------------------------------
# Session / orchestration
# ---------------------------------------------------------------------------


class SessionState(StrictModel):
    """In-memory only. Nothing here is written to disk unless the user opts in."""

    session_id: str
    language: Language = Language.HINDI
    created_at: datetime = Field(default_factory=_utcnow)
    transcript: Transcript | None = None
    parsed_document: ParsedDocument | None = None
    debt_analysis: DebtAnalysis | None = None
    compliance_report: ComplianceReport | None = None
    rights_explanation: RightsExplanation | None = None
    grievance_letter: GrievanceLetter | None = None
    persist_opt_in: bool = Field(
        default=False, description="False means uploads and transcripts die with the session."
    )


class VoiceIntakeResponse(StrictModel):
    session_id: str
    transcript: Transcript
    next_step: str = Field(description="What the UI should prompt for next.")


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    version: str
    demo_mode: bool
    adapters: dict[str, str] = Field(
        description="adapter name -> active implementation, e.g. {'stt': 'mock'}."
    )
    corpus_chunks: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(StrictModel):
    error: str
    detail: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
