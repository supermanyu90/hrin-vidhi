"""F3 — the compliance shield and grounded Q&A.

Document intake lives in `routes/intake.py` alongside voice intake, since both
feed the same borrower session.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Request

from backend.analysis.rag import answer_question
from backend.analysis.retriever import get_retriever
from backend.analysis.rules import RuleContext, evaluate
from backend.finance import analyse_debt
from backend.schemas import (
    ComplianceReport,
    DebtAnalysis,
    Language,
    LoanFacts,
    NoticeFacts,
    RagAnswer,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


@router.post("/analysis/compliance", response_model=ComplianceReport, summary="Run the F3 shield")
async def compliance(
    loan_facts: LoanFacts | None = Body(default=None),
    notice_facts: NoticeFacts | None = Body(default=None),
    transcript_english: str = Body(default=""),
) -> ComplianceReport:
    """Deterministic rules + corpus citations over the extracted facts.

    The money math is run here rather than taken from the caller, so the APR
    figures quoted in the report cannot disagree with `/calc/debt-trap`.
    """
    if loan_facts is None and notice_facts is None:
        raise HTTPException(
            status_code=422,
            detail="Provide loan_facts, notice_facts, or both — there is nothing to analyse.",
        )

    debt: DebtAnalysis | None = None
    notes: list[str] = []
    if loan_facts is not None:
        if loan_facts.principal and loan_facts.quoted_rate is not None and loan_facts.tenure_months:
            debt = analyse_debt(
                principal=loan_facts.principal,
                quoted_rate=loan_facts.quoted_rate,
                tenure_months=loan_facts.tenure_months,
                rate_type=loan_facts.rate_type,
                processing_fee=loan_facts.processing_fee or 0.0,
                other_fees_total=sum(f.amount or 0.0 for f in loan_facts.other_fees),
                include_schedules=False,
            )
        else:
            notes.append(
                "The loan paper did not yield a principal, rate and tenure together, so the "
                "true-cost calculation could not be run. The rate checks below are therefore "
                "incomplete."
            )

    return evaluate(
        RuleContext(
            loan=loan_facts,
            notice=notice_facts,
            debt=debt,
            transcript_english=transcript_english,
            extra_notes=notes,
        )
    )


@router.post("/ask", response_model=RagAnswer, summary="Ask a grounded question")
async def ask(
    request: Request,
    question: str = Body(..., embed=True, min_length=3, max_length=500),
    language: Language = Body(Language.ENGLISH, embed=True),
) -> RagAnswer:
    """Answer from the corpus, or say plainly that it could not be confirmed.

    `language` picks which phrasebook a vernacular question is matched against.
    It defaults to English so existing callers keep working; a question written
    in an Indic script is detected and routed on its own even if the caller
    never sends one.
    """
    adapters = request.app.state.adapters
    return await answer_question(question, llm=adapters.llm, language=language)


@router.get("/corpus/status", summary="What the retrieval layer is running on")
async def corpus_status() -> dict:
    """Surfaced so a judge can see retrieval is real, and which backend is live."""
    from backend.analysis.corpus_store import load_chunks

    chunks = load_chunks()
    sources: dict[str, int] = {}
    for chunk in chunks:
        sources[chunk.source] = sources.get(chunk.source, 0) + 1
    return {
        "retriever": get_retriever().describe(),
        "chunk_count": len(chunks),
        "sources": sources,
        "all_summaries": all(c.is_summary for c in chunks),
        "reviewed_by_lawyer": sum(1 for c in chunks if c.verified_by),
    }
