"""F4 — the spoken rights explainer. F6 — the grievance letter.

Both accept either a session id (the chat flow) or an explicit report (so each
stage can be demoed on its own, which is what the tests do).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import PlainTextResponse

from backend.explainer.script import explain_rights
from backend.grievance.drafter import draft_grievance
from backend.schemas import (
    ComplianceReport,
    DebtAnalysis,
    GrievanceDraftRequest,
    GrievanceLetter,
    Language,
    RightsExplanation,
)
from backend.sessions import get_store

log = logging.getLogger(__name__)

router = APIRouter(tags=["output"])


async def _from_session(session_id: str):
    state = await get_store().get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    if state.compliance_report is None:
        raise HTTPException(
            status_code=409,
            detail="Nothing has been analysed in this session yet. Send a document first.",
        )
    parsed = state.parsed_document
    return (
        state.compliance_report,
        parsed.loan_facts if parsed else None,
        parsed.notice_facts if parsed else None,
        state.debt_analysis,
        state.language,
        state,
    )


@router.post(
    "/explain/rights",
    response_model=RightsExplanation,
    summary="Plain-language rights script plus a voice note (F4)",
)
async def explain(
    request: Request,
    session_id: str | None = Body(default=None),
    report: ComplianceReport | None = Body(default=None),
    debt_analysis: DebtAnalysis | None = Body(default=None),
    language: Language | None = Body(default=None),
    speak: bool = Body(default=True),
) -> RightsExplanation:
    """Assemble the script from the report and speak it in the borrower's language."""
    adapters = request.app.state.adapters
    state = None

    if session_id:
        report, _, _, debt_analysis, session_language, state = await _from_session(session_id)
        language = language or session_language
    if report is None:
        raise HTTPException(
            status_code=422, detail="Provide a session_id or a compliance report."
        )

    explanation = await explain_rights(
        report=report,
        language=language or Language.HINDI,
        debt=debt_analysis,
        tts=adapters.tts if speak else None,
    )

    if state is not None:
        # Keep the script, drop the audio payload. A voice note is ~4 MB of
        # base64; retaining it in every session would let the store's own cap
        # (500 sessions) reach gigabytes. The borrower already has the audio in
        # this response, and asking again re-synthesises it.
        lightweight = explanation.model_copy(update={"voice_note": None})
        await get_store().update(state.model_copy(update={"rights_explanation": lightweight}))
    return explanation


@router.post(
    "/grievance/draft",
    response_model=GrievanceLetter,
    summary="Formal grievance letter plus a vernacular summary (F6)",
)
async def grievance(
    request: Request,
    session_id: str | None = Body(default=None),
    payload: GrievanceDraftRequest | None = Body(default=None),
) -> GrievanceLetter:
    adapters = request.app.state.adapters
    state = None

    if session_id:
        report, loan, notice, debt, language, state = await _from_session(session_id)
        borrower_name = None
    elif payload is not None:
        report = payload.report
        loan = payload.loan_facts
        notice = payload.notice_facts
        debt = payload.debt_analysis
        language = payload.language
        borrower_name = payload.borrower_name
    else:
        raise HTTPException(
            status_code=422, detail="Provide a session_id or a grievance draft request."
        )

    letter = await draft_grievance(
        report=report,
        loan=loan,
        notice=notice,
        debt=debt,
        language=language,
        borrower_name=borrower_name,
        llm=adapters.llm,
    )

    if state is not None:
        await get_store().update(state.model_copy(update={"grievance_letter": letter}))
    return letter


@router.get(
    "/grievance/{session_id}/download",
    response_class=PlainTextResponse,
    summary="Download the drafted letter as a text file",
)
async def download(session_id: str) -> PlainTextResponse:
    """Serves the already-drafted letter. Drafting happens in POST, so a
    download can never quietly produce something different from what the
    borrower was shown."""
    state = await get_store().get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    if state.grievance_letter is None:
        raise HTTPException(
            status_code=409, detail="No letter has been drafted in this session yet."
        )

    letter = state.grievance_letter
    body = (
        f"{letter.body_english}\n\n"
        f"{'-' * 72}\n"
        f"WHERE TO SEND THIS\n\n"
        + "\n\n".join(
            f"{index}. {a.role}\n{a.address_block}\n\n   Note: {a.note}"
            for index, a in enumerate(letter.addressees, start=1)
        )
        + f"\n\n{'-' * 72}\n{letter.disclaimer}\n"
    )
    return PlainTextResponse(
        content=body,
        headers={
            "Content-Disposition": f'attachment; filename="{letter.filename_suggestion}"'
        },
    )
