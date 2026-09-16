"""F5 — /calc/debt-trap.

The visualizer holds no loan mathematics of its own. It posts parameters here
and plots exactly what comes back, so the chart, the summary strip and any
figure quoted in a grievance letter are guaranteed to be the same numbers.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.finance import SolverError, analyse_debt
from backend.finance.emi import QuoteError
from backend.schemas import DebtAnalysis, DebtTrapRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/calc", tags=["money math"])


@router.post(
    "/debt-trap",
    response_model=DebtAnalysis,
    summary="Convert a flat-rate quote into its true reducing-balance cost",
)
async def debt_trap(request: DebtTrapRequest) -> DebtAnalysis:
    """Full analysis: EMI, effective APR, all-in APR, hidden gap, both curves.

    Pydantic bounds the inputs, so the handler only has to translate the
    finance module's own errors into a 422 the UI can show.
    """
    try:
        return analyse_debt(
            principal=request.principal,
            quoted_rate=request.quoted_rate,
            tenure_months=request.tenure_months,
            rate_type=request.rate_type,
            processing_fee=request.processing_fee,
            other_fees_total=request.other_fees_total,
        )
    except (QuoteError, SolverError) as exc:
        log.info("Rejected debt-trap request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
