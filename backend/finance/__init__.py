"""Standalone, dependency-light loan mathematics.

Importable on its own — nothing here touches FastAPI, the adapters, or any
network. `from backend.finance import analyse_debt` is the whole public API;
the submodules are exposed for unit testing and reuse.
"""

from backend.finance.analysis import analyse_debt
from backend.finance.emi import (
    QuoteError,
    emi_from_flat,
    emi_from_reducing,
    total_payable_flat,
)
from backend.finance.schedules import flat_schedule, reducing_schedule
from backend.finance.solver import SolverError, pv_factor, solve_monthly_rate

__all__ = [
    "QuoteError",
    "SolverError",
    "analyse_debt",
    "emi_from_flat",
    "emi_from_reducing",
    "flat_schedule",
    "pv_factor",
    "reducing_schedule",
    "solve_monthly_rate",
    "total_payable_flat",
]
