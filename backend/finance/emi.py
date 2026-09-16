"""Instalment arithmetic — the two ways a lender can quote the same loan.

The gap between these two functions *is* the product. A flat quote charges
interest on the original principal for the whole term, ignoring every rupee
already repaid; a reducing-balance quote charges only on what is still owed.
At the same headline percentage the flat version costs roughly twice as much.
"""

from __future__ import annotations


class QuoteError(ValueError):
    """Raised for inputs that cannot describe a real loan."""


def emi_from_flat(principal: float, annual_rate_percent: float, months: int) -> float:
    """EMI when interest is quoted flat on the original principal.

        total_payable = P + P * r * years
        EMI           = total_payable / months

    This is the arithmetic the borrower's paper actually encodes — no
    discounting, no amortisation.
    """
    _validate(principal, annual_rate_percent, months)
    years = months / 12.0
    total_payable = principal + principal * (annual_rate_percent / 100.0) * years
    return total_payable / months


def total_payable_flat(principal: float, annual_rate_percent: float, months: int) -> float:
    _validate(principal, annual_rate_percent, months)
    years = months / 12.0
    return principal + principal * (annual_rate_percent / 100.0) * years


def emi_from_reducing(principal: float, annual_rate_percent: float, months: int) -> float:
    """EMI when interest accrues on the outstanding balance (a normal bank loan).

        EMI = P * i * (1+i)^n / ((1+i)^n - 1),   i = annual / 12

    Used to answer "what should this loan have cost?", which is the honest
    baseline the hidden-cost gap is measured against.
    """
    _validate(principal, annual_rate_percent, months)
    monthly = annual_rate_percent / 100.0 / 12.0
    if monthly == 0.0:
        return principal / months
    growth = (1.0 + monthly) ** months
    return principal * monthly * growth / (growth - 1.0)


def _validate(principal: float, annual_rate_percent: float, months: int) -> None:
    if principal <= 0:
        raise QuoteError("principal must be positive")
    if annual_rate_percent < 0:
        raise QuoteError("annual rate cannot be negative")
    if months <= 0:
        raise QuoteError("tenure must be at least one month")
