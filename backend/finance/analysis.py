"""Assembles the full `DebtAnalysis` consumed by the visualizer (F5) and the
grievance letter (F6).

Three rates are reported, and the distinction matters enough to state plainly
because the UI labels them and a letter may quote them:

  * `effective_apr`          nominal APR on the EMI stream. i * 12. Directly
                             comparable to a bank's advertised "X% p.a.".
  * `effective_annual_rate`  the same rate compounded: (1+i)^12 - 1. Always
                             the larger number; it is the true annual cost of
                             money but is NOT what lenders advertise.
  * `all_in_apr`             nominal APR once fees are treated as a reduction
                             in the amount actually disbursed. This is the
                             closest thing to the RBI Digital Lending APR and
                             is the honest headline figure.

The hidden-cost gap is measured against what the loan *should* have cost if
the quoted percentage had been a genuine reducing-balance rate. That is the
comparison the borrower was implicitly promised.
"""

from __future__ import annotations

from backend.finance.emi import emi_from_flat, emi_from_reducing
from backend.finance.schedules import flat_schedule, reducing_schedule
from backend.finance.solver import solve_monthly_rate
from backend.schemas import DebtAnalysis, RateType


def analyse_debt(
    principal: float,
    quoted_rate: float,
    tenure_months: int,
    *,
    rate_type: RateType = RateType.FLAT,
    processing_fee: float = 0.0,
    other_fees_total: float = 0.0,
    include_schedules: bool = True,
) -> DebtAnalysis:
    """Full money-math analysis of one loan quote."""
    warnings: list[str] = []
    assumptions: list[str] = []

    fees_total = processing_fee + other_fees_total

    # --- EMI implied by the quote -----------------------------------------
    if rate_type is RateType.REDUCING:
        emi = emi_from_reducing(principal, quoted_rate, tenure_months)
        assumptions.append(
            "Lender quoted a reducing-balance rate, so the EMI is the standard "
            "annuity instalment."
        )
    else:
        emi = emi_from_flat(principal, quoted_rate, tenure_months)
        if rate_type is RateType.UNKNOWN:
            warnings.append(
                "The document does not say whether the rate is flat or reducing. "
                "Flat has been assumed because it is the norm for this kind of "
                "loan — confirm against the sanction letter before relying on "
                "the figures below."
            )
            assumptions.append("Rate type unstated; treated as flat.")
        else:
            assumptions.append(
                "Lender quoted a flat rate: interest is charged on the full "
                "original principal for the whole term, ignoring repayments."
            )

    total_payable = emi * tenure_months
    total_interest = total_payable - principal
    all_in_outflow = total_payable + fees_total

    # --- Solve the true rate on the EMI stream (no fees) -------------------
    solution = solve_monthly_rate(principal, emi, tenure_months)
    if not solution.converged:
        warnings.append(
            "The effective-rate solver did not fully converge; the rate shown is "
            "the closest bound found. Treat it as indicative."
        )

    # --- All-in rate: fees reduce what the borrower actually received ------
    net_disbursed = principal - fees_total
    if net_disbursed <= 0:
        all_in_apr = solution.nominal_apr
        warnings.append(
            "Fees equal or exceed the loan amount, so an all-in APR cannot be "
            "computed. The rate shown excludes fees."
        )
    elif fees_total > 0:
        all_in_solution = solve_monthly_rate(net_disbursed, emi, tenure_months)
        all_in_apr = all_in_solution.nominal_apr
        if not all_in_solution.converged:
            warnings.append("The all-in APR solver did not fully converge.")
        assumptions.append(
            f"Fees of Rs {fees_total:,.0f} are treated as deducted at disbursal, "
            f"so the borrower received Rs {net_disbursed:,.0f} but repays on "
            f"Rs {principal:,.0f}."
        )
    else:
        all_in_apr = solution.nominal_apr

    # --- The hidden-cost gap ----------------------------------------------
    # What an honest reducing-balance loan at the SAME advertised percentage
    # would have cost. The difference is what the flat quote plus undisclosed
    # fees actually took.
    honest_emi = emi_from_reducing(principal, quoted_rate, tenure_months)
    honest_total = honest_emi * tenure_months
    hidden_cost_gap = all_in_outflow - honest_total
    hidden_rate_gap = all_in_apr - quoted_rate

    assumptions.append(
        "The hidden-cost gap compares what is actually repaid (including fees) "
        f"with what a genuine reducing-balance loan at the same advertised "
        f"{quoted_rate:g}% would have cost."
    )

    flat_points = []
    reducing_points = []
    honest_points = []
    if include_schedules:
        flat_points = flat_schedule(
            principal, quoted_rate, tenure_months, emi, upfront_fees=fees_total
        )
        reducing_points = reducing_schedule(
            principal, solution.monthly_rate, tenure_months, emi, upfront_fees=fees_total
        )
        # The advertised-vs-actual baseline. No fees: this is the loan the
        # borrower believed they were signing.
        honest_points = reducing_schedule(
            principal, quoted_rate / 100.0 / 12.0, tenure_months, honest_emi
        )

    return DebtAnalysis(
        principal=principal,
        quoted_rate=quoted_rate,
        rate_type=rate_type,
        tenure_months=tenure_months,
        emi=emi,
        total_payable=total_payable,
        total_interest=total_interest,
        processing_fee=processing_fee,
        other_fees_total=other_fees_total,
        all_in_outflow=all_in_outflow,
        rate_solution=solution,
        effective_apr=solution.nominal_apr,
        effective_annual_rate=solution.effective_annual_rate,
        all_in_apr=all_in_apr,
        honest_emi=honest_emi,
        honest_total=honest_total,
        hidden_cost_gap=hidden_cost_gap,
        hidden_rate_gap=hidden_rate_gap,
        flat_schedule=flat_points,
        reducing_schedule=reducing_points,
        honest_schedule=honest_points,
        warnings=warnings,
        assumptions=assumptions,
    )
