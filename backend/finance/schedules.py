"""Month-by-month cost curves for the debt-trap visualizer (F5).

Two schedules over the *same* EMI stream:

  * `flat_schedule`      — how the lender books it. Interest is a fixed monthly
                           charge on the original principal; principal repays
                           in equal slices. This is what the borrower's paper says.
  * `reducing_schedule`  — the same payments re-expressed at the solved
                           effective rate. Early months are mostly interest;
                           the split shifts toward principal over the term.

Both end at the same cumulative outflow — they have to, since it is one set of
cash flows. What differs is the *shape*, and the shape is what shows a judge
(and a borrower) where the money actually goes.
"""

from __future__ import annotations

from backend.schemas import SchedulePoint


def flat_schedule(
    principal: float,
    annual_rate_percent: float,
    months: int,
    emi: float,
    *,
    upfront_fees: float = 0.0,
) -> list[SchedulePoint]:
    """The lender's own arithmetic, month by month.

    `upfront_fees` is charged at month 0 so the cumulative-cost curve starts
    at the true out-of-pocket amount rather than zero.
    """
    monthly_interest = principal * (annual_rate_percent / 100.0) / 12.0
    monthly_principal = principal / months

    points = [
        SchedulePoint(
            month=0,
            outstanding=principal,
            interest_paid_this_month=0.0,
            principal_paid_this_month=0.0,
            cumulative_interest=0.0,
            cumulative_paid=upfront_fees,
        )
    ]

    cumulative_interest = 0.0
    cumulative_paid = upfront_fees
    outstanding = principal

    for month in range(1, months + 1):
        cumulative_interest += monthly_interest
        cumulative_paid += emi
        outstanding -= monthly_principal
        points.append(
            SchedulePoint(
                month=month,
                # Guard the last month against float drift showing -0.0000001.
                outstanding=max(0.0, outstanding),
                interest_paid_this_month=monthly_interest,
                principal_paid_this_month=monthly_principal,
                cumulative_interest=cumulative_interest,
                cumulative_paid=cumulative_paid,
            )
        )
    return points


def reducing_schedule(
    principal: float,
    monthly_rate: float,
    months: int,
    emi: float,
    *,
    upfront_fees: float = 0.0,
) -> list[SchedulePoint]:
    """The same EMIs amortised at `monthly_rate` (the solved effective rate)."""
    points = [
        SchedulePoint(
            month=0,
            outstanding=principal,
            interest_paid_this_month=0.0,
            principal_paid_this_month=0.0,
            cumulative_interest=0.0,
            cumulative_paid=upfront_fees,
        )
    ]

    outstanding = principal
    cumulative_interest = 0.0
    cumulative_paid = upfront_fees

    for month in range(1, months + 1):
        interest = outstanding * monthly_rate
        principal_component = emi - interest

        if month == months:
            # Absorb accumulated float error into the final instalment so the
            # balance lands exactly on zero rather than a few paise either way.
            principal_component = outstanding

        outstanding = max(0.0, outstanding - principal_component)
        cumulative_interest += interest
        cumulative_paid += emi

        points.append(
            SchedulePoint(
                month=month,
                outstanding=outstanding,
                interest_paid_this_month=interest,
                principal_paid_this_month=principal_component,
                cumulative_interest=cumulative_interest,
                cumulative_paid=cumulative_paid,
            )
        )
    return points
