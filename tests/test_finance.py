"""Money math — the numbers a banker in the audience might check.

Organised as: closed-form identities, solver correctness (including a
round-trip that makes the solver prove itself against an independent formula),
edge cases, schedule invariants, and the end-to-end acceptance case.
"""

from __future__ import annotations

import math

import pytest

from backend.finance import (
    QuoteError,
    SolverError,
    analyse_debt,
    emi_from_flat,
    emi_from_reducing,
    flat_schedule,
    pv_factor,
    reducing_schedule,
    solve_monthly_rate,
    total_payable_flat,
)
from backend.schemas import RateType

# Rupee-level tolerance for display figures; the solver itself is far tighter.
RUPEE = 0.01


# ---------------------------------------------------------------------------
# Closed-form instalment arithmetic
# ---------------------------------------------------------------------------


def test_flat_emi_matches_the_hand_calculation() -> None:
    # 80,000 at 12% flat for 2 years: interest = 80000 * 0.12 * 2 = 19,200.
    assert total_payable_flat(80_000, 12.0, 24) == pytest.approx(99_200.0)
    assert emi_from_flat(80_000, 12.0, 24) == pytest.approx(99_200.0 / 24)


def test_reducing_emi_matches_the_annuity_formula() -> None:
    principal, annual, months = 80_000.0, 12.0, 24
    i = annual / 100 / 12
    expected = principal * i * (1 + i) ** months / ((1 + i) ** months - 1)
    assert emi_from_reducing(principal, annual, months) == pytest.approx(expected)


def test_flat_costs_more_than_reducing_at_the_same_headline_rate() -> None:
    """The entire premise of the product, asserted."""
    flat = emi_from_flat(80_000, 12.0, 24)
    reducing = emi_from_reducing(80_000, 12.0, 24)
    assert flat > reducing
    # Over two years the flat quote costs roughly 1.8-1.9x the honest interest.
    flat_interest = flat * 24 - 80_000
    reducing_interest = reducing * 24 - 80_000
    assert 1.7 < flat_interest / reducing_interest < 2.0


@pytest.mark.parametrize(
    "principal, rate, months",
    [(0, 12, 24), (-1, 12, 24), (80_000, -1, 24), (80_000, 12, 0), (80_000, 12, -3)],
)
def test_quote_rejects_impossible_inputs(principal: float, rate: float, months: int) -> None:
    with pytest.raises(QuoteError):
        emi_from_flat(principal, rate, months)
    with pytest.raises(QuoteError):
        emi_from_reducing(principal, rate, months)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def test_pv_factor_zero_rate_is_the_limit_not_a_division_error() -> None:
    assert pv_factor(0.0, 24) == 24.0


def test_pv_factor_is_strictly_decreasing_in_the_rate() -> None:
    values = [pv_factor(r, 24) for r in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)]
    assert all(a > b for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize("annual", [0.5, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0, 120.0])
@pytest.mark.parametrize("months", [2, 6, 12, 24, 36, 60, 120])
def test_solver_round_trips_against_the_annuity_formula(annual: float, months: int) -> None:
    """Build an EMI from a KNOWN reducing rate, then make the solver recover it.

    This is the strongest available check: the solver is validated against an
    independent closed form rather than against its own output.
    """
    principal = 100_000.0
    emi = emi_from_reducing(principal, annual, months)
    solution = solve_monthly_rate(principal, emi, months)
    assert solution.converged
    assert solution.nominal_apr == pytest.approx(annual, abs=1e-6)
    assert solution.residual < 1e-6


def test_solver_reports_both_annualisations_consistently() -> None:
    solution = solve_monthly_rate(80_000, emi_from_flat(80_000, 12.0, 24), 24)
    i = solution.monthly_rate
    assert solution.nominal_apr == pytest.approx(i * 12 * 100)
    assert solution.effective_annual_rate == pytest.approx(((1 + i) ** 12 - 1) * 100)
    # Compounding always makes the effective annual rate the larger number.
    assert solution.effective_annual_rate > solution.nominal_apr


def test_solver_handles_the_zero_interest_case() -> None:
    """0% BNPL: instalments exactly repay the principal."""
    solution = solve_monthly_rate(12_000, 1_000, 12)
    assert solution.converged
    assert solution.monthly_rate == 0.0
    assert solution.nominal_apr == 0.0
    assert solution.effective_annual_rate == 0.0
    assert solution.method == "closed_form"


def test_solver_handles_a_single_instalment() -> None:
    """One payment of 1,100 against 1,000 borrowed is 10% for that month."""
    solution = solve_monthly_rate(1_000, 1_100, 1)
    assert solution.converged
    assert solution.monthly_rate == pytest.approx(0.10)
    assert solution.nominal_apr == pytest.approx(120.0)
    assert solution.method == "closed_form"


def test_solver_rejects_a_schedule_that_never_repays_the_principal() -> None:
    with pytest.raises(SolverError, match="no non-negative interest rate"):
        solve_monthly_rate(100_000, 1_000, 24)  # 24,000 total against 100,000


@pytest.mark.parametrize("principal, emi, months", [(0, 100, 12), (1000, 0, 12), (1000, 100, 0)])
def test_solver_rejects_degenerate_inputs(principal: float, emi: float, months: int) -> None:
    with pytest.raises(SolverError):
        solve_monthly_rate(principal, emi, months)


def test_solver_converges_at_predatory_rates() -> None:
    """Informal lenders do charge 10%/month. The solver must not fall over."""
    principal = 10_000.0
    emi = emi_from_reducing(principal, 120.0, 12)  # 10% per month
    solution = solve_monthly_rate(principal, emi, 12)
    assert solution.converged
    assert solution.monthly_rate == pytest.approx(0.10, abs=1e-8)


def test_solver_always_converges_across_a_wide_sweep() -> None:
    """No input in a realistic range may leave converged=False."""
    failures = []
    for months in (1, 3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 240):
        for annual in (0.0, 1.0, 5.0, 9.5, 12.0, 15.0, 24.0, 36.0, 48.0, 72.0, 96.0):
            emi = emi_from_flat(50_000, annual, months)
            solution = solve_monthly_rate(50_000, emi, months)
            if not solution.converged or solution.residual > 1e-5:
                failures.append((months, annual, solution.residual))
    assert not failures, f"solver failed on {failures}"


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_schedules_have_one_point_per_month_plus_disbursal() -> None:
    emi = emi_from_flat(80_000, 12.0, 24)
    assert len(flat_schedule(80_000, 12.0, 24, emi)) == 25
    assert len(reducing_schedule(80_000, 0.0179, 24, emi)) == 25
    assert flat_schedule(80_000, 12.0, 24, emi)[0].month == 0


def test_both_schedules_fully_retire_the_debt() -> None:
    emi = emi_from_flat(80_000, 12.0, 24)
    solution = solve_monthly_rate(80_000, emi, 24)
    assert flat_schedule(80_000, 12.0, 24, emi)[-1].outstanding == pytest.approx(0.0, abs=1e-6)
    assert reducing_schedule(80_000, solution.monthly_rate, 24, emi)[-1].outstanding == (
        pytest.approx(0.0, abs=1e-6)
    )


def test_both_schedules_reach_the_same_total_because_the_cash_flows_are_identical() -> None:
    """Different shape, same money. If these diverged, one curve would be a lie."""
    emi = emi_from_flat(80_000, 12.0, 24)
    solution = solve_monthly_rate(80_000, emi, 24)
    flat = flat_schedule(80_000, 12.0, 24, emi)
    reducing = reducing_schedule(80_000, solution.monthly_rate, 24, emi)
    assert flat[-1].cumulative_paid == pytest.approx(reducing[-1].cumulative_paid, abs=RUPEE)
    assert flat[-1].cumulative_interest == pytest.approx(
        reducing[-1].cumulative_interest, abs=RUPEE
    )


def test_reducing_schedule_front_loads_interest_while_flat_does_not() -> None:
    """The visual the visualizer exists to show."""
    emi = emi_from_flat(80_000, 12.0, 24)
    solution = solve_monthly_rate(80_000, emi, 24)
    flat = flat_schedule(80_000, 12.0, 24, emi)
    reducing = reducing_schedule(80_000, solution.monthly_rate, 24, emi)

    # Flat books identical interest every month.
    assert flat[1].interest_paid_this_month == pytest.approx(flat[24].interest_paid_this_month)
    # Reducing charges far more interest early than late.
    assert reducing[1].interest_paid_this_month > reducing[24].interest_paid_this_month * 5


def test_cumulative_series_are_monotonic() -> None:
    emi = emi_from_flat(80_000, 12.0, 24)
    solution = solve_monthly_rate(80_000, emi, 24)
    for schedule in (
        flat_schedule(80_000, 12.0, 24, emi),
        reducing_schedule(80_000, solution.monthly_rate, 24, emi),
    ):
        for previous, current in zip(schedule, schedule[1:], strict=False):
            assert current.cumulative_paid >= previous.cumulative_paid
            assert current.cumulative_interest >= previous.cumulative_interest
            assert current.outstanding <= previous.outstanding + 1e-9


def test_fees_shift_the_curve_up_from_month_zero() -> None:
    emi = emi_from_flat(80_000, 12.0, 24)
    schedule = flat_schedule(80_000, 12.0, 24, emi, upfront_fees=3_200)
    assert schedule[0].cumulative_paid == pytest.approx(3_200)
    assert schedule[-1].cumulative_paid == pytest.approx(99_200 + 3_200)


# ---------------------------------------------------------------------------
# End-to-end analysis — the F5 acceptance case
# ---------------------------------------------------------------------------


def test_f5_acceptance_80k_12pc_24m() -> None:
    """Spec F5: 80,000 at flat 12% over 24 months => effective APR ~21-22%."""
    analysis = analyse_debt(80_000, 12.0, 24)
    assert 21.0 <= analysis.effective_apr <= 22.0, analysis.effective_apr
    assert analysis.emi == pytest.approx(4_133.33, abs=0.01)
    assert analysis.total_payable == pytest.approx(99_200.0)
    assert analysis.total_interest == pytest.approx(19_200.0)
    assert analysis.rate_solution.converged
    # The advertised rate understates the real one by roughly 9.5 points.
    assert analysis.hidden_rate_gap == pytest.approx(analysis.effective_apr - 12.0, abs=RUPEE)


def test_analysis_totals_are_internally_consistent() -> None:
    a = analyse_debt(80_000, 12.0, 24, processing_fee=3_200, other_fees_total=2_150)
    assert a.total_payable == pytest.approx(a.emi * a.tenure_months)
    assert a.total_interest == pytest.approx(a.total_payable - a.principal)
    assert a.all_in_outflow == pytest.approx(a.total_payable + 3_200 + 2_150)
    assert a.flat_schedule[-1].cumulative_paid == pytest.approx(a.all_in_outflow, abs=RUPEE)


def test_fees_raise_the_all_in_apr_above_the_fee_free_rate() -> None:
    without = analyse_debt(80_000, 12.0, 24)
    with_fees = analyse_debt(80_000, 12.0, 24, processing_fee=3_200, other_fees_total=2_150)
    assert with_fees.all_in_apr > with_fees.effective_apr
    assert with_fees.all_in_apr > without.all_in_apr
    assert with_fees.hidden_cost_gap > without.hidden_cost_gap


def test_hidden_cost_gap_is_measured_against_an_honest_loan() -> None:
    """The gap is actual outflow minus a true reducing-balance loan at 12%."""
    a = analyse_debt(80_000, 12.0, 24)
    honest_total = emi_from_reducing(80_000, 12.0, 24) * 24
    assert a.hidden_cost_gap == pytest.approx(a.all_in_outflow - honest_total, abs=RUPEE)
    assert a.hidden_cost_gap > 0


def test_an_honest_reducing_quote_shows_no_gap() -> None:
    """A lender who quotes reducing-balance correctly must not be accused."""
    a = analyse_debt(80_000, 12.0, 24, rate_type=RateType.REDUCING)
    assert a.effective_apr == pytest.approx(12.0, abs=1e-6)
    assert a.hidden_cost_gap == pytest.approx(0.0, abs=RUPEE)
    assert a.hidden_rate_gap == pytest.approx(0.0, abs=1e-6)


def test_unknown_rate_type_assumes_flat_and_says_so() -> None:
    a = analyse_debt(80_000, 12.0, 24, rate_type=RateType.UNKNOWN)
    assert a.emi == pytest.approx(emi_from_flat(80_000, 12.0, 24))
    assert any("flat or reducing" in w for w in a.warnings)
    assert a.assumptions


def test_zero_percent_loan_is_reported_as_genuinely_free() -> None:
    a = analyse_debt(24_000, 0.0, 12)
    assert a.effective_apr == pytest.approx(0.0)
    assert a.total_interest == pytest.approx(0.0)
    assert a.hidden_cost_gap == pytest.approx(0.0, abs=RUPEE)
    assert not a.warnings


def test_zero_percent_loan_with_fees_exposes_the_real_cost() -> None:
    """'0% EMI' plus a processing fee is not actually free — the APR must show it."""
    a = analyse_debt(24_000, 0.0, 12, processing_fee=1_500)
    assert a.effective_apr == pytest.approx(0.0)
    assert a.all_in_apr > 10.0
    assert a.hidden_cost_gap == pytest.approx(1_500, abs=RUPEE)


def test_single_month_loan_analyses_cleanly() -> None:
    a = analyse_debt(10_000, 24.0, 1)
    assert a.tenure_months == 1
    assert a.rate_solution.converged
    assert len(a.flat_schedule) == 2


def test_schedules_can_be_omitted_for_a_cheap_call() -> None:
    a = analyse_debt(80_000, 12.0, 24, include_schedules=False)
    assert a.flat_schedule == [] and a.reducing_schedule == []
    assert a.effective_apr > 0  # the rate work still happened


def test_every_reported_number_is_finite() -> None:
    """No NaN or inf may ever reach the UI or a letter."""
    for months in (1, 6, 24, 60, 240):
        for rate in (0.0, 12.0, 36.0, 120.0):
            a = analyse_debt(50_000, rate, months, processing_fee=1_000)
            for value in (
                a.emi,
                a.total_payable,
                a.total_interest,
                a.all_in_outflow,
                a.effective_apr,
                a.effective_annual_rate,
                a.all_in_apr,
                a.hidden_cost_gap,
                a.hidden_rate_gap,
            ):
                assert math.isfinite(value), (rate, months, value)
