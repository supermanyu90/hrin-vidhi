"""Numerical solver for the effective monthly interest rate.

Given a principal, a fixed instalment and a term, find the monthly rate `i`
satisfying the annuity present-value identity:

    P = EMI * (1 - (1 + i)^-n) / i

This is the credibility core of the whole product: it converts "12% flat" into
the reducing-balance rate a bank would have to advertise for the same cash
flows. The number goes on screen and into a letter addressed to a regulator, so
the solver reports whether it converged and what residual it left behind rather
than silently returning a plausible float.

Method: Newton-Raphson from a good initial guess, with an unconditional
bisection fallback. Bisection cannot fail here — `pv_factor` is continuous and
strictly decreasing in `i`, so a sign change brackets exactly one root.

Stretch (not in the MVP): this module is pure-Python and dependency-free by
design. If the solver ever became a hot path it could be swapped for a
Rust/PyO3 implementation behind the same `solve_monthly_rate` signature.
"""

from __future__ import annotations

from backend.schemas import RateSolution

#: Absolute tolerance on the present-value residual, in rupees. A tenth of a
#: paisa is far below anything we display or assert in a letter.
DEFAULT_TOLERANCE = 1e-7

#: Upper bracket for bisection: 100% per month (~130,000% APR). No informal
#: lender in the corpus approaches this; it exists so the bracket is provably
#: wide enough rather than tuned to a sample.
MAX_MONTHLY_RATE = 1.0

MAX_NEWTON_ITERATIONS = 60
MAX_BISECTION_ITERATIONS = 200


class SolverError(ValueError):
    """Raised when the inputs cannot describe a real loan."""


def pv_factor(monthly_rate: float, months: int) -> float:
    """Present value of 1 currency unit paid monthly for `months` months.

    The `i == 0` branch is the mathematical limit, not a special case bolted on:
    as i -> 0 the annuity factor tends to n. Handling it explicitly avoids a
    0/0 division for interest-free loans, which do occur (BNPL "0% EMI").
    """
    if months <= 0:
        raise SolverError("months must be positive")
    if monthly_rate == 0.0:
        return float(months)
    return (1.0 - (1.0 + monthly_rate) ** -months) / monthly_rate


def _pv_derivative(monthly_rate: float, months: int) -> float:
    """d/di of `pv_factor`. Strictly negative for i > 0."""
    if monthly_rate == 0.0:
        # Limit of the derivative as i -> 0.
        return -months * (months + 1) / 2.0
    n, i = months, monthly_rate
    compounded = (1.0 + i) ** -n
    return (n * compounded / (i * (1.0 + i))) - ((1.0 - compounded) / (i * i))


def solve_monthly_rate(
    principal: float,
    emi: float,
    months: int,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> RateSolution:
    """Solve for the monthly rate implied by (principal, emi, months).

    Returns a `RateSolution` carrying the rate, both annualisations, and the
    audit trail (method, iterations, residual, convergence flag).
    """
    if principal <= 0:
        raise SolverError("principal must be positive")
    if emi <= 0:
        raise SolverError("emi must be positive")
    if months <= 0:
        raise SolverError("months must be positive")

    total_paid = emi * months

    if total_paid < principal - tolerance:
        raise SolverError(
            f"instalments total {total_paid:.2f} but principal is {principal:.2f}; "
            "no non-negative interest rate can produce this schedule"
        )

    # Interest-free: the instalments exactly repay the principal.
    if abs(total_paid - principal) <= max(tolerance, principal * 1e-12):
        return _solution(0.0, iterations=0, converged=True, method="closed_form", residual=0.0)

    # Single instalment: solvable directly, no iteration needed.
    if months == 1:
        rate = emi / principal - 1.0
        return _solution(
            rate,
            iterations=0,
            converged=True,
            method="closed_form",
            residual=abs(emi * pv_factor(rate, 1) - principal),
        )

    def residual_at(rate: float) -> float:
        """Positive below the true rate, negative above it."""
        return emi * pv_factor(rate, months) - principal

    newton = _try_newton(principal, emi, months, residual_at, tolerance)
    if newton is not None:
        return newton
    return _bisect(residual_at, months, tolerance)


def _try_newton(
    principal: float,
    emi: float,
    months: int,
    residual_at,
    tolerance: float,
) -> RateSolution | None:
    """Newton-Raphson. Returns None if it wanders out of bracket or stalls."""
    # Initial guess: the flat-rate approximation, which is close enough to the
    # reducing-balance answer that Newton converges in a handful of steps.
    total_interest = emi * months - principal
    guess = (2.0 * total_interest) / (principal * (months + 1))
    guess = min(max(guess, 1e-9), MAX_MONTHLY_RATE * 0.5)

    rate = guess
    for iteration in range(1, MAX_NEWTON_ITERATIONS + 1):
        value = residual_at(rate)
        if abs(value) <= tolerance:
            return _solution(
                rate,
                iterations=iteration,
                converged=True,
                method="newton",
                residual=abs(value),
            )
        derivative = emi * _pv_derivative(rate, months)
        if derivative == 0.0 or not _is_finite(derivative):
            return None
        step = value / derivative
        next_rate = rate - step
        # Leaving the valid domain means the guess was poor; hand over to
        # bisection rather than clamping and pretending we converged.
        if not _is_finite(next_rate) or next_rate <= 0.0 or next_rate >= MAX_MONTHLY_RATE:
            return None
        if abs(next_rate - rate) <= 1e-15:
            value = residual_at(next_rate)
            return _solution(
                next_rate,
                iterations=iteration,
                converged=abs(value) <= tolerance,
                method="newton",
                residual=abs(value),
            )
        rate = next_rate
    return None


def _bisect(residual_at, months: int, tolerance: float) -> RateSolution:
    """Guaranteed fallback. `residual_at` is strictly decreasing in the rate."""
    low, high = 0.0, MAX_MONTHLY_RATE
    if residual_at(high) > 0:
        # The true rate exceeds 100%/month — report the ceiling, flagged.
        value = residual_at(high)
        return _solution(
            high,
            iterations=0,
            converged=False,
            method="bisection",
            residual=abs(value),
        )

    mid = low
    for iteration in range(1, MAX_BISECTION_ITERATIONS + 1):
        mid = (low + high) / 2.0
        value = residual_at(mid)
        if abs(value) <= tolerance or (high - low) < 1e-15:
            return _solution(
                mid,
                iterations=iteration,
                converged=abs(value) <= tolerance,
                method="bisection",
                residual=abs(value),
            )
        if value > 0:
            low = mid
        else:
            high = mid

    return _solution(
        mid,
        iterations=MAX_BISECTION_ITERATIONS,
        converged=False,
        method="bisection",
        residual=abs(residual_at(mid)),
    )


def _solution(
    monthly_rate: float,
    *,
    iterations: int,
    converged: bool,
    method: str,
    residual: float,
) -> RateSolution:
    return RateSolution(
        monthly_rate=monthly_rate,
        nominal_apr=monthly_rate * 12.0 * 100.0,
        effective_annual_rate=((1.0 + monthly_rate) ** 12 - 1.0) * 100.0,
        iterations=iterations,
        converged=converged,
        method=method,  # type: ignore[arg-type]
        residual=residual,
    )


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
