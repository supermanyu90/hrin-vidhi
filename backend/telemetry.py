"""Per-route latency, kept in memory and reported at /metrics.

Averages hide the request that matters. A mean of 600ms can be a hundred
fast answers and one borrower waiting nine seconds, and it is that borrower
who gives up. So this records percentiles: P50 is the ordinary experience,
P95 and P99 are the tail that decides whether the thing feels reliable.

Deliberately small. No exporter, no agent, no dependency — a ring of recent
durations per route, which is all that is needed to answer "is anything
slow?" during a demo or after a deploy. Per instance and lost on a cold
start, in exchange for costing nothing to run.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

#: How many recent requests to keep per route. Enough for a stable P99
#: without holding meaningful memory.
_WINDOW = 200

_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_WINDOW))

#: What each route should manage, in milliseconds, measured at P95. A budget
#: nobody wrote down is a budget nobody misses.
BUDGETS_MS: dict[str, float] = {
    "POST /ask": 2000,               # cached or vernacular; a model call blows this
    "POST /calc/debt-trap": 100,     # pure arithmetic
    "POST /analysis/compliance": 150,  # deterministic rules
    "GET /health": 100,
    "POST /intake/document": 30000,  # a vision call on a photographed page
    "POST /intake/voice": 15000,     # recognition then translation
}


def record(route: str, seconds: float) -> None:
    _samples[route].append(seconds * 1000.0)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest-rank: with a window this size, interpolation implies a
    # precision the sample does not have.
    index = min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1)
    return round(ordered[index], 1)


def snapshot() -> dict:
    """Percentiles per route, and whether each is inside its budget."""
    out: dict[str, dict] = {}
    for route, window in _samples.items():
        values = list(window)
        if not values:
            continue
        p95 = _percentile(values, 95)
        budget = BUDGETS_MS.get(route)
        out[route] = {
            "count": len(values),
            "p50_ms": _percentile(values, 50),
            "p95_ms": p95,
            "p99_ms": _percentile(values, 99),
            "max_ms": round(max(values), 1),
            "budget_ms": budget,
            "within_budget": None if budget is None else p95 <= budget,
        }
    return {
        "window": _WINDOW,
        "routes": dict(sorted(out.items())),
        "over_budget": sorted(r for r, v in out.items() if v["within_budget"] is False),
    }


def reset() -> None:
    _samples.clear()


class Timer:
    """Times a block and records it under `route`."""

    __slots__ = ("route", "_started")

    def __init__(self, route: str) -> None:
        self.route = route

    def __enter__(self) -> Timer:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        record(self.route, time.perf_counter() - self._started)
        return None
