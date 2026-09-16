"""F5 — the /calc/debt-trap contract the visualizer depends on."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from backend.main import app

    return TestClient(app)


BASE = {"principal": 80_000, "quoted_rate": 12, "tenure_months": 24}


def post(client: TestClient, **overrides) -> dict:
    response = client.post("/calc/debt-trap", json={**BASE, **overrides})
    assert response.status_code == 200, response.text
    return response.json()


def test_f5_acceptance_over_http(client: TestClient) -> None:
    """Spec F5: 80,000 / flat 12% / 24 months => effective APR ~21-22%."""
    body = post(client)
    assert 21.0 <= body["effective_apr"] <= 22.0
    assert body["emi"] == pytest.approx(4_133.33, abs=0.01)
    assert body["hidden_cost_gap"] > 0


def test_response_carries_all_three_curves_of_equal_length(client: TestClient) -> None:
    body = post(client)
    expected = BASE["tenure_months"] + 1  # month 0 is disbursal
    for key in ("flat_schedule", "reducing_schedule", "honest_schedule"):
        assert len(body[key]) == expected, key


def test_the_plotted_gap_equals_the_headline_gap(client: TestClient) -> None:
    """The chart's visual gap and the summary strip's number must be one value."""
    body = post(client, processing_fee=3_200, other_fees_total=2_150)
    plotted = (
        body["flat_schedule"][-1]["cumulative_paid"]
        - body["honest_schedule"][-1]["cumulative_paid"]
    )
    assert plotted == pytest.approx(body["hidden_cost_gap"], abs=0.01)


def test_an_honest_reducing_quote_with_no_fees_reports_no_gap(client: TestClient) -> None:
    body = post(client, rate_type="reducing")
    assert body["effective_apr"] == pytest.approx(12.0, abs=1e-6)
    assert body["hidden_cost_gap"] == pytest.approx(0.0, abs=0.01)


def test_fees_alone_raise_the_all_in_apr_without_inflating_the_rate(client: TestClient) -> None:
    """Drives the UI's 'the rate is honest, the fees are not' verdict."""
    body = post(client, rate_type="reducing", processing_fee=3_200, other_fees_total=2_150)
    assert body["effective_apr"] == pytest.approx(12.0, abs=1e-6)
    assert body["all_in_apr"] > 12.0
    assert body["hidden_cost_gap"] == pytest.approx(5_350.0, abs=0.01)


def test_unknown_rate_type_warns_that_flat_was_assumed(client: TestClient) -> None:
    body = post(client, rate_type="unknown")
    assert any("flat or reducing" in w for w in body["warnings"])


@pytest.mark.parametrize(
    "payload",
    [
        {"principal": 0},
        {"principal": -5},
        {"tenure_months": 0},
        {"tenure_months": 1000},
        {"quoted_rate": -1},
        {"quoted_rate": 500},
        {"processing_fee": -1},
        {"rate_type": "made_up"},
    ],
)
def test_bad_input_is_rejected_with_422(client: TestClient, payload: dict) -> None:
    response = client.post("/calc/debt-trap", json={**BASE, **payload})
    assert response.status_code == 422


def test_missing_fields_are_rejected(client: TestClient) -> None:
    assert client.post("/calc/debt-trap", json={"principal": 80_000}).status_code == 422


def test_fees_are_optional(client: TestClient) -> None:
    body = post(client)
    assert body["processing_fee"] == 0.0
    assert body["other_fees_total"] == 0.0
