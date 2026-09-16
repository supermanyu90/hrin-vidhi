"""HTTP contracts for F3 compliance and grounded Q&A.

Intake routes are covered in `test_intake.py`.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from backend.main import app

    return TestClient(app)


def upload(client: TestClient, *, hint: str, content: bytes = b"fake-jpeg-bytes",
           content_type: str = "image/jpeg") -> dict:
    response = client.post(
        "/intake/document",
        files={"file": ("loan.jpg", io.BytesIO(content), content_type)},
        data={"hint": hint},
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# F3 — /analysis/compliance
# ---------------------------------------------------------------------------


def test_compliance_on_the_seeded_story(client: TestClient) -> None:
    loan = upload(client, hint="loan paper")["loan_facts"]
    notice = upload(client, hint="notice")["notice_facts"]

    response = client.post(
        "/analysis/compliance", json={"loan_facts": loan, "notice_facts": notice}
    )
    assert response.status_code == 200, response.text
    report = response.json()

    rule_ids = {v["rule_id"] for v in report["violations"]}
    assert "R-001-recovery-hours" in rule_ids
    assert "R-002-threat-family-contact" in rule_ids
    assert "R-003-court-notice-misrepresentation" in rule_ids

    flag_ids = {f["flag_id"] for f in report["flags"]}
    assert "F-001-apr-understated" in flag_ids
    assert "F-003-nbfc-registration-unverified" in flag_ids

    assert report["unconfirmed"] == []
    assert report["citations"]
    assert report["overall_summary"]


def test_every_asserted_violation_ships_with_a_resolvable_citation(client: TestClient) -> None:
    notice = upload(client, hint="notice")["notice_facts"]
    report = client.post("/analysis/compliance", json={"notice_facts": notice}).json()
    for violation in report["violations"]:
        assert violation["citations"], violation["rule_id"]
        for citation in violation["citations"]:
            assert citation["chunk_id"]
            assert citation["is_summary"] is True


def test_compliance_needs_something_to_analyse(client: TestClient) -> None:
    response = client.post("/analysis/compliance", json={})
    assert response.status_code == 422


def test_partial_loan_facts_still_analyse_but_say_what_was_skipped(client: TestClient) -> None:
    """A photo that yields no principal must not silently drop the rate check."""
    response = client.post(
        "/analysis/compliance",
        json={"loan_facts": {"lender_name": "Some Finserv", "quoted_rate": 12.0}},
    )
    assert response.status_code == 200
    report = response.json()
    assert any("could not be run" in note for note in report["unconfirmed"])
    # The checks that do not need the money math still ran.
    assert any(f["flag_id"] == "F-003-nbfc-registration-unverified" for f in report["flags"])


def test_notice_only_analysis_works(client: TestClient) -> None:
    notice = upload(client, hint="notice")["notice_facts"]
    report = client.post("/analysis/compliance", json={"notice_facts": notice}).json()
    assert report["violations"]


# ---------------------------------------------------------------------------
# F3 — /ask
# ---------------------------------------------------------------------------


def test_ask_returns_a_grounded_cited_answer(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Can they call me at 6 in the morning?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["citations"]
    assert "rbi-fpc-recovery-hours" in {c["chunk_id"] for c in body["citations"]}


def test_ask_declines_an_off_topic_question(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Who won the cricket match last night?"})
    body = response.json()
    assert body["grounded"] is False
    assert body["citations"] == []
    assert "could not find" in body["answer"].lower()


def test_ask_validates_the_question(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "y" * 501}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


# ---------------------------------------------------------------------------
# /corpus/status
# ---------------------------------------------------------------------------


def test_corpus_status_shows_what_retrieval_is_running_on(client: TestClient) -> None:
    body = client.get("/corpus/status").json()
    assert body["chunk_count"] >= 25
    assert body["retriever"]
    assert body["all_summaries"] is True
    assert body["reviewed_by_lawyer"] == 0
    assert len(body["sources"]) >= 5


def test_health_reports_the_corpus_is_loaded(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["corpus_chunks"] >= 25
