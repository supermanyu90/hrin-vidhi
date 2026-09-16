"""Constraints that only bite once this is deployed to a serverless host.

Each of these was a real failure found by simulating Vercel: a response that
exceeded the platform's size cap, a voice note that was silently fake on
Linux, and a UI that depended on server memory surviving a cold start.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import FRONTEND_DIR, PROJECT_ROOT

#: Vercel refuses a serverless response larger than this.
SERVERLESS_RESPONSE_LIMIT = 4.5 * 1024 * 1024


@pytest.fixture
def client() -> TestClient:
    from backend.main import app

    return TestClient(app)


@pytest.fixture
def linux_host(monkeypatch):
    """Pretend no system speech binary exists — every Linux host, so every deploy."""
    real = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda n, *a, **k: None if n == "say" else real(n, *a, **k)
    )


def full_report(client: TestClient) -> tuple[dict, dict, dict]:
    loan = client.post(
        "/intake/document",
        files={"file": ("l.jpg", b"\xff\xd8x", "image/jpeg")},
        data={"hint": "loan paper"},
    ).json()["loan_facts"]
    notice = client.post(
        "/intake/document",
        files={"file": ("n.jpg", b"\xff\xd8y", "image/jpeg")},
        data={"hint": "notice"},
    ).json()["notice_facts"]
    report = client.post(
        "/analysis/compliance", json={"loan_facts": loan, "notice_facts": notice}
    ).json()
    return loan, notice, report


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def test_the_vercel_entry_point_exposes_the_app() -> None:
    from api.index import app as vercel_app
    from backend.main import app as local_app

    assert vercel_app is local_app, "the deployed app must be the app we test"


def test_deployment_config_is_present_and_routes_everything() -> None:
    import json

    config = json.loads((PROJECT_ROOT / "vercel.json").read_text())
    assert config["rewrites"][0]["source"] == "/(.*)"
    assert config["rewrites"][0]["destination"] == "/api/index"
    # The corpus, fixtures and frontend must ship with the function or the
    # deployed app answers with an empty corpus and no pages.
    included = config["functions"]["api/index.py"]["includeFiles"]
    for needed in ("backend", "frontend", "fixtures"):
        assert needed in included, needed


def test_data_files_resolve_independently_of_the_working_directory() -> None:
    """A serverless function is not invoked from the repo root."""
    from backend.analysis.corpus_store import CORPUS_DIR
    from backend.explainer.script import PHRASEBOOK_PATH

    for path in (CORPUS_DIR, PHRASEBOOK_PATH, FRONTEND_DIR):
        assert Path(path).is_absolute(), f"{path} is relative and will break when deployed"


# ---------------------------------------------------------------------------
# Response size
# ---------------------------------------------------------------------------


def test_no_response_approaches_the_serverless_size_limit(client: TestClient, linux_host) -> None:
    """The rights response once carried ~6MB of base64 audio and would have
    been rejected outright."""
    loan, notice, report = full_report(client)
    debt = client.post(
        "/calc/debt-trap",
        json={
            "principal": loan["principal"],
            "quoted_rate": loan["quoted_rate"],
            "tenure_months": loan["tenure_months"],
            "rate_type": loan["rate_type"],
            "processing_fee": loan["processing_fee"],
            "other_fees_total": sum(f["amount"] or 0 for f in loan["other_fees"]),
        },
    ).json()

    responses = {
        "/": client.get("/"),
        "/app": client.get("/app"),
        "/visualizer": client.get("/visualizer"),
        "/analysis/compliance": client.post(
            "/analysis/compliance", json={"loan_facts": loan, "notice_facts": notice}
        ),
        "/explain/rights": client.post(
            "/explain/rights",
            json={"report": report, "debt_analysis": debt, "language": "mr"},
        ),
        "/grievance/draft": client.post(
            "/grievance/draft",
            json={"payload": {"report": report, "loan_facts": loan,
                              "notice_facts": notice, "debt_analysis": debt,
                              "language": "mr"}},
        ),
    }
    for name, response in responses.items():
        assert response.status_code == 200, name
        assert len(response.content) < SERVERLESS_RESPONSE_LIMIT, (
            f"{name} is {len(response.content) / 1e6:.1f}MB"
        )


# ---------------------------------------------------------------------------
# Honest audio
# ---------------------------------------------------------------------------


def test_on_a_host_without_voices_no_audio_is_claimed(client: TestClient, linux_host) -> None:
    """Returning a silent WAV would look like a voice note to the UI and be
    ~6MB of base64. Returning nothing is smaller and truthful."""
    _, _, report = full_report(client)
    rights = client.post(
        "/explain/rights", json={"report": report, "language": "mr"}
    ).json()

    note = rights["voice_note"]
    assert note["audio_base64"] is None
    assert note["provider"] == "mock-unavailable"
    # The script still comes back — the browser speaks it.
    assert len(rights["script_native"]) > 200
    assert rights["disclaimer_native"].strip()


def test_the_page_can_speak_every_language_it_offers() -> None:
    """Marathi has no browser voice on any common platform, so it must fall
    back to a Devanagari-reading voice rather than silently doing nothing."""
    source = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    block = re.search(r"const SPEECH_TAGS = \{(.*?)\n\};", source, re.S)
    assert block, "the speech fallback table is gone"

    tags = dict(re.findall(r"(\w+):\s*\[([^\]]*)\]", block.group(1)))
    assert set(tags) == {"hi", "mr", "bho", "ta", "te", "en"}
    for code, chain in tags.items():
        entries = [t.strip().strip("'\"") for t in chain.split(",") if t.strip()]
        assert entries, code
        if code in ("mr", "bho"):
            assert any(t.startswith("hi") for t in entries), (
                f"{code} has no voice of its own and no Hindi fallback"
            )


# ---------------------------------------------------------------------------
# Statelessness
# ---------------------------------------------------------------------------


def test_the_ui_never_depends_on_a_server_session() -> None:
    """Serverless invocations do not share memory, so a session id in the UI
    would vanish on a cold start mid-demo."""
    source = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "session_id" not in source
    assert "/session/" not in source
    assert "sessionId" not in source


def test_the_whole_pipeline_runs_with_no_session(client: TestClient) -> None:
    loan, notice, report = full_report(client)
    assert len(report["violations"]) == 7

    rights = client.post("/explain/rights", json={"report": report, "language": "mr"})
    assert rights.status_code == 200
    assert rights.json()["key_points"]

    letter = client.post(
        "/grievance/draft",
        json={"payload": {"report": report, "loan_facts": loan,
                          "notice_facts": notice, "language": "mr"}},
    )
    assert letter.status_code == 200
    body = letter.json()
    assert body["violations_cited"]
    assert body["placeholders"]
    # The client builds the file from these, so they must all be present.
    assert body["body_english"] and body["addressees"] and body["disclaimer"]
