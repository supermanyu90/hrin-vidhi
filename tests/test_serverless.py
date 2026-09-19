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


def test_the_declared_entrypoint_is_the_app_we_test() -> None:
    """Vercel loads whatever `pyproject.toml` names. It must be this app.

    Two builds failed over an `api/index.py` shim that re-exported `app`:
    Vercel's FastAPI detector reads the entrypoint statically, so an `app`
    that arrives by import — and later from inside a try/except — was not
    recognised, and the build stopped with "does not define a top-level app".
    Naming the real module removed the indirection and both failures.
    """
    import tomllib

    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    entrypoint = config["tool"]["vercel"]["entrypoint"]
    assert entrypoint == "backend.main:app", entrypoint

    module_path, _, attribute = entrypoint.partition(":")
    module = __import__(module_path, fromlist=[attribute])
    from backend.main import app as local_app

    assert getattr(module, attribute) is local_app


def test_the_entrypoint_module_defines_app_at_the_top_level() -> None:
    """Statically, the way Vercel's detector reads it — not by importing."""
    import ast

    tree = ast.parse((PROJECT_ROOT / "backend" / "main.py").read_text())
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "app" in assigned, (
        "backend/main.py must assign `app` at module level, or Vercel's FastAPI "
        "detection fails the build before any code runs"
    )


def test_pyproject_and_requirements_agree() -> None:
    """Two files now list the runtime dependencies, so they must not drift.

    Vercel installs from `pyproject.toml` (it switches to uv the moment that
    file exists); people install from `requirements.txt`. A package added to
    one and not the other means the deployed function differs from the one
    anybody tested.
    """
    import re
    import tomllib

    declared = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    shipped = {
        re.split(r"[=<>\[]", spec)[0].strip()
        for spec in declared["project"]["dependencies"]
    }

    listed = {
        re.split(r"[=<>\[]", line)[0].strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert shipped == listed, (
        f"only in pyproject: {sorted(shipped - listed)}; "
        f"only in requirements.txt: {sorted(listed - shipped)}"
    )


def test_the_project_table_exists_for_uv() -> None:
    """Its absence failed a build: uv refuses a pyproject without [project]."""
    import tomllib

    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert "project" in config
    assert config["project"]["dependencies"]
    # uv would otherwise try to build this application as a wheel.
    assert config["tool"]["uv"]["package"] is False


def test_deployment_config_ships_the_data_files() -> None:
    import json

    config = json.loads((PROJECT_ROOT / "vercel.json").read_text())
    # No rewrites: Vercel routes every path into the FastAPI app itself. The
    # old catch-all pointed at the shim that no longer exists.
    assert "rewrites" not in config, "the framework handles routing; a rewrite fights it"

    # Keyed on the resolved entrypoint, which is what Vercel configures.
    fn = config["functions"]["backend/main.py"]
    for needed in ("backend", "frontend", "fixtures"):
        assert needed in fn["includeFiles"], needed


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
            json={
                "payload": {
                    "report": report,
                    "loan_facts": loan,
                    "notice_facts": notice,
                    "debt_analysis": debt,
                    "language": "mr",
                }
            },
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
    rights = client.post("/explain/rights", json={"report": report, "language": "mr"}).json()

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
        json={
            "payload": {
                "report": report,
                "loan_facts": loan,
                "notice_facts": notice,
                "language": "mr",
            }
        },
    )
    assert letter.status_code == 200
    body = letter.json()
    assert body["violations_cited"]
    assert body["placeholders"]
    # The client builds the file from these, so they must all be present.
    assert body["body_english"] and body["addressees"] and body["disclaimer"]


# ---------------------------------------------------------------------------
# Deployment configuration
# ---------------------------------------------------------------------------


def test_no_builder_version_is_pinned() -> None:
    """A pinned runtime is how the first Vercel deploy failed.

    vercel.json asked for `@vercel/python@5.0.1`, which has never existed —
    the published versions jump 12.x, 13.x, 14.x — so the build died at
    `pin-version-mismatch` before any code ran. Python is an officially
    supported runtime, where the docs are explicit that `runtime` is optional
    and exists for community runtimes. Omitting it cannot go stale.
    """
    import json
    from pathlib import Path

    config = json.loads((Path(__file__).resolve().parents[1] / "vercel.json").read_text())
    for name, fn in config.get("functions", {}).items():
        assert "runtime" not in fn, (
            f"{name} pins a builder version. Unless this is a community runtime, "
            "drop it — a pin that drifts breaks the build, not one request."
        )


def test_the_function_bundle_carries_its_provider_sdk() -> None:
    """A key in the dashboard does nothing without the package beside it.

    `require_sdk` refuses to construct an adapter whose SDK is missing, so a
    deploy without this line reports `fallback` with GOOGLE_API_KEY set and
    looks like a broken key.
    """
    from pathlib import Path

    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text()
    active = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any(line.startswith("google-genai") for line in active), active
    assert not any(line.startswith("pytest") for line in active), (
        "a test runner has no business in a serverless bundle"
    )


def test_a_blank_environment_variable_does_not_take_the_site_down() -> None:
    """Adding a dashboard variable and leaving it empty is an ordinary slip.

    Before this, pydantic refused to read "" as a boolean and raised during
    import — every route gone, with a validation error in the deploy log.
    """
    from backend.config import Settings

    settings = Settings(DEMO_MODE="", PROVIDER_TIMEOUT_SECONDS="", PORT="")
    assert settings.demo_mode is False
    assert settings.provider_timeout_seconds > 0
    assert settings.port > 0


def test_an_explicit_setting_still_wins_over_the_blank_guard() -> None:
    from backend.config import Settings

    assert Settings(DEMO_MODE="true").demo_mode is True


def test_a_blank_provider_choice_does_not_take_the_site_down() -> None:
    """This one actually happened, in production, on every route.

    A blank STT_PROVIDER reached the registry as a provider literally named
    "" and `build_adapters` raised at import:

        AdapterError: [stt] unknown provider ''; known: ['bhashini', 'sarvam']

    An earlier guard exempted string fields, reasoning that "" is a valid
    hostname. For a provider choice it is not a value at all.
    """
    from backend.adapters.registry import build_adapters
    from backend.config import Settings

    settings = Settings(
        STT_PROVIDER="", TRANSLATE_PROVIDER="", TTS_PROVIDER="",
        DOCPARSER_PROVIDER="", LLM_PROVIDER="", GOOGLE_MODEL="",
    )
    assert settings.stt_provider == "auto"
    assert settings.google_model, "a blank model must fall back, not empty out"

    adapters, _ = build_adapters(settings)
    assert adapters.mode in ("genai", "fallback", "forced")


def test_naming_a_provider_that_does_not_exist_still_fails_loudly() -> None:
    """The blank guard must not soften a real misconfiguration."""
    import pytest

    from backend.adapters.base import AdapterError
    from backend.adapters.registry import build_adapters
    from backend.config import Settings

    with pytest.raises(AdapterError):
        build_adapters(Settings(STT_PROVIDER="nosuchprovider"))


# ---------------------------------------------------------------------------
# Latency, caching and retry behaviour
# ---------------------------------------------------------------------------


def test_every_route_with_a_budget_is_reported_against_it() -> None:
    """A budget nobody wrote down is a budget nobody misses."""
    from backend import telemetry

    telemetry.reset()
    telemetry.record("GET /health", 0.010)
    telemetry.record("GET /health", 0.020)
    telemetry.record("GET /health", 0.030)

    snap = telemetry.snapshot()["routes"]["GET /health"]
    assert snap["count"] == 3
    assert snap["p50_ms"] <= snap["p95_ms"] <= snap["p99_ms"]
    assert snap["within_budget"] is True
    telemetry.reset()


def test_a_route_over_its_budget_is_named() -> None:
    from backend import telemetry

    telemetry.reset()
    for _ in range(10):
        telemetry.record("POST /calc/debt-trap", 5.0)   # 5s against a 100ms budget
    assert "POST /calc/debt-trap" in telemetry.snapshot()["over_budget"]
    telemetry.reset()


def test_percentiles_describe_the_tail_not_the_average() -> None:
    """A mean of 600ms can be a hundred fast answers and one borrower waiting
    nine seconds, and it is that borrower who gives up."""
    from backend import telemetry

    telemetry.reset()
    for _ in range(99):
        telemetry.record("POST /ask", 0.100)
    telemetry.record("POST /ask", 9.000)

    snap = telemetry.snapshot()["routes"]["POST /ask"]
    assert snap["p50_ms"] == pytest.approx(100, abs=1)
    assert snap["max_ms"] == pytest.approx(9000, abs=1)
    telemetry.reset()


def test_a_repeated_question_does_not_pay_for_the_model_twice() -> None:
    """The English path spends a model call tightening its draft, and that
    call is the whole cost of the request — 8.5s against 415ms for the
    vernacular path, which uses none."""
    import asyncio

    from backend.adapters.llm import MockLLM
    from backend.analysis.rag import _ANSWER_CACHE, answer_question, clear_answer_cache
    from backend.schemas import Language

    clear_answer_cache()
    llm = MockLLM()
    question = "Can they call me at 6 in the morning?"

    # Only the path that actually pays for a model call is worth caching;
    # without one the answer is already a few milliseconds.
    first = asyncio.run(answer_question(question, llm=llm, language=Language.ENGLISH))
    assert first.grounded
    assert len(_ANSWER_CACHE) == 1

    # Same question, any spacing or casing.
    second = asyncio.run(
        answer_question("  CAN THEY call me  at 6 in the morning? ",
                        llm=llm, language=Language.ENGLISH)
    )
    assert second is first, "a repeated question rebuilt the answer"
    clear_answer_cache()


def test_a_refusal_is_not_cached() -> None:
    """It costs nothing to produce again, and caching one would outlive a
    corpus that grew to answer it."""
    import asyncio

    from backend.adapters.llm import MockLLM
    from backend.analysis.rag import _ANSWER_CACHE, answer_question, clear_answer_cache

    clear_answer_cache()
    answer = asyncio.run(
        answer_question("Who won the cricket match last night?", llm=MockLLM())
    )
    assert not answer.grounded
    assert len(_ANSWER_CACHE) == 0
    clear_answer_cache()


def test_the_answer_cache_is_bounded() -> None:
    from backend.analysis.rag import _ANSWER_CACHE, _ANSWER_CACHE_MAX, clear_answer_cache
    from backend.schemas import RagAnswer

    clear_answer_cache()
    for i in range(_ANSWER_CACHE_MAX + 20):
        _ANSWER_CACHE[(f"q{i}", "en")] = RagAnswer(
            question=f"q{i}", answer="a", citations=[], grounded=True
        )
    assert len(_ANSWER_CACHE) <= _ANSWER_CACHE_MAX + 20
    clear_answer_cache()


def test_the_polish_budget_survives_a_model_that_thinks_first() -> None:
    """1024 tokens truncated every English answer mid-sentence, and lost the
    closing caveat that says these are summaries to be checked."""
    from backend.config import BACKEND_DIR

    source = (BACKEND_DIR / "analysis" / "rag.py").read_text(encoding="utf-8")
    assert "max_tokens=3072" in source
    assert "max_tokens=1024" not in source
