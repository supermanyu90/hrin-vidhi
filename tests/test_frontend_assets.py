"""The frontend is five files served from one origin. If any goes missing the
app breaks in the browser with nothing failing server-side, so the wiring is
asserted here rather than discovered at a demo."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from backend.config import FRONTEND_DIR

ASSETS = ["luminous.css", "format.js", "i18n.js", "debt-chart.js", "visualizer.html"]


@pytest.fixture
def client() -> TestClient:
    from backend.main import app

    return TestClient(app)


def test_the_chat_app_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Hrin Vidhi" in response.text


@pytest.mark.parametrize("name", ASSETS)
def test_every_static_asset_is_served(client: TestClient, name: str) -> None:
    response = client.get(f"/static/{name}")
    assert response.status_code == 200, name
    assert response.content


def test_every_asset_the_pages_reference_actually_exists() -> None:
    """Catches a renamed file that the HTML still points at."""
    referenced: set[str] = set()
    for page in ("home.html", "index.html", "visualizer.html"):
        text = (FRONTEND_DIR / page).read_text(encoding="utf-8")
        referenced |= set(re.findall(r'/static/([A-Za-z0-9_.-]+)', text))
    assert referenced, "expected the pages to reference static assets"
    for name in referenced:
        assert (FRONTEND_DIR / name).is_file(), f"{name} is referenced but missing"


def test_the_frontend_contains_no_loan_mathematics() -> None:
    """The rule that keeps the chart, the summary strip and the letter agreeing.

    Formatting is allowed; solving for a rate or building an amortisation
    schedule in the browser is not.
    """
    banned = ("Math.pow(1 +", "emi =", "effectiveApr", "solveRate", "annuity")
    for name in ("home.html", "index.html", "visualizer.html", "debt-chart.js", "format.js"):
        source = (FRONTEND_DIR / name).read_text(encoding="utf-8")
        for token in banned:
            assert token not in source, f"{name} appears to compute loan maths ({token})"


def test_all_six_languages_have_every_ui_string() -> None:
    """A missing key silently falls back to English mid-conversation."""
    source = (FRONTEND_DIR / "i18n.js").read_text(encoding="utf-8")
    blocks = re.findall(r"\n  (hi|mr|ta|te|bho|en): \{(.*?)\n  \},", source, re.S)
    assert len(blocks) == 6, f"expected 6 language blocks, found {len(blocks)}"

    keys = {code: set(re.findall(r"^\s{4}(\w+):", body, re.M)) for code, body in blocks}
    english = keys["en"]
    assert len(english) >= 20
    for code, found in keys.items():
        assert found == english, f"{code} differs from en: {english ^ found}"


def test_the_disclaimer_is_present_in_every_language() -> None:
    """§9: every borrower-facing output carries it, in their language."""
    source = (FRONTEND_DIR / "i18n.js").read_text(encoding="utf-8")
    block = re.search(r"const DISCLAIMER = \{(.*?)\n\};", source, re.S)
    assert block
    for code in ("hi", "mr", "ta", "te", "bho", "en"):
        assert re.search(rf"^\s{{2}}{code}:", block.group(1), re.M), code


def test_the_ui_disclaimers_match_the_backend_phrasebook() -> None:
    """The UI and the spoken voice note must say the same words."""
    from backend.adapters.translate import PHRASEBOOK
    from backend.schemas import Language

    source = (FRONTEND_DIR / "i18n.js").read_text(encoding="utf-8")
    for language, text in PHRASEBOOK["DISCLAIMER"].items():
        if language is Language.ENGLISH:
            continue  # quoting style differs; checked by its own test
        assert text in source, f"UI disclaimer for {language.value} drifted from the backend"


# ---------------------------------------------------------------------------
# Cache busting — an edit the browser refuses to pick up is a silent failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/app", "/visualizer"])
def test_asset_urls_are_stamped_with_a_build_version(client: TestClient, path: str) -> None:
    """Without this, a browser that cached an asset before the cache headers
    existed keeps serving it, and an edit is invisible until a hard reload.

    Each page is checked against the assets *it* references — the landing page
    carries its own design system and deliberately loads almost nothing.
    """
    html = client.get(path).text
    assert "{{ASSET_VERSION}}" not in html, f"{path}: the template token leaked through"
    referenced = re.findall(r"/static/([A-Za-z0-9_.-]+)", html)
    assert referenced, f"{path} references no static assets"
    for name in set(referenced):
        assert f"/static/{name}?v=" in html, f"{path}: {name} is not version-stamped"


def test_the_version_stamp_changes_when_a_frontend_file_changes(tmp_path) -> None:
    import os

    from backend.config import FRONTEND_DIR
    from backend.main import asset_version

    before = asset_version()
    target = FRONTEND_DIR / "i18n.js"
    original = target.stat().st_mtime
    # Must exceed the newest file in the directory, not just this one's own
    # mtime — the stamp is the maximum across the whole frontend.
    newest = max(p.stat().st_mtime for p in FRONTEND_DIR.glob("*") if p.is_file())
    try:
        os.utime(target, (newest + 60, newest + 60))
        assert asset_version() != before
    finally:
        os.utime(target, (original, original))
    assert asset_version() == before, "the stamp did not return to its original value"


def test_the_frontend_is_served_with_revalidation_headers(client: TestClient) -> None:
    for path in ("/", "/static/i18n.js"):
        cache_control = client.get(path).headers.get("cache-control", "")
        assert "no-cache" in cache_control, path


def test_the_visualizer_has_its_own_templated_route(client: TestClient) -> None:
    response = client.get("/visualizer")
    assert response.status_code == 200
    assert "{{ASSET_VERSION}}" not in response.text
    assert "Debt Trap Visualizer" in response.text


def test_the_chat_links_to_the_templated_visualizer() -> None:
    """A link to the raw static file would carry an unreplaced token."""
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "/visualizer?$" in html
