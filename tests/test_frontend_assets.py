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


# ---------------------------------------------------------------------------
# Phone layout
# ---------------------------------------------------------------------------
#
# Nearly everyone who uses this will only ever see it on a phone, often a
# 360px budget Android. These assert the few things that are checkable from
# the source; the layout itself was verified by measuring both pages at 320,
# 360, 414 and 768 with no element crossing either edge.

PAGES = ("index.html", "home.html")


@pytest.mark.parametrize("page", PAGES)
def test_every_page_declares_a_device_width_viewport(page: str) -> None:
    """Without this a phone renders at ~980px and scales the whole page down."""
    html = (FRONTEND_DIR / page).read_text(encoding="utf-8")
    assert "width=device-width" in html
    # The layout pads for the notch; that padding only applies with this set.
    assert "viewport-fit=cover" in html


@pytest.mark.parametrize("page", PAGES)
def test_every_page_has_a_phone_breakpoint(page: str) -> None:
    html = (FRONTEND_DIR / page).read_text(encoding="utf-8")
    assert "max-width: 460px" in html or "max-width: 520px" in html, (
        f"{page} has no phone-width rules"
    )


def test_the_chat_input_cannot_trigger_ios_zoom() -> None:
    """Safari zooms the whole page in when a focused input is under 16px,
    and the borrower then has to pinch back out to see what they typed."""
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "#askInput { font-size: 16px; }" in html


def test_phone_rules_come_after_the_rules_they_override() -> None:
    """Same specificity means source order decides.

    The phone block sat above the base styles and was silently overridden by
    them: the mode pill kept a 3px padding and a 22px tap height while the
    media query said otherwise.
    """
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    base = html.index("  .demo-pill {")
    phone = html.index("@media (max-width: 460px)")
    assert phone > base, (
        "the phone media block must come after the base rules it overrides"
    )


def test_the_landing_mascot_stays_silent() -> None:
    """Her bubble is positioned against her, and she sits in a row that wraps,
    so it landed on the subtitle above or the button below depending on where
    the row broke. It repeated that subtitle anyway."""
    html = (FRONTEND_DIR / "home.html").read_text(encoding="utf-8")
    assert ".sherni-hero .sh-say { display: none; }" in html
    assert "mascotGreet" not in html, "the hero must not queue a line it will not show"


def test_a_fixture_transcript_is_never_labelled_as_what_was_heard() -> None:
    """The mock returns a complete, convincing borrower's story.

    Shown under "this is what I heard" it is the application inventing a
    person and then reasoning about their debt. The interface checks the
    provider and labels the offline fixture as a sample instead.
    """
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "transcript.provider || ''" in html
    assert "sampleNotice" in html

    strings = (FRONTEND_DIR / "i18n.js").read_text(encoding="utf-8")
    assert strings.count("sampleNotice") == 6, "every language needs the notice"


#: Which script each language must be written in. Devanagari covers Hindi,
#: Marathi and Bhojpuri; Tamil and Telugu have their own blocks.
_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),
    "mr": (0x0900, 0x097F),
    "bho": (0x0900, 0x097F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
}


@pytest.mark.parametrize("lang", sorted(_SCRIPT_RANGES))
def test_each_language_block_is_written_in_its_own_script(lang: str) -> None:
    """Key parity alone does not catch a value in the wrong language.

    An edit that inserted strings block by block drifted, and Telugu text
    ended up under a Bhojpuri key. The key sets still matched, every existing
    test passed, and a Bhojpuri speaker would have been shown Telugu.
    """
    import re

    source = (FRONTEND_DIR / "i18n.js").read_text(encoding="utf-8")
    start = re.search(rf"^  {lang}: \{{$", source, re.M)
    assert start, lang
    block = source[start.end(): source.index("\n  },", start.end())]

    low, high = _SCRIPT_RANGES[lang]
    wrong = {"hi", "mr", "bho", "ta", "te"} - {lang}
    foreign = {w: _SCRIPT_RANGES[w] for w in wrong if _SCRIPT_RANGES[w] != (low, high)}

    for line in block.splitlines():
        match = re.match(r"\s*([a-zA-Z0-9]+): '((?:[^'\\]|\\.)*)'", line)
        if not match:
            continue
        key, value = match.groups()
        for other, (olow, ohigh) in foreign.items():
            if any(olow <= ord(ch) <= ohigh for ch in value):
                raise AssertionError(
                    f"{lang}.{key} contains {other} script: {value[:40]!r}"
                )


def test_the_asset_stamp_actually_changes_between_deployments(monkeypatch) -> None:
    """The old stamp was inert in production and the test could not tell.

    It was the newest frontend mtime, and Vercel normalises every file
    timestamp to a fixed sentinel for reproducible builds. Production served
    `?v=0.1.0-1540000000` — 20 October 2018 — identically on every deploy ever
    made, while this suite happily confirmed that a stamp was present.
    """
    from backend.main import asset_version

    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", "1111111111111111111111")
    first = asset_version()
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", "2222222222222222222222")
    second = asset_version()

    assert first != second, "two different commits produced the same asset stamp"
    assert "1540000000" not in first, "the frozen build sentinel is back"


def test_the_stamp_falls_back_to_mtime_for_local_development(monkeypatch) -> None:
    """No commit outside a deployment, and an edit must still bust the cache."""
    from backend.main import VERSION, asset_version

    monkeypatch.delenv("VERCEL_GIT_COMMIT_SHA", raising=False)
    stamp = asset_version()
    assert stamp.startswith(f"{VERSION}-")
    assert stamp != VERSION


def test_the_hero_names_the_whole_problem_not_only_the_interest_rate() -> None:
    """The opening line was "What is your loan really costing you?".

    True, and too narrow. Someone arrives here because the phone rang before
    dawn, because a letter arrived headed 'Court Notice', because they were
    told they would be jailed. Money was one fear among several and not
    usually the one that made them search.
    """
    import re

    html = (FRONTEND_DIR / "home.html").read_text(encoding="utf-8")
    arrays = re.findall(r"fears: \[(.*?)\]", html, re.S)
    assert len(arrays) == 6, f"every language needs the hero lines, found {len(arrays)}"

    counts = {len(re.findall(r"'(?:[^'\\]|\\.)*'", a)) for a in arrays}
    assert len(counts) == 1, f"languages disagree on how many lines: {counts}"
    assert counts.pop() >= 4, "too few to cover more than the money"


def test_hero_lines_are_things_the_corpus_can_actually_answer() -> None:
    """Each line is a promise. The app has to be able to keep it.

    Recovery hours, third-party shaming, a letter headed 'Court Notice',
    arrest for debt and an understated instalment are all concepts the
    vernacular layer answers, so none of these lines outruns the product.
    """
    from backend.analysis.vernacular import MESSAGE_TO_CHUNKS

    for concept in (
        "recovery_hours", "third_party", "fake_court_notice",
        "no_arrest", "hidden_charges",
    ):
        assert concept in MESSAGE_TO_CHUNKS, concept
