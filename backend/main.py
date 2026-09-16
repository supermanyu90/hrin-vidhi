"""Hrin Vidhi — FastAPI gateway.

Milestone 1 wires the scaffold: settings, redacting logs, adapter resolution,
a health route that reports exactly what is mocked, and the static frontend
mount. Pipeline routes land in later milestones.

Run it:  DEMO_MODE=true python -m backend.main
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.adapters.registry import build_adapters
from backend.config import FRONTEND_DIR, VERSION, get_settings
from backend.privacy import install_redacting_logging
from backend.routes import analysis as analysis_routes
from backend.routes import calc, intake, output
from backend.schemas import HealthResponse

log = logging.getLogger(__name__)


def asset_version() -> str:
    """Build stamp for cache-busting: version plus the newest frontend mtime.

    Recomputed per request so an edit during a demo is picked up without a
    restart; the directory listing is a handful of stat calls.
    """
    try:
        newest = max(p.stat().st_mtime for p in FRONTEND_DIR.glob("*") if p.is_file())
    except (ValueError, OSError):
        return VERSION
    return f"{VERSION}-{int(newest)}"


def create_app() -> FastAPI:
    settings = get_settings()
    install_redacting_logging(settings.log_level)

    adapters, adapter_warnings = build_adapters(settings)

    app = FastAPI(
        title="Hrin Vidhi",
        version=VERSION,
        description=(
            "Voice-first, vernacular legal and debt-rights assistant. "
            "Provides information and drafts only — not legal advice."
        ),
    )

    # The frontend is served from this same origin, so CORS is only here to keep
    # a separately-hosted dev UI workable. Localhost only, never a wildcard.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def revalidate_frontend(request, call_next):
        """Make the browser revalidate the UI instead of serving it from cache.

        Without this, editing index.html and refreshing shows the *old* page —
        the server sends the new file and Chrome ignores it. `no-cache` still
        allows a 304 via ETag, so this costs a round trip, not a re-download.
        """
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    app.state.settings = settings
    app.state.adapters = adapters
    app.state.adapter_warnings = adapter_warnings

    app.include_router(intake.router)
    app.include_router(calc.router)
    app.include_router(analysis_routes.router)
    app.include_router(output.router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        """What is wired up right now. The first thing to check at a demo."""
        from backend.analysis.corpus_store import corpus_chunk_count

        warnings = list(adapter_warnings)
        if not settings.demo_mode and adapters.all_mock:
            warnings.append(
                "DEMO_MODE is off but every adapter resolved to a mock — no provider "
                "keys were found in the environment."
            )
        return HealthResponse(
            version=VERSION,
            demo_mode=settings.demo_mode,
            adapters=adapters.describe(),
            corpus_chunks=corpus_chunk_count(),
            warnings=warnings,
        )

    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        def render_page(name: str) -> HTMLResponse:
            """Serve a page with its asset URLs stamped with a build version.

            Cache headers alone are not enough: a browser that cached an asset
            before those headers existed will keep serving it, so an edit is
            invisible until a manual hard reload. Stamping the URL sidesteps
            cache semantics entirely — a changed file means a changed URL.
            """
            html = (FRONTEND_DIR / name).read_text(encoding="utf-8")
            return HTMLResponse(html.replace("{{ASSET_VERSION}}", asset_version()))

        @app.get("/", include_in_schema=False)
        async def home() -> HTMLResponse:
            """The landing page. Its job is to get a borrower to press the mic."""
            return render_page("home.html")

        @app.get("/app", include_in_schema=False)
        async def assistant() -> HTMLResponse:
            return render_page("index.html")

        @app.get("/visualizer", include_in_schema=False)
        async def visualizer() -> HTMLResponse:
            return render_page("visualizer.html")

    banner = "DEMO MODE — all adapters mocked, no network required" if settings.demo_mode else "LIVE"
    log.info("Hrin Vidhi %s starting [%s] adapters=%s", VERSION, banner, adapters.describe())
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
