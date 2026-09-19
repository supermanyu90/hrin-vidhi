"""Hrin Vidhi — FastAPI gateway.

Milestone 1 wires the scaffold: settings, redacting logs, adapter resolution,
a health route that reports exactly what is mocked, and the static frontend
mount. Pipeline routes land in later milestones.

Run it:  DEMO_MODE=true python -m backend.main
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.adapters.registry import build_adapters
from backend.config import FRONTEND_DIR, VERSION, get_settings
from backend.privacy import install_redacting_logging
from backend.routes import analysis as analysis_routes
from backend.routes import calc, intake, output
from backend.schemas import CapabilityState, HealthResponse

log = logging.getLogger(__name__)


def asset_version() -> str:
    """Build stamp for cache-busting.

    The deployment's commit when there is one, otherwise the newest frontend
    mtime so an edit during local development is picked up without a restart.

    The mtime alone is not enough once deployed. Vercel normalises every file
    timestamp to a fixed sentinel for reproducible builds — observed in
    production as `?v=0.1.0-1540000000`, which is 20 October 2018 and is the
    same on every deploy ever made. The stamp changed for nobody.

    That was survivable rather than harmless: assets are served `no-cache`
    with a content ETag, so browsers revalidate and do get new files. But the
    stamp exists precisely so that updates propagate, and a mechanism that
    quietly does nothing is worse than no mechanism — the day someone marks
    these URLs `immutable`, which is the obvious optimisation for versioned
    assets, every borrower freezes on the build they first loaded.
    """
    commit = os.environ.get("VERCEL_GIT_COMMIT_SHA", "").strip()
    if commit:
        return f"{VERSION}-{commit[:12]}"
    try:
        newest = max(p.stat().st_mtime for p in FRONTEND_DIR.glob("*") if p.is_file())
    except (ValueError, OSError):
        return VERSION
    return f"{VERSION}-{int(newest)}"


def create_app() -> FastAPI:
    settings = get_settings()
    install_redacting_logging(settings.log_level)

    adapters, adapter_warnings = build_adapters(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        # Release the pooled connections rather than leaving them to the
        # runtime to reap.
        from backend.adapters.http import aclose_shared_client

        await aclose_shared_client()

    app = FastAPI(
        lifespan=lifespan,
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
        if adapters.mode == "fallback":
            warnings.append(
                "Running entirely on offline fallbacks: no provider key was usable. "
                "The full borrower flow still works; nothing is calling a model."
            )
        return HealthResponse(
            version=VERSION,
            demo_mode=settings.demo_mode,
            mode=adapters.mode,
            live_capabilities=adapters.live_count,
            total_capabilities=len(adapters.status),
            capabilities={
                name: CapabilityState(provider=st.provider, state=st.state, detail=st.detail)
                for name, st in adapters.status.items()
            },
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

    if adapters.mode == "genai":
        banner = f"GENAI — {adapters.live_count}/{len(adapters.status)} capabilities live"
    elif adapters.mode == "forced":
        banner = "FALLBACK (forced) — DEMO_MODE=true, staying offline on purpose"
    else:
        banner = "FALLBACK — no provider key usable; running offline"
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
