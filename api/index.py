"""Vercel serverless entry point.

Vercel's Python runtime looks for an ASGI app called `app` in this module and
serves it. Everything else — routes, adapters, corpus — is the same code that
runs locally under `python -m backend.main`; only the process model differs.

Three things about that difference matter, and each is handled rather than
hoped about:

  * **No shared memory between invocations.** The browser holds the
    conversation and passes facts explicitly, so nothing depends on a session
    surviving a cold start. The session routes still exist and still work
    within one warm instance; the UI simply does not rely on them.
  * **No system speech voices on Linux.** The TTS mock returns the script with
    no audio, and the page speaks it with the Web Speech API. Nothing tries to
    ship a silent file.
  * **A response size limit.** Because no audio crosses the wire, the largest
    response is the compliance report — a few tens of kilobytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The function's working directory is not the repo root, so make the project
# importable regardless of where Vercel invokes this from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from backend.main import app
except Exception:  # noqa: BLE001 - see below
    # An import failure here takes out every route at once, and the platform
    # reports it as an opaque FUNCTION_INVOCATION_FAILED with the traceback
    # only in runtime logs. Serving the traceback instead turns a blank 500
    # into something diagnosable from a terminal.
    #
    # It is guarded: a deployment that boots normally never reaches this, and
    # the detail is withheld unless DEPLOY_DEBUG is set, so a production
    # failure does not publish a stack trace to whoever asks for it.
    import os
    import traceback

    _DETAIL = traceback.format_exc()
    _SHOW = os.environ.get("DEPLOY_DEBUG", "").strip().lower() in ("1", "true", "yes")
    print("STARTUP FAILURE\n" + _DETAIL, flush=True)

    async def app(scope, receive, send):  # type: ignore[misc]
        if scope["type"] != "http":
            return
        body = (
            _DETAIL if _SHOW
            else "The application failed to start. Set DEPLOY_DEBUG=1 to see why."
        ).encode()
        await send({
            "type": "http.response.start",
            "status": 500,
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        })
        await send({"type": "http.response.body", "body": body})

__all__ = ["app"]
