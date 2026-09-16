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

from backend.main import app  # noqa: E402  (path setup must precede the import)

__all__ = ["app"]
