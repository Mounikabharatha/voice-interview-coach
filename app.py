"""ASGI entrypoint for hosting platforms.

Vercel resolves its entrypoint from the project root and looks for `app.py` (among a few other
names) exposing a top-level `app`. This package lives under `src/`, so `coach.web.app` is not
importable from the root without help — which is exactly what the first deploy failed on:

    [PYTHON_ENTRYPOINT_NOT_FOUND] "tool.vercel.entrypoint" is "coach.web.app:app"
    but no matching module file was found

Rather than flatten the `src/` layout, which is the right layout locally and for packaging,
this file bridges the two. `src` is put on the path explicitly instead of relying on the
editable install in requirements.txt: an editable install records an absolute build-time path,
and on a platform where the build and runtime filesystems differ that path may not resolve.

Local development is unaffected — `python -m coach` still goes through `__main__.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from coach.web.app import app  # noqa: E402  (path setup must come first)

__all__ = ["app"]
