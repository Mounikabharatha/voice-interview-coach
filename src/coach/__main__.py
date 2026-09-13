"""Entry point: python -m coach"""
from __future__ import annotations

import sys

import uvicorn

from .config import ConfigError, Settings


def main() -> int:
    try:
        settings = Settings.load()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"starting on http://{settings.host}:{settings.port}")
    uvicorn.run(
        "coach.web.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
