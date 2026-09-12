"""Entry point: python -m coach"""
from __future__ import annotations

import sys

from .config import ConfigError, Settings


def main() -> int:
    try:
        settings = Settings.load()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"config ok — will serve on http://{settings.host}:{settings.port}")
    print("server not implemented yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
