"""Spend limits for a publicly reachable deployment.

A hosted instance is a public WebSocket spending the developer's own free-tier quota. Groq
allows 1,000 requests a day and one session uses roughly twenty, so about fifty sessions
exhausts it — and a link on a CV that answers "rate limit exceeded" is worse than no link at
all. Anything reachable from the open internet gets crawled, so this is not a hypothetical.

Three limits, all failing **closed** with a message a visitor can understand rather than an
error:

- a cap on concurrent sessions, because the free tier's tokens-per-minute ceiling is the real
  constraint and two people talking at once is enough to hit it;
- a maximum session length, so an abandoned tab cannot hold a slot or stream audio forever;
- a daily session budget, deliberately set below the provider quota so the developer's own
  testing is never blocked by a stranger's traffic.

Everything is configurable by environment variable, so the local build can effectively disable
all of it while the hosted build stays protected.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Limits:
    """Tracks live sessions and the daily budget for one server process.

    **Process-local, and that matters when choosing where to host.** On a single long-running
    instance — Render, Fly, a VM — these counters mean exactly what they say. On a serverless
    platform that scales to many instances, each gets its own counter, so the real daily
    ceiling is the budget multiplied by however many instances happen to spin up. Divide the
    budget accordingly, or accept weaker protection. A shared counter would need a database
    this project does not otherwise require.
    """

    max_concurrent: int = field(default_factory=lambda: _env_int("COACH_MAX_CONCURRENT", 2))
    max_session_seconds: int = field(default_factory=lambda: _env_int("COACH_MAX_SESSION_SECONDS", 420))
    daily_budget: int = field(default_factory=lambda: _env_int("COACH_DAILY_SESSIONS", 40))

    _active: int = 0
    _used_today: int = 0
    _day: int = -1

    def _roll_day(self) -> None:
        today = int(time.time() // 86_400)
        if today != self._day:
            self._day, self._used_today = today, 0

    def try_acquire(self) -> tuple[bool, str]:
        """Claim a session slot. Returns (allowed, message_if_refused)."""
        self._roll_day()
        if self._active >= self.max_concurrent:
            return False, (
                "Someone else is using the demo right now — it runs on a free tier that only "
                "supports a couple of conversations at a time. Try again in a few minutes."
            )
        if self._used_today >= self.daily_budget:
            return False, (
                "The demo has reached its daily limit — it runs on free API credits. It resets "
                "tomorrow. In the meantime the README has a recorded demo, and the project "
                "runs locally with two free keys."
            )
        self._active += 1
        self._used_today += 1
        return True, ""

    def release(self) -> None:
        self._active = max(0, self._active - 1)

    def snapshot(self) -> dict:
        self._roll_day()
        return {
            "active": self._active,
            "used_today": self._used_today,
            "daily_budget": self.daily_budget,
            "max_concurrent": self.max_concurrent,
            "max_session_seconds": self.max_session_seconds,
        }
