"""Run a whole interview session end to end, without a microphone.

    python scripts/simulate_session.py

Answers every question the coach asks with a canned reply, streams it in as a fake microphone,
and prints the feedback report at the end. This is the closest thing to a real session that
can be run unattended, so it is what catches regressions in the session flow.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import websockets

from coach.stt.nvidia import SAMPLE_RATE_HZ as MIC_RATE
from coach.tts.nvidia import NvidiaTTS

WS_URL = os.getenv("COACH_WS", "ws://127.0.0.1:8000/ws")
FRAME_MS = 100

# Deliberately varied: the first is strong, the second is vague enough to earn a probe,
# the third sits in between. That exercises both branches of the probe decision.
ANSWERS = [
    "I rewrote our billing reconciliation job after I realised the retry logic was double "
    "charging about two hundred customers a week. I called the payments lead the same "
    "afternoon, we froze releases, and I shipped a fix in three days that cut failed payments "
    "by twelve percent. In hindsight I should have added alerting before the migration, not "
    "after it.",

    "We worked on it as a team and it went reasonably well in the end. Everyone pulled "
    "together and the stakeholders were happy with what we delivered.",

    "I took over a project that had already slipped twice. I spent the first week talking to "
    "each of the four engineers separately, found that nobody believed the deadline, and I "
    "went back to the director with a revised plan. We shipped six weeks later than the "
    "original date but with no rollbacks, and I learned to surface bad news earlier.",
]


async def main() -> int:
    tts = NvidiaTTS(os.environ["NVIDIA_API_KEY"], sample_rate_hz=MIC_RATE)
    fb = int(MIC_RATE * FRAME_MS / 1000) * 2
    silence = b"\x00\x00" * (fb // 2)

    async def render(text: str) -> list[bytes]:
        pcm = b"".join([c async for c in tts.synthesize(text)])
        return [pcm[i:i + fb] for i in range(0, len(pcm), fb)]

    print("rendering answers...")
    rendered = [await render(a) for a in ANSWERS]
    print(f"  {len(rendered)} answers ready\n")

    report: dict | None = None
    coach_idle = asyncio.Event()
    turns = 0

    async with websockets.connect(WS_URL, max_size=None) as ws:
        async def reader() -> None:
            nonlocal report, turns
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t == "session_start":
                    print(f"session questions ({len(m['questions'])}):")
                    for q in m["questions"]:
                        print(f"  - {q}")
                    print()
                elif t == "assistant_text":
                    print(f"  COACH: {m['text']}")
                elif t == "final":
                    turns += 1
                    print(f"  YOU:   {m['text'][:80]}... [{m.get('verdict')}]")
                elif t == "state":
                    if m["state"] == "listening":
                        coach_idle.set()
                    else:
                        coach_idle.clear()
                elif t == "report":
                    report = m
                elif t == "error":
                    print(f"  ERROR: {m['message']}")

        task = asyncio.create_task(reader())

        # Event-driven: every time the coach goes quiet, answer. Probes need answering
        # too, so the number of turns is not known in advance — cycle the canned answers
        # until the report arrives or the cap is hit.
        MAX_TURNS = 12
        for turn_no in range(MAX_TURNS):
            if report is not None:
                break
            coach_idle.clear()
            try:
                await asyncio.wait_for(coach_idle.wait(), timeout=45)
            except asyncio.TimeoutError:
                print("  (timed out waiting for the coach)")
                break
            if report is not None:
                break
            await asyncio.sleep(1.0)
            frames = rendered[turn_no % len(rendered)]
            print(f"\n-- turn {turn_no + 1} --")
            for f in frames:
                await ws.send(f)
                await asyncio.sleep(FRAME_MS / 1000)
            for _ in range(30):                       # trailing silence, lets the turn end
                await ws.send(silence)
                await asyncio.sleep(FRAME_MS / 1000)

        # keep the socket alive while the wrap turn and background scoring finish
        deadline = time.perf_counter() + 60
        while report is None and time.perf_counter() < deadline:
            await ws.send(silence)
            await asyncio.sleep(FRAME_MS / 1000)

        task.cancel()

    print("\n" + "=" * 60)
    if not report:
        print("NO REPORT RECEIVED — session did not reach the wrap step")
        return 1

    print("FEEDBACK REPORT\n")
    for key, s in report["summary"].items():
        bar = "#" * int(round(s["mean"] * 4)) + "." * (20 - int(round(s["mean"] * 4)))
        print(f"  {s['name']:22s} {bar} {s['mean']}/5  (n={s['n']})")
    print(f"\n  answers scored: {len(report['detail'])}")
    print(f"  turns recorded: {len(report['turns'])}")

    verified = sum(1 for turn in report["detail"] for s in turn if s["evidence_verified"])
    total = sum(len(turn) for turn in report["detail"])
    print(f"  evidence quotes verified against the transcript: {verified}/{total}")
    print("\nsession complete: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
