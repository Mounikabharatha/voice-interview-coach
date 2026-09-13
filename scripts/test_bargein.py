"""Barge-in test: talk over the coach and check it actually stops.

    python scripts/test_bargein.py

Answers a question, waits for the coach to start replying, then speaks over it. Passes only
if the server cancels the turn and tells the client to drop the audio it already scheduled.
"""
from __future__ import annotations

import asyncio, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import websockets
from coach.stt.nvidia import SAMPLE_RATE_HZ as MIC_RATE
from coach.tts.nvidia import NvidiaTTS

WS_URL = os.getenv("COACH_WS", "ws://127.0.0.1:8000/ws")
FRAME_MS = 100
ANSWER = "I handled a difficult stakeholder on the billing project last year."
INTERRUPTION = "Sorry, let me start that answer again from the beginning."


async def main() -> int:
    tts = NvidiaTTS(os.environ["NVIDIA_API_KEY"], sample_rate_hz=MIC_RATE)
    fb = int(MIC_RATE * FRAME_MS / 1000) * 2

    async def render(text):
        pcm = b"".join([c async for c in tts.synthesize(text)])
        return [pcm[i:i + fb] for i in range(0, len(pcm), fb)]

    answer, interrupt = await render(ANSWER), await render(INTERRUPTION)
    silence = b"\x00\x00" * (fb // 2)

    state = {"cancel_audio": False, "interrupted": False, "coach_started": False,
             "audio_after_cancel": 0, "cancel_at": None}

    async with websockets.connect(WS_URL, max_size=None) as ws:
        async def reader():
            async for msg in ws:
                if isinstance(msg, bytes):
                    if state["cancel_at"] is not None:
                        state["audio_after_cancel"] += len(msg)
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t == "assistant_text":
                    state["coach_started"] = True
                    print(f"  <- coach: {m['text'][:60]!r}")
                elif t == "cancel_audio":
                    state["cancel_audio"] = True
                    state["cancel_at"] = time.perf_counter()
                    print("  <- CANCEL_AUDIO")
                elif t == "interrupted":
                    state["interrupted"] = True
                    print("  <- INTERRUPTED")
                elif t in ("final", "state"):
                    print(f"  <- {t}: {m.get('text', m.get('state'))!r}")

        task = asyncio.create_task(reader())
        await asyncio.sleep(6)                     # let the greeting play out
        state["coach_started"] = False             # the greeting does not count

        print(f"\nanswering: {ANSWER!r}")
        for f in answer:
            await ws.send(f); await asyncio.sleep(FRAME_MS / 1000)
        for _ in range(25):                        # silence so the endpointer can fire
            await ws.send(silence); await asyncio.sleep(FRAME_MS / 1000)
            if state["coach_started"]:
                break

        # wait until the coach is actually talking
        t0 = time.perf_counter()
        while not state["coach_started"] and time.perf_counter() - t0 < 15:
            await ws.send(silence); await asyncio.sleep(FRAME_MS / 1000)
        if not state["coach_started"]:
            print("coach never replied — cannot test barge-in"); return 1

        print(f"\ntalking over the coach: {INTERRUPTION!r}")
        spoke_at = time.perf_counter()
        for f in interrupt:
            await ws.send(f); await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(3)
        task.cancel()

    print("\n--- result ---")
    if state["cancel_at"]:
        print(f"detected in {(state['cancel_at'] - spoke_at) * 1000:.0f} ms of speech")
    print(f"cancel_audio sent:        {state['cancel_audio']}")
    print(f"turn reported interrupted:{state['interrupted']}")
    print(f"audio sent after cancel:  {state['audio_after_cancel']:,} bytes")
    ok = state["cancel_audio"] and state["interrupted"]
    print(f"\nbarge-in working: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
