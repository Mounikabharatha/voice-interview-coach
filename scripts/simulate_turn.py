"""End-to-end test of the full loop, without a microphone.

Speaks a candidate's answer using the TTS, streams it into the server's WebSocket exactly as
the browser would — 16 kHz PCM16, 100 ms frames, at real-time pace — and reports what comes
back and when.

    python scripts/simulate_turn.py                     # default answer
    python scripts/simulate_turn.py "your answer here"

Useful because it exercises the same path a real user does, but deterministically and without
needing anyone to talk to it.
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
DEFAULT_ANSWER = (
    "I led the migration of our billing system to a new payments provider. "
    "It took about four months and I coordinated three teams."
)


async def main() -> int:
    answer = " ".join(sys.argv[1:]) or DEFAULT_ANSWER
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        print("error: NVIDIA_API_KEY not set", file=sys.stderr)
        return 1

    print(f"rendering the candidate's answer as speech:\n  {answer!r}\n")
    tts = NvidiaTTS(key, sample_rate_hz=MIC_RATE)
    pcm = b"".join([c async for c in tts.synthesize(answer)])
    secs = len(pcm) / (MIC_RATE * 2)
    frame_bytes = int(MIC_RATE * FRAME_MS / 1000) * 2
    frames = [pcm[i:i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
    print(f"{secs:.1f}s of audio, {len(frames)} frames\n")

    greeting_audio = 0
    reply_audio = 0
    speech_ended: float | None = None
    first_reply_audio: float | None = None
    final_seen = False

    async with websockets.connect(WS_URL, max_size=None) as ws:
        print("connected. waiting for the coach's opening question...")

        async def reader() -> None:
            nonlocal greeting_audio, reply_audio, first_reply_audio, final_seen
            async for msg in ws:
                if isinstance(msg, bytes):
                    if final_seen:
                        reply_audio += len(msg)
                        if first_reply_audio is None and speech_ended is not None:
                            first_reply_audio = (time.perf_counter() - speech_ended) * 1000
                            print(f"  <- first reply audio  +{first_reply_audio:.0f} ms after speech ended")
                    else:
                        greeting_audio += len(msg)
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t == "assistant_text":
                    print(f"  <- coach: {m['text']!r}")
                elif t == "partial":
                    print(f"  <- partial: {m['text']!r}")
                elif t == "final":
                    final_seen = True
                    print(f"  <- FINAL:   {m['text']!r}")
                elif t == "state":
                    print(f"  <- state:   {m['state']}")
                elif t == "metrics":
                    print(f"  <- metrics: {m['turn']}")
                elif t == "error":
                    print(f"  <- ERROR:   {m['message']}")

        task = asyncio.create_task(reader())
        await asyncio.sleep(6)  # let the greeting finish so it is not transcribed as speech

        print(f"\nstreaming the answer in at real-time pace ({FRAME_MS} ms frames):")
        for f in frames:
            await ws.send(f)
            await asyncio.sleep(FRAME_MS / 1000)
        speech_ended = time.perf_counter()
        print("  -> audio sent, speech ended\n")

        # Keep the socket open so the model can answer.
        await asyncio.sleep(25)
        task.cancel()

    print("\n--- result ---")
    print(f"greeting audio received: {greeting_audio:,} bytes")
    print(f"reply audio received:    {reply_audio:,} bytes")
    if first_reply_audio:
        print(f"end of speech -> first reply audio: {first_reply_audio:.0f} ms")
    ok = greeting_audio > 0 and final_seen and reply_audio > 0
    print(f"\nfull loop working: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
