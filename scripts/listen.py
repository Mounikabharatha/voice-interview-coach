"""Round-trip test: synthesize a sentence, then transcribe it back. Step 7's demo.

    python scripts/listen.py "your sentence here"

Feeds audio at real-time pace so partials arrive the way they will from a live microphone.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from coach.stt.nvidia import NvidiaSTT, SAMPLE_RATE_HZ
from coach.tts.nvidia import NvidiaTTS

FRAME_MS = 100


async def main() -> int:
    text = " ".join(sys.argv[1:]) or "I led the migration of our billing system over four months."
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        print("error: NVIDIA_API_KEY not set in .env", file=sys.stderr)
        return 1

    print(f"synthesizing at {SAMPLE_RATE_HZ} Hz: {text!r}")
    tts = NvidiaTTS(key, sample_rate_hz=SAMPLE_RATE_HZ)
    pcm = b"".join([c async for c in tts.synthesize(text)])
    audio_secs = len(pcm) / (SAMPLE_RATE_HZ * 2)
    print(f"got {audio_secs:.1f}s of audio\n")

    frame_bytes = int(SAMPLE_RATE_HZ * FRAME_MS / 1000) * 2
    frames = [pcm[i:i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]

    async def mic() -> AsyncIterator[bytes]:
        """Pace frames at wall-clock speed, the way a real microphone would."""
        for f in frames:
            await asyncio.sleep(FRAME_MS / 1000)
            yield f

    stt = NvidiaSTT(key)
    print("streaming it back in at real-time pace:")
    t0 = time.perf_counter()
    partials = 0
    first_partial = None
    final_text = None
    async for tr in stt.stream(mic()):
        elapsed = (time.perf_counter() - t0) * 1000
        if tr.is_final:
            final_text = tr.text
            print(f"  {elapsed:6.0f} ms  FINAL    {tr.text!r}")
        else:
            partials += 1
            if first_partial is None:
                first_partial = elapsed
            if partials <= 4 or partials % 8 == 0:
                print(f"  {elapsed:6.0f} ms  partial  {tr.text!r}")
        last = elapsed

    print(f"\n{partials} partials, first at {first_partial:.0f} ms")
    print(f"finalisation: {last - audio_secs * 1000:.0f} ms after the audio ended")
    if final_text:
        match = final_text.strip().lower().rstrip(".") == text.strip().lower().rstrip(".")
        print(f"transcript matches input exactly: {match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
