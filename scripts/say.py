"""Speak a sentence through the TTS provider. Step 6's demo.

    python scripts/say.py "Tell me about a time you led a project."

Writes out.wav and reports time to first audio byte, warm and cold.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from coach.tts.nvidia import NvidiaTTS, SAMPLE_RATE_HZ


async def speak(tts: NvidiaTTS, text: str) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    first_ms = None
    chunks = []
    async for chunk in tts.synthesize(text):
        if first_ms is None:
            first_ms = (time.perf_counter() - t0) * 1000
        chunks.append(chunk)
    return b"".join(chunks), first_ms or 0.0


async def main() -> int:
    text = " ".join(sys.argv[1:]) or "Tell me about a time you handled a difficult stakeholder."
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        print("error: NVIDIA_API_KEY not set in .env", file=sys.stderr)
        return 1

    tts = NvidiaTTS(key)

    print("cold call (connection not yet established):")
    _, cold = await speak(tts, "Ready.")
    print(f"  first audio byte: {cold:.0f} ms")

    print(f"\nwarm call — {text!r}")
    pcm, warm = await speak(tts, text)
    secs = len(pcm) / (SAMPLE_RATE_HZ * 2)
    print(f"  first audio byte: {warm:.0f} ms")
    print(f"  {secs:.1f}s of speech, {len(pcm):,} bytes")
    print(f"\nprewarming saved {cold - warm:.0f} ms on this call")

    out = Path("out.wav")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE_HZ)
        w.writeframes(pcm)
    print(f"\nwrote {out}  —  play it with:  afplay {out}   (macOS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
