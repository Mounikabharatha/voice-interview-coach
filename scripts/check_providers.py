"""Check that both providers answer before building anything on top of them.

Run this first, and any time something stops working:

    python scripts/check_providers.py

It reports what actually responded rather than what the docs claim. Free tiers in this
space change on a scale of weeks — models get retired, limits move — so a fast honest
check beats a stale assumption every time.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_BASE = "https://api.groq.com/openai/v1"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"

OK, BAD, INFO = "  ok  ", " FAIL ", " --   "


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f"  {detail}" if detail else ""))


async def groq_models(client: httpx.AsyncClient, key: str) -> list[str]:
    r = await client.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {key}"})
    if r.status_code != 200:
        line(BAD, "Groq models", f"HTTP {r.status_code} — {r.text[:120]}")
        return []
    ids = sorted(m["id"] for m in r.json().get("data", []))
    line(OK, "Groq models", f"{len(ids)} available")
    return ids


async def groq_chat(client: httpx.AsyncClient, key: str, model: str,
                    reasoning: str | None = None) -> None:
    """Stream a tiny completion and measure time to first token.

    `reasoning` maps to Groq's `reasoning_effort`. This matters more than it looks: with
    reasoning on, the first tokens streamed are the model's chain-of-thought, which is not
    speakable. What a voice agent cares about is time to first SPEAKABLE token, so the
    label below reports which of the two we actually measured.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say the single word: ready"}],
        "max_tokens": 8,
        "stream": True,
    }
    label = model if reasoning is None else f"{model} reasoning={reasoning!r}"
    if reasoning is not None:
        body["reasoning_effort"] = reasoning
    t0 = time.perf_counter()
    ttft: float | None = None
    text = ""
    try:
        async with client.stream(
            "POST", f"{GROQ_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=body,
        ) as r:
            if r.status_code != 200:
                line(BAD, f"Groq chat ({label})", f"HTTP {r.status_code} — {(await r.aread())[:160]!r}")
                return
            async for raw in r.aiter_lines():
                if not raw.startswith("data: "):
                    continue
                payload = raw[6:]
                if payload == "[DONE]":
                    break
                import json as _json
                delta = (_json.loads(payload).get("choices") or [{}])[0].get("delta", {})
                if chunk := delta.get("content"):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    text += chunk
            remaining = r.headers.get("x-ratelimit-remaining-requests", "?")
    except httpx.HTTPError as exc:
        line(BAD, f"Groq chat ({label})", f"{type(exc).__name__}: {exc}")
        return
    line(OK, f"Groq chat ({label})",
         f"first token {ttft:.0f} ms, replied {text.strip()!r}, {remaining} requests left this window")


async def nvidia_models(client: httpx.AsyncClient, key: str) -> list[str]:
    r = await client.get(f"{NVIDIA_BASE}/models", headers={"Authorization": f"Bearer {key}"})
    if r.status_code != 200:
        line(BAD, "NVIDIA models", f"HTTP {r.status_code} — {r.text[:160]}")
        return []
    ids = sorted(m["id"] for m in r.json().get("data", []))
    line(OK, "NVIDIA models", f"{len(ids)} available")
    speech = [m for m in ids if any(w in m.lower() for w in ("asr", "tts", "speech", "riva", "parakeet", "canary", "magpie"))]
    if speech:
        line(INFO, "NVIDIA speech models on this endpoint", ", ".join(speech))
    else:
        line(INFO, "NVIDIA speech models on this endpoint",
             "none — ASR/TTS live on a different host, see the note printed below")
    return ids


async def main() -> int:
    groq_key = os.getenv("GROQ_API_KEY", "")
    nv_key = os.getenv("NVIDIA_API_KEY", "")
    if not groq_key or not nv_key:
        print("error: GROQ_API_KEY and NVIDIA_API_KEY must both be set in .env", file=sys.stderr)
        return 1

    print("Checking providers. Nothing here is cached — every line is a live call.\n")
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("--- Groq (language model) ---")
        ids = await groq_models(client, groq_key)
        # Guard and safeguard models are text classifiers — they reject `stream: true`.
        chat = [m for m in ids if not any(w in m.lower() for w in ("guard", "whisper", "tts", "embed"))]
        # Order of preference from docs/research.md: qwen3.6 is the only free Groq model whose
        # reasoning can be switched off, so its first streamed token is speakable text.
        ranked = sorted(chat, key=lambda m: (0 if "qwen3.6" in m else 1 if "qwen" in m else 2, m))
        if ranked:
            line(INFO, "chat models", ", ".join(ranked))
            # The decisive test for this project: can reasoning be switched off so the
            # first streamed token is speakable? Run it both ways and compare.
            await groq_chat(client, groq_key, ranked[0])
            await groq_chat(client, groq_key, ranked[0], reasoning="none")

        print("\n--- NVIDIA NIM ---")
        await nvidia_models(client, nv_key)

    print("\nNote: NVIDIA's LLM endpoint and its speech (ASR/TTS) endpoints are different")
    print("services. This script confirms the key works and lists what the LLM endpoint")
    print("exposes; the speech endpoints are wired up in the next step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
