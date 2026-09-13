"""Groq chat completions.

`reasoning_effort="none"` is not a tuning knob here, it is load-bearing. Measured on this
project against `qwen/qwen3.6-27b`:

    default              first token  88 ms  ->  "<think>\\nHere's a thinking process:\\n\\n1"
    reasoning="none"     first token 363 ms  ->  "ready"

Left on, the first tokens streamed are the model's chain-of-thought. They arrive quickly and
cannot be spoken, so the user hears silence while the model thinks to itself. The number a
voice agent cares about is time to first *speakable* token, and switching reasoning off is what
makes that number exist at all.

As a second line of defence, any `reasoning_content` the API emits is counted and discarded
rather than yielded — a misconfigured request should degrade to a slow answer, never to the
coach reading its own notes aloud.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

log = logging.getLogger(__name__)

GROQ_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3.6-27b"


class GroqStream:
    """Async-iterable stream of text deltas, exposing `aclose()`.

    `aclose()` rather than `close()` is deliberate: it lets callers wrap this in
    `contextlib.aclosing()`, which drops the HTTP body on every exit path including
    cancellation. That matters — an abandoned stream keeps generating and keeps counting
    against the tokens-per-minute quota.
    """

    def __init__(self, ctx, response: httpx.Response) -> None:
        self._ctx = ctx
        self._response = response
        self.reasoning_chars = 0
        self.usage: dict | None = None
        h = response.headers
        self.ratelimit = {
            "remaining_requests": h.get("x-ratelimit-remaining-requests"),
            "remaining_tokens": h.get("x-ratelimit-remaining-tokens"),
        }

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        async for raw in self._response.aiter_lines():
            if not raw.startswith("data: "):
                continue
            payload = raw[6:]
            if payload == "[DONE]":
                return
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if usage := chunk.get("usage"):
                self.usage = usage
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Count it, never speak it. Non-zero here means reasoning_effort is misconfigured.
            if rc := delta.get("reasoning_content"):
                self.reasoning_chars += len(rc)
                continue
            if text := delta.get("content"):
                yield text

    async def aclose(self) -> None:
        if self.reasoning_chars:
            log.warning(
                "model emitted %d chars of reasoning — check reasoning_effort",
                self.reasoning_chars,
            )
        try:
            await self._response.aclose()
        finally:
            await self._ctx.__aexit__(None, None, None)


class GroqLLM:
    """Implements the LLMProvider protocol in `coach.llm.base`."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str | None = "none",
        max_tokens: int = 160,
        temperature: float = 0.6,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = httpx.AsyncClient(
            base_url=GROQ_BASE,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def stream_chat(self, messages: list[dict]) -> GroqStream:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort

        ctx = self._client.stream("POST", "/chat/completions", json=body)
        response = await ctx.__aenter__()
        if response.status_code != 200:
            detail = (await response.aread()).decode("utf-8", "replace")[:300]
            await ctx.__aexit__(None, None, None)
            raise RuntimeError(f"groq {response.status_code}: {detail}")
        return GroqStream(ctx, response)

    async def prewarm(self) -> None:
        """Open the TLS connection before the first real turn, as with the speech services."""
        try:
            stream = await self.stream_chat([{"role": "user", "content": "hi"}])
            async for _ in stream:
                break
            await stream.aclose()
            log.info("llm prewarmed")
        except Exception as exc:
            log.warning("llm prewarm failed: %s", exc)

    async def aclose(self) -> None:
        await self._client.aclose()
