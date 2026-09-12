"""Language model.

`stream_chat` is awaited and returns an async-iterable stream object exposing `aclose()`,
so callers can wrap it in `contextlib.aclosing()` and drop the HTTP body on every exit path
including cancellation. Closing matters: an abandoned stream keeps generating and keeps
counting against the tokens-per-minute quota.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol


class DeltaStream(Protocol):
    def __aiter__(self) -> AsyncIterator[str]: ...
    async def aclose(self) -> None: ...


class LLMProvider(Protocol):
    async def stream_chat(self, messages: list[dict]) -> DeltaStream:
        """Return a stream of text deltas. Reasoning tokens must never be yielded."""
        ...
