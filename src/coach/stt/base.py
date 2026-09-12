"""Speech to text.

Streaming is the whole point: `partials` must arrive WHILE the user is still speaking.
A provider that can only transcribe a finished recording cannot satisfy this interface,
which is why Groq's Whisper endpoint is not an option here (see docs/research.md).
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol


class Transcript(Protocol):
    text: str
    is_final: bool


class STTProvider(Protocol):
    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        """Consume PCM frames, yield partial transcripts then a final one."""
        ...

    async def aclose(self) -> None: ...
