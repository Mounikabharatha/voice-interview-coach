"""Speech synthesis.

`synthesize` yields audio in chunks rather than returning a finished file. Time to the FIRST
chunk is what the user experiences; total synthesis time is close to irrelevant. The stream
must also be cancellable mid-sentence so barge-in can stop the voice immediately.
"""
from __future__ import annotations

from typing import AsyncIterator, Protocol


class TTSProvider(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM chunks as they are produced."""
        ...
