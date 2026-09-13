"""Speech synthesis via NVIDIA Magpie TTS, served over Riva gRPC.

Two things here were found by measurement, not by reading documentation, and both matter.

**It is gRPC, not REST.** NVIDIA's speech models are not on the OpenAI-compatible endpoint at
`integrate.api.nvidia.com`; every HTTP path there returns 404. They are NVCF functions whose
record reports `"protocol": "gRPC"`, reached through `grpc.nvcf.nvidia.com:443` with the
function id passed as call metadata.

**Prewarming is worth ~940 ms.** Measured on this project, 44.1 kHz, four consecutive sentences:

    call 1   1141 ms   (cold: TLS + HTTP/2 setup)
    call 2    198 ms
    call 3    199 ms
    call 4    197 ms

The connection cost is paid once. Without `prewarm()` at startup the user pays it on the first
thing the coach ever says, which is the worst possible moment — the opening question, while a
recruiter is watching. `prewarm()` is not an optimisation to add later; it is the difference
between a 1.1 s and a 0.2 s first impression.
"""
from __future__ import annotations

import os

# Must be set before grpc is imported. grpc's default c-ares resolver fails to resolve
# grpc.nvcf.nvidia.com in some sandboxed and corporate-DNS environments while the system
# resolver handles it fine; "native" defers to the OS. Harmless where c-ares already works.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

import asyncio
import logging
from typing import AsyncIterator

import riva.client

log = logging.getLogger(__name__)

NVCF_GRPC = "grpc.nvcf.nvidia.com:443"
# ai-magpie-tts-multilingual, status ACTIVE as of 2026-09-13.
# Re-check with scripts/check_providers.py if synthesis starts failing — NVCF function ids
# are stable per deployment but the catalogue does change.
MAGPIE_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"

DEFAULT_VOICE = "Magpie-Multilingual.EN-US.Sofia"
SAMPLE_RATE_HZ = 44_100


class NvidiaTTS:
    """Implements the TTSProvider protocol in `coach.tts.base`."""

    def __init__(
        self,
        api_key: str,
        *,
        voice: str = DEFAULT_VOICE,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
        function_id: str = MAGPIE_FUNCTION_ID,
    ) -> None:
        self.voice = voice
        self.sample_rate_hz = sample_rate_hz
        self._auth = riva.client.Auth(
            uri=NVCF_GRPC,
            use_ssl=True,
            metadata_args=[
                ["function-id", function_id],
                ["authorization", f"Bearer {api_key}"],
            ],
        )
        self._service = riva.client.SpeechSynthesisService(self._auth)

    def _synthesize_blocking(self, text: str, out: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        """Riva's client is synchronous, so it runs off the event loop.

        Chunks are pushed to the queue as they arrive rather than collected, so the caller can
        start playing audio before synthesis has finished — which is the entire point.
        """
        try:
            for resp in self._service.synthesize_online(
                text=text,
                voice_name=self.voice,
                language_code="en-US",
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=self.sample_rate_hz,
            ):
                if resp.audio:
                    loop.call_soon_threadsafe(out.put_nowait, resp.audio)
        except Exception as exc:  # surfaced to the consumer, never swallowed
            loop.call_soon_threadsafe(out.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(out.put_nowait, None)  # sentinel: stream finished

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield 16-bit mono PCM chunks as they are produced.

        Cancellable mid-sentence: the consumer abandoning this generator stops playback
        immediately, which is what barge-in needs. The worker thread finishes its current
        response and exits on its own.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        worker = loop.run_in_executor(None, self._synthesize_blocking, text, queue, loop)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            worker.cancel()

    async def prewarm(self) -> None:
        """Pay the TLS and HTTP/2 setup cost at startup instead of on the first spoken word.

        Worth ~940 ms measured. Failure is logged and swallowed — a cold connection is a
        slow first sentence, not a reason to refuse to start.
        """
        try:
            async for _ in self.synthesize("Ready."):
                break  # first chunk is enough to establish the connection
            log.info("tts prewarmed")
        except Exception as exc:
            log.warning("tts prewarm failed, first utterance will be slow: %s", exc)
