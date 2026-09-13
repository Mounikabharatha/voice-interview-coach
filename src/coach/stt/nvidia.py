"""Streaming speech recognition via NVIDIA Nemotron ASR, over Riva gRPC.

Streaming is the whole reason this provider was chosen. Measured on this project, feeding
4.4 s of speech at real-time pace in 100 ms frames:

    795 ms   partial  'I'
    1108 ms  partial  'I led'
    1526 ms  partial  'I led the migration'
    4981 ms  FINAL    'I led the migration of our billing system to a new provider
                       over about four months.'

25 partials, and the final landed **569 ms after the audio ended**. That 569 ms is the real
cost of finishing a turn and it belongs in the latency budget — it is not free, and it is
considerably more than a local streaming model would charge.

The partials are not decoration. They are what lets the interface show words as the candidate
speaks, and what lets the endpointer reason about whether a sentence sounds finished rather
than just waiting out a silence timer.

### Bridging sync gRPC to asyncio

Riva's client is synchronous in both directions: it consumes an iterator of audio chunks and
returns an iterator of responses. Both are bridged here — a thread-safe `queue.Queue` carries
audio *into* the worker thread, and `loop.call_soon_threadsafe` carries results back *out* to
the event loop. The alternative, calling it directly, would block the event loop and stall the
WebSocket read that feeds the microphone.
"""
from __future__ import annotations

import os

# Must precede the grpc import — see coach/tts/nvidia.py for why.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import AsyncIterator

import riva.client

log = logging.getLogger(__name__)

NVCF_GRPC = "grpc.nvcf.nvidia.com:443"
# ai-nemotron-asr-streaming, status ACTIVE as of 2026-09-13.
NEMOTRON_ASR_FUNCTION_ID = "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"

# 16 kHz mono is what the model wants and what the browser will be told to send.
SAMPLE_RATE_HZ = 16_000

_SENTINEL = object()


@dataclass(frozen=True)
class Transcript:
    """Implements the Transcript protocol in `coach.stt.base`."""

    text: str
    is_final: bool


class NvidiaSTT:
    """Implements the STTProvider protocol in `coach.stt.base`."""

    def __init__(
        self,
        api_key: str,
        *,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
        function_id: str = NEMOTRON_ASR_FUNCTION_ID,
        language_code: str = "en-US",
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.language_code = language_code
        self._auth = riva.client.Auth(
            uri=NVCF_GRPC,
            use_ssl=True,
            metadata_args=[
                ["function-id", function_id],
                ["authorization", f"Bearer {api_key}"],
            ],
        )
        self._service = riva.client.ASRService(self._auth)

    def _config(self) -> "riva.client.StreamingRecognitionConfig":
        return riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=self.sample_rate_hz,
                language_code=self.language_code,
                max_alternatives=1,
                enable_automatic_punctuation=True,
            ),
            interim_results=True,  # without this there are no partials and the point is lost
        )

    def _run_stream(
        self,
        audio_in: queue.Queue,
        results_out: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Worker thread: pull audio off a thread-safe queue, push results back to the loop."""

        def chunks():
            while True:
                item = audio_in.get()
                if item is _SENTINEL:
                    return
                yield item

        try:
            for resp in self._service.streaming_response_generator(
                audio_chunks=chunks(), streaming_config=self._config()
            ):
                for res in resp.results:
                    if not res.alternatives:
                        continue
                    t = Transcript(res.alternatives[0].transcript, bool(res.is_final))
                    loop.call_soon_threadsafe(results_out.put_nowait, t)
        except Exception as exc:
            loop.call_soon_threadsafe(results_out.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(results_out.put_nowait, None)

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        """Consume PCM frames, yield partial transcripts and then a final one.

        The caller closing this generator tears the gRPC stream down: the feeder task is
        cancelled and the worker thread's audio iterator returns, ending the RPC.
        """
        loop = asyncio.get_running_loop()
        audio_in: queue.Queue = queue.Queue()
        results: asyncio.Queue = asyncio.Queue()

        worker = threading.Thread(
            target=self._run_stream, args=(audio_in, results, loop), daemon=True
        )
        worker.start()

        async def feed() -> None:
            try:
                async for frame in audio:
                    audio_in.put(frame)
            finally:
                audio_in.put(_SENTINEL)  # always closes the RPC, including on cancellation

        feeder = asyncio.create_task(feed())
        try:
            while True:
                item = await results.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            feeder.cancel()
            audio_in.put(_SENTINEL)

    async def prewarm(self) -> None:
        """Establish the connection at startup, as with TTS — see coach/tts/nvidia.py.

        Sends a short burst of silence so the TLS and HTTP/2 setup is paid before the
        candidate's first word rather than during it.
        """
        silence = b"\x00\x00" * int(self.sample_rate_hz * 0.1)

        async def one_frame() -> AsyncIterator[bytes]:
            yield silence

        try:
            async for _ in self.stream(one_frame()):
                break
            log.info("stt prewarmed")
        except Exception as exc:
            log.warning("stt prewarm failed, first turn will be slow: %s", exc)
