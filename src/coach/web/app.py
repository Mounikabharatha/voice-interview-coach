"""FastAPI server: browser microphone in, spoken reply out, over one WebSocket.

All audio hardware lives in the browser. The server never opens an audio device, which is
what keeps the install free of native dependencies and makes `pip install -r requirements.txt`
behave identically on Windows, Linux and macOS.

Wire protocol, v1:

    client -> server   binary   16 kHz mono PCM16 microphone frames
    client -> server   JSON     {"type": "start"}
    server -> client   binary   44.1 kHz mono PCM16 playback audio
    server -> client   JSON     {"type": "partial" | "final" | "assistant_text"
                                          | "state" | "metrics" | "error", ...}

Turn detection belongs to `pipeline/endpoint.py`. Recogniser finals are treated as
*fragments* of an answer, and the turn ends only after a run of silence whose required length
depends on whether the text sounds finished. Barge-in — interrupting the coach mid-sentence —
is not implemented here; endpointing is suspended while the coach speaks.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from ..llm.groq import GroqLLM
from ..pipeline.endpoint import Endpointer
from ..pipeline.turn import TurnController
from ..stt.nvidia import NvidiaSTT
from ..tts.nvidia import NvidiaTTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("coach")

STATIC = Path(__file__).parent / "static"
PLAYBACK_RATE_HZ = 44_100
MIC_FRAME_MS = 100  # must match FRAME_MS in static/mic-worklet.js

app = FastAPI(title="Voice Interview Coach")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
async def startup() -> None:
    settings = Settings.load()
    app.state.settings = settings
    app.state.stt = NvidiaSTT(settings.nvidia_api_key)
    app.state.tts = NvidiaTTS(settings.nvidia_api_key, sample_rate_hz=PLAYBACK_RATE_HZ)
    app.state.llm = GroqLLM(settings.groq_api_key)
    # Worth ~940 ms on the speech connection alone. Paid here so it is never paid in front
    # of a user — see coach/tts/nvidia.py.
    log.info("prewarming providers...")
    await asyncio.gather(
        app.state.tts.prewarm(), app.state.stt.prewarm(), app.state.llm.prewarm()
    )
    log.info("ready on http://%s:%d", settings.host, settings.port)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "playback_rate_hz": PLAYBACK_RATE_HZ}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    audio_in: asyncio.Queue[bytes | None] = asyncio.Queue()
    send_lock = asyncio.Lock()

    async def send_event(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_audio(pcm: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(pcm)

    controller = TurnController(
        app.state.llm, app.state.tts, send_audio=send_audio, send_event=send_event
    )

    async def mic_frames() -> AsyncIterator[bytes]:
        while (frame := await audio_in.get()) is not None:
            yield frame

    endpointer = Endpointer()
    responding = False

    async def transcribe() -> None:
        """Stream transcripts out.

        A recogniser final is a *fragment*, not the end of the turn — step 8 shipped the
        opposite and cut answers in half. Finals accumulate in the endpointer, which decides
        from silence and phrasing when the candidate has actually stopped.
        """
        try:
            async for tr in app.state.stt.stream(mic_frames()):
                if not tr.text.strip():
                    continue
                if tr.is_final:
                    endpointer.add_final(tr.text)
                    await send_event({"type": "fragment", "text": endpointer.pending_text})
                else:
                    await send_event({"type": "partial", "text": tr.text})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("transcription stream failed")
            with contextlib.suppress(Exception):
                await send_event({"type": "error", "message": str(exc)[:200]})

    async def handle_turn(decision) -> None:
        """Run one response. Kept off the receive loop so audio keeps flowing."""
        nonlocal responding
        responding = True
        try:
            await send_event({
                "type": "final", "text": decision.text,
                "verdict": decision.verdict, "waited_ms": round(decision.waited_ms),
            })
            await send_event({"type": "state", "state": "thinking"})
            await controller.respond(decision.text)
        except Exception as exc:
            log.exception("turn failed")
            with contextlib.suppress(Exception):
                await send_event({"type": "error", "message": str(exc)[:200]})
        finally:
            responding = False
            endpointer.reset()
            with contextlib.suppress(Exception):
                await send_event({"type": "state", "state": "listening"})

    task = asyncio.create_task(transcribe())
    turn_task: asyncio.Task | None = None
    try:
        await send_event({"type": "state", "state": "greeting"})
        await controller.say(controller.opening_line())
        await send_event({"type": "state", "state": "listening"})
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is None:
                continue
            await audio_in.put(data)
            # Endpointing pauses while the coach is speaking. Without barge-in there is
            # nothing useful to do with speech during playback, and the coach's own voice
            # leaking through the mic would otherwise trigger a turn. Step 10 changes this.
            if responding:
                continue
            if (decision := endpointer.feed_audio(data, MIC_FRAME_MS)) is not None:
                turn_task = asyncio.create_task(handle_turn(decision))
    except WebSocketDisconnect:
        pass
    finally:
        await audio_in.put(None)
        for t in (task, turn_task):
            if t is not None:
                t.cancel()
        await asyncio.gather(*[t for t in (task, turn_task) if t], return_exceptions=True)
        log.info("client disconnected")
