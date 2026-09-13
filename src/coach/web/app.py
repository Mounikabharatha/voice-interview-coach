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

Turn detection for now is the speech recogniser's own endpointing: its `is_final` transcript
*is* the end-of-turn signal. That is a deliberate shortcut — it is free, it is already
measured at 286-569 ms, and it keeps this step to one moving part. Proper voice-activity
detection and barge-in replace it in a later step, at which point the user will be able to
interrupt mid-sentence, which they cannot do here.
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
from ..pipeline.turn import TurnController
from ..stt.nvidia import NvidiaSTT
from ..tts.nvidia import NvidiaTTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("coach")

STATIC = Path(__file__).parent / "static"
PLAYBACK_RATE_HZ = 44_100

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

    async def transcribe() -> None:
        """Stream transcripts out; a final transcript is the end of the user's turn."""
        try:
            async for tr in app.state.stt.stream(mic_frames()):
                if not tr.text.strip():
                    continue
                if tr.is_final:
                    await send_event({"type": "final", "text": tr.text})
                    await send_event({"type": "state", "state": "thinking"})
                    try:
                        await controller.respond(tr.text)
                    except Exception as exc:
                        log.exception("turn failed")
                        await send_event({"type": "error", "message": str(exc)[:200]})
                    await send_event({"type": "state", "state": "listening"})
                else:
                    await send_event({"type": "partial", "text": tr.text})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("transcription stream failed")
            with contextlib.suppress(Exception):
                await send_event({"type": "error", "message": str(exc)[:200]})

    task = asyncio.create_task(transcribe())
    try:
        await send_event({"type": "state", "state": "greeting"})
        await controller.say(controller.opening_line())
        await send_event({"type": "state", "state": "listening"})
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await audio_in.put(data)
    except WebSocketDisconnect:
        pass
    finally:
        await audio_in.put(None)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        log.info("client disconnected")
