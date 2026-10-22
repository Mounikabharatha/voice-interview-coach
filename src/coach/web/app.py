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
depends on whether the text sounds finished. While the coach is speaking, `BargeInDetector`
watches for the user talking over it and cancels the turn — the client is told to drop audio
it has already scheduled, since the server cannot unsend it.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .limits import Limits

# Heavy imports behind a guard, and this is not defensiveness for its own sake.
#
# These pull in grpcio and the Riva client, native wheels that must match the host's Python
# version and libc. If any of them raises at module scope the ASGI app object is never
# created, every route 500s identically, and a serverless platform reports only
# FUNCTION_INVOCATION_FAILED — which says nothing about which import failed, or whether it was
# an import at all. Recording the traceback and serving it from /healthz turns one opaque
# deploy failure into an answer.
_BOOT_ERROR: str | None = None
try:
    from ..coachlogic import scoring
    from ..coachlogic.session import Session
    from ..config import Settings
    from ..llm.groq import GroqLLM
    from ..pipeline.endpoint import BargeInDetector, Endpointer
    from ..pipeline.turn import TurnController
    from ..stt.nvidia import NvidiaSTT
    from ..tts.nvidia import NvidiaTTS
except Exception:  # pragma: no cover - only on a broken deployment
    import traceback
    _BOOT_ERROR = traceback.format_exc()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("coach")


def friendly_error(exc: Exception) -> str:
    """Turn a provider exception into something a user can act on.

    A raw gRPC rendezvous dump on screen tells the person nothing and makes a working app look
    broken. The full exception still goes to the server log for whoever is debugging.
    """
    text = str(exc)
    if "Address lookup failed" in text or "StatusCode.UNAVAILABLE" in text:
        return ("Could not reach NVIDIA's speech service. Check your internet connection, "
                "then press Start again.")
    if "StatusCode.UNAUTHENTICATED" in text or "401" in text:
        return "NVIDIA rejected the API key. Check NVIDIA_API_KEY in your .env file."
    if "429" in text or "rate limit" in text.lower():
        return "Hit a free-tier rate limit. Wait a minute and press Start again."
    if "groq" in text.lower():
        return "The language model service returned an error. Check GROQ_API_KEY in .env."
    return "Something went wrong. The full error is in the terminal where you ran the server."

STATIC = Path(__file__).parent / "static"
# 22.05 kHz halves outbound bandwidth against 44.1 kHz and is indistinguishable for speech.
# Render's free workspace allows 5 GB/month outbound and then suspends the service until the
# 1st of the next month — a dead link on a CV — so the hosted build does not spend bandwidth
# it does not need. Override with COACH_PLAYBACK_HZ.
PLAYBACK_RATE_HZ = int(os.getenv("COACH_PLAYBACK_HZ", "22050"))
MIC_FRAME_MS = 100  # must match FRAME_MS in static/mic-worklet.js

app = FastAPI(title="Voice Interview Coach")

# StaticFiles raises at construction if the directory is missing, which would take the whole
# module down. On a platform that bundles files for us, "missing" is a deployment question, not
# a code one — so report it rather than crash.
if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
else:
    _BOOT_ERROR = (_BOOT_ERROR or "") + f"\nstatic directory not found at {STATIC}"


@app.on_event("startup")
async def startup() -> None:
    if _BOOT_ERROR:
        log.error("boot failed, not starting providers:\n%s", _BOOT_ERROR)
        return
    settings = Settings.load()
    app.state.settings = settings
    app.state.stt = NvidiaSTT(settings.nvidia_api_key)
    app.state.tts = NvidiaTTS(settings.nvidia_api_key, sample_rate_hz=PLAYBACK_RATE_HZ)
    app.state.llm = GroqLLM(settings.groq_api_key)
    # A SECOND client, for scoring only. The conversational one is capped at 160 tokens
    # because spoken replies must be short — sharing it silently truncated every scoring
    # response mid-JSON. Different jobs, different budgets.
    app.state.scorer = GroqLLM(settings.groq_api_key, max_tokens=900, temperature=0.2)
    # Worth ~940 ms on the speech connection alone. Paid here so it is never paid in front
    # of a user — see coach/tts/nvidia.py.
    # Resolve the model against the live catalogue BEFORE prewarming. Free-tier model ids
    # are not stable — the project's original choice was retired about a month after it was
    # measured — so a pinned id would fail on the first real turn instead of at startup.
    model = await app.state.llm.resolve_model()
    app.state.scorer.model = model
    log.info("using Groq model: %s", model)

    log.info("prewarming providers...")
    await asyncio.gather(
        app.state.tts.prewarm(), app.state.stt.prewarm(), app.state.llm.prewarm()
    )
    app.state.limits = Limits()
    log.info("limits: %s", app.state.limits.snapshot())
    log.info("ready on http://%s:%d", settings.host, settings.port)


@app.get("/")
async def index():
    if _BOOT_ERROR or not (STATIC / "index.html").is_file():
        return PlainTextResponse(
            "The server did not start correctly.\n\n"
            "Diagnostics: /healthz\n\n" + (_BOOT_ERROR or "index.html missing"),
            status_code=503,
        )
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    """Always answers, even when the app failed to boot — that is the whole point of it."""
    if _BOOT_ERROR:
        return {"ok": False, "error": "boot failed", "detail": _BOOT_ERROR.splitlines()[-12:]}
    ready = hasattr(app.state, "limits")
    return {
        "ok": ready,
        "playback_rate_hz": PLAYBACK_RATE_HZ,
        "static_dir_present": STATIC.is_dir(),
        "providers_ready": ready,
        "limits": app.state.limits.snapshot() if ready else None,
    }


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()

    allowed, refusal = app.state.limits.try_acquire()
    if not allowed:
        log.info("session refused: %s", refusal)
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "refused", "message": refusal})
            await websocket.close()
        return
    session_deadline = asyncio.get_running_loop().time() + app.state.limits.max_session_seconds
    audio_in: asyncio.Queue[bytes | None] = asyncio.Queue()
    send_lock = asyncio.Lock()

    async def send_event(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    speaking = {"audio_started": False}

    async def send_audio(pcm: bytes) -> None:
        speaking["audio_started"] = True
        async with send_lock:
            await websocket.send_bytes(pcm)

    controller = TurnController(
        app.state.llm, app.state.tts, send_audio=send_audio, send_event=send_event
    )

    async def mic_frames() -> AsyncIterator[bytes]:
        while (frame := await audio_in.get()) is not None:
            yield frame

    endpointer = Endpointer()
    barge_in = BargeInDetector()
    session = Session()
    # Barge-in is OFF by default. Without headphones the coach's own voice returns through
    # the microphone and cancels every reply — browser echo cancellation is not sufficient
    # on its own. A feature that silently breaks the app for most people is worse than a
    # feature behind a switch, so the client opts in and the UI says what it needs.
    barge_in_enabled = False
    # Set when a reply was cut off before anything was audible, so it can be re-issued.
    pending_instruction: dict[str, str | None] = {"text": None}
    all_scores: list[list[scoring.Score]] = []
    score_tasks: list[asyncio.Task] = []
    responding = False

    async def score_in_background(question: str, answer: str) -> None:
        """Full rubric scoring, off the hot path — nobody is waiting on this."""
        scores = await scoring.score_answer(app.state.scorer, question, answer)
        if scores:
            all_scores.append(scores)

    async def send_report() -> None:
        if score_tasks:
            await asyncio.gather(*score_tasks, return_exceptions=True)
        await send_event({
            "type": "report",
            "summary": scoring.summarise(all_scores),
            "turns": [
                {"question": t.question_text, "answer": t.answer, "was_probe": t.was_probe}
                for t in session.transcript
            ],
            "detail": [[scoring.score_to_dict(s) for s in turn] for turn in all_scores],
        })

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
                # Every transcript resets the stall clock, partials included. A partial is
                # proof the recogniser is still hearing words, which is exactly what the
                # endpointer needs to know.
                endpointer.note_activity()
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
                await send_event({"type": "error", "message": friendly_error(exc)})

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

            question = session.current.text if session.current else ""
            was_probe = session.state.value == "probing"
            session.record(decision.text, was_probe=was_probe)
            # Deep scoring runs in the background against the real rubric; the reply below
            # is decided by a local heuristic so the candidate is not left waiting.
            if question:
                score_tasks.append(
                    asyncio.create_task(score_in_background(question, decision.text))
                )

            if pending_instruction["text"]:
                kind, instruction = "question", pending_instruction["text"]
                pending_instruction["text"] = None
                log.info("re-issuing the instruction the interrupted turn never delivered")
            else:
                kind, instruction = session.next_move(decision.text)
            spoken_any = False
            try:
                metrics = await controller.respond(decision.text, instruction)
                spoken_any = metrics.first_sentence is not None
            finally:
                if not spoken_any:
                    # Interrupted (or failed) before a single word was audible. The session
                    # has already advanced past this question, so without this the candidate
                    # is simply never asked it.
                    pending_instruction["text"] = instruction
                    log.info("turn produced no audio; will re-issue: %s", instruction[:60])
            if kind == "wrap":
                await send_report()
        except asyncio.CancelledError:
            # Barge-in. Expected, not an error. TurnController's own finally block has
            # already closed the LLM stream and recorded what was said.
            with contextlib.suppress(Exception):
                await send_event({"type": "interrupted"})
        except Exception as exc:
            log.exception("turn failed")
            with contextlib.suppress(Exception):
                await send_event({"type": "error", "message": friendly_error(exc)})
        finally:
            responding = False
            endpointer.reset()
            barge_in.reset()
            with contextlib.suppress(Exception):
                await send_event({"type": "state", "state": "listening"})

    task = asyncio.create_task(transcribe())
    turn_task: asyncio.Task | None = None
    try:
        await send_event({"type": "state", "state": "greeting"})
        await send_event({"type": "session_start",
                          "questions": [q.text for q in session.questions]})
        await controller.say(session.opening_line())
        await send_event({"type": "state", "state": "listening"})
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (text := message.get("text")) is not None:
                with contextlib.suppress(Exception):
                    cfg = json.loads(text)
                    if cfg.get("type") == "config":
                        barge_in_enabled = bool(cfg.get("barge_in"))
                        log.info("barge-in %s by client",
                                 "enabled" if barge_in_enabled else "disabled")
                continue
            if (data := message.get("bytes")) is None:
                continue
            if asyncio.get_running_loop().time() > session_deadline:
                log.info("session hit the time limit")
                with contextlib.suppress(Exception):
                    await send_event({"type": "refused", "message":
                        "That's the end of this demo session — it is capped to keep the free "
                        "tier available for others. Press Start to begin a new one."})
                break
            await audio_in.put(data)
            if responding:
                # Only once real audio is going out. Arming at turn-end meant the whole
                # think-and-synthesize pause counted as a window in which room noise could
                # cancel a reply that had not started yet.
                if not barge_in_enabled or not speaking["audio_started"]:
                    continue
                if barge_in.feed(data, MIC_FRAME_MS) and turn_task and not turn_task.done():
                    log.info("barge-in")
                    turn_task.cancel()
                    # Tell the client first: it has audio already scheduled on the device
                    # clock that the server cannot unsend, and it has to drop that itself.
                    await send_event({"type": "cancel_audio"})
                    endpointer.reset()
                continue
            if (decision := endpointer.feed_audio(data, MIC_FRAME_MS)) is not None:
                # Hand barge-in the noise floor the endpointer has learned from this room,
                # plus a grace period. A fixed floor fired on room tone and cancelled every
                # reply before a word of it was audible.
                speaking["audio_started"] = False
                barge_in.arm(endpointer.noise_floor)
                turn_task = asyncio.create_task(handle_turn(decision))
    except WebSocketDisconnect:
        pass
    finally:
        await audio_in.put(None)
        for t in [task, turn_task, *score_tasks]:
            if t is not None:
                t.cancel()
        await asyncio.gather(*[t for t in [task, turn_task, *score_tasks] if t],
                             return_exceptions=True)
        app.state.limits.release()
        log.info("client disconnected (%s)", app.state.limits.snapshot())
