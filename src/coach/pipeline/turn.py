"""One conversational turn: transcript in, spoken audio out.

The ordering here is the whole point of the project. A naive implementation waits for the
model to finish, synthesizes the reply, then plays it — which adds generation time and
synthesis time to the silence the user sits through. This one overlaps them: the first
complete clause goes to speech synthesis while the model is still writing the rest.

The marks recorded on `TurnMetrics` are all server-side. The endpoint the user actually
experiences — first audible sample — happens in the browser, so the honest end-to-end number
needs a client mark too. That is wired up in a later step; until then these are labelled as
server-side and the README says so.
"""
from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable

from .sentence import Sentencizer

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a concise, warm interview coach running a mock behavioural interview.

You are being SPOKEN ALOUD. Write only what a person would say out loud.

Rules:
- At most two short sentences. Usually one.
- No markdown, bullet points, numbered lists, emoji or headings. Ever.
- No stage directions, no "as an AI", no meta-commentary about the interview.
- Be warm but never flattering. Do not praise an answer that has not earned it; "that's a
  good start" is fine, "what a fantastic answer" is not.
- Never score the candidate out loud and never tell them a number. Scoring is written
  feedback they read afterwards.
- Never comment on their voice, accent, confidence, pace or personality. Only on what they
  actually said.

Each turn you are given an INSTRUCTION describing what to do next. Follow it exactly, but
phrase it so it follows naturally from what the candidate just said. When the instruction
contains a question to ask word for word, ask it word for word."""


@dataclass
class TurnMetrics:
    """Server-side timings for one turn. All values are milliseconds from `t0`."""

    t0: float = field(default_factory=time.perf_counter)
    llm_request_sent: float | None = None
    llm_first_token: float | None = None
    first_sentence: float | None = None
    tts_first_byte: float | None = None
    turn_done: float | None = None
    reasoning_chars: int = 0

    def mark(self, name: str) -> None:
        if getattr(self, name, "missing") is None:
            setattr(self, name, (time.perf_counter() - self.t0) * 1000)

    def as_dict(self) -> dict:
        return {
            k: (round(v, 1) if isinstance(v, float) else v)
            for k, v in self.__dict__.items()
            if k != "t0" and v is not None
        }


class TurnController:
    """Runs the response half of a turn: LLM -> sentences -> speech -> caller."""

    def __init__(self, llm, tts, *, send_audio, send_event) -> None:
        self.llm = llm
        self.tts = tts
        self._send_audio: Callable[[bytes], Awaitable[None]] = send_audio
        self._send_event: Callable[[dict], Awaitable[None]] = send_event
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def respond(self, user_text: str, instruction: str | None = None) -> TurnMetrics:
        m = TurnMetrics()
        self.history.append({"role": "user", "content": user_text})
        if instruction:
            # Sent as a system turn rather than folded into the user message, so the
            # candidate's own words stay clean in the history the model sees.
            self.history.append({"role": "system", "content": f"INSTRUCTION: {instruction}"})
        sentencizer = Sentencizer()  # one per turn — see Sentencizer's docstring
        spoken: list[str] = []

        m.mark("llm_request_sent")
        stream = await self.llm.stream_chat(self.history)
        try:
            async with contextlib.aclosing(stream) as deltas:
                async for delta in deltas:
                    m.mark("llm_first_token")
                    for sentence in sentencizer.push(delta):
                        await self._speak(sentence, spoken, m)
            for sentence in sentencizer.flush():  # mandatory — models stop without punctuation
                await self._speak(sentence, spoken, m)
        finally:
            m.reasoning_chars = getattr(stream, "reasoning_chars", 0)
            reply = " ".join(spoken)
            if reply:
                self.history.append({"role": "assistant", "content": reply})
            m.mark("turn_done")
            await self._send_event({"type": "metrics", "turn": m.as_dict()})
        return m

    async def _speak(self, sentence: str, spoken: list[str], m: TurnMetrics) -> None:
        if not sentence.strip():
            return
        m.mark("first_sentence")
        spoken.append(sentence)
        await self._send_event({"type": "assistant_text", "text": sentence})
        async for chunk in self.tts.synthesize(sentence):
            m.mark("tts_first_byte")
            await self._send_audio(chunk)

    async def say(self, text: str) -> None:
        """Speak a fixed line without involving the model — used for the opening question."""
        m = TurnMetrics()
        await self._speak(text, [], m)
        self.history.append({"role": "assistant", "content": text})
