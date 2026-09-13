"""Deciding when the candidate has actually finished speaking.

This is the piece that separates a usable interview coach from an annoying one, and step 8
demonstrated why. The speech recogniser finalises a transcript whenever it hears a sentence
end — so an answer like:

    "I led the migration of our billing system to a new payments provider.
     [thinks for a second]
     It took about four months, and I coordinated three teams."

arrives as *two* finals, and treating each one as end-of-turn interrupts the candidate
mid-answer. Interview answers are exactly the case where this matters: people pause to
recall a number, or to decide how much credit to claim.

The fix has two halves.

**Silence, not sentence ends.** Recogniser finals accumulate into a pending transcript. The
turn only ends after a genuine run of silence, measured from the audio itself.

**How long to wait depends on what was said.** A fixed threshold is either too eager (cuts
people off) or too slow (every reply feels laggy). So the wait adapts to whether the text
sounds finished:

    "...and I coordinated three teams."     complete    -> 900 ms
    "I led the migration, and"              incomplete  -> 2500 ms
    "It took about four months"             unclear     -> 1500 ms

A trailing conjunction is the strongest signal there is that someone is mid-thought, and it
costs nothing to check. Saying "and" then pausing buys the candidate an extra 1.6 seconds.

Voice activity detection here is plain RMS energy against an adaptive noise floor. A learned
VAD would be better in a noisy room, and would be the natural upgrade; energy is chosen
because it has no dependency, no model download, and is honest about what it is.
"""
from __future__ import annotations

import array
import logging
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Milliseconds of silence required to end a turn, by how finished the text sounds.
SILENCE_COMPLETE_MS = 900
SILENCE_UNCLEAR_MS = 1500
SILENCE_INCOMPLETE_MS = 2500

# Words that almost always have more coming after them.
TRAILING_CONTINUATIONS = frozenset(
    """
    and but so or because since while although though however
    which that who when where if then plus also
    um uh er erm like basically actually
    the a an my our their his her its your
    to of for with from about into onto
    was were is are been being had has have did do does
    i we they he she it you
    """.split()
)

_WORD = re.compile(r"[A-Za-z']+")


def looks_complete(text: str) -> str:
    """Classify a pending transcript as 'complete', 'incomplete' or 'unclear'."""
    stripped = text.strip()
    if not stripped:
        return "incomplete"
    # Terminal punctuation is checked FIRST and outranks the word test. Plenty of complete
    # sentences end on a word that is usually a continuation — "That's it!", "I did it." —
    # and the recogniser only emits that punctuation when it heard a falling, final
    # intonation. Testing the word first misreads those as mid-thought.
    if stripped[-1] in ".!?":
        return "complete"
    words = _WORD.findall(stripped.lower())
    if words and words[-1] in TRAILING_CONTINUATIONS:
        return "incomplete"  # nobody finishes an answer on "and"
    return "unclear"


def required_silence_ms(text: str) -> int:
    return {
        "complete": SILENCE_COMPLETE_MS,
        "unclear": SILENCE_UNCLEAR_MS,
        "incomplete": SILENCE_INCOMPLETE_MS,
    }[looks_complete(text)]


def frame_rms(frame: bytes) -> float:
    """RMS of a little-endian PCM16 frame. Pure stdlib: `audioop` is gone in 3.13."""
    if len(frame) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


@dataclass
class TurnDecision:
    text: str
    waited_ms: float
    verdict: str  # which completeness class triggered the wait


@dataclass
class Endpointer:
    """Feed it audio frames and recogniser finals; it tells you when the turn is over.

    Not thread-safe and not reusable across connections — one per WebSocket.
    """

    noise_floor: float = 120.0        # adapts upward in a noisy room
    speech_factor: float = 2.5        # RMS must exceed floor * this to count as speech
    min_speech_ms: float = 300.0      # ignore coughs and door slams

    _pending: list[str] = field(default_factory=list)
    _silence_started: float | None = None
    _speech_ms: float = 0.0
    _speaking: bool = False

    def add_final(self, text: str) -> None:
        """A recogniser final is a fragment of the turn, not the end of it."""
        if text.strip():
            self._pending.append(text.strip())
            self._silence_started = None  # new words cancel any silence run

    @property
    def pending_text(self) -> str:
        return " ".join(self._pending).strip()

    def feed_audio(self, frame: bytes, frame_ms: float) -> TurnDecision | None:
        """Returns a decision when the turn is complete, otherwise None."""
        rms = frame_rms(frame)

        # Track the noise floor only while nobody is speaking, so a loud answer does not
        # drag the threshold up and deafen the detector.
        is_speech = rms > self.noise_floor * self.speech_factor
        if not is_speech:
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms

        now = time.monotonic()
        if is_speech:
            self._speaking = True
            self._speech_ms += frame_ms
            self._silence_started = None
            return None

        if not self._speaking and not self._pending:
            return None  # never spoke; nothing to end

        if self._silence_started is None:
            self._silence_started = now
            return None

        silent_ms = (now - self._silence_started) * 1000
        text = self.pending_text
        if not text or self._speech_ms < self.min_speech_ms:
            return None

        needed = required_silence_ms(text)
        if silent_ms >= needed:
            decision = TurnDecision(text, silent_ms, looks_complete(text))
            log.info(
                "turn end: %s after %.0f ms silence (needed %d) — %r",
                decision.verdict, silent_ms, needed, text[:60],
            )
            self.reset()
            return decision
        return None

    def reset(self) -> None:
        self._pending.clear()
        self._silence_started = None
        self._speech_ms = 0.0
        self._speaking = False
