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
# Milliseconds with NO NEW WORDS from the recogniser before a turn ends. The recogniser's own
# finalisation lag (300-500 ms) is inside these, not added to them, so wall-clock from the
# candidate falling silent is roughly these numbers minus that lag.
SILENCE_COMPLETE_MS = 1300
SILENCE_UNCLEAR_MS = 1800
SILENCE_INCOMPLETE_MS = 2600

# Below this many words, a grammatically complete sentence is treated as the opening clause of
# an answer rather than the whole thing. Tuned for behavioural interview answers, which run
# 40-120 words; 20 is deliberately conservative.
MIN_ANSWER_WORDS = 20

# Words that CANNOT end an English sentence: conjunctions, articles, prepositions, fillers.
# These override punctuation, and they have to, because the recogniser runs with automatic
# punctuation on and cheerfully emits "...to a new payments provider, and." — period included —
# for a candidate who is plainly mid-sentence. Trusting that period cut answers in half.
NEVER_FINAL = frozenset(
    """
    and but so or nor yet because since while although though however whereas
    plus also therefore thus hence moreover furthermore
    the a an my our their his her its your this these those
    to of for with from about into onto upon than as at by
    um uh er erm like basically actually literally
    which when where if then whether
    """.split()
)

# Usually mid-thought, but they genuinely can end a sentence — "That's it.", "I did it."
# For these, punctuation is trusted, because the recogniser only emits it on falling intonation.
WEAK_CONTINUATIONS = frozenset(
    """
    that who whom it they he she you we i
    was were is are am been being had has have did do does
    could would should might must can will
    """.split()
)

_WORD = re.compile(r"[A-Za-z']+")


def looks_complete(text: str) -> str:
    """Classify a pending transcript as 'complete', 'incomplete' or 'unclear'.

    Order matters, and it is not the obvious one. `NEVER_FINAL` is checked before punctuation
    precisely because the recogniser's punctuation is unreliable mid-answer; `WEAK_CONTINUATIONS`
    is checked after, because there punctuation is the better signal.
    """
    stripped = text.strip()
    if not stripped:
        return "incomplete"

    words = _WORD.findall(stripped.lower())
    last = words[-1] if words else ""

    # Outranks punctuation: no English sentence ends on "and", whatever the recogniser wrote.
    if last in NEVER_FINAL:
        return "incomplete"

    # Domain rule, and the one that does the real work here. A behavioural interview answer is
    # a story: situation, action, result. Nobody tells one in twelve words. So a short,
    # grammatically complete sentence is far more likely to be the first clause of an answer
    # than the whole of it.
    #
    # This rule exists because the lexical rules above cannot fire. The recogniser segments on
    # its own endpointing and punctuates what it emits, so a candidate who says
    # "...to a new payments provider, and" then pauses is transcribed as
    # "...to a new payments provider." — trailing conjunction deleted, period added. The
    # evidence of being mid-sentence is destroyed before this code ever sees the text.
    #
    # Length is the signal that survives that. It is a blunt instrument, and the honest fix is
    # a semantic turn detector that judges completeness from the audio; see the README.
    if len(words) < MIN_ANSWER_WORDS:
        return "incomplete"

    if stripped[-1] in ".!?":
        return "complete"
    if last in WEAK_CONTINUATIONS:
        return "incomplete"
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
    _last_text: float | None = None

    def add_final(self, text: str) -> None:
        """A recogniser final is a fragment of the turn, not the end of it.

        Deliberately does NOT reset the silence timer. A final arrives 300-500 ms *after* the
        audio it describes — it is confirmation of speech that has already stopped, not
        evidence of new speech. Resetting on it made every turn pay the recogniser's
        finalisation latency on top of the silence threshold, roughly doubling the wait.
        Silence is measured from the audio, which is the only signal that knows the truth.
        """
        if text.strip():
            self._pending.append(text.strip())
            self.note_activity()

    @property
    def pending_text(self) -> str:
        return " ".join(self._pending).strip()

    def note_activity(self) -> None:
        """Call for EVERY transcript, partial or final — this is the clock that matters."""
        self._last_text = time.monotonic()

    def feed_audio(self, frame: bytes, frame_ms: float) -> TurnDecision | None:
        """Returns a decision when the turn is complete, otherwise None.

        **The signal is silence from the recogniser, not silence from a loudness threshold.**

        The first version measured RMS energy against an adaptive noise floor and ended a turn
        after enough quiet frames. On a real microphone that failed in both directions: in one
        room the floor sat low enough that ordinary background noise read as speech forever and
        the turn never ended at all; in another, speech read as silence.

        The recogniser already contains a far better speech model than an energy threshold ever
        will. It emits text only when it hears words, so "no new text for N milliseconds" means
        the person stopped talking — regardless of fans, traffic, room tone, or microphone gain.
        That is the signal used now. Energy is kept only to populate the UI meter.

        The cost is the recogniser's own finalisation delay, roughly 300-500 ms, which is now
        inside the wait rather than added after it. The thresholds below are set accordingly.
        """
        self.noise_floor = 0.98 * self.noise_floor + 0.02 * frame_rms(frame)  # meter only

        text = self.pending_text
        if not text or self._last_text is None:
            return None

        quiet_ms = (time.monotonic() - self._last_text) * 1000
        needed = required_silence_ms(text)
        if quiet_ms >= needed:
            decision = TurnDecision(text, quiet_ms, looks_complete(text))
            log.info(
                "turn end: %s after %.0f ms with no new words (needed %d) — %r",
                decision.verdict, quiet_ms, needed, text[:60],
            )
            self.reset()
            return decision
        return None

    def reset(self) -> None:
        self._pending.clear()
        self._last_text = None


@dataclass
class BargeInDetector:
    """Decides whether the user is talking over the coach.

    Separate from `Endpointer` because the question is different. Endpointing asks "have they
    finished?" and can afford to be patient. Barge-in asks "have they started?" and must be
    fast — every millisecond of delay is the coach still talking over someone.

    The hard part is not detecting speech, it is not detecting *the coach's own voice*
    arriving back through the microphone. Three things guard against that:

    1. The browser's acoustic echo canceller, which does most of the work.
    2. A higher energy bar than normal endpointing — `speech_factor` here is deliberately
       stricter, because residual echo is quieter than a real speaker.
    3. `min_speech_ms`, so a cough, a keyboard click or one leaked syllable cannot cancel a
       reply. This is the main reason the threshold is not simply "any frame above the floor".

    Headphones remove the problem entirely, which is why the UI recommends them.

    **The noise floor must be learned, not assumed.** An earlier version hardcoded it at 120,
    which survived every test because the tests fed digitally silent frames (RMS exactly 0).
    On a real microphone with automatic gain control, room tone alone sits well above that, so
    barge-in fired on the first frame the coach spoke and cancelled every reply instantly. The
    floor is now taken from `Endpointer`, which already adapts to the room, with an absolute
    minimum so a very quiet room cannot drive the threshold down to nothing.
    """

    speech_factor: float = 4.0      # stricter than Endpointer: residual echo is quiet
    min_speech_ms: float = 240.0    # sustained, not a click
    grace_ms: float = 400.0         # see `arm()`
    min_floor: float = 60.0         # a silent room must not make the threshold ~0

    _speech_ms: float = 0.0
    _floor: float = 120.0
    _since_armed_ms: float = 0.0

    def arm(self, noise_floor: float) -> None:
        """Called when the coach starts speaking.

        The grace period matters for two reasons. The browser's echo canceller needs a moment
        to converge on a signal it has not heard before, so the first few hundred milliseconds
        of playback are the most likely to leak. And a barge-in in the first instant cancels a
        reply before a single word of it is audible, which reads as the app being broken rather
        than as responsive.
        """
        self._speech_ms = 0.0
        self._since_armed_ms = 0.0
        self._floor = max(noise_floor, self.min_floor)

    def feed(self, frame: bytes, frame_ms: float) -> bool:
        """True the moment sustained speech is confirmed. Call only while the coach speaks."""
        self._since_armed_ms += frame_ms
        if self._since_armed_ms < self.grace_ms:
            return False
        if frame_rms(frame) > self._floor * self.speech_factor:
            self._speech_ms += frame_ms
            if self._speech_ms >= self.min_speech_ms:
                self._speech_ms = 0.0
                return True
        else:
            # Decay rather than reset: real speech has gaps between words, and a hard reset
            # on the first quiet frame makes the detector miss anyone speaking slowly.
            self._speech_ms = max(0.0, self._speech_ms - frame_ms * 0.5)
        return False

    def reset(self) -> None:
        self._speech_ms = 0.0
        self._since_armed_ms = 0.0
