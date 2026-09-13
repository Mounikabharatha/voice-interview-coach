"""The interview session: what to ask, when to probe, when to move on, what to report.

**Two scoring paths, deliberately.** Deciding whether to probe has to happen *now* — the
candidate is sitting in silence waiting for the coach to speak. Scoring properly needs an LLM
call, which is a whole extra round trip. So:

- `quick_gaps()` is a cheap, deterministic, local heuristic. It decides probe-vs-advance in
  microseconds and never adds latency to a turn.
- `scoring.score_answer()` runs in the background against the full rubric and feeds the
  end-of-session report, where nobody is waiting.

The heuristics are crude on purpose and they are not pretending otherwise: counting "we"
against "I" is a blunt proxy for ownership. It is right often enough to pick a sensible
follow-up question, and picking a slightly wrong follow-up costs the candidate one question,
not a wrong assessment. The report, which is the thing they take away, uses the real rubric.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .questions import Question, pick_session
from .rubric import BY_KEY

MAX_PROBES_PER_QUESTION = 2
QUESTIONS_PER_SESSION = 3

# A result claim usually carries a number, a unit, or a before/after word.
_NUMBERISH = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|dozen|half|double|"
    r"percent|per cent|%|x|times|weeks?|months?|days?|hours?|years?|"
    r"reduced|increased|cut|grew|saved|shipped|launched|from|to)\b",
    re.I,
)
_I_WORDS = re.compile(r"\b(i|my|me|myself)\b", re.I)
_WE_WORDS = re.compile(r"\b(we|our|us|the team|they)\b", re.I)
_CONCRETE_VERBS = re.compile(
    r"\b(wrote|built|called|met|changed|moved|rewrote|escalated|proposed|refused|"
    r"scheduled|tested|measured|asked|told|drafted|negotiated|migrated|deployed|"
    r"removed|added|split|merged|documented|trained|hired|fired|presented)\b",
    re.I,
)


class State(str, Enum):
    GREETING = "greeting"
    ASKING = "asking"
    PROBING = "probing"
    WRAPPING = "wrapping"
    DONE = "done"


def quick_gaps(answer: str) -> list[str]:
    """Cheap local guess at which competencies this answer is weakest on, worst first."""
    words = re.findall(r"[A-Za-z']+", answer)
    gaps: list[tuple[int, str]] = []

    if not _NUMBERISH.search(answer):
        gaps.append((0, "result"))          # most common gap, and the most useful to probe

    i_count, we_count = len(_I_WORDS.findall(answer)), len(_WE_WORDS.findall(answer))
    if we_count > i_count:
        gaps.append((1, "ownership"))

    if not _CONCRETE_VERBS.search(answer):
        gaps.append((2, "action"))

    if len(words) < 45:
        gaps.append((3, "situation"))       # too short to have set any scene

    if not re.search(r"\b(learn|learned|next time|differently|realised|realized|"
                     r"mistake|wrong|should have|in hindsight)\b", answer, re.I):
        gaps.append((4, "reflection"))

    gaps.sort()
    return [k for _, k in gaps]


@dataclass
class TurnRecord:
    question_id: str
    question_text: str
    answer: str
    was_probe: bool


@dataclass
class Session:
    questions: list[Question] = field(default_factory=lambda: pick_session(QUESTIONS_PER_SESSION))
    state: State = State.GREETING
    index: int = 0
    probes_used: int = 0
    probed_keys: list[str] = field(default_factory=list)
    transcript: list[TurnRecord] = field(default_factory=list)

    @property
    def current(self) -> Question | None:
        return self.questions[self.index] if self.index < len(self.questions) else None

    def opening_line(self) -> str:
        q = self.questions[0]
        self.state = State.ASKING
        return (
            f"Hi. I'll ask you {len(self.questions)} behavioural questions and give you "
            f"feedback at the end. Take your time — I won't cut you off if you pause. "
            f"First one: {q.text}"
        )

    def record(self, answer: str, *, was_probe: bool) -> None:
        q = self.current
        if q is not None:
            self.transcript.append(TurnRecord(q.id, q.text, answer, was_probe))

    def next_move(self, answer: str) -> tuple[str, str]:
        """Decide what happens after an answer.

        Returns (kind, text) where kind is 'probe', 'question' or 'wrap'. `text` is the
        instruction handed to the conversational model, not a line read verbatim — the coach
        rephrases it so it follows naturally from what was just said.
        """
        q = self.current
        gaps = [g for g in quick_gaps(answer) if g not in self.probed_keys]

        if gaps and self.probes_used < MAX_PROBES_PER_QUESTION:
            key = gaps[0]
            self.probes_used += 1
            self.probed_keys.append(key)
            self.state = State.PROBING
            template = BY_KEY[key].probe_templates[
                (self.probes_used - 1) % len(BY_KEY[key].probe_templates)
            ]
            return "probe", template

        # Done with this question.
        self.index += 1
        self.probes_used = 0
        self.probed_keys.clear()
        if self.current is None:
            self.state = State.WRAPPING
            return "wrap", (
                "Thank the candidate briefly, say that's the last question, and tell them "
                "their written feedback is on screen now. Two sentences at most."
            )
        self.state = State.ASKING
        return "question", (
            f"Acknowledge their answer in at most one short sentence, then ask this next "
            f"question, word for word: {self.current.text}"
        )

    @property
    def finished(self) -> bool:
        return self.state in (State.WRAPPING, State.DONE)
