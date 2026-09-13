"""The scoring rubric: five competencies, each anchored at levels 1, 3 and 5.

Anchors exist so that scoring is a judgement against a written standard rather than a vibe.
They are also what makes the feedback useful — telling someone their answer scored 2 is
worthless; telling them "you described what the team did, not what you did" is actionable.

The competencies are drawn from how behavioural interviews are actually assessed: a STAR-shaped
story (situation, task, action, result) plus the reflection that distinguishes a candidate who
learned something from one who merely survived.

**A deliberate omission.** There is no "hireability", "confidence" or "communication style"
score. Those are where automated interview scoring has historically gone wrong — they correlate
with accent, gender and background far more than with competence, and a portfolio project that
shipped them would deserve the criticism. Everything scored here is a property of the *answer*:
whether it names a result, whether it says what the candidate personally did. Nothing scores the
person, their voice, or how they sound.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Competency:
    key: str
    name: str
    question: str          # what the scorer is being asked to judge
    anchor_1: str
    anchor_3: str
    anchor_5: str
    probe_templates: tuple[str, ...]   # asked when this competency scores low


RUBRIC: tuple[Competency, ...] = (
    Competency(
        key="situation",
        name="Situation and stakes",
        question="Does the answer establish a specific situation, and why it mattered?",
        anchor_1="No context. The listener cannot tell what happened, where, or when.",
        anchor_3="A situation is described, but generically — no timeframe, scale, or reason it was hard.",
        anchor_5="A specific situation with enough detail to picture it, and a clear reason the stakes were real.",
        probe_templates=(
            "Before we go on, set the scene for me. What was the situation, and why did it matter?",
            "What made that situation difficult in the first place?",
        ),
    ),
    Competency(
        key="ownership",
        name="Personal ownership",
        question=(
            "Does the answer make clear what the CANDIDATE personally did, as opposed to what "
            "their team or company did? Count uses of 'we' that hide an individual contribution."
        ),
        anchor_1="Entirely 'we'. The candidate's own contribution is invisible.",
        anchor_3="Some personal action, but blurred into the team's work.",
        anchor_5="Unambiguous about what they personally decided, did, or argued for — including where they were wrong.",
        probe_templates=(
            "You said 'we' a few times there. What did you personally do?",
            "Of everything you described, which part was your own decision?",
        ),
    ),
    Competency(
        key="action",
        name="Action specificity",
        question="Are the actions concrete and specific, or abstract and generic?",
        anchor_1="Abstract only — 'I communicated better', 'I managed expectations'.",
        anchor_3="Some concrete steps, but the how is missing.",
        anchor_5="Specific, sequenced actions a listener could repeat. Names the hard choices.",
        probe_templates=(
            "Walk me through what you actually did, step by step.",
            "What was the first thing you changed, specifically?",
        ),
    ),
    Competency(
        key="result",
        name="Measurable result",
        question="Does the answer state an outcome, ideally with a number or a clear before-and-after?",
        anchor_1="No outcome at all. The story stops before it resolves.",
        anchor_3="An outcome is claimed but not quantified — 'it went well', 'they were happy'.",
        anchor_5="A concrete result with a number, a timeframe, or an unambiguous before-and-after.",
        probe_templates=(
            "What was the measurable outcome?",
            "How did you know it had worked? What changed?",
        ),
    ),
    Competency(
        key="reflection",
        name="Reflection",
        question="Does the candidate show what they learned, or what they would do differently?",
        anchor_1="No reflection. The story is presented as an unqualified success.",
        anchor_3="A general lesson, stated but not connected to anything specific.",
        anchor_5="A specific lesson tied to a specific mistake or surprise, with what they would change.",
        probe_templates=(
            "Looking back, what would you do differently?",
            "What surprised you about how that went?",
        ),
    ),
)

BY_KEY = {c.key: c for c in RUBRIC}

# Below this, a competency is considered a gap worth probing.
PROBE_THRESHOLD = 3


def anchors_block() -> str:
    """The rubric rendered for the scoring prompt. Kept in one place so it cannot drift."""
    parts = []
    for c in RUBRIC:
        parts.append(
            f"{c.key} — {c.name}\n"
            f"  Judge: {c.question}\n"
            f"  1: {c.anchor_1}\n"
            f"  3: {c.anchor_3}\n"
            f"  5: {c.anchor_5}"
        )
    return "\n\n".join(parts)
