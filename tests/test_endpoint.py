"""Endpointer tests.

An important distinction runs through these: `looks_complete` does not ask "is this a
grammatical English sentence?" It asks "has this candidate finished their interview answer?"
Those differ, and the difference is the point. "We shipped it in March." is a complete
sentence and an obviously unfinished answer.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from coach.pipeline.endpoint import (
    MIN_ANSWER_WORDS,
    BargeInDetector,
    frame_rms,
    looks_complete,
    required_silence_ms,
)

LONG_COMPLETE = (
    "I led the migration of our billing system to a new payments provider, and it took about "
    "four months with three teams involved, and we cut failed payments by twelve percent in "
    "the first quarter."
)

CASES = [
    # --- trailing conjunction beats punctuation -------------------------------------------
    # The recogniser runs with automatic punctuation and appends a period even mid-sentence,
    # so the word has to outrank the period. This is what broke step 9 in practice.
    ("I led the migration to a new payments provider, and.", "incomplete"),
    ("It took about four months with the.", "incomplete"),
    ("I was responsible for.", "incomplete"),
    ("We spoke about.", "incomplete"),
    ("I think it went well, but", "incomplete"),
    (LONG_COMPLETE[:-1] + ", and", "incomplete"),

    # --- short but grammatical: unfinished ANSWERS, whatever the grammar says --------------
    ("We shipped it in March.", "incomplete"),
    ("That's it!", "incomplete"),
    ("I did it.", "incomplete"),
    ("That is what I would do.", "incomplete"),
    ("So the outcome was a twelve percent reduction in churn", "incomplete"),
    ("um", "incomplete"),

    # --- a full-length answer that lands ---------------------------------------------------
    (LONG_COMPLETE, "complete"),
    # long, no terminal punctuation -> genuinely ambiguous
    (LONG_COMPLETE.rstrip("."), "unclear"),
    # long, ends on a weak continuation with no punctuation -> still mid-thought
    (LONG_COMPLETE.rstrip(".") + " which is what I", "incomplete"),
]


def main() -> int:
    failures = 0

    print("completeness — a judgement about ANSWERS, not grammar:")
    for text, want in CASES:
        got = looks_complete(text)
        ok = got == want
        failures += not ok
        label = f"{text[:46]}..." if len(text) > 46 else text
        print(f"  [{'ok  ' if ok else 'FAIL'}] {want:10s} <- {label!r}" + ("" if ok else f"  got {got}"))

    print(f"\nsilence budget (MIN_ANSWER_WORDS = {MIN_ANSWER_WORDS}):")
    for text in (LONG_COMPLETE, "We shipped it in March.", LONG_COMPLETE.rstrip(".")):
        label = f"{text[:40]}..." if len(text) > 40 else text
        print(f"  {required_silence_ms(text):5d} ms  {looks_complete(text):10s} {label!r}")

    print("\nRMS:")
    checks = [
        ("silence is zero", frame_rms(b"\x00\x00" * 1600) == 0.0),
        ("loud frame reads high", frame_rms(b"\x00\x40" * 1600) > 1000),
        ("empty frame handled", frame_rms(b"") == 0.0),
        ("odd-length frame handled", frame_rms(b"\x00\x40\x00") > 0),
    ]
    for name, ok in checks:
        failures += not ok
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")

    print("\nbarge-in detector:")
    import struct
    loud = struct.pack("<1600h", *([8000] * 1600))    # someone talking
    quiet = b"\x00\x00" * 1600                        # digital silence
    # Real microphones never produce digital silence. This is room tone at a level that a
    # real mic with automatic gain control routinely reports — and it is ABOVE the fixed
    # threshold the first version used, which is why barge-in fired on an empty room and
    # cancelled every reply. This case is the regression test for that bug.
    room_tone = struct.pack("<1600h", *([700] * 1600))

    def run(frames, floor=120.0):
        d = BargeInDetector()
        d.arm(floor)
        return [d.feed(f, 100) for f in frames]

    # grace_ms=400 at 100 ms a frame means the first four frames can never fire
    grace_frames = 4
    bchecks = [
        ("digital silence never triggers", not any(run([quiet] * 20))),
        ("ROOM TONE never triggers (the bug)", not any(run([room_tone] * 20, floor=700))),
        ("room tone IS above the old fixed threshold",
         700 > 120 * 4),
        ("a single loud frame does not trigger",
         not any(run([quiet] * grace_frames + [loud]))),
        ("sustained speech does trigger",
         any(run([quiet] * grace_frames + [loud] * 5))),
        ("nothing fires during the grace period",
         not any(run([loud] * grace_frames))),
        ("fires once per arm, then rearms",
         sum(run([quiet] * grace_frames + [loud] * 5)) == 1),
    ]
    for name, ok in bchecks:
        failures += not ok
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")

    total = len(CASES) + len(checks) + len(bchecks)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
