import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from coach.pipeline.endpoint import looks_complete, required_silence_ms, frame_rms

cases = [
    # the step-8 bug: the first half of a real answer must NOT read as finished
    ("I led the migration of our billing system to a new payments provider, and", "incomplete"),
    ("It took about four months, and I coordinated three teams.", "complete"),
    ("I was responsible for the", "incomplete"),
    ("We shipped it in March.", "complete"),
    ("um", "incomplete"),
    ("So the outcome was a twelve percent reduction in churn", "unclear"),
    ("What I did was", "incomplete"),
    ("That's it!", "complete"),
    ("I think it went well, but", "incomplete"),
]
bad = 0
print("completeness classification:")
for text, want in cases:
    got = looks_complete(text)
    ok = got == want
    bad += not ok
    print(f"  [{'ok ' if ok else 'FAIL'}] {want:10s} <- {text[:52]!r}" + ("" if ok else f"  got {got}"))

print("\nsilence budget:")
for text in ["We shipped it in March.", "I was responsible for the", "The outcome was good"]:
    print(f"  {required_silence_ms(text):5d} ms  {looks_complete(text):10s} {text[:40]!r}")

print("\nRMS:")
silence = b"\x00\x00" * 1600
loud = (b"\x00\x40" * 1600)
r_s, r_l = frame_rms(silence), frame_rms(loud)
ok = r_s == 0.0 and r_l > 1000
bad += not ok
print(f"  [{'ok ' if ok else 'FAIL'}] silence={r_s:.0f}  loud={r_l:.0f}")
print(f"  [{'ok ' if frame_rms(b'') == 0.0 else 'FAIL'}] empty frame handled")
print(f"\n{len(cases)+2-bad}/{len(cases)+2} passed")
