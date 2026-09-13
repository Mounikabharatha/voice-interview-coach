import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
from coach.pipeline.sentence import Sentencizer

def run(deltas, **kw):
    s = Sentencizer(**kw); out=[]
    for d in deltas: out += s.push(d)
    out += s.flush(); return out

cases = [
    ("splits on sentence end",
     ["That's a good start. ","What was the outcome?"],
     ["That's a good start.", "What was the outcome?"]),
    ("does not split a decimal",
     ["You paused for 3.5 seconds there. ","Try again."],
     ["You paused for 3.5 seconds there.", "Try again."]),
    ("does not split an abbreviation",
     ["Talk about e.g. a conflict you handled. ","Go on."],
     ["Talk about e.g. a conflict you handled.", "Go on."]),
    ("opening clause breaks early at a comma",
     ["That is a reasonably solid answer overall, but I want more detail on your role."],
     ["That is a reasonably solid answer overall,", "but I want more detail on your role."]),
    ("later clauses do NOT break at commas",
     ["First sentence here. ","Second one is quite long indeed, and keeps going without a stop"],
     ["First sentence here.", "Second one is quite long indeed, and keeps going without a stop"]),
    ("tail with no punctuation still flushes",
     ["No terminal punctuation at all"],
     ["No terminal punctuation at all"]),
    ("handles question mark plus quote",
     ['She asked "what happened?" ', "Then left."],
     ['She asked "what happened?"', "Then left."]),
]
bad=0
for name, deltas, want in cases:
    got = run(deltas)
    ok = got == want
    bad += not ok
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"          want {want}")
        print(f"          got  {got}")

# hard cap
long_run = ["word " * 60]
got = run(long_run)
ok = all(len(c) <= 180 for c in got) and len(got) > 1
bad += not ok
print(f"  [{'ok ' if ok else 'FAIL'}] hard cap splits a run-on ({len(got)} chunks, max {max(len(c) for c in got)} chars)")
print(f"\n{len(cases)+1-bad}/{len(cases)+1} passed")
