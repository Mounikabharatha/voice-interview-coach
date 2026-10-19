"""quick_gaps tests — the heuristic that decides whether to ask a follow-up.

It is matched against SPEECH RECOGNISER OUTPUT, not written English. That distinction is the
source of the one bug here worth a regression test: a candidate said "I learnt to test earlier",
the recogniser wrote the British spelling, and the American-only word list concluded they had
not reflected at all and probed them for it.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from coach.coachlogic.session import quick_gaps

# Close to a real transcript: 60+ words, which is what a genuine interview answer runs to.
# Keep it above 45 words or the length rule fires and the test is measuring the wrong thing.
STRONG = ("I built a real time voice interview coach. You speak an answer out loud and it "
          "listens while you talk, replies out loud, and scores you at the end. I chose the "
          "providers, wrote the pipeline, and measured every stage. I cut the reply time from "
          "about 3 s to one and a half seconds, and I {reflect} to test with a real microphone "
          "much earlier.")

checks = []
def check(name, cond): checks.append((name, cond))

# the exact transcript from a real session, which used to trigger a spurious probe
check("British 'learnt' counts as reflection",
      "reflection" not in quick_gaps(STRONG.format(reflect="learnt")))
check("American 'learned' counts as reflection",
      "reflection" not in quick_gaps(STRONG.format(reflect="learned")))
for phrase in ("realised", "realized", "in hindsight I needed", "looking back I needed",
               "my takeaway was", "next time I will try"):
    check(f"{phrase!r} counts as reflection",
          "reflection" not in quick_gaps(STRONG.format(reflect=phrase)))

# genuinely missing reflection must still be caught
check("no reflection is still detected",
      "reflection" in quick_gaps(STRONG.format(reflect="went on")))

# the other four heuristics still behave
VAGUE = "We worked on it as a team and it went pretty well in the end."
g = quick_gaps(VAGUE)
check("vague answer asks for the result first", g and g[0] == "result")
check("vague answer flags ownership ('we' over 'I')", "ownership" in g)
check("vague answer flags it is too short", "situation" in g)
check("a full strong answer needs no follow-up at all",
      quick_gaps(STRONG.format(reflect="learnt")) == [])

bad = 0
for name, ok in checks:
    bad += not ok
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
print(f"\n{len(checks) - bad}/{len(checks)} passed")
raise SystemExit(1 if bad else 0)
