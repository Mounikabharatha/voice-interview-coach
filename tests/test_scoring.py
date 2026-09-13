"""Scoring validation tests.

The model is asked to quote the candidate verbatim. In a real session only 13 of 35 quotes
actually appeared in the transcript — the rest were paraphrases. These tests pin down what
happens to the other 22, because showing a candidate words they never said, attributed to
them, would be worse than showing no evidence at all.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from coach.coachlogic.scoring import parse_scores, summarise

ANSWER = ("I rewrote the billing job after I realised the retry logic was double charging "
          "about two hundred customers a week, and I shipped a fix in three days.")

def j(*items):
    import json
    return json.dumps({"scores": list(items)})

def sc(comp, level, ev, rat="because"):
    return {"competency": comp, "level": level, "evidence": ev, "rationale": rat}

checks = []
def check(name, cond):
    checks.append((name, cond))

# verbatim quote survives intact
r = parse_scores(j(sc("result", 5, "I shipped a fix in three days")), ANSWER)
check("verbatim quote verifies", len(r) == 1 and r[0].evidence_verified and r[0].level == 5)

# paraphrase: high score capped to 3, quote suppressed
r = parse_scores(j(sc("result", 5, "he fixed the billing problem quickly")), ANSWER)
check("paraphrased quote is not verified", len(r) == 1 and not r[0].evidence_verified)
check("unverified level 5 is capped to 3", r[0].level == 3)
check("unverified quote is blanked, never shown", r[0].evidence == "")

# a LOW score with no evidence is legitimate — nothing to quote
r = parse_scores(j(sc("reflection", 1, "")), ANSWER)
check("low score with empty evidence is kept", len(r) == 1 and r[0].level == 1)

# out-of-range and unknown competencies are dropped
r = parse_scores(j(sc("result", 7, "x"), sc("nonsense", 3, "y"), sc("action", 0, "z")), ANSWER)
check("level 7 rejected", not any(s.level == 7 for s in r))
check("unknown competency rejected", not any(s.competency == "nonsense" for s in r))
check("level 0 rejected", len(r) == 0)

# duplicates: first wins
r = parse_scores(j(sc("action", 2, ""), sc("action", 5, "")), ANSWER)
check("duplicate competency deduped", len(r) == 1 and r[0].level == 2)

# prose-wrapped JSON still parses — models ignore "return ONLY JSON"
r = parse_scores("Here you go:\n```json\n" + j(sc("action", 3, "")) + "\n```", ANSWER)
check("JSON wrapped in prose still parses", len(r) == 1)

# truncated JSON yields nothing rather than garbage
r = parse_scores('{"scores": [{"competency": "action", "level": 3, "evi', ANSWER)
check("truncated JSON returns empty, not garbage", r == [])

# garbage in, empty out — scoring must never raise
check("garbage returns empty", parse_scores("not json at all", ANSWER) == [])
check("empty string returns empty", parse_scores("", ANSWER) == [])

# aggregation
s = summarise([parse_scores(j(sc("result", 4, "I shipped a fix in three days")), ANSWER),
               parse_scores(j(sc("result", 2, "")), ANSWER)])
check("mean aggregates across turns", s["result"]["mean"] == 3.0 and s["result"]["n"] == 2)
check("weakest is tracked", s["result"]["weakest"] == 2)

bad = 0
for name, ok in checks:
    bad += not ok
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
print(f"\n{len(checks) - bad}/{len(checks)} passed")
raise SystemExit(1 if bad else 0)
