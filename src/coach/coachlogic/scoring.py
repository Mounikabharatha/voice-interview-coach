"""Score one answer against the rubric.

Runs *off* the conversational hot path. The candidate never waits for a score: the coach's
spoken reply is generated separately and immediately, while scoring happens in the background
and lands in the end-of-session report. Putting the scorer in the turn loop would add a full
round trip to every reply for information nobody hears.

**JSON mode, not strict schema.** Groq is OpenAI-compatible, and OpenAI-compatible strict
`json_schema` rejects several common keywords (`minimum`, `maximum`, `maxLength`) depending on
the backend. Rather than depend on that, this asks for `json_object` and validates in Python.
The validation has to exist either way — a model can return well-formed JSON that is still
nonsense, like a level of 7 or evidence quoting words the candidate never said.
"""
from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import asdict, dataclass

from .rubric import BY_KEY, RUBRIC, anchors_block

log = logging.getLogger(__name__)

SCORING_PROMPT = f"""You are scoring one answer from a mock behavioural interview.

Score the answer against each of these five competencies on a 1 to 5 scale, using the anchors.

{anchors_block()}

Rules:
- Score only what is in the answer. Never infer, never give credit for what the candidate
  probably meant.
- For each competency, quote the exact words from the answer that justify the score. If there
  is nothing to quote, the evidence must be an empty string and the score must be 1 or 2.
- Judge the ANSWER, never the person. Do not comment on tone, confidence, accent, personality
  or communication style. Do not estimate whether they should be hired.
- A short answer is not automatically bad, but an answer with no result genuinely cannot score
  above 2 on `result`.

Return ONLY a JSON object, no prose, in exactly this shape:

{{"scores": [
  {{"competency": "situation", "level": 3, "evidence": "exact quote from the answer",
    "rationale": "one sentence, max 20 words"}},
  ... one entry for each of the five competencies ...
]}}"""

VALID_KEYS = {c.key for c in RUBRIC}


@dataclass
class Score:
    competency: str
    level: int
    evidence: str
    rationale: str
    evidence_verified: bool   # did the quote actually appear in the answer?


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def _evidence_appears(evidence: str, answer: str) -> bool:
    """Check the quote is really in the answer, allowing for punctuation and casing drift.

    Models paraphrase when asked to quote. An unverified quote is not a reason to throw the
    score away, but it IS a reason not to show it to the candidate as something they said.
    """
    ev = _normalise(evidence).split()
    if len(ev) < 3:
        return False
    hay = _normalise(answer)
    return " ".join(ev) in hay


def parse_scores(raw: str, answer: str) -> list[Score]:
    """Parse and validate the model's JSON. Never raises — returns what survived."""
    with contextlib.suppress(Exception):
        # Models sometimes wrap JSON in prose or a code fence despite instructions.
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0) if match else raw)
        out: list[Score] = []
        seen: set[str] = set()
        for item in data.get("scores", []):
            key = str(item.get("competency", "")).strip().lower()
            if key not in VALID_KEYS or key in seen:
                continue
            seen.add(key)
            try:
                level = int(item.get("level", 0))
            except (TypeError, ValueError):
                continue
            if not 1 <= level <= 5:
                continue
            evidence = str(item.get("evidence") or "").strip()
            verified = _evidence_appears(evidence, answer)
            # A high score with no verifiable evidence is exactly the failure mode this
            # guards against: a model being agreeable. Cap it rather than drop it.
            if level >= 4 and not verified:
                log.info("capping %s from %d to 3 — evidence not found in answer", key, level)
                level = 3
            out.append(Score(key, level, evidence if verified else "",
                             str(item.get("rationale") or "").strip()[:160], verified))
        return out
    return []


async def score_answer(llm, question: str, answer: str) -> list[Score]:
    """Score one answer. Returns [] on any failure — scoring must never break a session."""
    messages = [
        {"role": "system", "content": SCORING_PROMPT},
        {"role": "user", "content": f"Question asked:\n{question}\n\nCandidate's answer:\n{answer}"},
    ]
    try:
        stream = await llm.stream_chat(messages)
        try:
            raw = "".join([chunk async for chunk in stream])
        finally:
            await stream.aclose()
    except Exception as exc:
        log.warning("scoring call failed: %s", exc)
        return []
    scores = parse_scores(raw, answer)
    if not scores:
        # The classic cause is a max_tokens budget too small for five competencies, which
        # truncates the JSON mid-object. Say so rather than making someone re-derive it.
        truncated = raw.count("{") > raw.count("}")
        log.warning(
            "scoring returned nothing usable%s: %r",
            " (JSON looks TRUNCATED — raise max_tokens on the scoring client)" if truncated else "",
            raw[-160:] or raw[:160],
        )
    return scores


def summarise(all_scores: list[list[Score]]) -> dict:
    """Aggregate per-competency across a session, for the report."""
    by_key: dict[str, list[int]] = {k: [] for k in VALID_KEYS}
    for turn in all_scores:
        for s in turn:
            by_key[s.competency].append(s.level)
    summary = {}
    for key, levels in by_key.items():
        if not levels:
            continue
        summary[key] = {
            "name": BY_KEY[key].name,
            "mean": round(sum(levels) / len(levels), 1),
            "n": len(levels),
            "weakest": min(levels),
        }
    return summary


def score_to_dict(s: Score) -> dict:
    return asdict(s)
