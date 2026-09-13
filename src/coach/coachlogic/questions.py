"""The question bank.

Each question is tagged with the competencies it most naturally exercises, so a session can
cover different ground rather than asking three variations of the same thing.

Questions are phrased to be *spoken*, not read: short, one clause where possible, no
parentheticals. A question that reads well on a page often sounds convoluted out loud.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    targets: tuple[str, ...]   # rubric competency keys this question exercises


BANK: tuple[Question, ...] = (
    Question("stakeholder", "Tell me about a time you handled a difficult stakeholder.",
             ("situation", "ownership", "result")),
    Question("conflict", "Describe a disagreement you had with a colleague. How did it end?",
             ("ownership", "reflection")),
    Question("failure", "Tell me about a time something you owned went badly wrong.",
             ("ownership", "reflection", "result")),
    Question("deadline", "Describe a time you had to ship something under a deadline you did not believe in.",
             ("situation", "action", "result")),
    Question("influence", "Tell me about a time you changed someone's mind without any authority over them.",
             ("action", "ownership")),
    Question("ambiguity", "Describe a project where the goal was unclear when you started.",
             ("situation", "action")),
    Question("priorities", "Tell me about a time you had to drop something that mattered to someone.",
             ("ownership", "result", "reflection")),
    Question("feedback", "Describe a piece of critical feedback you received and what you did about it.",
             ("reflection", "action")),
    Question("scale", "Tell me about the largest or most complex thing you have worked on.",
             ("situation", "action", "result")),
    Question("initiative", "Describe something you started that nobody asked you to do.",
             ("ownership", "result")),
    Question("teaching", "Tell me about a time you helped someone else get better at something.",
             ("action", "reflection")),
    Question("tradeoff", "Describe a technical or product trade-off you made that you would defend today.",
             ("action", "reflection", "ownership")),
    Question("pressure", "Tell me about a time you were the only person who thought something was a bad idea.",
             ("ownership", "action", "reflection")),
    Question("recover", "Describe a relationship at work you had to repair.",
             ("situation", "ownership", "reflection")),
)

BY_ID = {q.id: q for q in BANK}


def pick_session(count: int = 3, *, seed: int | None = None) -> list[Question]:
    """Choose questions that between them exercise as much of the rubric as possible.

    Greedy by marginal coverage: each pick is the question adding the most competencies not
    yet covered, so a three-question session usually touches all five. Ties are broken
    randomly so consecutive sessions are not identical.
    """
    rng = random.Random(seed)
    pool = list(BANK)
    rng.shuffle(pool)
    chosen: list[Question] = []
    covered: set[str] = set()
    while pool and len(chosen) < count:
        pool.sort(key=lambda q: len(set(q.targets) - covered), reverse=True)
        pick = pool.pop(0)
        chosen.append(pick)
        covered |= set(pick.targets)
    return chosen
