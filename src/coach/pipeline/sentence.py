"""Incremental sentencizer: turn a stream of text deltas into speakable chunks.

This sits directly on the critical path. The language model streams a word at a time; speech
synthesis wants a whole clause. Waiting for the complete reply before speaking adds the entire
generation time to the pause the user hears, so the job here is to release the *smallest*
chunk that still sounds natural, as early as possible.

Four rules, in priority order:

1. **Terminal punctuation** ends a sentence — except where it does not. `Dr.`, `e.g.` and
   `3.5` all contain a period that does not end anything, and splitting on them produces
   audible stutter.
2. **The opening clause is special.** For the first chunk only, a comma past ~40 characters is
   good enough. Getting *something* speaking sooner is worth more than the slightly better
   prosody of a full sentence, because this is the gap the user actually experiences.
3. **A hard character cap** forces a split. Without it, a model that produces a long run-on
   clause stalls audio indefinitely.
4. **`flush()` is mandatory** at end of stream. Models routinely stop without final
   punctuation, and the tail would otherwise never be spoken.

Batch sentence splitters are not usable here: they assume the complete text is available,
which is the one thing that is not true.
"""
from __future__ import annotations

import re

TERMINALS = ".!?"

# Tokens whose trailing period does not end a sentence.
ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sr jr st
    e.g i.e etc vs approx est
    inc ltd co corp dept univ
    jan feb mar apr jun jul aug sep sept oct nov dec
    mon tue wed thu fri sat sun
    a.m p.m u.s u.k ph.d b.sc m.sc
    fig no vol pp
    """.split()
)

OPENING_CLAUSE_MIN_CHARS = 40
MAX_CHUNK_CHARS = 180

_WORD_BEFORE_PERIOD = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


def _is_abbreviation(text_up_to_period: str) -> bool:
    m = _WORD_BEFORE_PERIOD.search(text_up_to_period)
    if not m:
        return False
    return m.group(1).lower().rstrip(".") in ABBREVIATIONS


def _is_decimal(buf: str, i: int) -> bool:
    """True when buf[i] is a period between two digits, as in 3.5 or 99.9."""
    return 0 < i < len(buf) - 1 and buf[i - 1].isdigit() and buf[i + 1].isdigit()


class Sentencizer:
    """Push deltas in, get complete speakable chunks out.

    One instance per turn. Reusing one across turns is a real bug, not a style preference:
    the opening-clause rule is gated on `emitted == 0`, so a reused instance applies it once
    per session and every later turn silently loses the early-audio win.
    """

    def __init__(
        self,
        *,
        opening_clause_min_chars: int = OPENING_CLAUSE_MIN_CHARS,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
    ) -> None:
        self._buf = ""
        self.emitted = 0
        self._opening_min = opening_clause_min_chars
        self._max_chars = max_chunk_chars

    def push(self, delta: str) -> list[str]:
        """Add a delta, return any chunks that are now complete."""
        self._buf += delta
        out: list[str] = []
        while (chunk := self._take()) is not None:
            out.append(chunk)
            self.emitted += 1
        return out

    def flush(self) -> list[str]:
        """Release whatever is left. Call this at end of stream, always."""
        tail = self._buf.strip()
        self._buf = ""
        if tail:
            self.emitted += 1
            return [tail]
        return []

    def _take(self) -> str | None:
        buf = self._buf
        if not buf.strip():
            return None

        for i, ch in enumerate(buf):
            if ch not in TERMINALS:
                continue
            if ch == "." and (_is_decimal(buf, i) or _is_abbreviation(buf[: i + 1])):
                continue
            # Consume any run of terminals and closing quotes, e.g. `?"` or `!!`
            end = i + 1
            while end < len(buf) and buf[end] in TERMINALS + "\"')]":
                end += 1
            # A period at the very end of the buffer waits: the next delta may reveal it was
            # the `3.` of `3.5`, or the `e.` of `e.g.`.
            if end >= len(buf):
                break
            # A real sentence boundary is followed by whitespace. This is what stops the split
            # happening *inside* a multi-part abbreviation — at the first period of `e.g.` the
            # next character is `g`, so the abbreviation check above never gets a chance to see
            # the whole token. By the second period it does.
            if not buf[end].isspace():
                continue
            return self._emit(end)

        # Rule 2: the opening clause may break at a comma.
        if self.emitted == 0 and len(buf) >= self._opening_min:
            idx = buf.find(",", self._opening_min)
            if idx != -1:
                return self._emit(idx + 1)

        # Rule 3: hard cap. Prefer the last space so a word is not cut in half.
        if len(buf) >= self._max_chars:
            cut = buf.rfind(" ", 0, self._max_chars)
            return self._emit(cut if cut > 0 else self._max_chars)

        return None

    def _emit(self, end: int) -> str:
        chunk, self._buf = self._buf[:end], self._buf[end:]
        return chunk.strip()
