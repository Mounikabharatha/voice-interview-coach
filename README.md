# Voice Interview Coach

A real-time voice agent that runs mock behavioural interviews. You speak your answer out loud;
it listens while you talk, decides when you have actually finished, asks a follow-up when your
answer is thin, and gives you written feedback against a scoring rubric at the end.

You can interrupt it mid-sentence.

> **Demo:** _(recording to be added — see [DEMO.md](DEMO.md))_

Runs on free API tiers. No GPU, no model downloads, no native audio libraries. Clone it, add
two free keys, and it runs the same on Windows, Linux and macOS.

---

## Quickstart

```bash
git clone https://github.com/Mounikabharatha/voice-interview-coach
cd voice-interview-coach
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                   # also installs the package itself
cp .env.example .env                              # then paste your two keys into it
python -m coach
```

Open <http://localhost:8000>, press Start, allow the microphone. Headphones recommended.

Two free keys, neither needing a card: **[Groq](https://console.groq.com/keys)** for the
language model, **[NVIDIA NIM](https://build.nvidia.com)** for speech.

Check them before anything else — this reports what actually answered, rather than what the
docs claim:

```bash
python scripts/check_providers.py
```

---

## Why this is harder than it looks

Speech recognition, language models and speech synthesis are all solved problems with free
tiers. Wiring them together takes an afternoon. The result is unusable, and it takes about
three seconds per reply.

The actual problems are **latency** and **turn-taking**, and neither is fixed by choosing a
faster model.

**Latency is fixed by overlapping the stages.** Transcription runs while you are still
speaking. The moment the model produces one complete clause — not the whole reply — that clause
is already being synthesized and played. Each stage overlaps the next.

**Turn-taking is fixed by not trusting the obvious signal**, which is the subject of the
section below and was the hardest part of this project.

---

## Measured

Every figure came from a live call, reproducible with the scripts in `scripts/`. Nothing here
is from a datasheet.

| Stage | Measured |
|---|---|
| Language model, first **speakable** token | **363 ms** |
| Language model, first token of any kind | 88 ms — but it is chain-of-thought, so it cannot be spoken |
| Speech synthesis, first audio byte (warm) | 200–800 ms |
| Speech synthesis, first audio byte (cold) | 1,141–2,384 ms |
| Speech recognition, first partial | 875 ms |
| Speech recognition, finalisation after speech stops | 286–569 ms |
| Barge-in, speech to cancellation | **306 ms** |
| End of speech to first reply audio | ~1.9–2.8 s |

Three of these changed the design.

**Reasoning has to be switched off explicitly.** `qwen/qwen3.6-27b` returns its first token in
88 ms — and that token is the model thinking out loud, which cannot be spoken. With
`reasoning_effort="none"` the first *speakable* token takes 363 ms. The honest measure for a
voice agent is the second number. A `reasoning_chars` counter asserts this stays at zero in
production; it does.

**Prewarming the speech connection is worth about a second.** The TLS and HTTP/2 handshake is
paid once, at startup, so it never lands on the first thing the coach says to a user.

**Speech synthesis latency is not stable, and that is itself the finding.** An early run
measured a warm first byte at 198 ms, four calls in a row. Hours later the same code on the
same connection measured 563–810 ms. A/B against raw synchronous calls showed the async wrapper
costs almost nothing and the variance is service-side, so the README now states a range instead
of the flattering number. Two suspected causes were chased and found not to exist: an
idle-connection timeout did not reproduce across 15 s and 25 s gaps, and gRPC keepalive changed
nothing, so it was **not** added. A fix for a problem that is not there is worse than no fix.

---

## The hard part: knowing when someone has finished talking

The speech recogniser emits a final transcript at every sentence end. Treating those as
end-of-turn is the obvious design, and it cuts people off mid-answer — which is exactly what
interview candidates trigger, because they pause to recall a number or decide how much credit
to claim.

So finals are treated as **fragments**, and a turn ends only after measured silence.

The second idea was to read the wording: nobody ends an answer on "and", so a trailing
conjunction should buy extra patience. It does — when the word survives. But the recogniser
segments on its own endpointing **and punctuates what it emits**, so a candidate who says

> "…to a new payments provider, **and**" *[pauses to think]*

is transcribed as `"…to a new payments provider."` — trailing conjunction deleted, period
added. The evidence of being mid-sentence is destroyed upstream of any logic that could use it.

What survives is **length**. A behavioural answer is a story — situation, action, result — and
nobody tells one in twelve words, so a short grammatically complete sentence is far more likely
to be an opening clause than a whole answer.

| Pending text | Waits |
|---|---|
| Full-length, ends `.` `!` `?` | 900 ms |
| Long, no terminal punctuation | 1,500 ms |
| Ends on "and" / "the" / "um", **or under 20 words** | 2,500 ms |

**The cost is real.** Short replies like "Yes, that's right" wait the full 2.5 s. For an
interview coach that is the right way round — being interrupted mid-answer is a much worse
failure than a slow reply — but it does make quick back-and-forth feel sluggish.

**The known better answer is semantic turn detection**: a small model judging from the *audio*
whether a turn sounds finished, rather than inferring it from a transcript that has already
thrown the evidence away. That is the next real improvement, and it would replace the
word-count rule entirely.

---

## Scoring, and why every quote is checked

Answers are scored against five competencies — situation, ownership, action specificity,
result, reflection — each anchored at levels 1, 3 and 5, so a score is a judgement against a
written standard rather than a vibe.

**Nothing scores the person.** There is no hireability, confidence or communication-style
score. Those track accent, gender and background far more than competence, and they are where
automated interview scoring has historically gone wrong. Every competency is a property of the
*answer*: does it name a result, does it say what the candidate personally did.

**Every quote is verified against the transcript.** The scorer is instructed to quote the
candidate verbatim. In a real session, **13 of 35 quotes actually appeared in what was said** —
the rest were paraphrases. So quotes are checked by normalised token match, unverified ones are
never displayed, and a score of 4 or 5 resting on an unverified quote is capped to 3. Showing
someone words they never said, attributed to them, is worse than showing no evidence.

Two paths, deliberately: deciding whether to probe happens while the candidate sits in silence,
so it uses a cheap local heuristic and adds no latency. Full rubric scoring runs in the
background and lands in the report, where nobody is waiting.

---

## How it is put together

```
browser ──mic──▶ AudioWorklet ──16 kHz PCM16──▶ WebSocket ──▶ FastAPI
  ▲   resample 48k→16k                                          │
  │                                          ┌───────────────────┼──────────────────┐
  │                                          ▼                   ▼                  ▼
  │                                   Endpointer          NVIDIA ASR            Session
  │                                  (silence + text)     (streaming)        (rubric, probes)
  │                                          │                   │                  │
  │                                          └────── turn end ───┴──────────────────┘
  │                                                         │
  │                                                         ▼
  │                                           Groq LLM ──▶ Sentencizer ──▶ NVIDIA TTS
  └────────────────◀── WebSocket ◀── PCM16 ◀───────────────────────────────────┘
```

**All audio hardware lives in the browser.** The server never opens an audio device, so there
is no PortAudio or native dependency to break on someone else's operating system. That single
decision is what makes the cross-platform promise true rather than aspirational.

| Layer | Choice | Why |
|---|---|---|
| Speech to text | NVIDIA Nemotron ASR (Riva gRPC) | True streaming partials. Groq's Whisper is batch-only, which cannot support barge-in |
| Language model | Groq `qwen/qwen3.6-27b` | The only free Groq model whose reasoning can be switched off |
| Speech synthesis | NVIDIA Magpie (Riva gRPC) | Streams audio in chunks and cancels mid-sentence |

Each layer sits behind a small `Protocol` in `base.py`, so a provider can be swapped without
touching the pipeline.

```
src/coach/
  stt/ llm/ tts/        one provider interface + one implementation each
  pipeline/             endpoint.py  turn detection and barge-in
                        sentence.py  incremental sentence splitting
                        turn.py      one conversational turn
  coachlogic/           rubric.py  questions.py  scoring.py  session.py
  web/                  app.py + static/
```

---

## Tests

```bash
for t in tests/test_*.py; do python "$t"; done
```

46 unit tests covering turn detection, the incremental sentencizer, and scoring validation
(capping, quote verification, malformed and truncated JSON).

Three integration scripts drive the real path unattended, with no microphone — they synthesize
a candidate's speech and stream it into the WebSocket as if it came from a mic:

```bash
python scripts/simulate_turn.py      # one turn, including a mid-answer thinking pause
python scripts/test_bargein.py       # talk over the coach, assert it stops
python scripts/simulate_session.py   # a full interview, ending in the report
```

`simulate_turn.py` is a regression test for a real bug: it reproduces the 1.6-second thinking
pause that used to split one answer into two turns.

---

## Limitations

- **Turn detection is heuristic**, not learned. See the section above for what it costs and
  what would replace it.
- **Echo.** Browser echo cancellation does most of the work, but without headphones the coach's
  own voice can occasionally trigger barge-in.
- **Free-tier limits.** Groq allows 1,000 requests/day. A session uses roughly 10–20.
- **English only.** The recogniser and rubric assume English.
- **Interrupted replies** are recorded as spoken up to the sentence boundary, not to the exact
  audio frame the user heard. Precise truncation needs playback position reported by the
  client.

## Design notes

The architecture research behind these choices, including the provider comparison and the
options rejected, is in [`docs/research.md`](docs/research.md), with its own audit in
[`docs/research-open-questions.md`](docs/research-open-questions.md). Sections of that document
predate the decision to go cloud-only and are marked accordingly.

## Licence

MIT
