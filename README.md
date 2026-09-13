# Voice Interview Coach

A real-time voice agent that runs mock behavioural interviews. You speak your answer out loud;
it listens while you talk, scores the answer against a competency rubric, asks a follow-up, and
speaks back.

**Status: in development.** This README describes what is being built and why. Measured numbers
and a demo will replace this section as they exist — nothing here is claimed as working until it
is.

## The problem this is actually solving

The hard part of a voice agent is not speech recognition, language modelling, or speech
synthesis. Free, high-quality services exist for all three. The hard part is **latency and
turn-taking**.

Humans leave roughly 200 ms between conversational turns. Past about 800 ms a pause stops feeling
like conversation and starts feeling like a slow computer. A naive implementation — record,
upload, transcribe, generate, synthesise, play — lands around 3 seconds and feels broken.

Closing that gap is not a matter of picking faster models. It comes from **overlapping the
stages**: transcribing while the user is still speaking, and speaking the first sentence of a
reply while the rest is still being generated.

Interview coaching makes the turn-taking problem harder in a useful way. Candidates pause
mid-answer to think. A system that treats one second of silence as "they're finished" will cut
people off constantly — so deciding when a turn has actually ended is a real problem here, not a
detail.

## Design constraints

- **$0.** Free tiers only, no credit card.
- **Clone-and-run on Windows, Linux and macOS.** No native audio libraries, no build step, no
  model downloads. All audio capture and playback happens in the browser, so the Python server
  never touches an audio device.
- Two API keys, both free, both in a `.env` file.

## Planned architecture

```
browser mic ──▶ WebSocket ──▶ FastAPI
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                  speech       language     speech
                  to text       model      synthesis
                    │            │            │
                    └────────────┼────────────┘
                                 ▼
browser speaker ◀── WebSocket ◀──┘
```

| Layer | Service |
|---|---|
| Speech to text | NVIDIA NIM (streaming) |
| Language model | Groq |
| Speech synthesis | NVIDIA NIM |

Every layer sits behind a small interface, so a provider can be swapped without touching the
pipeline.

## Measured so far

Every figure below came from a live call on 2026-09-13, not from a datasheet. Scripts to
reproduce them are in `scripts/`.

| Stage | Measured | Notes |
|---|---|---|
| LLM, first *speakable* token | **363 ms** | `qwen/qwen3.6-27b` with `reasoning_effort="none"` |
| LLM, first token of any kind | 88 ms | but it is chain-of-thought — not speakable, so not the number that counts |
| Speech synthesis, first audio byte (warm) | **200–800 ms** | NVIDIA Magpie over Riva gRPC — see the note on variance below |
| Speech synthesis, first audio byte (cold) | 1,141–2,384 ms | why the connection is prewarmed at startup |
| Speech recognition, first partial | 875 ms | partials stream while you speak |
| Speech recognition, finalisation after speech ends | **286–569 ms** | varies run to run |

Two of these changed the design:

**Reasoning has to be switched off explicitly.** Left on, the model's first streamed tokens are
its own chain-of-thought — fast to arrive and impossible to speak. The honest measure for a voice
agent is time to first *speakable* token, and switching reasoning off moves it from unusable to
363 ms.

**Prewarming the speech connection is worth roughly a second.** The TLS and HTTP/2 handshake is
paid once; without prewarming it lands on the first thing the coach ever says.

**Speech synthesis latency is not stable, and that is the finding.** An early run measured a
warm first audio byte at 198 ms, repeatably, four calls in a row. Hours later the same code on
the same connection measured 563–810 ms, and a cold call reached 2,384 ms. Re-measuring
carefully — raw synchronous calls against the async wrapper, back to back — showed the wrapper
costs almost nothing and the variance is service-side.

Two things were chased and found not to exist: a suspected idle-connection timeout did not
reproduce across 15 s and 25 s gaps, and gRPC keepalive made no measurable difference, so it
was not added. A fix for a problem that is not there is worse than no fix.

The practical consequence is that a single flattering measurement should not be designed
around. Fixed lines the coach says often — the opening question, the move-on lines — are worth
pre-rendering precisely because synthesis latency cannot be relied on.

Not yet measured: end-to-end latency through the browser at the device clock.

## Turn-taking, and where it is still weak

Deciding when a candidate has *finished* is the hardest problem here, and it is not solved —
it is traded off, deliberately, and the trade is worth understanding.

The obvious approach fails. The speech recogniser emits a final transcript at every sentence
end, so treating those as end-of-turn cuts people off mid-answer — which is exactly what
candidates trigger, because they pause to recall a number or decide how much credit to claim.
So finals are treated as *fragments*, and a turn ends only after measured silence.

The second approach fails too, less obviously. The plan was to read the wording: nobody ends
an answer on "and", so a trailing conjunction should buy extra patience. It does, when the word
survives. But the recogniser segments on its own endpointing **and punctuates what it emits**,
so a candidate who says

> "…to a new payments provider, **and**" *[pauses to think]*

is transcribed as `"…to a new payments provider."` — trailing conjunction deleted, period
added. The evidence of being mid-sentence is destroyed upstream of any logic that could use it.

What survives is length. A behavioural answer is a story — situation, action, result — and
nobody tells one in twelve words, so a short grammatically complete sentence is far more likely
to be an opening clause than a whole answer. That is the rule doing most of the work:

| Pending text | Waits |
|---|---|
| Full-length and ends `.` `!` `?` | 900 ms |
| Long, no terminal punctuation | 1,500 ms |
| Ends on "and" / "the" / "um", **or under 20 words** | 2,500 ms |

**The cost is real and it is not hidden.** Short exchanges — "Yes, that's right." — wait the
full 2.5 s. For an interview coach that is the right way round, because being interrupted
mid-answer is a much worse failure than a slow reply, but it does make quick back-and-forth
feel sluggish.

**The known better answer is semantic turn detection** — a small model that judges from the
audio whether a turn sounds finished, rather than inferring it from a transcript that has
already thrown the evidence away. That is the next real improvement, and it would replace the
word-count rule entirely.

## Scoring, and why it is checked

The coach scores each answer against five competencies — situation, ownership, action
specificity, result, reflection — each anchored at levels 1, 3 and 5 so a score is a judgement
against a written standard rather than a vibe.

**Nothing scores the person.** There is no hireability, confidence or communication-style
score. Those correlate with accent, gender and background far more than with competence, and
they are where automated interview scoring has historically gone wrong. Every competency here
is a property of the answer: does it name a result, does it say what the candidate personally
did.

**Every quote is checked against the transcript.** The model is told to quote the candidate
verbatim. In a real session, **13 of 35 quotes actually appeared in what was said** — the rest
were paraphrases. So quotes are verified by normalised token match, unverified ones are never
shown, and a score of 4 or 5 resting on an unverified quote is capped to 3. Showing someone
words they never said, attributed to them, is worse than showing no evidence.

Two paths, deliberately. Deciding whether to ask a follow-up happens *now*, while the candidate
sits in silence, so it uses a cheap local heuristic — counting numbers, "we" against "I",
concrete verbs. Full rubric scoring runs in the background and lands in the report, where
nobody is waiting.

## Design notes

The architecture research behind these choices is in [`docs/research.md`](docs/research.md),
including the provider comparison, the latency budget, and the options that were rejected.

## Licence

MIT
