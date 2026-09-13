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
| Speech synthesis, first audio byte (warm) | **198 ms** | NVIDIA Magpie over Riva gRPC |
| Speech synthesis, first audio byte (cold) | 1,141 ms | why the connection is prewarmed at startup |
| Speech recognition, first partial | 875 ms | partials stream while you speak |
| Speech recognition, finalisation after speech ends | **286–569 ms** | varies run to run |

Two of these changed the design:

**Reasoning has to be switched off explicitly.** Left on, the model's first streamed tokens are
its own chain-of-thought — fast to arrive and impossible to speak. The honest measure for a voice
agent is time to first *speakable* token, and switching reasoning off moves it from unusable to
363 ms.

**Prewarming the speech connection is worth ~940 ms.** The TLS and HTTP/2 handshake is paid once;
without prewarming it lands on the first thing the coach ever says.

Not yet measured: turn detection, and end-to-end latency through the browser.

## Design notes

The architecture research behind these choices is in [`docs/research.md`](docs/research.md),
including the provider comparison, the latency budget, and the options that were rejected.

## Licence

MIT
