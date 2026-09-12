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

## Design notes

The architecture research behind these choices is in [`docs/research.md`](docs/research.md),
including the provider comparison, the latency budget, and the options that were rejected.

## Licence

MIT
