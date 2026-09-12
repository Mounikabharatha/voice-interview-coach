# Real-Time Voice Mock Interview Coach — Architecture Research

**Research date: 2026-09-13.** Every provider number below was checked against a live page on
that date. Model availability and free-tier limits in this space change on a scale of weeks —
Groq retired two of its most-used free models on 2026-08-16, Hugging Face closed free CPU
Spaces around July 2026, and Deepgram's Flux TTS promotion expired the week this was written.
**Re-verify the numbers in [§13 Sources](#13-sources) before you write code, and read
[§14 Unverified claims](#14-unverified-claims) first.**

Constraints this document is written against: $0 budget, no credit card, LLM must come from
Groq or NVIDIA NIM free tiers, Python, macOS on Apple Silicon with no NVIDIA GPU, solo
developer, part time, portfolio project that must survive hiring-manager scrutiny and be demoed
live without breaking.

**Contents.**
§1 Executive summary · §2 How this works, in plain language · §3 LLM layer · §4 STT layer ·
§5 TTS layer · §6 Reference architecture (protocols, wire protocol, concurrency, sentencizer,
instrumentation, deployment, hosted posture) · §7 Latency budget · §8 Turn-taking (flush
protocol, barge-in, endpointer evaluation, AEC) · §9 Interview-coach domain design (rubric,
prompts, score schema, question bank, calibration, ethics) · §10 Build plan · §11 Failure modes
and the degradation ladder · §12 Interview answers · §13 Sources · §14 Unverified claims.

---

## 1. Executive summary

**Run speech recognition and speech synthesis locally on the Mac, and use Groq only for the
language model.** That is not a preference; it is what the free tiers force, and it happens to
produce a better product.

- **STT — Moonshine v2 Medium Streaming (245M), local (MIT).** Decisive reason: it is the only
  option whose accuracy *and* latency are both measured on the right hardware class by a primary
  source — **6.65% average WER** across eight Open ASR Leaderboard datasets, beating Whisper
  large-v3's 7.44%, with a **published finalization latency of 107–258 ms on Apple hardware** —
  and it is a genuine streaming architecture (position-free encoder, sliding-window attention),
  not a Whisper wrapper re-decoding a sliding buffer. Both figures are the Medium model.
  "Finalization latency" means end-of-speech → final transcript: the encoder work overlaps
  speech, the decode does not. The vendor's paper says 258 ms on an M3 and its own repo says
  107 ms on a "MacBook Pro"; this project budgets 258 ms and publishes what
  `TranscriptLine.last_transcription_latency_ms` actually reports (§4.1). Groq's Whisper endpoint
  is batch-only with no interim transcripts and a 20 requests/minute free cap — structurally
  incapable of a barge-in loop regardless of how fast it transcribes.
- **LLM — Groq `qwen/qwen3.6-27b` with `reasoning_effort: "none"` on the hot path; a local
  4B-class model as the latency/quota fallback; `openai/gpt-oss-20b` at `reasoning_effort: "low"`
  for post-turn rubric scoring only.** Decisive reason: gpt-oss-20b **cannot disable reasoning**
  (the floor is `"low"`), so its first streamed tokens are chain-of-thought — Artificial Analysis
  measures time-to-first-*answer*-token at **3.55 s** against a TTFT of 0.81 s. That disqualifies
  it from the voice loop and qualifies it for scoring, where latency is free and its prompt
  caching and 908 t/s throughput are worth having — and where it is the only free Groq model that
  supports strict `json_schema` structured outputs, which `qwen/qwen3.6-27b` does not support at
  all. When Groq 429s or stalls, the ladder falls to a **local** model (§11.1), not to a slower
  cloud one.
- **TTS — Kokoro-82M, local, sentence-chunked (Apache-2.0).** Decisive reason: Groq's TTS is
  arithmetically unusable — Orpheus is capped at **200 characters per request, 100 requests per
  day, WAV only, no streaming**, i.e. a hard ceiling of 20,000 characters/day shared across all
  development and every demo. Kokoro is the only strong option that is simultaneously fully
  permissive, rate-limit-free, fast on Apple Silicon (~90 ms to first audio measured on an M5
  Max), and cancellable in-process for instant barge-in.

**Response target: ~1,230 ms `t_v2v`** — acoustic end of speech to first audible sample,
measured across a reconciled client/server clock and checked against a loopback recording
(§6.7, §10). The server-only figure, `t_server_v2v` (end of speech at the server → first PCM
byte on the socket), is ~1,133 ms; the last mile is ~31–103 ms (plan 67) and is reported separately rather
than hidden. Felt as faster via a pre-rendered opening, early acknowledgement, and sub-60 ms
barge-in. **Do not claim sub-1s in the README.** The measured P50 with N and a confidence
interval is the claim; three named terms, each defined by the two events it spans, is what makes
it checkable.

### The stack

| Layer | Choice | Where it runs | Licence | Why |
|---|---|---|---|---|
| Transport | Browser mic → WebSocket → FastAPI, wire protocol v1 | Local (demo) / Render (invite-gated link) | — | Browser gives free AEC; WS is the only free-tier-friendly long-lived transport |
| VAD | `silero-vad` 6.2.1 ONNX weights, loaded into our own ORT session | Local CPU | MIT | <1 ms per 32 ms frame, ~2 MB, robust to room noise; ONNX-only keeps PyTorch off the shipped path |
| Endpointing | `pipecat-ai/smart-turn-v3.2` ONNX int8 | Local CPU | BSD-2 | Semantic completeness from the waveform; ~12–30 ms; open weights (LiveKit's is licence-locked) |
| STT | Moonshine v2 Medium Streaming (245M) | Local (no PyTorch, no MLX) | MIT (English) | 6.65% WER (Medium); 107–258 ms finalization, sources disagree — measure; true partials |
| LLM (hot path) | Groq `qwen/qwen3.6-27b`, `reasoning_effort:"none"` | Groq free tier | — | Only free Groq model that can turn reasoning off |
| LLM (fallback) | `gemma-3n-e4b-it-text` via LM Studio at `127.0.0.1:1234/v1` | **Local** | Gemma Terms of Use | ~4 GB RAM; no quota, no network, no 429. The only fallback that is *faster*, not slower |
| LLM (scoring) | Groq `openai/gpt-oss-20b`, `reasoning_effort:"low"`, strict `json_schema` | Groq free tier | — | Off the hot path. Production tier, prompt caching, the only free model with structured outputs |
| TTS | Kokoro-82M via `mlx-audio`, int16 LE in ~50 ms chunks | Local | Apache-2.0 | ~90 ms first audio, no quota, instant cancel |
| AEC | `getUserMedia({echoCancellation:true})` + loopback playback | Browser | — | Only free path; headphones as the demo-day guarantee |
| Metrics | JSONL turn log (`turnlog/v2`, reconciled two-clock) + `bench/` harnesses | Local | — | The portfolio differentiator |

**Explicitly rejected:** Groq Whisper on the hot path (batch, 20 RPM); Groq Orpheus TTS (100
requests/day); **`openai/gpt-oss-20b` as a hot-path fallback** (3.55 s to first answer token — it
does not degrade the product, it breaks it); `faster-whisper` (CTranslate2 has no Metal backend —
CPU-only on a Mac); Coqui XTTS-v2 (its licence URL and entire `coqui.ai` domain return HTTP 404);
LiveKit's turn detector (MODEL_LICENSE forbids use outside LiveKit Agents); Vocode (last commit
2024-11-15); PyTorch anywhere on the shipped path (§6.5); any "hireability" or confidence scoring
(the HireVue failure mode).

---

## 2. How this works, in plain language

The system is ears, a brain, and a mouth, wired in a loop.

The **ears** listen through your laptop microphone. Software called a *voice activity detector*
decides, thirty times a second, whether the sound it hears is speech or just room noise. When
it hears speech it starts writing down what you said — a *speech-to-text* model turns the
waveform into words as you talk, not after you stop.

The **brain** is a large language model. It reads the transcript of your answer, compares it
against an interview scoring rubric, and writes a reply — a follow-up question, or feedback.

The **mouth** is a *text-to-speech* model that turns those written words back into audio and
plays them through your speakers.

The hard part is not any single step. It is that all three must happen inside the gap that
humans naturally leave between conversational turns — which research across ten languages puts
at **about 200 milliseconds**. This system lands at about 1.2 seconds, and the document says so
rather than rounding it down; every voice agent built on a cloud language model is in that range,
and the ones that claim otherwise are usually measuring a different pair of events. What closes
the felt gap is not a smaller number but a different shape: the opening question is pre-recorded
so it is instant, the first clause of every reply is spoken while the rest is still being
written, and interrupting the bot stops it inside 60 ms.

So the trick is never to do the steps one at a time. Transcription runs *while* you are still
speaking. The moment you finish, the brain starts writing, and as soon as it has produced one
complete sentence — not the whole answer — that sentence is already being spoken aloud. Each
stage overlaps the next.

The other hard part is knowing *when you have finished*. A naive system waits for a fixed
second of silence, which means it cuts off anyone who pauses to think — exactly what interview
candidates do. And when you interrupt the bot mid-sentence, it has to stop instantly, discard
the audio already queued up, and remember only the words you actually heard it say.

---

## 3. LLM layer

Both providers are OpenAI-API-compatible, both stream via server-sent events, both support
function calling on the relevant models. They differ on limits, legal terms, and how honestly
those limits are published.

### Groq vs NVIDIA NIM

| | **Groq** | **NVIDIA NIM (build.nvidia.com)** |
|---|---|---|
| Base URL | `https://api.groq.com/openai/v1` | `https://integrate.api.nvidia.com/v1` |
| Credit card | **Not required.** Free plan is $0; a payment method is needed only to upgrade to Developer | **Not required.** `build.nvidia.com/llms.txt`: "All models offer a free trial tier with no credit card required." |
| Published free rate limits | **Yes, per model ID.** Chat models: **30 RPM / 1,000 RPD / 8,000 TPM / 200,000 TPD** — ⚠ but see the trap below, the page now self-labels as Developer-plan base limits | **Yes, but soft.** Site copy: **"Up to 40 rpm"** and **"10,000 requests per day"**, caveated "Rate limits may vary by model and traffic from other users may cause throttling" |
| Credits / quota model | No credits — pure rate limits, org-level | Credit-based. 1,000 on signup, +4,000 with a business email (5,000 total), initial grant **expires after 30 days** — *all from NVIDIA staff forum posts (Sept 2024 / Jan 2025), not current docs.* Per-call credit cost is published nowhere |
| Legal use limits | Nothing restricts portfolio demos, open-sourcing, or benchmarks. AUP forbids exceeding limits via multiple accounts, and forbids automated employment decisions without human supervision | **ToS §1.4: "you may only use the API Service for internal testing and evaluation purposes, not in production."** A local demo is fine; a public always-on instance on the trial key is not |
| Data / training | "By default, Groq does not retain customer data for inference requests." Services Agreement 4.2 forbids training on your inputs/outputs | Anti-competition clause ToS §4.12; service can be withdrawn without notice (ToU §2) |
| Prompt caching | Yes, on `gpt-oss-*` only. Automatic, 2 h expiry, 50% discount, and **cached tokens do not count toward rate limits**. Minimum cacheable prefix "128 to 1024 tokens depending on the specific model" | Not documented |
| Structured outputs | `json_schema` with `strict: true` on `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` **only**. All models support the weaker `json_object`. **Cannot be combined with streaming or tool use** | Not documented per model |
| Streaming mechanism | `stream: true` in body → SSE deltas | `"stream": true` **in the body**, with `Accept: application/json`. *Do not* set `Accept: text/event-stream` — that is the stale pre-OpenAI-compat pattern |
| Model churn | Severe. 8 shutdowns in 12 months, incl. both Llama chat models on 2026-08-16 | Severe. `meta/llama-3.3-70b-instruct` absent from `/v1/models` (end-of-support 08/25/2026); free endpoints deprecate per-model |

### Groq free-tier chat models (verified 2026-09-13)

| Model ID | Tier | Context / max out | Reasoning off? | Prompt cache | Strict JSON | Free limits (RPM/RPD/TPM/TPD) | Measured out t/s | TTFT | **Time to first *answer* token** |
|---|---|---|---|---|---|---|---|---|---|
| `qwen/qwen3.6-27b` | **Preview** | 131,072 / 16,384 | **Yes (`none`)** | No | **No** | 30 / 1K / 8K / 200K | ~437–439 | 1.29 s | **2.44 s** (reasoning off) vs 15.36 s (on) |
| `openai/gpt-oss-20b` | Production | 131,072 / 65,536 | **No** (floor `low`) | **Yes** | **Yes** | 30 / 1K / 8K / 200K | ~908–909 | 0.81–0.83 s | **3.55 s** (high) / 3.58 s (low) |
| `openai/gpt-oss-120b` | Production | 131,072 / 65,536 | **No** (floor `low`) | **Yes** | **Yes** | 30 / 1K / 8K / 200K | ~471 | 0.73–0.74 s | **6.04 s** |
| `qwen/qwen3.8-27b` | **Preview** | 131,072* / 16,384 | **Yes (`none`)** | No | **Yes** | 30 / 1K / 8K / 200K | n/a | n/a | n/a |
| `openai/gpt-oss-safeguard-20b` | **Preview** | 131,072 / 65,536 | No | Yes | n/a | 30 / 1K / 8K / 200K | n/a | n/a | n/a |
| `groq/compound-mini` | Production | 131,072 / 8,192 | n/a | No | n/a | 30 / **250** / 70K / — | ~450 | n/a | n/a |
| ~~`llama-3.1-8b-instant`~~ | **Shut down 2026-08-16** | — | — | — | — | Enterprise only | — | — | — |
| ~~`llama-3.3-70b-versatile`~~ | **Shut down 2026-08-16** | — | — | — | — | Enterprise only | — | — | — |

\* printed as 131,042 in the docs; almost certainly a typo.
Throughput/TTFT/first-answer-token figures are from Artificial Analysis, a **live rolling
benchmark** — treat as ±5% and re-check. Groq publishes no official TTFT and explicitly refers
developers to Artificial Analysis.

**Three traps on Groq's own docs.** First, `console.groq.com/docs/models.md` has a rate-limit
column headed **"RATE LIMITS (DEVELOPER PLAN)"** showing 250K TPM / 1K RPM — 31× the free TPM —
and a "MAX FILE SIZE" column showing 100 MB, which is also the developer figure (free is 25 MB).
Second, `rate-limits.md` notes some organisations get split **ITPM/OTPM** limits instead of one
TPM. Third, and new as of 2026-09-13: **`rate-limits.md` itself now carries the line "the limits
shown below are the base limits for the Developer plan."** That page was the authoritative
free-tier source in earlier drafts of this document and it no longer describes itself that way.
Every token figure in §7, §9, §11 and §12 is therefore phrased as "against a published 8,000 TPM"
so it can be redone with one substitution. **`console.groq.com/settings/limits` on your actual
logged-in account is now the only authority. Check it before writing the README.**

### NVIDIA NIM candidates (verified 2026-09-13)

| Model ID | Tool calling | Context | Latency knob |
|---|---|---|---|
| `nvidia/nemotron-3.5-lightning-30b-a3b` | Yes | 1,048,576 | **Must set** `extra_body={"chat_template_kwargs":{"enable_thinking":False}}` — thinking is on by default and streams `delta.reasoning_content` before any `delta.content` |
| `openai/gpt-oss-20b` | Yes | 131,072 | Same reasoning problem as on Groq |
| `nvidia/nemotron-3-super-120b-a12b` | Yes | — | 12B active; heavier |
| `deepseek-ai/deepseek-v4-flash-0731` | Yes | — | "Flash" tier |
| ~~`meta/llama-3.3-70b-instruct`~~ | — | — | **Gone** from `/v1/models`; page carries a deprecation notice dated 08/25/2026 |

### Decision

**Hot path: Groq `qwen/qwen3.6-27b`, `reasoning_effort: "none"`, `stream=True`,
`max_completion_tokens=120`, `temperature=0.6`.**

**Hot-path fallback: a local 4B-class model (`gemma-3n-e4b-it-text` via LM Studio), not a second
cloud model.** The full four-rung ladder — `qwen3.6-27b` → `qwen3.8-27b` (availability only) →
local → scripted probes — is specified in §11.1.

**Scoring (off hot path): Groq `openai/gpt-oss-20b`, `reasoning_effort:"low"`, `response_format`
`json_schema` with `strict:true`, `temperature=0`, prompt caching on.** Verified 2026-09-13:
`qwen/qwen3.6-27b` supports neither strict nor best-effort `json_schema` — it is absent from
Groq's structured-outputs model list entirely. The hot-path model and the scoring model are
different models for that reason, and the difference costs nothing because scoring is not
latency-bound. Fallback `qwen/qwen3.8-27b` (strict, no caching).

**Secondary provider: NVIDIA NIM `nvidia/nemotron-3.5-lightning-30b-a3b` with
`enable_thinking: False`, behind the same `LLMProvider` interface.**

The decisive reason is the reasoning-token trap. For a voice agent the only latency number that
matters is time to the first token you can hand to TTS, and gpt-oss-20b cannot stop emitting
chain-of-thought — Artificial Analysis puts 2.7 seconds between its TTFT and its first answer
token, and that gap barely moves between `reasoning_effort` `low` and `high`. qwen3.6-27b is the
only free Groq model with a documented `none` setting, and AA shows the payoff directly:
2.44 s to first answer token with reasoning off versus 15.36 s with it on.

That same fact is why gpt-oss-20b is **not** the fallback. Earlier drafts of this document named
it in §1, §3 and §11 as the one-line config-flag fallback while simultaneously proving it takes
3.55 s to its first answer token. Flipping that flag does not degrade the coach from 1,230 ms to
1,500 ms; it degrades it to ~4,000 ms, which is not a slower product but a different, broken one.
The two axes are separated in §11.1: the **latency/quota** fallback goes local, and the
**availability** fallback (Groq retires the Preview model) goes to `qwen/qwen3.8-27b`.

**But settle the primary by measurement, not by my table.** AA measures with its own prompts and
harness, and a 2.44 s first-answer-token for a non-reasoning model against a 1.29 s TTFT is
hard to reconcile with a short conversational prompt. Day 0, before any pipeline code, run
`bench/spike_llm.py` against three candidates with your real payload (§10 Milestone 1). Decision
rule:

- If gpt-oss-20b at `reasoning_effort:"low"` lands under ~600 ms to first *content* token in your
  own measurement, **use it on the hot path** — it is a Production model with prompt caching and
  strict JSON. (Expect this not to happen; the 3.55 s figure says it will not.)
- Otherwise use qwen3.6-27b and accept the Preview risk, mitigated by config + startup check.
- If the **local** rung's P90 first token is under 700 ms, the ladder in §11.1 is correct as
  written and the local rung is a genuine peer, not a parachute. 700–1,200 ms: keep the ladder but
  pre-render the rung-1 filler and drop `max_tokens` to 70. Over 1,200 ms: the local rung becomes
  the scripted-probe path only, and you say so in the README.

**Both paths require:** the model ID lives in config, never as a literal; a startup health check
that calls `GET /openai/v1/models` and fails loudly if the configured ID is absent; and history
capped in *tokens*, not messages.

**The binding constraint is 8,000 TPM, not 30 RPM.** 30 requests/minute is generous for a voice
loop. 8,000 tokens/minute is not: a system prompt plus a rubric plus five turns of interview
transcript reaches 1,500–3,000 input tokens per turn, which is four to five turns a minute
before a 429. The measured design figures for this specific product are in §9.6 — **~1,585 tokens
per probe turn, ~530 per cached scoring call, ≈28,050 tokens per 17-minute session, ≈7 sessions
per day against the 200,000 TPD cap.** TPD is the binding *daily* constraint; TPM binds only on
bursts, which the reserved-floor bucket in §9.6 handles. Cap history in *tokens*, and read
`x-ratelimit-remaining-tokens` off **every** response into the turn log — not just off 429s.
(`retry-after` is returned only on a 429; the two `x-ratelimit-*` headers are on every response.)

**But "summarise older turns" and "prompt caching" are in direct conflict, and the conflict has
to be designed around rather than left in a bullet list.** Groq's cache matches on a stable
prompt prefix; rewriting old turns into a summary changes the prefix and invalidates everything
downstream of the edit — at exactly the moment you are closest to the TPM ceiling and most need
the discount and the rate-limit exemption. §3.1 resolves it: the prompt is three regions, the
prefix is append-only, and summarisation is a scheduled epoch rollover at a question boundary
rather than a per-turn rewrite.

### §3.1 Prompt layout, and the one place it is allowed to change

Verified on `console.groq.com/docs/prompt-caching`, 2026-09-13: caching is **automatic**, applies
to **`openai/gpt-oss-20b`, `gpt-oss-120b`, `gpt-oss-safeguard-20b` only**, expires after **2 hours
without use**, gives a **50% input discount**, **cached tokens do not count toward rate limits**,
and the minimum cacheable prefix is **"128 to 1024 tokens depending on the specific model"**.
Hits are attempted, not guaranteed.

Two consequences. First, **the hot-path model, `qwen/qwen3.6-27b`, has no caching at all** — so
caching is not a property of the recommended hot path; it is a property of the *scoring* path
(§9.4) and one of the inputs to the Day-0 model decision. Second, the layout below costs nothing
on qwen (it is just prompt hygiene) and pays off on gpt-oss, so it is built regardless.

**Three regions. This order and no other.**

| Region | Contents | Mutability | Size |
|---|---|---|---|
| **R1 STATIC** | system prompt + the full rubric YAML verbatim + the question bank | **Byte-identical for the life of the process, and across sessions** | Target ≥1,200 tokens so it clears the 1,024-token worst-case cacheable minimum. Keeping the rubric verbatim rather than compressed gets you there without padding — and the judge prompts want it verbatim anyway |
| **R2 LOG** | epoch summary (if any) + verbatim turns since the last rollover | **Append-only.** Never edited in place | Bounded by the rollover, ~1,400 tokens |
| **R3 TAIL** | STAR slot state, probes remaining, current question id | Rewritten **every turn** | ~120 tokens |

R3 changes every turn, so it is concatenated onto the **current user message** — the one message
that is uncached by definition — inside a `<state>…</state>` delimiter. Mutating it therefore
costs **zero** cached tokens. Putting live state anywhere earlier is the mistake that silently
kills caching, and it is the mistake most agent codebases make.

**Render every rendered block with sorted keys and no timestamp.** Prefix matching is exact, and
one reordered dict key silently costs you every cache hit for two hours.

**Epoch rollover, not per-turn summarisation.** When R1+R2 exceeds the budget (2,600 tokens),
fold everything but the last 2 turns into a single `<session_so_far>` summary and rebuild R2.
That costs **one** cache miss, and it is only ever invoked at a **question boundary** —
`CoachSession` advancing from one behavioural question to the next — where a pre-rendered
transition line is already playing and masks the extra TTFT. It is never invoked mid-answer or
between a probe and its follow-up.

**The tradeoff, named.** A 17-minute session will roll over once or twice. Each rollover discards
a warm ~1,400-token cached prefix and pays full price on the next turn. The alternative —
summarising every turn to keep the prompt small — would invalidate the cache on **every** turn
and lose the rate-limit exemption entirely, which on gpt-oss is worth more than the token
reduction: cached tokens are exempt from the 8,000 TPM cap, so a cache hit does not just cost
less, it does not count. Rolling over twice a session is strictly better than rolling over forty
times. On qwen, where there is no cache, the rollover still bounds the prompt and nothing is lost.

**Measure it, do not assume it.** A startup probe sends the static prefix twice and asserts the
second call reports `cached_tokens > 0`. If it does not, `cache_confirmed` stays `False` and the
README says so instead of claiming a benefit. Every turn logs `prompt_tokens`, `cached_tokens`,
`cache_hit_ratio` and `billable_prompt_tokens = prompt_tokens − cached_tokens` — the last one
being the number that actually races the 8,000 TPM cap, and the number the HUD should show next
to `x-ratelimit-remaining-tokens`.

```python
"""src/coach/llm/prompt.py - cache-safe prompt assembly.

Three regions, in this order and no other:

  R1 STATIC   system prompt + full rubric YAML + question bank. Byte-identical
              for the life of the process and across sessions. The only region
              Groq can cache across turns, so it goes first and never changes.
              Must clear ~1,024 tokens (Groq's documented cacheable prefix
              minimum is "128 to 1024 tokens depending on the model") - the
              rubric verbatim gets you there without padding.

  R2 LOG      epoch summary (if any) + verbatim turns since the last rollover.
              APPEND-ONLY. Nothing in here is ever edited in place.

  R3 TAIL     STAR slot state, probe budget, current question. Rewritten every
              turn, so it is concatenated onto the CURRENT user message - the
              one message that is uncached by definition. Cost of mutating it:
              zero cached tokens.

Summarisation never rewrites R2 in place. It happens as an EPOCH ROLLOVER: fold
everything but the last `keep_verbatim_turns` into a new epoch summary, rebuild
R2 once, eat one cache miss. Rollover is called only at a question boundary,
where pre-rendered transition audio covers the extra TTFT.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass
class CachedPrompt:
    static: str                       # R1
    budget_tokens: int = 2_600        # ceiling on R1 + R2
    keep_verbatim_turns: int = 2
    epoch_summary: str = ""
    log: list[dict] = field(default_factory=list)   # R2, append-only
    rollovers: int = 0
    _chars_per_token: float = 4.0     # self-calibrating; see observe_usage()

    @property
    def static_sha(self) -> str:
        """Log this per turn. If it ever changes mid-session, caching is dead
        and you want to know why."""
        return hashlib.sha256(self.static.encode()).hexdigest()[:12]

    # ---- R2: append-only -------------------------------------------------
    def append_user(self, text: str) -> None:
        self.log.append({"role": "user", "content": text})

    def append_assistant(self, spoken: str) -> None:
        """Only what the user actually heard. Called from TurnController.finally."""
        if spoken:
            self.log.append({"role": "assistant", "content": spoken})

    # ---- assembly --------------------------------------------------------
    def messages(self, user_text: str, tail: str = "") -> list[dict]:
        head = self.static
        if self.epoch_summary:
            head += f"\n\n<session_so_far>\n{self.epoch_summary}\n</session_so_far>"
        tail_block = f"\n\n<state>\n{tail}\n</state>" if tail else ""
        return (
            [{"role": "system", "content": head}]
            + self.log
            + [{"role": "user", "content": user_text + tail_block}]
        )

    # ---- budget ----------------------------------------------------------
    def est_tokens(self) -> int:
        chars = (len(self.static) + len(self.epoch_summary)
                 + sum(len(m["content"]) for m in self.log))
        return int(chars / self._chars_per_token)

    def over_budget(self) -> bool:
        return self.est_tokens() > self.budget_tokens

    def observe_usage(self, usage: dict, sent_chars: int) -> None:
        """Correct the chars-per-token estimate from what the API actually
        counted, so no tokenizer dependency is needed for either model."""
        pt = usage.get("prompt_tokens") or 0
        if pt > 0 and sent_chars > 0:
            self._chars_per_token += 0.3 * (sent_chars / pt - self._chars_per_token)

    # ---- the one place R2 is ever rewritten ------------------------------
    async def rollover(self, summarise: Callable[[list[dict]], Awaitable[str]]) -> bool:
        """Question boundaries only - never mid-answer, never between a probe
        and its follow-up. Returns True if the cache was invalidated."""
        if not self.over_budget() or len(self.log) <= self.keep_verbatim_turns:
            return False
        keep = self.log[-self.keep_verbatim_turns:]
        self.epoch_summary = await summarise(self.log[: -self.keep_verbatim_turns])
        self.log = keep
        self.rollovers += 1
        return True


class CacheMeter:
    """Records what caching actually bought, per turn, into the turn log.
    Nothing in the README claims a caching benefit this did not measure."""

    def __init__(self, model: str, supported: bool):
        self.model, self.supported, self.confirmed = model, supported, False
        self.prompt_tokens = self.cached_tokens = 0

    def observe(self, usage: dict) -> dict:
        p = usage.get("prompt_tokens", 0) or 0
        c = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0
        self.prompt_tokens += p
        self.cached_tokens += c
        if c > 0:
            self.confirmed = True
        return {
            "prompt_tokens": p,
            "cached_tokens": c,
            "cache_hit_ratio": round(c / p, 3) if p else None,
            "cache_supported": self.supported,
            "billable_prompt_tokens": p - c,     # what races the 8,000 TPM cap
        }

    def session(self) -> dict:
        return {
            "model": self.model,
            "cache_supported": self.supported,
            "cache_confirmed": self.confirmed,
            "prompt_tokens_total": self.prompt_tokens,
            "cached_tokens_total": self.cached_tokens,
            "cache_hit_ratio_session":
                round(self.cached_tokens / self.prompt_tokens, 3) if self.prompt_tokens else None,
        }


async def confirm_caching(llm, static_block: str) -> bool:
    """Startup probe. Send the static prefix twice; the second call must report
    cached_tokens > 0. If it does not, the README says caching is unconfirmed."""
    msgs = [{"role": "system", "content": static_block},
            {"role": "user", "content": "ready?"}]
    await llm.complete(msgs, max_completion_tokens=1)
    usage = (await llm.complete(msgs, max_completion_tokens=1)).get("usage", {})
    return ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0) > 0
```

---

## 4. STT layer

The single most important distinction here is **true streaming versus batch**. A batch model
transcribes a complete audio file after the user stops talking, so its entire round-trip is
added to your response latency. A true streaming model emits partial transcripts *while* the
user speaks, so most of the transcription cost is paid before the user stops — but, as §4.1
establishes, not all of it.

Whisper is a batch encoder-decoder model. Everything marketed as "streaming Whisper" is a
wrapper re-decoding a sliding window, which produces unstable partials and seconds of latency.

### §4.1 What Moonshine's latency number actually measures

Earlier drafts of this document quoted two Moonshine numbers as if they were the same kind of
thing. They are not, and one of them was invented. Settle it before the table.

**The definition, from the primary source.** Both the paper and the vendor's own benchmark docs
define the metric identically:

> "Response latency for both models is measured as the time between a phrase being identified as
> complete by the VAD segmenter and the transcribed text being returned."
> — `docs/using/benchmarks.md`, moonshine-ai/moonshine

> "the amount of time taken between detecting the end of a speech segment in an audio stream (via
> a voice activity detection (VAD) model) and the transcript text being returned."
> — arXiv:2602.12241, §Evaluation

So **258 ms is end-of-speech → final transcript, and it sits squarely on the critical path after
endpointing.** It is not a throughput figure, not an RTF, and not "overlapped with speech." What
*is* overlapped with speech is the encoder work — which is precisely why the number is 258 ms and
not Whisper large-v3's 11,286 ms on the same harness. The streaming architecture collapses the
finalization cost; it does not eliminate it. The vendor's stated design goal for this metric is
"keeping it below 200 ms."

**The number belongs to Medium Streaming — and the sources disagree with each other.** Both the
6.65% WER and the 258 ms latency are **Moonshine v2 Medium Streaming (245M)**. Full
correspondence, from arXiv:2602.12241 Table 2 plus the WER table:

| Model (v2) | Params | Avg WER, 8 Open ASR datasets | Response latency, Apple M3 (paper) | Response latency, "MacBook Pro" (repo README) |
|---|---|---|---|---|
| Tiny Streaming | 34M | 12.01% | 50 ms | 34 ms |
| Small Streaming | 123M | 7.84% | 148 ms | 73 ms |
| **Medium Streaming** | **245M** | **6.65%** | **258 ms** | **107 ms** |
| Whisper large-v3 (faster-whisper, CPU) | 1.5B | 7.44% | 11,286 ms | 11,286 ms |
| Whisper small | 244M | 8.59% | 1,940 ms | 1,940 ms |
| Whisper tiny | 39M | 12.81% | 289 ms | 277 ms |

**The two Moonshine columns are 1.5–2.4× apart and both are first-party.** The Whisper rows are
byte-identical between them while every Moonshine row differs, which tells you what happened: the
README's Moonshine cells were refreshed by `scripts/test-mobile-latency.sh` (native C++ core, CDN
models, averaging the library's own `lastTranscriptionLatencyMs`) while the Whisper baselines were
carried over from the Python harness `scripts/run-benchmarks.py` that produced the paper's table.
Two harnesses, one table.

A third first-party number contradicts both. The same benchmarks doc says of the README cells:

> "Medium Streaming on the Mac varied between 56 and 83 ms across three runs … The published cells
> are medians of three runs taken with the device given time to cool between them, so expect a
> single run to disagree with them a little."

So the vendor publishes 107 ms, says the underlying runs measured 56–83 ms, and the paper says
258 ms. **Nobody's published Moonshine Mac latency figure can be trusted to better than ~3×.** Say
that in the README rather than picking a flattering one.

**Decision: budget 258 ms, measure it on day one.** Three reasons, in order:

1. The paper's harness is `run-benchmarks.py` — **the Python path**, which is the path this
   project ships. The README's faster cells come from the native C++ core benchmark. You are not
   running the native core benchmark.
2. It is the conservative figure. A budget that degrades gracefully when reality is better is
   worth more than one that collapses when reality is worse.
3. It is the only one attached to a named chip (Apple M3) and a stated methodology.

**And then stop guessing, because the library measures this for you.** `TranscriptLine` carries
`last_transcription_latency_ms: int` — the same field the vendor's own benchmark script averages:

```python
# src/coach/stt/moonshine_local.py — inside the LineCompleted handler
def on_line_completed(self, event: LineCompleted) -> None:
    line = event.line
    self._log.mark("stt_final_ms")
    self._log.set("stt_engine_latency_ms", line.last_transcription_latency_ms)  # int, ms
    self._log.set("stt_line_id", line.line_id)
    self._log.set("stt_words", len(line.words or []))
    self._emit_final(line.text, line.words)
```

One field, logged every turn, settles a contradiction three published tables could not. Publish
the distribution of `stt_engine_latency_ms` against 258 / 107 / 56–83 and you have converted a
documentation mess into a measurement result — which is a better portfolio artifact than either
number would have been.

**Three implementation consequences that fall out of the source.**

**1. Force the finalization pass; do not wait for Moonshine's own segmenter.** The 258 ms starts
when *Moonshine's* VAD segmenter declares the phrase complete, which is a different instant from
when *your* Silero + smart-turn endpointer does. Drive it yourself:

```python
from moonshine_voice.transcriber import MOONSHINE_FLAG_FORCE_UPDATE

# Fired the moment Silero reports speech->silence (entering CANDIDATE_END),
# NOT when END_TURN is declared. This runs concurrently with smart-turn.
def force_finalize(self) -> None:
    self._log.mark("stt_force_update_ms")
    self._stream.update_transcription(MOONSHINE_FLAG_FORCE_UPDATE)
```

This buys the whole smart-turn inference (12–30 ms) for free by overlapping it, and it is why the
budget in §7 charges `max(smart_turn, stt_final)` rather than their sum. Note precisely what this
does and does not do: it forces a **decode pass over audio already fed**, it does not inject audio
and it does not close the line, so no word timestamp moves. Closing the line is a separate
operation with different consequences — see the flush protocol in §8.1, which is where END_TURN is
handled. If smart-turn then says "incomplete" and the candidate resumes, Moonshine opens a new
line; concatenate the line texts at END_TURN. That is the same pattern as Deepgram Flux's
`TurnResumed`, hand-rolled.

**2. `update_transcription()` blocks the calling thread.** It is a synchronous FFI call into the
C++ core (`moonshine_transcribe_stream`). Called from the FastAPI event loop it stalls the WS
reader and the TTS pump for its whole duration — 258 ms of stall per turn, on the turn where you
can least afford it. Run the Moonshine stream on a dedicated single-worker thread (`stt_pool`,
§6.5) and hand finals back with `loop.call_soon_threadsafe`.

**3. Set `update_interval=0.25`, not the default 0.5.** The interval is a *floor that adapts
upward*, not a cadence. From `Stream.add_audio`:

```python
needed = min(max(self._update_interval, self._last_pass),
             self._update_interval * _MAX_UPDATE_INTERVAL_FACTOR)   # factor = 10
```

A pass must cover at least as much audio as the previous pass took to compute, so a machine
without headroom automatically backs off instead of falling further behind each pass. That makes
0.25 safe to ask for: if the Mac cannot afford twice-a-second passes, the library ignores you.
What you gain is a shorter un-transcribed tail at the moment you force finalization, and the tail
is what the finalization pass costs. The source gives the cost model's shape — for the Tiny model,
"102 ms of a pass goes on getting started and 269 ms on each second of audio it looks at."
**Medium's constants are published nowhere; do not extrapolate them.** Halving the tail is the
lever; measure the effect with `stt_engine_latency_ms` at `update_interval` 0.5 vs 0.25 and put
the two CDFs in the README.

### The options

| Option | **Streaming?** | Accuracy (avg WER, Open ASR) | Speed on Apple Silicon | Free tier | Licence | Verdict |
|---|---|---|---|---|---|---|
| **Moonshine v2 Medium Streaming** (`moonshine-voice` 0.1.5) | **True streaming** — `on_text` partials, `on_line` finals | **6.65%** (Medium, 245M; beats Whisper large-v3) | **107–258 ms finalization latency on Apple hardware — first-party sources disagree (§4.1); budget 258** | Unlimited (local) | **MIT** (English) | **Recommended.** No PyTorch; deps are numpy + sounddevice. **Wheel is `macosx_15_0_arm64` — requires macOS 15.0+, and there is no Linux wheel at all** |
| **parakeet-mlx** (Parakeet TDT 0.6b v3) | **True streaming** — `transcribe_stream` with `finalized_tokens` + `draft_tokens` | **6.34%** (best here) | **No published benchmark on any Apple Silicon chip.** One secondary test measured 0.50 s for a ~2–3 s utterance on M4 | Unlimited (local) | Apache-2.0 code / CC-BY-4.0 weights | Fallback if macOS <15. Requires `ffmpeg`, Python ≥3.10 |
| **Kyutai STT 1B** (`stt-1b-en_fr`) | **True streaming** + semantic VAD for end-of-turn | Not published | Fixed **0.5 s model delay** | Unlimited (local) | MIT/Apache code, CC-BY-4.0 weights | 0.5 s delay is twice Moonshine's budgeted finalization cost |
| **Deepgram Flux** (`flux-general-en`) | **True streaming + turn-taking as an API primitive** | Not published | N/A (cloud) | **$200 credit, no card, "No expiration"** ≈ 512 h @ $0.0065/min promo (433 h @ $0.0077 regular) | Commercial | **Best cloud option, and the STT layer of the HOSTED build (§6.8).** A credit pot, not a free tier |
| **AssemblyAI Universal-Streaming** | **True streaming**, immutable transcripts + `end_of_turn` | Not published per model | N/A (cloud) | $50 credit, no card ≈ 333 h @ $0.15/hr. Limit is **5 new sessions/minute** on free (not 5 concurrent) | Commercial | Credible, but **billed on WebSocket connection-open time, not audio sent** — a forgotten socket burns credit |
| **Groq Whisper** (`whisper-large-v3-turbo`) | ❌ **Batch only.** No `stream` param, no WebSocket, no partials | 12% WER (10.3% for `large-v3`) | 216× real-time server-side | **20 RPM / 2K RPD / 7,200 audio-sec/hr / 25 MB** | Commercial | **Not viable on the hot path.** 20 RPM = one request per 3 s; chunked pseudo-streaming needs 60–120 RPM |
| **whisper.cpp** `examples/stream` | ⚠ **Pseudo-streaming** — ring buffer, re-decodes, churning partials | ~7.4% (large-v3) | ~1.8× real-time large-v3 on M3, ~2.6× on M4 with Metal | Unlimited | MIT | The walkie-talkie feel you are trying to avoid |
| **mlx-whisper** | ❌ **Batch only**; streaming needs `whisper_streaming` (headline latency 3.3 s) | ~7.4% | 1.02 s for a ~2–3 s utterance on M4 | Unlimited | MIT | Batch — but see §4.2 measurement 4: it is the right tool for building in-domain references |
| **faster-whisper** | ❌ **Batch only** | ~7.4% | ⚠ **CPU-only on a Mac — CTranslate2 supports CPU and CUDA only, no Metal/MPS** | Unlimited | MIT | Wrong hardware. 6.96 s for a short utterance on M4 |
| **WhisperKit / argmax-oss-swift** | OSS server streams *output* over a complete upload (SSE); the real-time WebSocket server is **Pro (paid)** | ~7.4% | Good (CoreML/ANE) | OSS MIT, streaming server paid | MIT (OSS tier) | Swift-first; awkward from Python |
| **Apple SpeechAnalyzer** | True streaming (`AsyncStream<AnalyzerInput>`) | Third-party benchmark only | Very fast (OS model) | Free, on-device | OS framework | ⚠ **Swift-only, macOS 26** — needs a sidecar process. Stretch goal, not a dependency |
| **NVIDIA `nemotron-asr-streaming`** | True streaming (gRPC, interim results, tunable endpointing) | 6.93% @ 1.12 s chunk | ❌ **GPU-only**; hosted path is gRPC at `grpc.nvcf.nvidia.com:443` | 40 rpm / 10k per day (soft) | NVIDIA Open Model Licence | Hosted-only, gRPC plumbing, ToS non-production. Ruled out |

### Decision

**Primary: Moonshine v2 Medium Streaming, local. Fallback: parakeet-mlx. Cloud comparison and
HOSTED-build backend: Deepgram Flux, behind the same interface.**

Verify `sw_vers -productVersion` on your Mac before committing — if it is below 15.0 there is no
`moonshine-voice` wheel and the fallback becomes primary.

Moonshine's API gives you exactly the events a barge-in loop needs. The rich listener interface
(`TranscriptEventListener` with `on_line_started`, `on_line_text_changed`, `on_line_updated`,
`on_line_completed`) carries line ids, `start_time` and word timings — use that rather than the
simple `on_text`/`on_line` callbacks, because word timings are also what your delivery metrics
and evidence-span rubric need. `add_audio()` accepts audio of any length and handles conversion
internally, so you do not need to match its chunk size.

**Keep Groq Whisper wired in anyway**, as an optional high-accuracy pass over the complete
recorded answer after the turn ends. It costs nothing on the latency path and it gives you a
second transcript to diff against. But the comparison it supports is
**transcript disagreement rate and latency delta, local streaming vs cloud batch** — not WER.
There are no reference transcripts for your own recorded answers, and the only reference on
offer would be Groq Whisper's own output, which the table above scores at 12% WER against
Moonshine's 6.65%. Scoring the more accurate system against the less accurate one produces a
disagreement rate wearing a WER label, and any interviewer who reads the table carefully will
catch it. Report the disagreement rate as what it is, and confine real WER to corpora that ship
human references — see §4.2.

**If you use Deepgram Flux**, note its event taxonomy carefully — the messages are `TurnInfo`
envelopes whose `event` field takes the values **`Update`** (~every 0.25 s of audio),
**`StartOfTurn`**, **`EagerEndOfTurn`**, **`TurnResumed`**, **`EndOfTurn`**. `TurnResumed` is the
cancellation signal when the user keeps talking after an eager end-of-turn; build without it and
the bot speaks over every mid-sentence pause. Eager events **do not fire at all** unless you set
`eager_eot_threshold` (range 0.3–0.9, unset by default). `eot_threshold` is 0.5–1.0, default 0.7;
`eot_timeout_ms` is 500–60000, default 5000. Control messages are `Configure`, `CloseStream`,
`ForceEndTurn`. Flux publishes end-of-turn detection "under 400 ms" (vendor); Coval's independent
benchmark, published by Deepgram, puts median EOT under 300 ms.

### §4.2 What gets measured, and on what

Three separate measurements. Only the first is a word error rate.

**1. Real WER — ungated public test sets with human references.** `hf-audio/esb-datasets-test-only`
carries eight Open ASR Leaderboard corpora. Three are gated behind acceptance forms
(`common_voice`, `gigaspeech`, `spgispeech` — the last needs the Kensho user agreement), verified
2026-09-13. **Use the five ungated ones: `librispeech` (test.clean and test.other), `voxpopuli`,
`tedlium`, `earnings22`, `ami`.** Subsample 500 utterances per split with a fixed seed, print N
and total audio duration next to every number, and report **pooled (corpus) WER** — total edits
over total reference words — not the mean of per-utterance WERs, which lets a 4-word utterance
outweigh a 40-word one. `ami` and `earnings22` are the two that matter for this project: far-field
meeting speech and spontaneous earnings-call speech are the closest public proxies for a nervous
person telling a story into a laptop mic. `librispeech test.clean` is read audiobook speech and
will flatter every model; report it for comparability to the leaderboard and say what it is.

This is where the 6.65% and 12% figures in the table above live, and it is the only corpus on
which "Moonshine beats Groq Whisper on WER" is a sentence you are entitled to write.

**2. Transcript disagreement rate — your own 40–60 in-domain answers, no references.** Neither
side is ground truth, so the metric is an edit distance between two hypotheses, reported in both
directions plus symmetrically, after Whisper's `EnglishTextNormalizer` on both sides (without it
you will be counting "40%" vs "forty percent" as an error — measured on a two-utterance smoke
test, normalisation took the disagreement rate from three edits to one).

- `D(local-denominator)` = edit ops ÷ local word count
- `D(cloud-denominator)` = edit ops ÷ cloud word count
- `D(symmetric)` = edit ops ÷ mean word count — the one to headline
- substitution / deletion / insertion counts, because the *shape* of the disagreement is the
  interesting part: batch Whisper deleting disfluencies is a different finding from the two
  models hearing different nouns

Label it in the README as *"the two systems disagreed on 7.1% of words (symmetric, N = 52
utterances, 31 min)"*. Never as WER.

**3. Adjudicated win rate — the only directional accuracy claim this corpus supports.** Draw 50
disagreement sites uniformly at random from measurement 2, listen to each one (Moonshine's
`on_line_updated` gives word timings, so the site jumps straight to the audio), and mark
`local` / `cloud` / `both_wrong` / `tie`. Report counts with a Wilson 95% interval. Cost: about
25 minutes. At N = 50 the interval on a 60% win rate is roughly [46%, 72%] — wide, and worth
stating, but it is an honest directional claim where the disagreement rate alone is none.

**4. In-domain WER — optional, and only if you build references properly.** If you want a real
in-domain WER, the reference cannot be derived from either system under test, or it launders one
model's errors into the ground truth. Build it from a **third** system: run `mlx-whisper`
large-v3 locally over the answers, hand-correct its output against the audio, and use the
corrected text as the reference for both Moonshine and Groq Whisper. Do this on a 20-answer
subset (~25 min of audio, ~1 h of post-editing at 2–3× real time), report in-domain WER with
N = 20 and a bootstrap CI, and say the references are self-produced and single-pass.

**Pricing the alternative, because someone will ask.** Rev human transcription is **$1.99/minute**,
99%+ accuracy, 12-hour turnaround (verified 2026-09-13). 50 answers × ~75 s ≈ 62 minutes of audio
≈ **$124**. That is outside a $0 budget, so it is your hour or nothing — which is exactly why the
in-domain WER is scoped to 20 answers and labelled as such.

**Latency delta.** Both systems measured off the same monotonic `t0` in the same turn log, same
utterances, N ≥ 200 turns through `bench/replay.py`, P50/P95 with bootstrap CIs:

- local: `stt_final_ms` — the §7 row 4 cost, ~258 ms budgeted, because the encoder already
  consumed the audio during speech and only the finalization pass remains
- cloud batch: `t(HTTP response received) − t(end_of_speech)` — includes the whole-utterance
  upload, queue time and inference

Operational note for the replay run: Groq Whisper's free cap is 20 RPM, so the cloud leg must be
spaced ≥3 s. A 200-utterance comparison run takes **10+ minutes of wall clock** and cannot be
parallelised. Budget it, and log `x-ratelimit-remaining-requests` so a silent 429 does not become
a missing row that quietly biases the P95.

```python
#!/usr/bin/env python3
"""bench/disagree.py - local streaming STT vs cloud batch STT, WITHOUT pretending
either one is ground truth.

  1. Directional disagreement rates D(local-denom) and D(cloud-denom) plus a
     symmetric rate. These are edit distances between two hypotheses. They are
     NOT word error rates and are never labelled as such.
  2. An adjudication sample: K disagreement sites drawn uniformly at random with
     surrounding context, written to CSV for you to listen to and mark. Marked
     results give a win rate with a Wilson 95% interval - the only directional
     accuracy claim this corpus supports.
  3. Nothing else. Real WER lives in bench/wer.py and runs only on splits that
     ship human references.

    python bench/disagree.py runs/2026-10-04-3f9a2c1.jsonl --sites 50
    python bench/disagree.py runs/2026-10-04-3f9a2c1.jsonl --adjudicated sites.csv
"""
from __future__ import annotations

import argparse, csv, json, math, pathlib, random
from jiwer import process_words
from whisper_normalizer.english import EnglishTextNormalizer

NORM = EnglishTextNormalizer()


def rates(a: list[str], b: list[str]) -> dict:
    """a, b are parallel transcripts of the SAME utterances from two systems."""
    ab = process_words([NORM(x) for x in a], [NORM(y) for y in b])
    ba = process_words([NORM(y) for y in b], [NORM(x) for x in a])
    edits = ab.substitutions + ab.deletions + ab.insertions
    n_a = sum(len(t) for t in ab.references)
    n_b = sum(len(t) for t in ab.hypotheses)
    return {
        "n_utterances": len(a),
        "words_local": n_a, "words_cloud": n_b, "edit_ops": edits,
        "D_local_denom": round(ab.wer, 4),
        "D_cloud_denom": round(ba.wer, 4),
        "D_symmetric": round(edits / ((n_a + n_b) / 2), 4),
        "sub": ab.substitutions, "del": ab.deletions, "ins": ab.insertions,
    }


def sites(a: list[str], b: list[str], ids: list[str], k: int, seed: int = 0) -> list[dict]:
    out = []
    for uid, x, y in zip(ids, a, b):
        o = process_words([NORM(x)], [NORM(y)])
        ref, hyp = o.references[0], o.hypotheses[0]
        for c in o.alignments[0]:
            if c.type == "equal":
                continue
            out.append({
                "utterance_id": uid,
                "type": c.type,
                "local_word_index": c.ref_start_idx,   # -> Moonshine word timing -> audio
                "local": " ".join(ref[max(0, c.ref_start_idx - 4):c.ref_end_idx + 4]),
                "cloud": " ".join(hyp[max(0, c.hyp_start_idx - 4):c.hyp_end_idx + 4]),
                "verdict": "",   # you fill: local | cloud | both_wrong | tie
            })
    random.Random(seed).shuffle(out)
    return out[:k]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="JSONL with turn_id, stt_local_text, stt_cloud_text")
    ap.add_argument("--sites", type=int, default=50)
    ap.add_argument("--adjudicated", help="CSV returned with the verdict column filled")
    a = ap.parse_args()

    rows = [json.loads(l) for l in pathlib.Path(a.run).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("stt_local_text") and r.get("stt_cloud_text")]
    loc = [r["stt_local_text"] for r in rows]
    cld = [r["stt_cloud_text"] for r in rows]
    ids = [r["turn_id"] for r in rows]

    print(json.dumps(rates(loc, cld), indent=2))

    if a.adjudicated:
        v = [r["verdict"].strip() for r in csv.DictReader(open(a.adjudicated)) if r["verdict"].strip()]
        n, w = len(v), v.count("local")
        lo, hi = wilson(w, n)
        print(f"\nadjudicated sites N={n}: local correct {w} ({w/n:.0%}, "
              f"Wilson 95% [{lo:.0%}, {hi:.0%}]), cloud {v.count('cloud')}, "
              f"both wrong {v.count('both_wrong')}, tie {v.count('tie')}")
        return

    out = pathlib.Path(a.run).with_suffix(".sites.csv")
    s = sites(loc, cld, ids, a.sites)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s[0].keys()))
        w.writeheader(); w.writerows(s)
    print(f"\nwrote {len(s)} adjudication sites -> {out}")


if __name__ == "__main__":
    main()
```

---

## 5. TTS layer

| Option | Type | **Time to first audio** | Streams? | Runs on Apple Silicon? | Free-tier ceiling | **Licence for a public portfolio demo** |
|---|---|---|---|---|---|---|
| **Kokoro-82M** (`mlx-audio` / `kokoro-onnx`) | Local | **~90 ms first audio, RTF 0.08** (M5 Max, third-party) | Yes — `KPipeline` is a generator yielding per segment at 24 kHz; `mlx-audio` has `--stream` | Yes — MPS / MLX / ONNX / CoreML | None | **Apache-2.0 — fully clear.** (`espeak-ng` dep is GPL-3.0; matters only if you ship a binary) |
| **Piper** (`piper1-gpl`) | Local | ~40 ms, RTF 0.03 (M5 Max, third-party) | Yes (chunked) | Yes (ONNX, CPU) | None | ⚠ Engine is now **GPL-3.0**; the MIT `rhasspy/piper` repo was archived. Official VOICES.md: *"Piper is intended for personal use and text to speech research only."* Voices are licensed individually |
| **KittenTTS** | Local | Very low | Yes | Yes (<25 MB model) | None | Apache-2.0 — clear. Quality is the trade-off |
| **Chatterbox Turbo** | Local | Vendor: "75 ms Latency — real-time voice synthesis on a GPU", "6× faster than real-time … on a single GPU". **No Apple Silicon figure published; repo device options are `cuda`/`cpu` only** | Yes | Unverified on MPS | None | MIT — clear |
| **Coqui XTTS-v2** | Local | High | Partial | Yes, slow | None | ❌ **Licence link `coqui.ai/cpml` returns 404 — the entire `coqui.ai` domain is dead.** Repo last pushed 2024-08-16. Unusable for anything public |
| **Orpheus TTS (local)** | Local | ~200 ms — **but measured with vLLM on GPU** | Yes | ❌ 3B Llama backbone | None | Apache-2.0 |
| **Groq Orpheus** (`canopylabs/orpheus-v1-english`) | Hosted | n/a — **no streaming**, returns a finished WAV | **No** | n/a | **10 RPM / 100 RPD / 200 chars per request / WAV only** → hard ceiling ~20,000 chars/day | Groq ToS |
| **Deepgram Aura-2** | Hosted | Sub-200 ms TTFB (vendor) | Yes (TTS WebSocket) | n/a | $200 credit ≈ 6.6 M chars @ $0.030/1k. 15 REST / 45 WS concurrent | Credit-funded commercial product — no demo restriction |
| **Rime** (Mist v3) | Hosted | ~70 ms claimed | Yes | n/a | **3,000 free minutes on Starter (~3 M characters)** — the most generous hosted allowance found | Check plan terms |
| **ElevenLabs Flash v2.5** | Hosted | ~75 ms model latency (excludes network) | Yes (`/stream-input` WS) | n/a | 10k credits/mo ≈ 20k chars / ~20 min on Flash (50% lower per-char) | ❌ **ToS §1(c): Free Users "may only use the Services for non-commercial purposes."** Commercial starts at $6/mo |
| **Cartesia Sonic-3.6** | Hosted | Not stated | Yes — best incremental API (`continue` flag) | n/a | 20k credits/mo ≈ 27 min | ❌ **No commercial licence below Pro ($5/mo).** ⚠ `max_buffer_delay_ms` **defaults to 3000 ms** — a self-inflicted 3 s latency floor if left unset |
| **NVIDIA Magpie TTS** | Hosted | Not stated | Yes (gRPC `SynthesizeOnline`) | n/a — **no public weights; NIM container needs an NVIDIA GPU (CC 8.0+, 16 GB VRAM)** | 40 rpm / 10k per day (soft) | ⚠ NVIDIA ToU §7: trial services are "for evaluation purposes only and not for production use" |
| **edge-tts** | Hosted (unofficial) | ~200–500 ms, network-bound | Yes (WS) | n/a | Unpublished | ⚠ LGPLv3 library calling an **undocumented Microsoft endpoint** with no public contract. Weak answer to "why did you choose this?" |

### Decision

**Kokoro-82M, local, synthesised sentence by sentence, driven through `mlx-audio`, emitting
24 kHz mono int16 LE in ~50 ms chunks.**

Decisive reason: it is the only option simultaneously Apache-2.0 (zero licence risk in a public
repo), fast enough on Apple Silicon, and subject to no quota that can fail you in front of a
hiring manager. Note that the `hexgrad/kokoro` inference library itself was last pushed
2025-08-06 (~13 months stale) while `Blaizzy/mlx-audio` was pushed 2026-09-11 — run the weights
through `mlx-audio`, which is the actively maintained Apple Silicon path and ships Kokoro in
bf16/8/6/4-bit with a `--stream` flag.

**The chunk size is part of the decision, not an implementation detail.** `kokoro_local.py` must
re-chunk the generator's output into ~50 ms (1,200-frame) int16 pieces and convert from float32 at
that boundary. A generator that yields one array per sentence makes "first byte" and "last byte"
the same instant, which silently destroys both the 90–200 ms TTS row in §7 and the
`tts_first_byte_ms` mark that proves it. Int16 rather than float32 also halves the bytes on the
WebSocket — the demo path — and `sd.RawOutputStream(dtype="int16")` consumes it directly.

**Do not plan on Groq for TTS.** Combine its two published limits: 100 requests/day × 200
characters = **20,000 characters/day absolute ceiling**, organisation-wide, shared across every
test run and the live demo — and the 3,600 TPD cap probably binds sooner. At ~150 characters per
spoken sentence that is roughly 130 utterances per day. Worse, the 200-character cap plus no
streaming compound: a multi-sentence coach reply needs 3–5 *sequential* whole-file requests, so
at 10 RPM your real ceiling is two to three bot replies per minute, each with whole-file
latency and no partial-audio escape hatch.

**Add Deepgram Aura-2 as a flag-switched second backend.** It buys you a pluggable-TTS talking
point and a measured local-vs-hosted TTFB comparison for the README — and it is the TTS layer of
the HOSTED build (§6.8), because Kokoro cannot run on 0.1 CPU. Note for that build: Aura-2's TTS
WebSocket accepts only `linear16`, `mulaw` and `alaw`; **`opus` is REST-only**, so there is no
cheap compressed downlink.

---

## 6. Reference architecture

### Data flow

```
 ┌───────────────────────── BROWSER (Chrome) ───────────────────────────┐
 │                                                                      │
 │  getUserMedia({audio:{echoCancellation:true,                         │
 │                       noiseSuppression:true,                         │
 │                       autoGainControl:true}})                        │
 │        │                                                             │
 │        ▼                                                             │
 │  AudioWorklet ──► downsample 48k→16k, Int16 ──► 512-sample frames     │
 │        │              (32 ms each — matches Silero exactly)          │
 │        │              + capture_ctx_s stamp per frame (§6.7)         │
 │        │                                                             │
 │        │  ◄── TTS PCM chunks ── played via loopback RTCPeerConnection │
 │        │      so the browser AEC treats the bot's voice as a remote   │
 │        │      stream and cancels it  (fallback: headphones)           │
 │        │      explicitly scheduled; client marks returned (§6.7)      │
 └────────┼─────────────────────────────────────────────────────────────┘
          │  WebSocket, wire protocol v1 (§6.2)
          │    binary: 8-byte header + audio frames up / int16 PCM down
          │    json:   control frames — turn events, partials, metrics, HUD
 ┌────────▼──────────────────── PYTHON SERVER (FastAPI + uvicorn) ───────┐
 │                                                                       │
 │  ┌──────────┐   512-sample frames                                     │
 │  │ Silero   │──────────────┬──────────────────────────────┐           │
 │  │ VAD 6.2.1│  inline on   │                              │           │
 │  │ (ONNX)   │  the loop    ▼                              ▼           │
 │  └────┬─────┘        ┌──────────────┐              ┌──────────────┐   │
 │       │              │  Moonshine   │              │  8s ring     │   │
 │       │              │  v2 stream   │              │  buffer      │   │
 │       │              │  (stt_pool)  │              │  (16k mono)  │   │
 │       │              └──────┬───────┘              └──────┬───────┘   │
 │       │  speech/silence     │ partials + finals           │           │
 │       ▼                     │                             ▼           │
 │  ┌─────────────────────┐    │                    ┌──────────────────┐ │
 │  │   ENDPOINTER        │    │                    │ smart-turn v3.2  │ │
 │  │  SPEAKING           │    │                    │ ONNX int8 8.7 MB │ │
 │  │   └200ms silence──► │    │  ─ ─ ─ arm ─ ─ ─►  │  (turn_pool)     │ │
 │  │  CANDIDATE_END ─────┼────┼─► FORCE_UPDATE     │  threshold 0.5   │ │
 │  │   └p>0.5 ─► END     │    │  ◄── p(complete) ──┤  12–30 ms CPU    │ │
 │  │   └3.0s ─► END (SLA)│    │                    └──────────────────┘ │
 │  └──────────┬──────────┘    │  the forced pass and the classifier    │
 │             │ END ──► flush protocol (§8.1) ──► transcript, t0       │
 │             ▼                                                         │
 │  ┌─────────────────────────────────────────────────────────────┐      │
 │  │            TurnController  (generation-id guard)            │      │
 │  │   cancel() ─► player.abort() FIRST ─► task.cancel() ─► await│      │
 │  └──────────┬──────────────────────────────────────────────────┘      │
 │             ▼                                                         │
 │  ┌────────────────┐  SSE deltas   ┌──────────────┐  sentences         │
 │  │  LLM ladder    │──────────────►│  Incremental │──────────┐         │
 │  │  qwen3.6-27b   │  §11.1        │  sentencizer │          │         │
 │  │  → local       │               └──────────────┘          ▼         │
 │  └────────────────┘                                  ┌────────────┐   │
 │          ▲                                           │  Kokoro-82M│   │
 │          │ R1 static + R2 log + R3 tail (§3.1)       │ (tts_pool) │   │
 │  ┌───────┴────────┐                                  └─────┬──────┘   │
 │  │ CoachSession   │  STAR slot state, probe selection       │ 24 kHz  │
 │  │ + rubric.yaml  │  — the STATE MACHINE picks, not the LLM │ int16   │
 │  └────────────────┘                                         ▼         │
 │                                                      ┌────────────┐   │
 │   every turn ──► TurnLog (turnlog/v2 JSONL) ──►      │ AudioPlayer│   │
 │                  t0 + 40 fields, two clocks          │ enqueue/   │   │
 │                  reconciled (§6.7)                   │ abort/drain│   │
 │                                                      └─────┬──────┘   │
 └────────────────────────────────────────────────────────────┼──────────┘
                                                              │ WS down
                                                              ▼ browser
```

### §6.1 Protocols and formats — pin these

| Hop | Format |
|---|---|
| Browser mic → server | **16 kHz mono Int16 PCM, 512-sample (32 ms) frames**, binary WS messages. 512 because Silero VAD raises `ValueError` on anything else at 16 kHz |
| Server → Moonshine | Same buffer, any length (`add_audio()` converts internally) |
| Server → smart-turn | 16 kHz mono, **last 8 s**, front-padded so speech sits at the *end* of the vector |
| Server → Groq | HTTPS, one persistent `httpx.AsyncClient`, SSE via `stream=True` in the body |
| Server → Deepgram Flux (`HOSTED`) | WS, `linear16` @ 16 kHz, `eot_threshold=0.7`, `eot_timeout_ms=4000` |
| Kokoro → player (`LOCAL`) | **24 kHz mono int16 LE PCM, ~50 ms (1,200-frame) chunks.** The float32 → int16 conversion and the re-chunking both live in `kokoro_local.py`; see §5 for why |
| Aura-2 → server (`HOSTED`) | WS, `encoding=mulaw&sample_rate=24000`. ⚠ Deepgram lists `mulaw` and `24000` as independently valid but does not document which *combinations* are accepted, and mu-law is conventionally 8 kHz telephony. **Test on day 1.** If rejected: `linear16` @ 24 kHz (doubles egress, halves the monthly session ceiling) or `mulaw` @ 8000 (telephone-grade). Aura-2's WS does **not** support `opus` — compressed encodings are REST-only |
| Server → browser | **Wire protocol v1**, §6.2 |
| Native path instead of the browser | livekit `rtc.AudioProcessingModule` requires **exactly 10 ms frames (160 samples @ 16 kHz)** — re-chunk to 512 afterwards for Silero |

### §6.2 Wire protocol v1

One socket, two channels. **The binary channel carries audio and nothing else; the text channel
carries JSON and nothing else.** No base64, no JSON-wrapped audio, no sniffing. Every frame on
both channels is versioned.

**Binary framing — 8-byte header, big-endian:**

```
byte  0      uint8    frame type
byte  1      uint8    protocol version (== 1; mismatch closes the socket)
bytes 2–3    uint16   utt_id   — the utterance this audio belongs to; 0 for uplink mic frames
bytes 4–7    uint32   seq      — monotonic within (channel, utt_id); wraps are a protocol error
bytes 8–     payload
```

| Type | Name | Dir | Payload |
|---|---|---|---|
| `0x01` | `MIC_PCM` | C→S | Exactly 1,024 bytes: 512 samples, 16 kHz mono Int16 LE. Any other length is a protocol error |
| `0x02` | `TTS_PCM` | S→C | 20–100 ms of audio in the codec announced by `session_resume.audio_out`. Never inferred per frame |
| `0x03` | `TTS_PRERENDERED` | S→C | Same payload rules; flags cached audio (greeting, questions, move-on lines) so the HUD excludes it from TTS latency statistics |
| `0x00`, `0x04`–`0xFF` | — | — | Reserved. Receiving one is `error{code:"bad_frame"}` and an immediate close |

8 bytes on a 1,024-byte payload is 0.8% overhead. `seq` is what makes `playback_ack` and resume
reconciliation possible; without it there is no way to know what the browser actually played.

**JSON control frames.** Every frame, both directions:
`{"v":1, "t":<type>, "seq":<uint32>, "ts":<int ms from session t0>, ...}`. Unknown `t` values are
ignored (forward compatibility); `v != 1` is fatal.

| `t` | Dir | Fields | Semantics |
|---|---|---|---|
| `hello` | C→S | `sid`, `last_seq`, `played_through`:{utt_id→frames}, `invite_key`, `client`:{`ua`,`sr_in`,`sr_out`,`vad`:bool}, `consent`:{`accepted`:bool,`text_sha256`} | First frame after open. **No audio is accepted until the server answers.** `text_sha256` pins which consent text was agreed to |
| `session_resume` | S→C | `sid`, `resumed`:bool, `reason`, `replayed`:int, `build`:`"LOCAL"\|"HOSTED"`, `audio_out`:{`codec`:`"pcm_s16le"\|"mulaw"`,`rate`,`channels`}, `limits`:{`max_session_s`,`idle_timeout_s`,`sessions_left_today`}, `providers`:{`stt`,`llm`,`tts`} | **Always the server's first frame.** Carries the downlink codec contract — the client must not guess |
| `clock_ping` / `clock_pong` / `clock_offset` | C→S / S→C / C→S | see §6.7 | NTP-style four-timestamp offset estimation on the socket that already exists |
| `client_marks` | C→S | `turn_id` plus the client-clock marks in §6.7's table | The browser's half of `t_v2v` |
| `vad_state` | C→S (`HOSTED`) / S→C (`LOCAL`) | `speaking`:bool, `p_speech`:float, `frame_seq`:uint32 | Direction is decided by `build`. Exactly one side is authoritative; the other ignores its own copy. Drives the HUD's speech bar |
| `partial` | S→C | `line_id`, `text`, `words`:[{`w`,`t0_ms`,`t1_ms`}], `stable_prefix_len`:int | Moonshine `on_line_text_changed` / Flux `Update`. Superseded by any later `partial` with the same `line_id`; the client replaces, never appends |
| `final` | S→C | `line_id`, `text`, `words`:[…], `t_start_ms`, `t_end_ms`, `source`:`"moonshine"\|"flux"\|"groq_whisper"` | Immutable. `source` is what makes the local-vs-cloud disagreement visible in the UI |
| `turn_p` | S→C | `p`:float, `decision`:`"continue"\|"end"\|"timeout"`, `model`:`"smart-turn-v3.2"\|"flux"`, `infer_ms`:int | One per candidate endpoint. This is the §8 visualisation; emit it even when `decision == "continue"` |
| `turn_end` | S→C | `line_id`, `transcript`, `reason`:`"semantic"\|"timeout"\|"forced"`, `gen`:int | Turn closed, generation `gen` started |
| `bot_speaking_start` | S→C | `utt_id`, `gen`, `text`, `prerendered`:bool | Sent immediately **before** the first binary frame of `utt_id`. The client uses it to pre-open the playback path |
| `bot_speaking_stop` | S→C | `utt_id`, `reason`:`"complete"\|"barge_in"\|"cancelled"\|"error"`, `frames_sent`:int, `chars`:int | `frames_sent` is the denominator `reconcile_playback` compares acks against |
| `playback_ack` | C→S | `utt_id`, `frames_played`:int, `played_ms`:int | Every 250 ms while playing and once at utterance end. The only evidence the server has that audio was heard |
| `barge_in` | C→S | `utt_id`, `client_stop_ms`:int | `HOSTED` only. Authoritative: the client already stopped. Server cancels the generation and reports `client_stop_ms` as the audible-stop metric |
| `metrics` | S→C | `turn_id`, the `turnlog/v2` field set (§6.7), `n`:int | One per completed turn. **This frame is byte-identical to the JSONL row** written by `metrics/turn_log.py` — one schema, two sinks. Excluded from the resume replay ring |
| `provider_switch` | S→C | `from`, `to`, `reason`, `detail`, `ui`, `filler_audio` — full schema in §11.1 | The visible graceful-degradation path. The HUD shows a badge; the demo never shows a traceback |
| `error` | S→C | `code`, `message` (user-safe, no internals), `retriable`:bool, `retry_after_s`:int\|null | Codes: `bad_frame`, `unsupported_version`, `unauthorized`, `no_consent`, `rate_limited`, `session_cap`, `budget_exhausted`, `session_expired`, `server_restarting`, `internal` |
| `end_session` | C→S, S→C | `reason`:`"user"\|"deadline"\|"idle"\|"error"` | Clean close. Server finalises the report and drops the session |

```python
# src/coach/wire.py  — the whole framing layer
from __future__ import annotations
import json, struct, time
from collections import deque
from dataclasses import dataclass

PROTOCOL_VERSION = 1
_HDR = struct.Struct("!BBHI")                 # type, version, utt_id, seq -> 8 bytes
HDR_LEN = _HDR.size

MIC_PCM, TTS_PCM, TTS_PRERENDERED = 0x01, 0x02, 0x03
_BINARY_TYPES = frozenset((MIC_PCM, TTS_PCM, TTS_PRERENDERED))

MIC_PAYLOAD_BYTES = 1024                      # 512 samples * int16
MAX_BINARY_BYTES  = HDR_LEN + 4096            # cap before uvicorn allocates; 512 MB instance
MAX_JSON_BYTES    = 4096

class BadFrame(ValueError):
    """Never leaks to the client verbatim — it becomes error{code:"bad_frame"}."""

@dataclass(slots=True)
class BinaryFrame:
    type: int
    utt_id: int
    seq: int
    payload: memoryview

def pack(type_: int, utt_id: int, seq: int, payload: bytes) -> bytes:
    return _HDR.pack(type_, PROTOCOL_VERSION, utt_id & 0xFFFF, seq & 0xFFFFFFFF) + payload

def unpack(data: bytes) -> BinaryFrame:
    n = len(data)
    if n < HDR_LEN or n > MAX_BINARY_BYTES:
        raise BadFrame(f"length {n}")
    t, v, utt_id, seq = _HDR.unpack_from(data)
    if v != PROTOCOL_VERSION:
        raise BadFrame(f"version {v}")
    if t not in _BINARY_TYPES:
        raise BadFrame(f"type 0x{t:02x}")
    if t == MIC_PCM and n - HDR_LEN != MIC_PAYLOAD_BYTES:
        raise BadFrame(f"mic payload {n - HDR_LEN} != {MIC_PAYLOAD_BYTES}")
    return BinaryFrame(t, utt_id, seq, memoryview(data)[HDR_LEN:])


class ControlChannel:
    """Owns the JSON seq counter and the replay ring. Outlives any single socket."""
    RING = 256
    NO_REPLAY = frozenset(("metrics", "vad_state"))   # re-derivable / stale on reconnect

    def __init__(self, t0_ns: int) -> None:
        self.t0_ns, self.seq = t0_ns, 0
        self.ring: deque[dict] = deque(maxlen=self.RING)
        self.ws = None

    def attach(self, ws) -> None:
        self.ws = ws

    async def send(self, t: str, **fields) -> dict:
        self.seq += 1
        frame = {"v": PROTOCOL_VERSION, "t": t, "seq": self.seq,
                 "ts": (time.monotonic_ns() - self.t0_ns) // 1_000_000, **fields}
        if t not in self.NO_REPLAY:
            self.ring.append(frame)
        if self.ws is not None:
            await self.ws.send_text(json.dumps(frame, separators=(",", ":")))
        return frame

    async def replay_after(self, last_seq: int) -> int:
        pending = [f for f in self.ring if f["seq"] > last_seq]
        for f in pending:
            await self.ws.send_text(json.dumps(f, separators=(",", ":")))
        return len(pending)


async def send_audio(chan: ControlChannel, utt_id: int, seq: int,
                     pcm: bytes, *, prerendered: bool = False) -> None:
    t = TTS_PRERENDERED if prerendered else TTS_PCM
    await chan.ws.send_bytes(pack(t, utt_id, seq, pcm))
```

```js
// web/protocol.js — the browser half. 40 lines, no dependencies.
export const V = 1;
export const MIC_PCM = 0x01, TTS_PCM = 0x02, TTS_PRERENDERED = 0x03;

export function pack(type, uttId, seq, payload /* Int16Array | Uint8Array */) {
  const bytes = payload instanceof Int16Array
    ? new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength)
    : payload;
  const out = new Uint8Array(8 + bytes.length);
  const dv = new DataView(out.buffer);
  dv.setUint8(0, type); dv.setUint8(1, V);
  dv.setUint16(2, uttId, false); dv.setUint32(4, seq, false);
  out.set(bytes, 8);
  return out;
}

export function unpack(buf /* ArrayBuffer */) {
  if (buf.byteLength < 8) throw new Error("short frame");
  const dv = new DataView(buf);
  const v = dv.getUint8(1);
  if (v !== V) throw new Error(`protocol v${v}, expected v${V}`);
  return { type: dv.getUint8(0), uttId: dv.getUint16(2, false),
           seq: dv.getUint32(4, false), payload: new Uint8Array(buf, 8) };
}

// G.711 mu-law -> Int16, 256-entry table. The hosted downlink is mu-law; decoding costs the
// browser nothing and halves Render egress. Built once at module load.
// Check: 0xFF -> 0 (silence), 0x00 -> -32124, 0x80 -> +32124.
const MULAW = new Int16Array(256);
for (let i = 0; i < 256; i++) {
  const u = ~i & 0xff;                                  // BIAS = 0x84
  const t = (((u & 0x0f) << 3) + 0x84) << ((u & 0x70) >> 4);
  MULAW[i] = (u & 0x80) ? (0x84 - t) : (t - 0x84);
}
export const decodeMulaw = (b) => { const o = new Int16Array(b.length);
  for (let i = 0; i < b.length; i++) o[i] = MULAW[b[i]]; return o; };
```

**Version rule.** `v` is an integer on every JSON frame and byte 1 of every binary frame, and the
path prefix `/ws/v1/` carries it a third time. A breaking change bumps all three: old clients get a
404 on the upgrade rather than a silent mis-parse. Additive changes (a new optional field, a new
`t` value) do not bump `v` — unknown `t` is ignored by contract, which is what makes the metrics
schema extensible without a client deploy.

### §6.3 Module layout

```
voice-interview-coach/
├── README.md                     # methodology block + measured numbers, see §12
├── pyproject.toml                # PIN every version; Pipecat-adjacent deps move weekly
├── .env.example                  # GROQ_API_KEY, NVIDIA_API_KEY, DEEPGRAM_API_KEY,
│                                 #   INVITE_KEY, IP_HASH_SALT, PUBLIC_ORIGIN
├── config/
│   ├── default.yaml              # model IDs live HERE, never as literals in code
│   ├── local.yaml                # the LOCAL build
│   ├── hosted.yaml               # the HOSTED build (Flux + Aura-2 + browser VAD)
│   ├── prerendered/              # WAVs: intro, 14 questions, 5 move-on lines, filler, 6 probes
│   ├── questions/behavioral_v1.yaml
│   └── rubric/behavioral_v1.yaml # competencies + behavioural anchors
├── schemas/
│   ├── score_response.schema.json   # what the model returns (strict json_schema)
│   └── score_record.schema.json     # what is stored and what the report reads
├── docs/
│   ├── archive_sources.sh        # snapshot the 15 load-bearing pages (§13.1)
│   └── snapshots/                # committed copies + dated screenshots
├── src/coach/
│   ├── app.py                    # FastAPI, /ws/v1/<sid>, /healthz warm-up, executor wiring
│   ├── settings.py               # pydantic-settings; validate_models() at startup
│   ├── wire.py                   # binary framing + ControlChannel + replay ring (§6.2)
│   ├── sessions.py               # SessionStore, resume, playback reconciliation (§6.9)
│   ├── guard.py                  # invite key, origin, IP-hash limiter, daily ledgers (§6.10)
│   ├── sentence.py               # incremental sentencizer: push(delta)->[sentences], flush()
│   ├── audio/
│   │   ├── frames.py             # ring buffer, 512-sample re-chunker (public .pending), resampling
│   │   ├── player.py             # sounddevice OutputStream(latency='low'); enqueue/abort/drain/played_frames
│   │   └── aec.py                # optional native path: livekit rtc APM / pywebrtc-audio
│   ├── turn/
│   │   ├── vad.py                # SileroVAD.is_speech(frame) — ONNX only, no torch
│   │   ├── endpointer.py         # FSM + SmartTurn.p_complete(window) as a separable callable
│   │   └── controller.py         # TurnController: generation id, abort-first cancel, heard-only commit
│   ├── stt/
│   │   ├── base.py               # Protocol: start_turn(), feed(frame), flush(timeout)->FlushResult,
│   │   │                         #   seal(), snapshot()
│   │   ├── moonshine_local.py    # PRIMARY
│   │   ├── parakeet_mlx.py       # fallback for macOS < 15
│   │   ├── deepgram_flux.py      # HOSTED build + cloud comparison; handle TurnResumed
│   │   └── groq_whisper.py       # post-hoc second transcript, off the hot path
│   ├── llm/
│   │   ├── base.py               # Protocol: stream_chat(messages) -> DeltaStream
│   │   ├── stream.py             # DeltaStream: SSE → text deltas, first-token + stall timeouts
│   │   ├── ladder.py             # LLMLadder: rung switching, hysteresis, provider_switch frame
│   │   ├── prompt.py             # CachedPrompt (R1/R2/R3), CacheMeter (§3.1)
│   │   ├── groq_llm.py           # rung 0; reasoning_effort from config; logs usage.queue_time
│   │   ├── local_llm.py          # rung 1; LM Studio / Ollama OpenAI-compatible
│   │   └── nim_llm.py            # enable_thinking=False; base_url from config
│   ├── tts/
│   │   ├── base.py               # Protocol: synthesize(sentence) -> AsyncIterator[bytes]
│   │   ├── kokoro_local.py       # PRIMARY (mlx-audio); float32→int16, ~50 ms re-chunk
│   │   └── deepgram_aura.py      # hosted comparison + HOSTED build
│   ├── coachlogic/
│   │   ├── session.py            # interview state machine: intro → Q → probes → wrap
│   │   ├── rubric.py             # SpanMatcher + attach_evidence (rejects unquoted claims)
│   │   ├── prompts.py            # COACH_SYSTEM_PROMPT, SCORING_SYSTEM_PREFIX, builders
│   │   ├── probe.py              # rubric-gap detectors → follow-up selection
│   │   └── report.py             # timestamped HTML session report
│   └── metrics/
│       ├── clock.py              # ClockSync: NTP-style browser/server offset
│       ├── turn_log.py           # turnlog/v2 JSONL row, client-mark merge, derived headlines
│       └── stats.py              # percentiles + bootstrap CIs
├── web/
│   ├── index.html                # transcript, VAD state, smart-turn p, latency HUD, rung badge
│   ├── mic-worklet.js            # 48k→16k, Int16, 512-sample frames + capture_ctx_s
│   ├── protocol.js               # wire protocol v1 (§6.2)
│   ├── clock.js                  # ClockSync client half (§6.7)
│   └── app.js                    # WS client + loopback-RTCPeerConnection playback + scheduling
├── bench/
│   ├── spike_llm.py              # Day-0: three candidates, N=20 each, P50/P90
│   ├── replay.py                 # stub mode N=200; live-LLM mode N=60/day
│   ├── wer.py                    # REAL WER only. jiwer 4.x + EnglishTextNormalizer, POOLED,
│   │                             #   ungated esb splits: librispeech/voxpopuli/tedlium/earnings22/ami
│   ├── disagree.py               # reference-free local-vs-cloud disagreement + adjudication
│   ├── endpoint.py               # false-cutoff rate, added latency, SLA fallback, words lost
│   ├── latency_report.py         # three columns (t_endpoint / t_server_v2v / t_v2v), CDF plot
│   ├── calibrate.py              # quadratically-weighted kappa, per dimension, bootstrap CI
│   ├── power_sim.py              # what N buys you; run before hand-labelling
│   ├── resources.py              # fills the RSS column in §6.5
│   └── scrub.py                  # scrub.yaml → transcripts + WAV bleeps; pre-commit check mode
├── data/
│   ├── calibration/              # 50 hand-scored answers + label passes
│   ├── fixtures/                 # replay WAVs
│   ├── fixtures/endpoint/        # 40 labelled endpointing clips + JSON sidecars
│   └── scrub.yaml                # literal → replacement map
├── runs/                         # JSONL per run; commit one sample, gitignore the rest
└── tests/
    ├── conftest.py               # FakeClock, FakePlayer, FakeLLM, FakeTTS
    ├── test_turn_controller.py   # cancellation, ordering, history commit
    ├── test_ladder.py            # rung switching, stickiness, no mid-stream switch
    ├── test_sentencizer.py       # golden files
    ├── test_frames.py            # Hypothesis property test for the 512-sample re-chunker
    ├── golden/sentencizer/*.jsonl
    └── integration/              # @pytest.mark.integration: real Groq, real ONNX, real device
```

### §6.4 Why not a framework

**Pipecat 1.10.0** (2026-09-12, BSD-2, 15.5k stars) and **LiveKit Agents 1.8.1** (2026-09-10,
Apache-2.0, 14.2k stars) are both excellent and both actively shipping. **Vocode is dead** —
last stable release 2024-06-17, last commit 2024-11-15 — do not use it; a hiring manager who
opens GitHub sees a 22-month-stale dependency.

For a *portfolio* project, hand-roll the turn-taking loop but **read Pipecat's source for the
defaults**, which are cited throughout this document. Pipecat's own default user-turn stop
strategy is literally `TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)` — the same
design recommended here. "I built the endpointing state machine and the cancellation path"
interviews better than "I configured a framework," and Pipecat is moving fast enough that its
`MinWordsInterruptionStrategy` API was deprecated and removed inside one minor cycle.

Do borrow one thing directly: Pipecat's `LocalSmartTurnAnalyzerV3` computes log-mel features
with a vendored numpy implementation, so smart-turn needs only `onnxruntime` + `numpy` + `soxr`
rather than pulling in `transformers`. §6.5 applies the same trick to Silero, and the result is a
process with exactly two inference runtimes.

---

### §6.5 Concurrency and resource model

Nothing above says which thread anything runs on, and four models — silero-vad, smart-turn ONNX,
Moonshine v2, Kokoro-82M — are all synchronous, and all four live in a `uvicorn` process whose
event loop is also reading the microphone socket. Call any of them on the loop and the WebSocket
reader stops; frames back up in the kernel buffer; the VAD sees them late and in bursts; the
endpointer fires at the wrong time. That is the entire mechanism behind "it worked yesterday and
today the demo is choppy."

**The rule: the event loop owns I/O and state. Every model call except one runs on a dedicated
single-worker executor. `asyncio.to_thread` is never used for inference.**

| Work | Thread | Why |
|---|---|---|
| WebSocket read/write, endpointer FSM, `TurnController`, `Sentencizer`, `TurnLog` | event loop | Pure Python, microseconds |
| Groq SSE stream (`httpx.AsyncClient`) | event loop | Genuinely async; never blocks |
| `player.enqueue()` / `abort()` | event loop | Deque ops and one `sd.RawOutputStream.abort()`, sub-ms |
| **Silero VAD** | **event loop, inline** | <1 ms per 32 ms frame = **3% duty cycle**. See below |
| smart-turn v3.2 ONNX | `turn_pool`, `max_workers=1` | 12–30 ms, once per candidate endpoint |
| Moonshine `add_audio()` / `update_transcription()` / `stop()` | `stt_pool`, `max_workers=1` | Stateful decoder; two concurrent calls corrupt the stream |
| Kokoro generation | `tts_pool`, `max_workers=1` | Sentences must synthesise in order; MLX saturates the GPU anyway |
| TurnLog file rotation, `bench/` WAV reads | default executor, capped at 4 | Genuinely incidental |

**Silero VAD is the one exception, and it is justified by its own measured number, not by
convenience.** It costs <1 ms per 512-sample frame on one CPU thread (§8 VAD table), frames
arrive every 32 ms, so it occupies 3% of the frame budget. Handing it to a thread would cost
~60–100 µs of executor scheduling per frame for no gain and — worse — a multi-worker pool
reorders frames, and a VAD that sees frames out of order is not a VAD. **Guard it:** record
`vad_ms` per frame in the turn log and alarm if p99 exceeds 5 ms. If it ever does, move it to
`turn_pool` (`max_workers=1` already guarantees FIFO).

**Why `max_workers=1` everywhere and not `asyncio.to_thread`.** `to_thread` uses the loop's
default executor, whose default size is `min(32, os.cpu_count() + 4)` — **12 threads on an
8-core Mac**. Twelve concurrent ONNX sessions on eight cores is how a 12 ms classifier becomes a
200 ms one. One worker per model also makes ordering a property of the type system rather than a
thing you hope for: `ThreadPoolExecutor` dispatches from a FIFO queue, so with a single worker,
submission order *is* execution order.

**Why three pools and not one.** `stt_pool` is busy continuously while the user speaks.
smart-turn fires *at the exact moment* the user stops — i.e. when `stt_pool`'s queue is deepest.
Sharing one worker would put the end-of-turn classification behind a backlog of `add_audio()`
calls and add tens of milliseconds to the single most latency-sensitive decision in the system.
It would also serialise the two operations §7 charges as `max(smart_turn, stt_final)` rather than
as a sum, which is worth 12–30 ms on every turn.

#### Thread counts inside each runtime

These must be set **before the first import of the library**, not after. `onnxruntime`'s default
`intra_op_num_threads=0` means "one thread per physical core", so two sessions silently create 16
threads on an 8-core machine.

```python
# src/coach/app.py — the first six lines of the file, above every other import
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")             # before onnxruntime import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # mlx-audio pulls transformers>=5.14
```

| Runtime | Setting | Value | Reason |
|---|---|---|---|
| onnxruntime — silero session | `intra_op_num_threads` / `inter_op_num_threads` | **1 / 1**, `ORT_SEQUENTIAL` | 2 MB model, 512 samples. Thread fan-out costs more than the op. Must not steal cores from the loop it runs on |
| onnxruntime — smart-turn session | `intra_op_num_threads` / `inter_op_num_threads` | **2 / 1** | The only ONNX graph big enough to benefit (Whisper-Tiny encoder over 8 s). Runs once per turn. Raise to 4 only if measured p95 > 40 ms |
| MLX (Kokoro) | `mx.set_default_device(mx.gpu)` | explicit | Dispatches to the Metal command queue; no CPU thread knob, and it does not contend with the ONNX sessions |
| PyTorch | **not imported** | — | See below |
| asyncio default executor | `loop.set_default_executor(ThreadPoolExecutor(max_workers=4))` | **4** | Overrides the 12-thread default |

**PyTorch is not on the shipped path, and that is a decision, not an accident.**
`silero-vad` 6.2.1 lists `torch>=1.12.0` and `torchaudio>=0.12.0` as *core* dependencies
(`onnxruntime` is only the `onnx-cpu` extra), and importing the package imports torch. So do not
import the package: load the `silero_vad.onnx` weights that ship inside the wheel into our own
`onnxruntime.InferenceSession`, the same way §6.4 borrows Pipecat's trick of running smart-turn on
`onnxruntime` + `numpy` + `soxr` instead of pulling in `transformers`. That leaves the process
with exactly two inference runtimes — onnxruntime and MLX — instead of three, and it removes
torch's own intra-op thread pool, which ignores `OMP_NUM_THREADS` and sizes itself to `cpu_count`.
Moonshine v2 ships no PyTorch (§4) and `mlx-audio` 0.5.3's dependency list contains no torch
either, so nothing else drags it in.

```python
# src/coach/turn/vad.py
import importlib.resources as res, onnxruntime as ort
_ONNX = next(p for p in res.files("silero_vad").rglob("*.onnx"))   # wheel ships .jit and .onnx
_OPTS = ort.SessionOptions()
_OPTS.intra_op_num_threads = 1
_OPTS.inter_op_num_threads = 1
_OPTS.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
_SESSION = ort.InferenceSession(str(_ONNX), _OPTS, providers=["CPUExecutionProvider"])
```

If that resource lookup fails on a future release, the documented fallback is
`silero_vad.load_silero_vad(onnx=True)` — correct, but it imports torch, so add
`torch.set_num_threads(2)` and `torch.set_num_interop_threads(1)` at the very top of `app.py`.
Both raise `RuntimeError` if called after any parallel work has started, so they cannot go
anywhere else.

#### Wiring

```python
# src/coach/app.py
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=4, thread_name_prefix="misc"))
    app.state.stt_pool  = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
    app.state.turn_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turn")
    app.state.tts_pool  = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    # Warm-up (§7 optimisation #5). MLX compiles Metal kernels on first use; ORT allocates
    # its arenas on first run. Pay both now, not in front of an audience.
    await loop.run_in_executor(app.state.turn_pool, smart_turn.infer, np.zeros(128_000, "f4"))
    await loop.run_in_executor(app.state.stt_pool,  stt.warmup)            # 1 s of silence
    async for _ in tts.synthesize("Ready."):                               # discard
        pass
    await prerender_fixed_clips()          # 21 WAVs to runs/tts_cache/ (§9.6)
    await groq.get("/openai/v1/models")    # opens the keepalive connection AND validates the ID
    await local_llm.warm()                 # rung 1 weights resident before it is ever needed
    yield
    for p in (app.state.stt_pool, app.state.turn_pool, app.state.tts_pool):
        p.shutdown(wait=False, cancel_futures=True)
```

**Run `uvicorn` with `--workers 1 --loop uvloop`.** More than one worker means more than one copy
of every model in RAM, and the table below shows there is no room for two.

#### Backpressure: TTS runs *ahead* of playback, not behind

Kokoro's reported RTF is 0.08 (§5) — it produces roughly twelve seconds of audio per second of
compute. Left alone, the synthesis loop will race a whole 120-token reply into the player queue
in under a second. Three costs: memory, GPU spent on audio a barge-in will throw away, and a
large window of *generated but unheard* text that the commit-truncation rule then has to undo.

**Rule: the synthesis pipeline may run at most `MAX_LOOKAHEAD_S = 2.0` seconds of audio ahead of
the playout head.** `AudioPlayer.enqueue()` awaits an `asyncio.Event` while
`queued_seconds() > 2.0`. Backpressure then propagates for free — the `async for chunk in
tts.synthesize(...)` loop stalls, which stalls the sentencizer, which stalls consumption of the
Groq SSE stream.

Why 2.0 s: long enough to absorb one full sentence of synthesis jitter (a 30-word clause is
~9 s of audio but only ~700 ms of Kokoro compute at RTF 0.08, so the queue never underruns),
and short enough that a barge-in discards at most two seconds of wasted work. It is also why
the sentencizer's hard cap exists — see §6.6.

Two correctness details:

1. **Check the ceiling *before* appending, never after.** A single chunk larger than the
   ceiling must still be accepted, or a long sentence deadlocks on its own first chunk.
2. **The `await` on that Event is the cancel point.** `asyncio.CancelledError` lands there
   instantly on barge-in. Blocking in a thread instead would make the cancel wait for a
   synthesis call to return — which is exactly the failure §8.2 is written to prevent.

Stalling the SSE consumer holds the Groq connection open longer than strictly necessary. See
§14 item 6 — whether tokens generated before you close a stream still count against TPM is
undocumented. This is a reason to close the stream promptly on barge-in, which §8.2's controller
does.

#### Resource table

The demo Mac is assumed to be an 8-core M-series (4 performance + 4 efficiency),
`os.cpu_count() == 8`. Scale the core column linearly; the RAM column does not scale.

| Component | Runtime | Weights on disk | **RSS (estimate)** | Threads |
|---|---|---|---|---|
| Silero VAD 6.2.1 | onnxruntime | **~2 MB** (§8, verified) | ~25 MB | inline on loop, intra-op 1 |
| smart-turn v3.2 int8 | onnxruntime | **8.68 MB** (§8, verified) | ~60 MB | `turn_pool` ×1, intra-op 2 |
| Moonshine v2 Medium Streaming | bundled native (no torch, no MLX) | **245 MB / 245M params** (arXiv 2602.12241) | ~450 MB | `stt_pool` ×1 |
| Kokoro-82M | MLX → Metal | **~163 MB bf16** (82M params); the ONNX build is ~326 MB fp32 / ~92 MB int8 | ~500 MB incl. MLX + `transformers` import | `tts_pool` ×1 + GPU |
| Python 3.11 + FastAPI + uvicorn + httpx + numpy + soxr | — | — | ~120 MB | loop |
| **Peak total** | | **≈ 420 MB** | **≈ 1.15 GB** | **≈ 6 runnable threads at peak, 3 steady-state** |

**The RSS column is an estimate — weights plus typical runtime overhead — not a measurement.**
No source measures resident memory for any of these four on Apple Silicon, and inventing four
numbers to two significant figures would be worse than saying so. Fill it in before the README
quotes it:

```python
# bench/resources.py
import psutil, os, time
p = psutil.Process(os.getpid()); base = p.memory_info().rss
for name, load in [("vad", load_vad), ("smart_turn", load_turn),
                   ("moonshine", load_stt), ("kokoro", load_tts)]:
    before = p.memory_info().rss; load(); time.sleep(0.5)
    print(f"{name:12} +{(p.memory_info().rss - before) / 2**20:7.1f} MiB")
print(f"{'total':12}  {(p.memory_info().rss - base) / 2**20:7.1f} MiB over a bare interpreter")
```

**Core budget at peak** — a barge-in landing mid-reply, the worst concurrent moment: loop thread
(VAD + socket, ~5% duty), `stt_pool` decoding, `turn_pool` classifying with 2 intra-op threads,
`tts_pool` dispatching to Metal. Six runnable threads, eight cores, two cores of headroom.
Compare the naive default: two ORT sessions at `intra_op_num_threads=0` (8 + 8) plus a
12-thread default executor plus torch's 8 = **36 threads on 8 cores**.

**This table is also the arithmetic behind §6.8's deployment decision.** Render's free instance is
0.1 CPU / 512 MB. ~1.15 GB of models does not fit in 512 MB and four inference threads do not fit
in 0.1 CPU — that, not a hunch, is why the hosted build must be a thin relay with browser-side
VAD. State the number in the README rather than the adjective.

### §6.6 `sentence.py`: the incremental sentencizer

§7 ranks this as the #2 latency optimisation, worth 500–2,000 ms. It sits on the critical path
between the LLM stream and the first audible word, and naive `text.split(". ")` breaks on every
one of these, all of which a coach actually says: *"a 3.5 second pause"*, *"Dr. Chen"*,
*"i.e. the measurable outcome"*, *"wait... say more"*, *"two things: first, ..."*, and a reply
whose last sentence has no trailing space.

#### Rules, in priority order

1. **Terminals are `. ! ?`**, optionally followed by a run of closers `" ' ) ] } ” ’ »`.
2. **A terminal at end-of-buffer is never a boundary.** This is the streaming rule and it
   subsumes most of the decimal problem: when delta *n* ends `"a 3."` and delta *n+1* begins
   `"5 second pause"`, holding the boundary until a following character arrives costs one token
   (~2 ms at ~440 t/s) and prevents speaking `"a 3."` aloud. **Consequence: the final sentence of
   every reply is always held back, so `flush()` on stream end is mandatory, not defensive. Omit
   it and every reply loses its last sentence.**
3. **The following character must be whitespace.** Kills `3.5`, `v1.2`, URLs, `$4.5`.
4. **Digit-period-digit guard** for the case a whitespace slips in: `"Section 4. 3 was"`.
5. **Abbreviation exception list**, matched on the whitespace-delimited token before the period,
   lowercased. `"e.g."` → token `"e.g"`; `"Dr."` → `"dr"`.
6. **Single-letter guard**: a one-character alphabetic token before a period is an initial
   (`"J. Smith"`). One line, covers every initial without enumerating any.
7. **Bare-number guard**: token before the period is all digits → list marker (`"1."`), not a
   sentence.
8. **Ellipsis is a continuation, never a boundary** — `"..."` and `"…"` both. An ellipsis marks
   trailing off, which is precisely when a full-stop prosody and a synthesis seam are wrong:
   *"Wait... say more about that"* is one utterance, and splitting it yields a clipped one-word
   audio chunk. The failure direction is one fewer split, which is the cheap direction.
9. **Opening-clause break**, first emitted chunk of a turn only: first `, : ;` **at index ≥ 40**
   followed by whitespace. That is the strict reading of §7 optimisation #2's "first comma past
   ~40 chars" — the first clause is guaranteed to be at least 40 characters, not merely "the
   first comma once the buffer passes 40." **Only the first chunk** — once ~1.5 s of audio is
   queued there is no latency left to buy, and splitting every subsequent comma makes Kokoro emit
   choppy fragments with falling intonation. The cut lands *after* the comma so the first clause
   keeps its continuation prosody.
10. **Forced split at `MAX_CHARS = 180`**, at the last whitespace before 180, never mid-word.
    180 chars ≈ 30 words ≈ 9 s of speech at the 196 WPM baseline §9 already cites — deliberately
    longer than the 2.0 s backpressure window, so the cap is a stall guard, not a prosody knob.
    Without it, `max_completion_tokens=120` of unpunctuated run-on stalls audio until the stream
    closes.
11. **`flush()` returns the stripped remainder and resets.** Always called, on the normal path
    and after a Groq error.

Deliberately not implemented: quote-internal sentence detection, and abbreviation
disambiguation by capitalisation of the next word. Both are ambiguous even for humans, and the
failure mode is one *fewer* split — a longer chunk, never a wrong transcript. The known cost is
an abbreviation that legitimately ends a sentence: *"We shipped at 9 a.m. The rollback was at
noon"* stays one chunk. That is the right trade; the alternative mis-splits *"9 a.m. was the
deploy window."*

**`no` is deliberately absent from `ABBREV`.** It is tempting to include it for `"No. 4"`, but
"no" ending a sentence is enormously more common in speech than "No." meaning "number", and
including it silently swallows the boundary in *"Dr. Chen said no. What did you do?"*

#### Implementation

```python
# src/coach/sentence.py
TERMINALS = ".!?"
CLOSERS   = "\"')]}”’»"
ABBREV = {"mr","mrs","ms","dr","prof","sr","jr","st","vs","etc","e.g","i.e","approx",
          "inc","ltd","co","dept","fig","al","cf","u.s","u.k","ph.d","a.m","p.m",
          "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec"}   # NOT "no"
MAX_CHARS, OPENING_BREAK_MIN = 180, 40


class Sentencizer:
    def __init__(self) -> None:
        self.buf, self.emitted = "", 0

    def push(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while (cut := self._boundary()) is not None:
            piece, self.buf = self.buf[:cut].strip(), self.buf[cut:].lstrip()
            if piece:
                out.append(piece)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        """MANDATORY on stream end — rule 2 guarantees the last sentence is still buffered."""
        tail, self.buf = self.buf.strip(), ""
        if tail:
            self.emitted += 1
        return [tail] if tail else []

    def _boundary(self) -> int | None:
        b = self.buf
        for i, c in enumerate(b):
            if c not in TERMINALS:
                continue
            j = i + 1
            while j < len(b) and b[j] in CLOSERS:
                j += 1
            if j >= len(b):                     # rule 2: hold; more tokens may follow
                break
            if b[j].isspace() and not _suppressed(b, i):
                return j
        if self.emitted == 0:                   # rule 9: opening-clause break
            for k in range(OPENING_BREAK_MIN, len(b) - 1):
                if b[k] in ",:;" and b[k + 1].isspace():
                    return k + 1
        if len(b) > MAX_CHARS:                  # rule 10: forced split, never mid-word
            k = b.rfind(" ", 0, MAX_CHARS)
            return k + 1 if k > 0 else MAX_CHARS
        return None


def _suppressed(b: str, i: int) -> bool:
    """True if the terminal at b[i] is not a sentence end."""
    if b[i] != ".":
        return False                                    # ! and ? are never abbreviations
    if b[i - 1:i] == "." or b[i + 1:i + 2] == ".":
        return True                                     # rule 8: "..." is a continuation
    prev = b[:i]
    if prev[-1:].isdigit() and b[i + 1:].lstrip()[:1].isdigit():
        return True                                     # rule 4: "Section 4. 3 was"
    tok = prev.rsplit(maxsplit=1)[-1].lower() if prev.strip() else ""
    return (tok in ABBREV                               # rule 5
            or tok.endswith("…")                        # rule 8: unicode ellipsis
            or (len(tok) == 1 and tok.isalpha())        # rule 6: "J. Smith"
            or tok.isdigit())                           # rule 7: "1." list marker
```

**This implementation was executed against the fixtures below, not just written.** All 19 cases
produce the intended split, and — the property that actually matters — feeding each case one, two,
three, five and seven characters at a time produces byte-identical output to feeding it whole.
Two bugs surfaced only by running it: `"no"` in `ABBREV` swallowed the boundary in *"Dr. Chen said
no. What did you do?"*, and treating `…` as a terminal emitted *"Wait..."* as a standalone
one-word audio chunk. Both fixes are in the code above.

Rescanning the whole buffer on every delta is deliberate: the buffer never exceeds ~180
characters and deltas arrive at ~440 tokens/s, so this is under 100 K character comparisons per
second — three orders of magnitude below anything that matters, and an incremental scan index is
the kind of state that breaks exactly once, in front of an audience.

`tests/test_sentence.py` — the fixture list, with verified expected output:

| Input | Expected chunks |
|---|---|
| `That was a 3.5 second pause, which is fine. Tell me the outcome.` | 2 — the decimal does not split |
| `Dr. Chen said no. What did you do?` | 2 |
| `Name the metric, i.e. the measurable outcome. Then stop.` | 2 |
| `Wait... say more about that. Please.` | 2 — the ellipsis does not split |
| `Two things: first, you never said what the target was, and second, ...` | 2, cut after `was,` (opening-clause rule) |
| `1. Ship it. 2. Measure it.` | 2 — list markers do not split |
| `J. Smith owned the rollout. Who owned the rollback?` | 2 |
| `She said "that is the number." Then she left.` | 2 — closer run handled |
| `Revenue grew $4.5 million that quarter. Good.` | 2 |
| `It was a U.S. customer, not an EU one. Fine.` | 2 |
| `Really?! Tell me more about the constraint you hit.` | 2 |
| `I don't know… tell me what you would try next time.` | 1 |
| `(He owned it.) She reviewed it. Good.` | 3 |
| `We shipped at 9 a.m. The rollback was at noon.` | **1 — accepted limitation, see above** |
| `Done.` / `No.` / `""` | 1 / 1 / 0 |
| 200-char run-on, no punctuation | 2, cut at the last space before 180 |

**And every one of the above re-run split across delta boundaries at each character position.**
That last test is the one that finds bugs — it is what proves rule 2 actually holds a terminal at
end-of-buffer instead of speaking `"a 3."` aloud.

### §6.7 Instrumentation: two clocks, one timeline

Every mark in §8.2's reference implementation is a `time.monotonic()` reading taken inside the
Python process. "First audible word" happens in the browser, on a different clock, after a
WebSocket hop, a jitter buffer and a DAC. **`time.monotonic()` and `performance.now()` have
unrelated, arbitrary epochs — they cannot be subtracted.** Reconcile them explicitly or do not
quote `t_v2v`.

#### Clock offset: NTP-style, over the socket that already exists

Four timestamps per probe. The client owns t1 and t4, the server stamps t2 and t3:

```
offset_ms = ((t2 - t1) + (t3 - t4)) / 2      # add to a client performance.now() to get server ms
rtt_ms    = (t4 - t1) - (t3 - t2)
```

Both formulas assume the uplink and downlink delays are equal. They are not, exactly, and the
residual error is bounded by **±rtt_min/2**. That bound is the honest uncertainty on every
client-side mark, and it is why the headline number is measured on the local build:

| Deployment | `rtt_min_ms` | Offset uncertainty | Effect on a ~1,230 ms claim |
|---|---|---|---|
| Local (loopback WS) | 0.3–1.5 | ±0.2–0.8 ms | Negligible |
| Render, home broadband | 30–90 | **±15–45 ms** | ±1.2–3.7% — quote `t_v2v ± 45 ms` or quote `t_server_v2v` |

**Probe schedule:** a burst of **5 probes 200 ms apart on WS open**, then **1 probe at the start of
every turn** (fired at END_TURN, so it is in flight during the LLM call and costs nothing). Keep the
**minimum-RTT sample** from the last 30 s — the least-delayed probe suffered the least asymmetric
queuing, so it carries the least biased offset. Log `clock_offset_age_ms` with every turn; if it
exceeds 30,000 ms the turn's client marks are marked `clock_stale: true` and excluded from `t_v2v`
percentiles.

```python
# src/coach/metrics/clock.py
from __future__ import annotations

import time
from dataclasses import dataclass


def server_ms() -> float:
    """The one server clock. Never time.time() — it can step."""
    return time.monotonic() * 1000.0


@dataclass(frozen=True)
class ClockOffset:
    offset_ms: float        # add to a client performance.now() value -> server ms
    rtt_ms: float           # round-trip of the probe this came from
    taken_server_ms: float

    @property
    def uncertainty_ms(self) -> float:
        return self.rtt_ms / 2.0


class ClockSync:
    """NTP-style offset between the browser's performance.now() and the
    server's time.monotonic(), carried on the audio WebSocket.

    Control frames (JSON, same socket as the PCM):
      c->s {"type":"clock_ping","seq":int,"t1_client_ms":float}
      s->c {"type":"clock_pong","seq":int,"t1_client_ms":float,
            "t2_server_ms":float,"t3_server_ms":float}
      c->s {"type":"clock_offset","seq":int,"offset_ms":float,"rtt_ms":float}

    The client computes the offset because only it holds t1 and t4.
    The server stamps t2/t3 and keeps the best estimate it is told about.
    """

    MAX_AGE_MS = 30_000.0

    def __init__(self) -> None:
        self._best: ClockOffset | None = None

    def on_ping(self, msg: dict) -> dict:
        t2 = server_ms()
        return {
            "type": "clock_pong",
            "seq": msg["seq"],
            "t1_client_ms": msg["t1_client_ms"],
            "t2_server_ms": t2,
            "t3_server_ms": server_ms(),   # stamped last, after serialisation work
        }

    def on_offset(self, msg: dict) -> None:
        cand = ClockOffset(float(msg["offset_ms"]), float(msg["rtt_ms"]), server_ms())
        # Minimum-RTT filter, and always accept if the incumbent has expired.
        if self._best is None or self.is_stale or cand.rtt_ms < self._best.rtt_ms:
            self._best = cand

    @property
    def is_stale(self) -> bool:
        return (self._best is None
                or server_ms() - self._best.taken_server_ms > self.MAX_AGE_MS)

    def to_server_ms(self, client_ms: float) -> float | None:
        """Convert a browser performance.now() reading to the server clock."""
        return None if self._best is None else client_ms + self._best.offset_ms

    def snapshot(self) -> dict:
        if self._best is None:
            return {"clock_offset_ms": None, "clock_rtt_min_ms": None,
                    "clock_offset_age_ms": None, "clock_stale": True}
        return {
            "clock_offset_ms": round(self._best.offset_ms, 3),
            "clock_rtt_min_ms": round(self._best.rtt_ms, 3),
            "clock_offset_age_ms": round(server_ms() - self._best.taken_server_ms, 1),
            "clock_stale": self.is_stale,
        }
```

```js
// web/clock.js
export class ClockSync {
  constructor(ws) {
    this.ws = ws;
    this.seq = 0;
    this.offsetMs = null;    // add to performance.now() -> server ms
    this.rttMs = Infinity;
    this.takenAt = -Infinity;
  }

  // 5 probes 200 ms apart on open; 1 per turn thereafter.
  async burst(n = 5, gapMs = 200) {
    for (let i = 0; i < n; i++) { this.probe(); await new Promise(r => setTimeout(r, gapMs)); }
  }

  probe() {
    this.ws.send(JSON.stringify({
      type: "clock_ping", seq: ++this.seq, t1_client_ms: performance.now(),
    }));
  }

  onPong(m) {
    const t4 = performance.now();
    const rtt = (t4 - m.t1_client_ms) - (m.t3_server_ms - m.t2_server_ms);
    const offset = ((m.t2_server_ms - m.t1_client_ms) + (m.t3_server_ms - t4)) / 2;
    const stale = performance.now() - this.takenAt > 30000;
    if (this.offsetMs === null || stale || rtt < this.rttMs) {
      this.offsetMs = offset; this.rttMs = rtt; this.takenAt = performance.now();
      this.ws.send(JSON.stringify({
        type: "clock_offset", seq: m.seq, offset_ms: offset, rtt_ms: rtt,
      }));
    }
  }

  toServerMs(clientMs) { return this.offsetMs === null ? null : clientMs + this.offsetMs; }
}
```

#### Getting "first audible word" right in the browser

Do not let the audio free-run and stamp `performance.now()` when you enqueue — that is the same
error as stamping on the server, moved 2 ms closer. **Schedule every buffer explicitly** and convert
the scheduled time out of the audio clock:

```js
// web/app.js — playback, 24 kHz mono from Kokoro (int16 on the wire, converted on arrival)
const ctx = new AudioContext({ latencyHint: "interactive", sampleRate: 24000 });
const JITTER_S = 0.040;          // row 9 of the §7 budget
let nextStart = 0;
let turnMarks = null;

function enqueuePcm(f32, isFirstOfTurn) {
  const buf = ctx.createBuffer(1, f32.length, 24000);
  buf.copyToChannel(f32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(loopbackDest);     // the RTCPeerConnection sink, so AEC sees the bot's voice
  if (nextStart < ctx.currentTime + JITTER_S) nextStart = ctx.currentTime + JITTER_S; // re-arm after underrun
  src.start(nextStart);

  if (isFirstOfTurn) {
    // Two mappings out of the audio clock, and the difference between them matters.
    // getOutputTimestamp() maps to performance.now() AT THE DEVICE OUTPUT: contextTime
    // is the frame the device is playing, performanceTime is when it got there. So that
    // mapping ALREADY contains outputLatency — adding ctx.outputLatency on top of it is
    // a ~25 ms double-count. ctx.currentTime is the graph clock and does not.
    const ts = ctx.getOutputTimestamp();
    const K_out   = ts.performanceTime - ts.contextTime * 1000;   // -> audible at the device
    const K_graph = performance.now()  - ctx.currentTime  * 1000; // -> graph, no device latency
    turnMarks.client_first_audio_scheduled_ms = nextStart * 1000 + K_graph;
    turnMarks.client_first_audio_played_ms    = nextStart * 1000 + K_out;
    // Free in-browser consistency check: these two should agree to a few ms. If they
    // don't, one of the mappings is wrong and every t_v2v is biased by the difference.
    turnMarks.output_latency_ms         = (ctx.outputLatency ?? 0) * 1000;
    turnMarks.output_latency_implied_ms = K_out - K_graph;
    turnMarks.base_latency_ms  = ctx.baseLatency * 1000;
    turnMarks.jitter_buffer_ms = JITTER_S * 1000;
    ws.send(JSON.stringify({ type: "client_marks", turn_id: turnMarks.turn_id, ...turnMarks }));
  }
  nextStart += buf.duration;
}
```

Three things this gets right that the naive version does not:

1. **`getOutputTimestamp()` already includes device latency.** `contextTime` is the frame the device
   is rendering; `performanceTime` is when it got there. Adding `outputLatency` on top is a
   ~25 ms double-count. The code derives the same scheduled instant through both mappings and logs
   `output_latency_implied_ms = K_out − K_graph`; assert in `bench/latency_report.py` that its median
   agrees with the browser's own `output_latency_ms` to within 5 ms. That catches the double-count in
   the browser, before the loopback calibration has to.
2. **Fall back explicitly.** `getOutputTimestamp()` and `outputLatency` are Chrome/Firefox; Safari
   support is not verified here. If `getOutputTimestamp` is missing, use
   `performance.now() + (nextStart - ctx.currentTime) * 1000 + (ctx.outputLatency ?? ctx.baseLatency) * 1000`
   and set `client_clock_method: "fallback"` on the row so those turns are separable in analysis.
   The demo browser is Chrome; pin it.
3. **The mic side has no equivalent API.** There is no `inputLatency`. Stamp the AudioWorklet's own
   `currentTime` at the start of the quantum that closed each 512-sample frame, ship it with the
   frame, and accept that the hardware capture delay underneath it is unmeasurable from JavaScript.
   That residual is exactly what §10's room recording bounds.

```js
// web/mic-worklet.js — inside process(), after the 512th sample lands
this.port.postMessage({ frame: int16, capture_ctx_s: currentTime }, [int16.buffer]);
```

#### The mark table — every field, unit, and clock

One JSONL row per turn, `schema: "turnlog/v2"`. Every `*_ms` field is **float milliseconds relative
to `t0`**, where `t0` = server receipt of the last mic frame Silero scored as speech. Client marks
are converted to the server clock with `ClockSync.to_server_ms()` before being written, and the raw
client value is kept alongside so the conversion is auditable.

| Field | Unit | Clock | Taken when |
|---|---|---|---|
| `schema` | `"turnlog/v2"` | — | — |
| `session_id`, `turn_id` | str (uuid4) | — | WS open / turn start |
| `t0_server_monotonic_s` | float s | server `time.monotonic()` | absolute anchor for every `*_ms` below |
| `t0_wall_utc` | ISO-8601 str | server `time.time()` | correlating with the loopback recording |
| `frame_quantum_ms` | 32.0 | — | constant; the acoustic uncertainty on `t0` |
| **Capture** | | | |
| `mic_frame_capture_ctx_s` | float s | client `AudioContext.currentTime` | worklet quantum that closed the last speech frame |
| `mic_frame_sent_ms` | float ms | **client** → server | `performance.now()` at `ws.send()` of that frame |
| `mic_frame_sent_client_raw_ms` | float ms | client, unconverted | same instant, pre-offset |
| `uplink_ms` | float ms | derived | `0 - mic_frame_sent_ms`; main-thread jitter + WS transit |
| **Endpointing** | | | |
| `vad_silence_armed_ms` | float ms | server | CANDIDATE_END entered (`stop_secs` elapsed) |
| `smart_turn_started_ms`, `smart_turn_done_ms` | float ms | server | around the ONNX call |
| `smart_turn_p` | float 0–1 | — | classifier output; also rendered live in the HUD |
| `turn_end_ms` | float ms | server | END_TURN declared |
| **STT** | | | |
| `stt_force_update_ms` | float ms | server | `update_transcription(FORCE_UPDATE)` issued at CANDIDATE_END |
| `stt_flush_started_ms` | float ms | server | flush protocol entered at END_TURN (§8.1) |
| `stt_final_ms` **or** `stt_flush_timeout_ms` | float ms | server | complete line delivered, or the 250 ms ceiling fired |
| `stt_used_partial` | bool | — | true means the transcript sent to the LLM was not final |
| `stt_engine_latency_ms` | int ms | **Moonshine** | `TranscriptLine.last_transcription_latency_ms` |
| `stt_late_finals` | list[str] | — | finals that arrived after `seal()`; free accuracy metric (§8.1) |
| `stt_model_arch`, `stt_update_interval_s` | str, float | — | so runs at 0.25 and 0.5 are separable |
| **LLM** | | | |
| `llm_request_sent_ms` | float ms | server | `httpx` request written |
| `llm_first_token_ms` | float ms | server | first SSE delta carrying any content |
| `llm_first_speakable_token_ms` | float ms | server | first delta that is answer text, not reasoning |
| `llm_first_sentence_ms` | float ms | server | sentencizer emits sentence 1 |
| `llm_rung` | str | — | which ladder rung served this turn (§11.1). **Never average rungs together** |
| `groq_tokens_remaining`, `groq_queue_time_ms` | int, float | Groq headers | `x-ratelimit-remaining-tokens`, `usage.queue_time` |
| `prompt_tokens`, `cached_tokens`, `billable_prompt_tokens` | int | Groq usage | §3.1's `CacheMeter` |
| **TTS + downlink** | | | |
| `tts_request_ms`, `tts_first_byte_ms` | float ms | server | Kokoro call issued / **first** ~50 ms chunk returned |
| `ws_first_pcm_sent_ms` | float ms | server | first PCM chunk written to the socket |
| `audio_first_frame_out_ms` | float ms | server, PortAudio callback | **LOCAL build only.** First frame actually handed to the device. This is `t_v2v`'s endpoint when audio plays on the server |
| `client_first_pcm_recv_ms` | float ms | **client** → server | `performance.now()` in `ws.onmessage` for that chunk |
| `client_first_audio_scheduled_ms` | float ms | **client** → server | `nextStart*1000 + K_graph` — graph clock, excludes device latency |
| `client_first_audio_played_ms` | float ms | **client** → server | `nextStart*1000 + K_out` — device output clock; **`t_v2v`'s endpoint in the browser build** |
| `output_latency_implied_ms` | float ms | derived, client | `K_out − K_graph`; must match `output_latency_ms` |
| `client_clock_method` | `"getOutputTimestamp"` \| `"fallback"` | — | which path produced the two marks above |
| `output_latency_ms`, `base_latency_ms`, `jitter_buffer_ms` | float ms | client | diagnostics |
| **Clock reconciliation** | | | |
| `clock_offset_ms`, `clock_rtt_min_ms`, `clock_offset_age_ms` | float ms | derived | from `ClockSync.snapshot()` |
| `clock_stale` | bool | — | true → exclude from `t_v2v` percentiles |
| **Derived headline numbers** | | | |
| `t_endpoint_ms` | float ms | server only | `turn_end_ms` |
| `t_server_v2v_ms` | float ms | server only | `ws_first_pcm_sent_ms` |
| `t_v2v_ms` | float ms | **mixed** | `client_first_audio_played_ms`, or `audio_first_frame_out_ms` on the LOCAL build |
| `t_v2v_uncertainty_ms` | float ms | — | `clock_rtt_min_ms/2 + frame_quantum_ms/2` |
| `last_mile_ms` | float ms | derived | `t_v2v_ms - t_server_v2v_ms` — WS + jitter + DAC |
| **Barge-in** | | | |
| `barge_in_detected_ms`, `audio_abort_ms` | float ms | server | decision, and `player.abort()` returning |
| `turn_unwound_ms` | float ms | server | the cancelled task's `finally` completed; not user-visible |
| `client_audio_silent_ms` | float ms | **client** → server | last scheduled buffer's end, or the cancel instant |

```python
# src/coach/metrics/turn_log.py
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coach.metrics.clock import ClockSync, server_ms

CLIENT_MARKS = {
    "mic_frame_sent_ms",
    "client_first_pcm_recv_ms",
    "client_first_audio_scheduled_ms",
    "client_first_audio_played_ms",
    "client_audio_silent_ms",
}
FRAME_QUANTUM_MS = 32.0


class TurnLog:
    """One row per turn. All *_ms fields are float ms relative to t0."""

    def __init__(self, session_id: str, clock: ClockSync, path: Path):
        self.session_id = session_id
        self.turn_id = str(uuid.uuid4())
        self.clock = clock
        self.path = path
        self.t0_monotonic_ms: float | None = None
        self.fields: dict[str, Any] = {}

    def set_t0(self) -> None:
        """Called with the last mic frame Silero scored as speech."""
        self.t0_monotonic_ms = server_ms()
        self.fields["t0_server_monotonic_s"] = self.t0_monotonic_ms / 1000.0
        self.fields["t0_wall_utc"] = datetime.now(timezone.utc).isoformat()

    def mark(self, name: str) -> None:
        self.fields[name] = server_ms() - self.t0_monotonic_ms

    def mark_once(self, name: str) -> None:
        if name not in self.fields:
            self.mark(name)

    def set(self, name: str, value: Any) -> None:
        self.fields[name] = value

    def merge_client(self, msg: dict) -> None:
        """Fold a client_marks control frame in, converting each timestamp
        from performance.now() to the server clock, then to t0-relative ms.
        The raw client value is kept so the conversion can be re-derived."""
        for name, raw in msg.items():
            if name not in CLIENT_MARKS or raw is None:
                continue
            self.fields[f"{name.removesuffix('_ms')}_client_raw_ms"] = raw
            converted = self.clock.to_server_ms(float(raw))
            if converted is not None:
                self.fields[name] = converted - self.t0_monotonic_ms
        for passthrough in ("output_latency_ms", "output_latency_implied_ms",
                            "base_latency_ms", "jitter_buffer_ms", "client_clock_method"):
            if passthrough in msg:
                self.fields[passthrough] = msg[passthrough]

    def _derive(self) -> None:
        f = self.fields
        f["t_endpoint_ms"] = f.get("turn_end_ms")
        f["t_server_v2v_ms"] = f.get("ws_first_pcm_sent_ms")
        if f.get("mic_frame_sent_ms") is not None:
            f["uplink_ms"] = -f["mic_frame_sent_ms"]     # t0 is zero by definition
        # Row 1 of the §7 budget — everything BEFORE t0. `t0` is server receipt of the last speech
        # frame, so an ACOUSTIC t_v2v has to add the audio's age at that instant: the mean unclosed
        # part of the 32 ms worklet quantum, plus the uplink. Both were already logged and neither
        # was added to anything — which is why the tooling produced ~1,200 ms for a figure §7
        # DEFINES as ~1,230 ms. Calibration A (§10) exists to check this term against a microphone.
        f["pre_t0_ms"] = (f.get("uplink_ms") or 0.0) + FRAME_QUANTUM_MS / 2.0
        endpoint = f.get("client_first_audio_played_ms") or f.get("audio_first_frame_out_ms")
        # Server-anchored: t0-relative, the right basis for A/B-ing server-side work between runs.
        f["t_v2v_server_anchored_ms"] = endpoint
        # Published figure: acoustic end of speech → first audible sample. This is the headline.
        f["t_v2v_ms"] = None if endpoint is None else endpoint + f["pre_t0_ms"]
        if endpoint is not None and f.get("t_server_v2v_ms") is not None:
            # Rows 8–10 only: WS transit + jitter buffer + device out. Both terms are t0-relative,
            # so pre_t0 must NOT appear here — that was the old bug's second symptom.
            f["last_mile_ms"] = endpoint - f["t_server_v2v_ms"]
        rtt = f.get("clock_rtt_min_ms")
        if rtt is not None:
            f["t_v2v_uncertainty_ms"] = rtt / 2.0 + FRAME_QUANTUM_MS / 2.0

    # Fail-safe split (§6.10 item 4). runs/*.jsonl is COMMITTED, so it may hold only machine
    # values: numbers, booleans, and a short allow-list of identifier strings. Any other string —
    # every transcript field included — goes to runs/*.text.jsonl, which .gitignore excludes
    # wholesale. Deliberately an allow-list on the STRING side only: a new numeric metric needs no
    # registration, but a new text field cannot reach the committed file by forgetting one.
    SAFE_STR_KEYS = frozenset({
        "t0_wall_utc", "build", "llm_provider", "llm_model", "llm_rung",
        "stt_provider", "tts_provider", "stt_flush_outcome",
    })

    def _committable(self, key: str, value: object) -> bool:
        if isinstance(value, bool) or isinstance(value, (int, float)) or value is None:
            return True
        if isinstance(value, str):
            return key in self.SAFE_STR_KEYS
        if isinstance(value, (list, tuple)):
            return not any(isinstance(x, str) for x in value)
        return False                                   # dicts, objects: never committed

    def write(self) -> None:
        self.fields.update(self.clock.snapshot())
        self.fields["frame_quantum_ms"] = FRAME_QUANTUM_MS
        self._derive()
        head = {"schema": "turnlog/v2", "session_id": self.session_id,
                "turn_id": self.turn_id}
        safe, private = {}, {}
        for k, v in self.fields.items():
            (safe if self._committable(k, v) else private)[k] = v
        with self.path.open("a") as fh:                          # runs/*.jsonl — committable
            fh.write(json.dumps({**head, **safe}, separators=(",", ":")) + "\n")
        if private:                                              # runs/*.text.jsonl — gitignored
            with self.path.with_suffix(".text.jsonl").open("a") as fh:
                fh.write(json.dumps({**head, **private}, separators=(",", ":")) + "\n")
```

#### Analysis rules for `bench/latency_report.py`

- Report **`t_endpoint`, `t_server_v2v`, and `t_v2v` as three separate columns**, each with P50/P95,
  N, and a bootstrap 95% CI. Never collapse them into one "latency".
- **Split every distribution by `llm_rung`.** A rung-0 turn and a rung-1 turn are different
  configurations; averaging them into one P95 is the same error as averaging cold and warm runs.
- **Exclude rows where `clock_stale` is true** from `t_v2v` percentiles, and print how many were
  excluded. If that count is more than ~5% of N, the probe schedule is broken.
- Print `median(t_v2v_uncertainty_ms)` next to the `t_v2v` percentiles. A 1,230 ms P50 with a
  ±0.5 ms uncertainty is a measurement; the same P50 with ±45 ms is an estimate, and the README must
  say which one it is.
- Print `last_mile_ms` as its own distribution. It is the number the old instrumentation could not
  see, and its P95 is what a hiring manager who has built a voice agent will ask about.
- Assert `median(output_latency_implied_ms) ≈ median(output_latency_ms)` within 5 ms, and fail the
  report if not — that is the double-count check described above.

---

### §6.8 Deployment

**Demo live from the Mac. The always-on public artifact is a static session report plus a
recorded walkthrough. The live hosted link exists, but it is invite-gated, capped, and it is
not the product.**

That is a reversal of the obvious plan, and the arithmetic below is why.

#### The hosted build is a different stack, not a smaller one

Two configs behind the same `STTProvider` / `LLMProvider` / `TTSProvider` interfaces:
`config/local.yaml` and `config/hosted.yaml`. Nothing is shared below the interface line except
the LLM.

| Layer | `LOCAL` (the recommended stack) | `HOSTED` (Render free) |
|---|---|---|
| VAD | `silero-vad` 6.2.1 ONNX, **server**, 512-sample frames | silero-vad v6 ONNX in the **browser** (`onnxruntime-web`, WASM SIMD), same 512-sample contract, ~2 MB one-time download |
| Endpointing | `smart-turn-v3.2` int8, **server**, 8.68 MB, 12–30 ms | **Deepgram Flux** `EndOfTurn` (`eot_threshold: 0.7`, `eot_timeout_ms: 4000`). No model on the server |
| STT | **Moonshine v2 Medium Streaming**, server, MIT | **Deepgram Flux** `flux-general-en` over WS, linear16 @ 16 kHz |
| LLM | Groq `qwen/qwen3.6-27b`, `reasoning_effort:"none"` | **Identical.** The only layer that does not change |
| TTS | **Kokoro-82M** int8 ONNX, server | **Deepgram Aura-2** TTS WebSocket, `encoding=mulaw&sample_rate=24000` |
| AEC | browser `echoCancellation` + loopback `RTCPeerConnection` | Identical |
| Barge-in decision | Server (VAD + generation id) | **Browser.** Local VAD fires, playback stops immediately, a `barge_in` frame follows. No round trip on the stop path |
| Downlink codec | `pcm_s16le` @ 24 kHz — 48 KB/s | `mulaw` @ 24 kHz — 24 KB/s |
| Report | Written to `runs/` and `reports/` | Held in memory, offered as a download, **never touches disk** |
| Provider cost | $0 | ~$0.017 per active minute of Deepgram credit |

Present this as what it is: the hosted build is the executable proof that the provider
abstraction is real. "I have two complete backends behind one interface and here is the measured
latency and disagreement delta between them" is a stronger claim than a single deployment, and it
is the same work the §4/§5 comparison backends already required.

#### Why the local stack cannot run on Render Free — the arithmetic

Render Free is **0.1 CPU / 512 MB** (`render.com/docs/compute-plans`). Three independent blockers,
any one of which is fatal:

1. **Moonshine v2 does not exist on Linux.** `moonshine-voice` 0.1.5 publishes a single wheel,
   `macosx_15_0_arm64`. There is no x86-64 Linux wheel and no source build documented. This is not
   a performance argument; the package cannot be installed on Render at all.
2. **Kokoro does not fit the CPU quota.** `kokoro-v1.0.int8.onnx` is 92,361,271 bytes (88 MiB) and
   `voices-v1.0.bin` is 28,214,398 bytes (27 MiB) — with an `onnxruntime` arena, roughly 230–260 MB
   RSS, which *would* fit in 512 MB. The CPU does not. Kokoro's measured RTF is 0.08 on an M5 Max.
   Render Free is a 10% cgroup quota on one shared vCPU. Even assuming a generous 1:3 per-core
   ratio against an M5 Max performance core, RTF becomes `0.08 × 3 ÷ 0.1 ≈ 2.4` — **2.4 seconds of
   compute per second of speech.** The scaling factor is an estimate (§14); the order of magnitude
   is not in doubt. Real-time synthesis needs RTF < 1 with headroom.
3. **The always-on models would eat the memory before a session starts.** §6.5's resource table
   puts the local stack at ~1.15 GB RSS against a 512 MB instance. Even the ONNX-only subset is
   ~395 MB before FastAPI, `httpx`, per-session ring buffers, or the Python interpreter.

The relay, by contrast, is small and measurable:

| Component | Hosted relay RSS (estimate — measure with `resource.getrusage` on first deploy) |
|---|---|
| CPython 3.11 interpreter | ~15 MB |
| FastAPI + starlette + uvicorn + `websockets` | ~35 MB |
| `httpx` AsyncClient (HTTP/2 off) + certifi | ~15 MB |
| Per concurrent session: 2 upstream WS clients + 8 s uplink ring (256 KB) + 2 s downlink jitter queue (48 KB) + transcript/history | ~3 MB |
| **Total at the enforced concurrency of 1** | **~70 MB of 512 MB** |

Headroom is deliberate: 0.1 CPU means a GC pause or a `json.dumps` burst is a real latency event,
and you want the allocator never under pressure.

#### Burn rate, and what the credit pot actually buys

Define a **demo-minute** as one minute of wall clock inside an active session.

| Cost line | Rate | Per demo-minute |
|---|---|---|
| Deepgram Flux STT | $0.0077/min regular ($0.0065 promo) — billed on stream duration, so browser VAD gating does **not** reduce it | **$0.0077** |
| Deepgram Aura-2 TTS | $0.030 / 1,000 chars. A coach reply ≈ 260 chars; ~1.2 replies per active minute ≈ 310 chars/min | **$0.0093** |
| Groq LLM | $0 cash — consumes the shared free quota (below) | $0 |
| Render egress | ~0.57 MB/min (mulaw downlink + control frames); free under 5 GB/mo | $0 until the cliff |
| **Total** | | **$0.017 / demo-minute** |

- **$200 Deepgram credit ÷ $0.017 = 11,765 demo-minutes ≈ 196 hours ≈ 1,960 six-minute sessions.**
- **Render egress: 5 GB ÷ 2.6 MB per six-minute session ≈ 1,970 sessions/month.** (With
  `linear16` instead of `mulaw`, 5.0 MB/session → ~1,024 sessions.)

Neither of those is the binding constraint. **Groq's free tier is**, and it is the number that
decides the whole design. Using the measured per-call figures from §9.6 and the 6-minute hosted
cap (one question, up to three probes, one scored block, one wrap):

```
3 probe turns        3 × ~1,585                                    ≈  4,755
1 block scored       1,730 (uncached first call) + 4 × 530         ≈  3,850
1 wrap                                                             ≈    920
                                                    SESSION TOTAL  ≈  9,525 tokens
```

- **200,000 TPD ÷ ~9,500 ≈ 21 hosted sessions per day, organisation-wide, shared with your own
  development and with the full 17-minute local sessions that cost ~28,050 each.** Reserve ~60%
  for development and demo rehearsal and the public ceiling is **5 sessions/day**. The 120,000-token
  daily ledger in §6.10 is the fail-closed backstop, not the binder — at ~9,500 a session it would
  allow ~12, so the session counter trips first, by design.
- **8,000 TPM makes concurrency 1.** One session's worst minute is a probe turn (~1,585) landing in
  the same minute as a five-call scoring batch (~2,650) = **~4,235 tokens**. Two visitors hitting
  that minute together is ~8,470 — over the cap. Two concurrent sessions do not degrade each other;
  they 429 each other, and they 429 you.
- 1,000 RPD is not binding: ~14 requests/session × 5 = 70.

**And the uncapped-cost attack the free tiers do not stop.** Flux is billed on **stream duration**,
not on speech. An attacker who opens five sockets and says nothing burns
`5 × 1,440 min × $0.0077 = $55.44 per day`. **The entire $200 pot is gone in 3.6 days without
anyone uttering a word.** Separately, five concurrent `linear16` downlinks at 48 KB/s exhaust the
5 GB egress allowance in **under 6.2 hours**, after which — with no payment method on file —
**Render spins down every free service in the workspace until the 1st of next month.**

#### The decision

**An ungated public WebSocket cannot be reconciled with these numbers.** Five public sessions a
day is not a link you post; it is a link you hand to a named person. So:

**Tier 0 — always on, no backend, no quota, no cold start. This is the link in the README and on
the CV.** GitHub Pages, static:

1. A **real HTML session report** generated by the real pipeline from the developer's own session:
   per-dimension scores, each citing a verbatim transcript span with word-level timestamps, every
   feedback item clickable to its audio moment. Not a mockup — `coachlogic/report.py` output,
   committed.
2. A **3-minute screen recording** with the latency HUD and the live smart-turn `p` visible,
   including one deliberate barge-in and one three-second thinking pause. `assets/demo.mp4` in the
   repo, and a GIF of the barge-in in the README above the fold.
3. The **measured artifacts**: the three-column latency table (`t_endpoint` / `t_server_v2v` /
   `t_v2v`) with CDFs, N and bootstrap CIs; the acoustic-calibration agreement figure; the
   pooled-WER table; the κ_w confusion matrices; the endpointer FCR/latency curve; and the
   LOCAL-vs-HOSTED provider comparison.
4. A README quickstart that gets the full local stack running in one command.

**Tier 1 — the live hosted link, invite-gated.** `https://<app>.onrender.com/#k=<INVITE_KEY>`, one
concurrent session, 6-minute hard cap, 5 sessions/day, fails closed with a friendly message that
points back at Tier 0. Warm it yourself (`GET /healthz`) before you send it.

This is the honest recommendation and it is also the stronger one. A recruiter who clicks a cold
free instance waits ~60 seconds, then gets a voice loop over their hotel wifi running a stack the
README says is not the recommended stack. A recruiter who clicks Tier 0 sees the actual output in
under a second. **Say all of this in the README** — "here is the free-tier arithmetic, here is why
the live link is gated, here is the invite" is an engineering judgement call you want to be seen
making, not a gap you want them to find.

#### Cold start and warm-up

Render spins a free service down after 15 minutes with no inbound traffic and takes **about one
minute** to spin back up; since the changelog entry of **2026-02-24** an inbound WebSocket message
resets the idle timer, so an in-progress interview cannot be spun down mid-turn. Everything else
surveyed fails outright: **Fly.io** requires a card on file; **Modal** states "you must have a
payment method on file in order to use Modal"; **Railway** free is $1 of credit per month (under a
day of uptime); **Vercel Hobby** caps every function invocation at 300 s so the socket dies every
five minutes; **Cloudflare Workers** free gives 10 ms CPU per request under Pyodide;
**Hugging Face** now requires a paid plan for new Gradio/Docker Spaces (free accounts get up to 2
ZeroGPU Gradio Spaces, 5 GPU-min/day).

- The invite page fires `GET /healthz` on load and shows a determinate "waking the free instance —
  about 60 seconds" progress state. The mic button stays disabled until `/healthz` returns 200.
- **`/healthz` must not touch Groq or Deepgram.** It returns process uptime, the resolved provider
  IDs from config, `resource.getrusage(RUSAGE_SELF).ru_maxrss`, and the remaining daily budget. A
  crawler hitting it must cost nothing.
- **Do not run an external cron pinger.** 730 of your 750 monthly instance-hours go to keeping one
  service awake 24/7, leaving nothing, and it defeats the gate.
- Free-tier sockets die on **every deploy** (30 s SIGTERM grace). The SIGTERM handler broadcasts
  `error {code:"server_restarting", retriable:true, retry_after_s:45}` to every open socket before
  closing, so clients reconnect with backoff instead of showing a dead UI. See §6.9.

### §6.9 Session lifecycle and resume

Render free sockets die on every deploy, and a visitor's wifi will drop at least once in six
minutes. Resume is not a nicety; without it the demo's failure mode is "it just stopped."

**The URL carries the session id.** `wss://<host>/ws/v1/<sid>` — path, not query string, so the
version is a routable prefix and a breaking change returns 404 to old clients instead of silently
mis-parsing. `sid` is 22 chars of `secrets.token_urlsafe(16)`, minted by the **client** on first
connect so that a reconnect needs no prior round trip. The invite key is **never** in the URL (see
§6.10); it travels in the `hello` frame.

**Server-side state lives in one process.** Render Free is a single instance with no autoscaling,
so an in-process dict is correct here — and would be wrong on any paid multi-instance plan. Say so
in the code comment; it is the kind of thing an interviewer probes.

```python
# src/coach/sessions.py
from __future__ import annotations
import asyncio, secrets, time
from collections import deque
from dataclasses import dataclass, field

@dataclass(slots=True)
class SessionState:
    sid: str
    ip_hash: str
    build: str                        # "LOCAL" | "HOSTED"
    t0_ns: int                        # monotonic origin; every `ts` in the protocol is ms from here
    hard_deadline_ns: int             # t0 + MAX_SESSION_S; never extended by a resume
    chan: "ControlChannel"            # owns the seq counter and the replay ring; survives reconnects
    history: "History"
    pending_spoken: dict[int, str] = field(default_factory=dict)   # utt_id -> text NOT yet acked
    tokens_spent: int = 0
    deepgram_ms: int = 0
    disconnected_at: float | None = None

class SessionStore:
    """In-process. Render Free runs exactly one instance; this would need Redis on a paid plan."""
    RESUME_TTL_S = 900                # a dropped session is resumable for 15 minutes
    REAPER_PERIOD_S = 30

    def __init__(self) -> None:
        self._s: dict[str, SessionState] = {}

    def get(self, sid: str) -> SessionState | None:
        return self._s.get(sid)

    def put(self, st: SessionState) -> None:
        self._s[st.sid] = st

    def drop(self, sid: str) -> None:
        self._s.pop(sid, None)

    async def reap_forever(self) -> None:
        while True:
            await asyncio.sleep(self.REAPER_PERIOD_S)
            now_w, now_n = time.time(), time.monotonic_ns()
            for sid, st in list(self._s.items()):
                expired = st.disconnected_at and now_w - st.disconnected_at > self.RESUME_TTL_S
                if expired or now_n > st.hard_deadline_ns:
                    self._s.pop(sid, None)        # transcript, history and buffers go with it
```

**Reconnect behaviour, in order:**

1. Client reconnects to `/ws/v1/<same sid>` with exponential backoff: immediate, 0.5 s, 1 s, 2 s,
   4 s, 8 s, then 8 s flat; it gives up after 60 s and shows the Tier 0 link.
2. Client sends `hello` with `sid`, `last_seq` (the highest control-frame `seq` it processed), and
   `played_through` (a map of `utt_id` → frames actually rendered to the output device).
3. Server replies `session_resume` **always, as its first frame**, with `resumed: true|false`.
   `resumed:false` carries `reason` ∈ `{"unknown","expired","hard_deadline","version"}` and the
   client starts a fresh session with a new `sid`.
4. On `resumed:true` the server replays every JSON control frame with `seq > last_seq` from a
   bounded 256-frame ring, in order, then resumes normal operation. **Audio is never replayed.**
   `metrics` frames are excluded from the ring — they are re-derivable from the JSONL.
5. Any bot utterance in flight at disconnect is abandoned. The `TurnController`'s `finally` block
   already commits only what it handed to the player; resume tightens that to **only what the
   client acknowledged playing**:

```python
    # SessionState, on resume — extends §8.2's "commit only what the user heard" across a dropped socket
    def reconcile_playback(self, played_through: dict[int, int], frames_per_sentence: dict[int, int]) -> None:
        for utt_id, text in list(self.pending_spoken.items()):
            played = played_through.get(utt_id, 0)
            if played >= frames_per_sentence.get(utt_id, 1):
                self.history.commit_assistant(text)   # heard in full -> it is real
            # else: dropped. The model must never reference audio that died in a jitter buffer.
            del self.pending_spoken[utt_id]
```

That closes the exact hallucination-shaped bug §8.2 warns about, for the case where the audio was
generated, sent, and then never played.

**The hard deadline is not extended by a resume.** `hard_deadline_ns` is set once at `hello` and a
resumed session inherits it. Otherwise reconnecting is an unbounded session extension, which is
precisely the abuse path the 6-minute cap exists to close.

### §6.10 Hosted deployment posture: auth, abuse, and visitor privacy

The hosted build is a public WebSocket spending the developer's Deepgram credit and Groq quota.
Everything below is in `src/coach/guard.py` and is a no-op when `build == "LOCAL"`.

#### Admission

**Invite key in the URL fragment.** `https://<app>.onrender.com/#k=<INVITE_KEY>`. The fragment is
never sent in the HTTP request line, so it never reaches Render's access logs, a proxy, or a
`Referer` header. The page reads `location.hash`, calls `history.replaceState` to strip it, and
sends it in the `hello` frame — **not** in the WebSocket query string, which does get logged. The
key is a single rotating shared secret, 20 chars of `secrets.token_urlsafe(15)`, compared with
`hmac.compare_digest`. Rotate by changing one env var; every old link dies. This is not
authentication — it is a speed bump against crawlers and scanners, and it is honest to call it
that in the README.

**Origin check.** Reject the upgrade if `Origin` is not the deployed origin. Two lines, stops
hotlinking from someone else's page.

**Rate-limit key.** On Render the client IP arrives in `X-Forwarded-For`. ⚠ **The left-most entry
is attacker-controlled and must not be the rate-limit key.** Use the right-most entry that Render
itself appended, and verify Render's exact append behaviour before trusting the limiter (§14).
Store only `sha256(SALT + ip)[:16]` — never the address.

| Control | Value | Why this number |
|---|---|---|
| Global concurrent sessions | **1** | Groq 8,000 TPM. One session's worst minute is ~4,235 tokens; two of them collide at ~8,470 and 429 each other |
| Concurrent sessions per IP hash | 1 | — |
| Sessions per IP hash per rolling 24 h | 3 | Enough for a recruiter to retry twice |
| Public sessions per UTC day | **5** | 200,000 TPD ÷ ~9,500 tokens/hosted session ≈ 21; ~60% reserved for development and rehearsal |
| Hard session length | **360 s** | 1 question + 3 probes + wrap. Set once at `hello`, never extended by a resume |
| Idle timeout | 45 s with no inbound frame | Closes abandoned tabs that keep the Flux stream billing |
| Max mic frames/second | 40 (32 ms frames are 31.25/s) | 2 s over the cap → close with `rate_limited`. Stops a flood that pins 0.1 CPU and burns Flux minutes |
| Max binary frame | 4,104 bytes | A single oversized frame must not OOM a 512 MB instance |
| Max JSON frame | 4,096 bytes | — |
| Daily Groq token ceiling | **120,000** | Fails closed. A backstop behind the 5-session counter, not the binding limit |
| Daily Deepgram ceiling | **200 cents ($2.00)** | The $200 pot survives ≥100 days even under sustained abuse |

The two daily ceilings are the load-bearing ones. Everything else limits how fast the pot drains;
these limit how much of it can drain at all.

```python
# src/coach/guard.py
from __future__ import annotations
import hashlib, hmac, os, time
from collections import defaultdict, deque
from dataclasses import dataclass

class Denied(Exception):
    def __init__(self, code: str, message: str, retry_after_s: int | None = None):
        super().__init__(code)
        self.code, self.message, self.retry_after_s = code, message, retry_after_s

class BudgetExhausted(Denied):
    pass


@dataclass(slots=True)
class DailyLedger:
    """UTC-day spend ceiling. reserve() BEFORE the upstream call, settle() after.

    Fails closed by construction: reserve() charges the pessimistic worst case up front, so a
    crash between reserve and settle over-counts (safe) and never under-counts (unsafe).
    Not persisted: a Render restart resets the day. That is a deliberate, stated trade-off --
    the per-IP and concurrency caps bound how fast a restart can be exploited, and the
    alternative on a $0 budget is a database this project does not otherwise need.
    """
    ceiling: int
    unit: str
    deny_code: str = "budget_exhausted"
    day: str = ""
    used: int = 0

    def _roll(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self.day:
            self.day, self.used = today, 0

    def remaining(self) -> int:
        self._roll()
        return max(0, self.ceiling - self.used)

    def reserve(self, worst_case: int) -> int:
        self._roll()
        if self.used + worst_case > self.ceiling:
            raise BudgetExhausted(self.deny_code, _exhausted_message(), _secs_to_utc_midnight())
        self.used += worst_case
        return worst_case

    def settle(self, reserved: int, actual: int) -> None:
        self.used += actual - reserved      # give back the unspent reservation


def _secs_to_utc_midnight() -> int:
    now = time.gmtime()
    return 86_400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)

def _exhausted_message() -> str:
    h, m = divmod(_secs_to_utc_midnight() // 60, 60)
    return (f"Today's free demo quota is spent — it resets in {h}h {m:02d}m. "
            f"Nothing broke; this runs on free API tiers and the ceiling is deliberate. "
            f"A full recorded walkthrough and a real session report are linked on the page, "
            f"and the repo runs the complete local stack with one command.")


class Guard:
    MAX_CONCURRENT_GLOBAL   = 1
    MAX_CONCURRENT_PER_IP   = 1
    MAX_SESSIONS_PER_IP_24H = 3
    MAX_SESSIONS_PER_DAY    = 5
    MAX_SESSION_S           = 360
    IDLE_TIMEOUT_S          = 45
    MAX_MIC_FRAMES_PER_S    = 40

    def __init__(self) -> None:
        self.invite_key = os.environ["INVITE_KEY"].encode()
        self.ip_salt    = os.environ["IP_HASH_SALT"].encode()
        self.origin     = os.environ["PUBLIC_ORIGIN"]
        self.tokens   = DailyLedger(ceiling=120_000, unit="groq_tokens")
        self.deepgram = DailyLedger(ceiling=200,     unit="deepgram_cents")
        self.sessions = DailyLedger(ceiling=self.MAX_SESSIONS_PER_DAY, unit="sessions",
                                    deny_code="session_cap")
        self._live: set[str] = set()
        self._per_ip: dict[str, int] = defaultdict(int)
        self._recent: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.MAX_SESSIONS_PER_IP_24H))

    def ip_hash(self, xff: str | None, peer: str) -> str:
        # The RIGHT-most X-Forwarded-For entry is the one Render appended; the left-most is
        # whatever the client typed. Verify Render's behaviour before trusting this (see §14).
        ip = xff.split(",")[-1].strip() if xff else peer
        return hashlib.sha256(self.ip_salt + ip.encode()).hexdigest()[:16]

    def admit(self, *, origin: str | None, invite_key: str, iph: str, consent_ok: bool) -> None:
        if origin != self.origin:
            raise Denied("unauthorized", "This demo only runs from its own page.")
        if not hmac.compare_digest(invite_key.encode(), self.invite_key):
            raise Denied("unauthorized",
                         "This link needs an invite key. The recorded demo and a sample "
                         "session report are public and need nothing.")
        if not consent_ok:
            raise Denied("no_consent", "The microphone stays off until you accept the notice.")
        if len(self._live) >= self.MAX_CONCURRENT_GLOBAL:
            raise Denied("session_cap",
                         "One session at a time — the free LLM tier is 8,000 tokens/minute and "
                         "two at once would rate-limit both. Try again in a few minutes.", 120)
        if self._per_ip[iph] >= self.MAX_CONCURRENT_PER_IP:
            raise Denied("session_cap", "You already have a session open in another tab.", 30)
        now = time.time()
        recent = self._recent[iph]
        if len(recent) == recent.maxlen and now - recent[0] < 86_400:
            raise Denied("rate_limited", "Three sessions in 24 hours is the cap here.",
                         int(86_400 - (now - recent[0])))
        self.sessions.reserve(1)
        recent.append(now)

    def acquire(self, sid: str, iph: str) -> None:
        self._live.add(sid); self._per_ip[iph] += 1

    def release(self, sid: str, iph: str) -> None:
        self._live.discard(sid)
        self._per_ip[iph] = max(0, self._per_ip[iph] - 1)
```

Wiring it to the LLM call, so the ceiling is enforced where the money is spent:

```python
# src/coach/llm/groq_llm.py — inside stream_chat()
worst_case = count_tokens(messages) + self.cfg.max_completion_tokens
reserved = guard.tokens.reserve(worst_case)                 # raises BudgetExhausted -> error frame
try:
    ...  # stream
finally:
    actual = usage.prompt_tokens + usage.completion_tokens if usage else worst_case
    guard.tokens.settle(reserved, actual)
    turn_log.set("groq_tokens_remaining", resp.headers.get("x-ratelimit-remaining-tokens"))
```

`BudgetExhausted` is caught once, at the socket boundary, and becomes
`error {code:"budget_exhausted", retriable:true, retry_after_s:<to UTC midnight>}` followed by
`end_session`. **It is never an exception in the browser console and never a traceback on screen.**

#### Consent — one screen, before the mic opens

Rendered as the only thing on the page. The mic button is disabled until the checkbox is ticked;
`hello.consent.text_sha256` pins which version of this text was accepted, so the wording is
auditable against the deployed build.

> **Before you turn on the microphone** — 20 seconds, please read it.
>
> - Your microphone audio is streamed to this server and forwarded to **Deepgram** (speech
>   recognition and speech synthesis, United States); your **transcript text only** goes to **Groq**
>   (the language model, United States). Nothing is sent anywhere else, and no audio goes to Groq.
> - **No audio is written to disk, here or by us.** On this server your voice lives in an 8-second
>   in-memory ring buffer that is continuously overwritten.
> - Your transcript, the coach's replies and the per-turn timings are held in server memory for the
>   session and **deleted 15 minutes after you disconnect**. There is no database, no account, and
>   no cookie.
> - The only thing that outlives your session is one **aggregate timing row**: millisecond
>   measurements, token counts, and the provider names. No text, no audio, no IP address.
> - Your IP address is **hashed** (SHA-256 with a server-side salt) purely to enforce a
>   one-session-at-a-time limit. The hash is dropped after 15 minutes. The address itself is never
>   stored or logged.
> - Sessions are capped at **6 minutes**, and the whole demo is capped at 5 sessions a day, because
>   it runs on free API tiers.
> - This scores **your own practice answers** against a published rubric, for you. It is not a
>   hiring tool. It does not score confidence, emotion, accent, or "hireability". No employer sees
>   anything, because nothing is kept.
> - Deepgram and Groq have their own terms. Groq states it does not retain inference inputs by
>   default and does not train on them.
>
> ☐ I understand. **[ Start the microphone ]**

Two sentences of this belong in §9.9's ethics block as well: that section reasons about the *user*
scoring *themselves*, and says nothing about a stranger whose voice transits two US clouds on
someone else's API keys. The honest framing is that the hosted build makes every visitor a third
party to a contract they did not sign, which is a second, independent reason the link is gated
rather than posted.

#### Recording and scrubbing

**The hosted build has no audio writer.** Not a flag set to false — the object is not constructed
when `build == "HOSTED"`, and a unit test asserts it (`test_hosted_has_no_disk_writer`). A flag can
be flipped by a stray env var; an absent code path cannot. **No visitor audio is ever recorded, so
there is nothing to scrub from the hosted path.**

Everything in `data/fixtures/`, `data/calibration/` and the published report is the developer's own
voice, and it still gets scrubbed before it is committed:

1. `data/scrub.yaml` is a hand-maintained map of literal strings → replacements: names, employers,
   schools, clients, project code names, and every numeric identifier.
2. `bench/scrub.py` applies it to the transcript JSON **and bleeps the matching spans in the WAV** —
   Moonshine's word-level timings give you the exact sample ranges, so the replacement is silence
   over `[t0_ms, t1_ms]`, not a re-record.
3. The same script runs as a pre-commit hook in **check mode**: if any `scrub.yaml` key appears in
   any tracked transcript, JSONL, or report, the commit fails. A scrubber you have to remember to
   run is a scrubber that leaks.
4. `metrics/turn_log.py` **splits every row by value type** (see `_committable()` in §6.7).
   `runs/*.jsonl` — the committed file — takes numbers, booleans, and a named allow-list of
   identifier strings only. Every other string, which is every transcript field
   (`generated_text`, `committed_text`, `stt_late_finals`, `stt_local_text`, `stt_cloud_text`),
   is written instead to `runs/*.text.jsonl`. `.gitignore` excludes `runs/*.text.jsonl`
   wholesale, and the §6.10 item 3 pre-commit hook fails the commit if one is ever staged.
   The allow-list is on the string side only, which is the fail-safe direction: a new timing
   metric needs no registration, and a new text field cannot leak by someone forgetting to
   register it. One sample `runs/*.jsonl` is committed; no `*.text.jsonl` ever is.
5. `coachlogic/report.py` refuses to render a shareable report unless `session.consent_scope ==
   "self"`, which only the `LOCAL` build ever sets.

---

## 7. Latency budget

Two reference points to anchor against. Pipecat's author publishes an **800 ms** voice-to-voice
target split as network 200 / turn-detection+transcription 400 / LLM 500 / TTS 200. And an
instrumented production Pipecat trace showed named service latencies summing to **570 ms while
wall-clock was 1,048 ms** — the missing ~480 ms being VAD silence windows, audio buffering, and
network hops. **Model compute is not the bottleneck. Endpointing and plumbing are.**

### Say which two events you mean, every time

Earlier drafts said "end of speech → first audible word", which mixed a server-side event with a
browser-side one and then measured both on the server. Three named terms, three definitions, used
consistently from here on:

| Term | From | To | Clock |
|---|---|---|---|
| **`t_endpoint`** | server receipt of the last mic frame Silero scored as speech (**`t0`**) | END_TURN declared, final transcript in hand | server only |
| **`t_server_v2v`** | `t0` | first TTS PCM byte written to the WebSocket | server only |
| **`t_v2v`** | acoustic end of speech at the microphone | first TTS sample audible at the output device | both, reconciled (§6.7) |

`t_server_v2v` is what the §8.2 code actually measures on its own. `t_v2v` is what a README
normally claims. The gap between them is 31–103 ms (plan 67) and it is not optional to account for.

### The budget

Both columns are strict sums of their row ranges. The naive column keeps its original values and
gains the two last-mile rows.

| # | Stage | **Naive** | **Tuned — range** | **Plan** | How the gap closes |
|---|---|---|---|---|---|
| 1 | Mic capture + 32 ms worklet quantum + WS uplink *(before `t0`)* | 50 | 20–35 | **28** | 32 ms frames, no extra buffering |
| 2 | Silero silence → CANDIDATE_END | **800** (fixed threshold) | 200 | **200** | `stop_secs=0.2`; the semantic classifier does the rest |
| 3 | smart-turn v3.2 ONNX | — | 12–30 | **(overlapped)** | Runs concurrently with row 4 |
| 4 | **STT finalization** — `update_transcription(FORCE_UPDATE)` at CANDIDATE_END → complete line | **300–400** (Groq Whisper batch round-trip) | **107–258 ⚠ MEASURE** | **258** | Encoder work already done during speech (§4.1) |
| 3∪4 | Charged as `max(row 3, row 4)` | 300–400 | 107–258 | **258** | Forcing the pass at CANDIDATE_END buys row 3 free |
| — | *(`t_endpoint` subtotal, from `t0`)* | *1,100–1,200* | *307–458* | ***458*** | |
| 5 | LLM → first **speakable** token | **~3,550** (gpt-oss-20b, reasoning first) | **300–800 ⚠ MEASURE** | **500** | `reasoning_effort:"none"` |
| 6 | Sentence 1 assembled | 600–1,200 (waits for completion) | 30–45 | **35** | ~15 tokens @ ~440 t/s |
| 7 | TTS → first PCM byte | 400–800 (Orpheus whole-WAV) | **90–200 ⚠ MEASURE** | **140** | Kokoro local, first clause only, ~50 ms chunks |
| — | ***`t_server_v2v` subtotal, from `t0`*** | ***5,650–6,750*** | ***727–1,503*** | ***1,133*** | **the number the server's own marks produce** |
| 8 | Server → browser WS transit (local; Render adds 20–60) | 5 | 1–3 | **2** | Same socket as audio |
| 9 | Browser jitter buffer before first `start()` | 100 | 20–60 | **40** | Your constant — tune against underrun rate |
| 10 | Device output latency (`AudioContext.outputLatency`) | 50 | 10–40 | **25** | `latencyHint:'interactive'` |
| — | ***`t_v2v` total, acoustic end of speech → first audible sample*** | ***≈5,850–6,950 ms*** | ***≈780–1,640 ms*** | ***≈1,230 ms*** | |
| — | Barge-in: decision → `abort()` | 300–1,000 | **20–60** | **40** | `abort()` discards buffers; `stop()` drains them |
| — | Barge-in: decision → last audible sample | 300–1,000 | **50–160** | **105** | Adds rows 9–10; report it **separately** |

**Three rows carry a ⚠ and they are 60% of the plan.** Rows 5 and 7 were always flagged. Row 4 is
the change: earlier drafts booked STT at "30–120 ms (local streaming, already overlapped with
speech)", a figure that appears in no source and contradicts §1 and §4 of this same document.

Two footnotes on the rows:

- **Row 4 is the expected value, not the ceiling.** The END_TURN flush protocol (§8.1) has a hard
  250 ms timeout, and the fast path — a final line already covering end-of-speech, because the
  forced pass at CANDIDATE_END has had 200 ms plus the smart-turn window to complete — usually
  hits and costs 0 ms. The p95 of `stt_final_ms` is its own column in the report. If that p95 sits
  at the 250 ms ceiling on a meaningful fraction of turns, row 4 is 258 + up to 250 and the
  headline moves again; say so rather than quietly keeping 1,230.
- **The barge-in rows are the LOCAL build.** In the browser build the audible stop adds one
  one-way socket trip (<2 ms on localhost, ~30–60 ms over the Render link) plus one 128-frame
  worklet block (2.7 ms at 48 kHz).

### What changed, and why the headline moved from ~900 ms to ~1,230 ms

Old tuned total: **≈710–1,250 ms, plan ~900 ms**. New: **≈780–1,640 ms, plan ~1,230 ms.** Four
corrections, largest first — and the largest one is not the STT number:

1. **+135 ms — the old plan figure was below its own table.** Summing the old tuned column at its
   row midpoints gives ~1,035 ms. It advertised ~900 ms. That gap was never sourced to anything.
2. **+183 ms — STT.** 258 ms replaces an invented 30–120 ms (midpoint 75).
3. **+32 ms — the last mile.** The old single "playback enqueue + device latency: 20–50 ms" row
   described a server-side `enqueue()` call that returns before anything is audible. Rows 8, 9 and
   10 replace it with 67 ms of WS transit, jitter buffer and device output, measured on the client
   clock.
4. **−25 ms — smart-turn is now free.** Forcing the STT pass at CANDIDATE_END puts the classifier
   inside the STT window instead of after it.

Rows 1, 2, 5, 6 and 7 are unchanged. 900 + 135 + 183 + 32 − 25 = **1,225**.

**~1,230 ms is a better number than ~900 ms even though it is larger**, and not only for honesty's
sake: it is consistent with the two independent anchors this section opens with. Pipecat's
instrumented production trace measured 1,048 ms wall-clock against 570 ms of summed service
latencies — the same ~480 ms of VAD windows, buffering and hops that rows 1–2 and 8–10 make
explicit here. A cascaded pipeline with a cloud LLM landing at 1.0–1.3 s is the normal result.
Claiming 900 ms invited an interviewer to find the 300 ms you hid; publishing 1,230 ms with a CDF
invites them to ask which row you would attack next.

### If you need to get under 1 s, here is the order

| Lever | Saves | Costs |
|---|---|---|
| **Moonshine Small Streaming** instead of Medium (`ModelArch.SMALL_STREAMING`) | 110 ms (258→148) | 7.84% WER vs 6.65% — and you lose the "beats Whisper large-v3 (7.44%)" line. Config flag, not the default |
| `update_interval` 0.5 → 0.25 | unquantified; shortens the finalization tail | ~2× per-pass fixed overhead; library backs off on its own if unaffordable |
| First **clause** to TTS (first comma past ~40 chars) rather than first sentence | 200–600 ms on long openings | Occasional unnatural prosody break |
| Pre-rendered audio for the greeting, the 14 questions and the 5 move-on lines | 1,230 ms → ~40 ms on those turns | Only applies to fixed phrases — but that is 20 of ~34 utterances per session (§9.6) |
| Jitter buffer 40 → 20 ms | 20 ms | Underruns; only after you have measured the underrun rate |
| A local LLM (LM Studio, Gemma-3n-4B class) | row 5: 500 → ~150 ms ⚠ unverified (§14) | Answer quality, and it breaks the "Groq free tier" story |

Do not cut row 2. The 200 ms VAD silence is what makes the semantic endpointer possible, and the
whole domain argument in §8 rests on not cutting off a thinking candidate.

### Optimisations ranked by impact per effort

| # | Optimisation | Saves | Effort | Note |
|---|---|---|---|---|
| **1** | **Turn reasoning off** — `reasoning_effort:"none"` (Groq qwen) / `enable_thinking:False` (NIM) | **~2,700–3,300 ms** | **One config value** | The single highest-leverage line in the project |
| **2** | **Sentence-boundary chunking, LLM stream → TTS** — speak clause one while the model is still writing clause three | 500–2,000 ms | Low | Split on the first sentence end, or the first comma past ~40 chars for the opening clause. §6.6 is the implementation |
| **3** | **Semantic endpointing** (smart-turn v3.2) replacing a fixed silence threshold | 300–600 ms | Low–Med | 8.68 MB ONNX, ~12.6 ms on a fast server CPU, ~12–30 ms expected on M-series. **And now free**, because §4.1 overlaps it with the STT finalization pass. §8.3 is how you prove the 300–600 ms rather than assert it |
| **4** | **Local streaming STT** instead of cloud batch | 50–150 ms, **not 300–500** | Medium | Corrected: Moonshine's 258 ms finalization against a 300–400 ms Groq batch round-trip is a smaller win than earlier drafts claimed. The real reason is the 20 RPM ceiling and the absence of partials |
| **5** | **Connection prewarming + warm models** — one persistent `httpx.AsyncClient` with keepalive, a warm-up request, preloaded weights, and the 21 pre-rendered clips generated at startup | 100–300 ms **every turn**, and fixes the first turn (the one your audience sees) | Very low | §6.5's `lifespan` |
| **6** | **`abort()` not `stop()`, aborting *before* awaiting the cancelled task, plus `latency='low'`** on the output device | Barge-in 300 ms → ~50 ms | Very low | `sounddevice.stop()` *waits* for pending buffers; `latency` defaults to `'high'`. Ordering matters — see §8.2 defect 4 |
| **7** | **Prompt caching + static-first prompt + token-capped history** | TTFT, plus rate-limit headroom | Low | gpt-oss models only — `qwen/qwen3.6-27b` has no cache, so on the hot path this is prompt hygiene and on the **scoring** path it is the thing that makes 25 calls affordable. Requires the three-region layout in §3.1; the naive "summarise older turns" mitigation invalidates the cache on every turn and is strictly worse. Measured via `cached_tokens`, not assumed |
| **8** | **Pre-rendered audio for fixed phrases** (greeting, all 14 questions, all 5 move-on lines, the rung-1 filler, the 429 fallback) | ~1,190 ms on applicable turns | Very low | Makes the demo's opening instant and removes 20 of ~34 utterances from the TTS path entirely |
| **9** | ~~**Speculative / preemptive generation**~~ | 200–600 ms | High | **Rank last.** LiveKit issue #4219 documents duplicate LLM requests; doubling token spend against an 8,000 TPM free cap converts a latency win into a mid-demo 429 |

### LOCAL vs HOSTED — the same budget, measured the same way

Browser and Render instance in the same US region. Rows correspond one-to-one with the table above.

| # | Stage | `LOCAL` plan | `HOSTED` range | `HOSTED` plan | Why it differs |
|---|---|---|---|---|---|
| 1 | Mic capture + quantum + uplink | 28 | 40–75 | **55** | Real network instead of loopback |
| 2 | End-of-turn decision | 200 | 250–400 | **320** | Flux `EndOfTurn`; it subsumes its own silence window, so it is not additive to a separate VAD wait. Vendor says <400 ms, Coval median <300 ms |
| 3∪4 | smart-turn ∪ STT finalization | 258 | **0** | **0** | The `EndOfTurn` message carries the final transcript |
| 5 | LLM → first speakable token | 500 | 300–800 ⚠ | **500** | Same Groq call, same model |
| 6 | Sentence 1 assembled | 35 | 30–45 | **35** | — |
| 7 | TTS → first byte | 140 | 120–250 | **185** | Aura-2 vendor sub-200 ms TTFB + one RTT |
| 8 | Server → browser transit | 2 | 15–40 | **25** | Real network |
| 9 | Jitter buffer | 40 | 40–100 | **70** | Deliberately larger over the public internet |
| 10 | Device output latency | 25 | 10–40 | **25** | — |
| — | **`t_v2v` total** | **≈1,230** | **≈805–1,750** | **≈1,215** | |
| — | Barge-in audible stop | 105 | 30–80 | **55** | Client-side decision; no round trip |

**The medians are within noise of each other, and that is a finding, not a coincidence.** The
hosted path pays ~110 ms more in network and jitter and ~45 ms more in TTS, and gets it all back
because Flux folds endpointing and transcription into one event with no separate finalization
pass. What the hosted path does *not* match is the tail: its high end is 1,750 ms against 1,640,
and the variance is the visitor's home wifi plus two third-party clouds — not yours to fix.
**Publish both columns and both CDFs.** "The hosted link has the same median and a much heavier
tail, and here is the distribution showing why" is a better answer than a single unqualified
number, and the clock-offset uncertainty in §6.7 is ±0.2–0.8 ms locally against ±15–45 ms hosted,
which is itself a reason the headline is quoted from the local build.

---

## 8. Turn-taking

This is where the project is won or lost, and it is the specific thing the commercial leader in
this space is criticised for: Final Round AI sits at **2.9/5 across 276 Trustpilot reviews**,
where recurring complaints include lag and that the AI **"doesn't properly recognize when
interviewers finish speaking."**

### Why endpointing matters especially here

Interview answers are not conversational turns. A candidate telling a STAR story pauses two to
four seconds mid-answer to recall a number, to decide how to frame a conflict, or simply from
nerves. A fixed silence threshold forces an unwinnable trade: short (500 ms) and you cut off a
thinking candidate mid-sentence, which is both a broken product and actively demoralising for
the user; long (1,500 ms) and every turn has a second of dead air.

The fix is a two-stage endpointer. A short VAD silence *arms* a candidate endpoint; a semantic
classifier then decides whether the utterance actually sounds finished, judged from the
waveform's prosody rather than from silence length.

### VAD

| Option | Licence / install | Cost | Frame contract | Verdict |
|---|---|---|---|---|
| **`silero-vad` 6.2.1** (PyPI 2026-02-24) | MIT, `pip install silero-vad` — but load the bundled `.onnx` yourself (§6.5) | <1 ms per 30+ ms chunk on one CPU thread; ~2 MB | **Exactly 512 samples @16 kHz** (256 @8 kHz) or `ValueError`; 8/16 kHz only | **Recommended.** v6 claims 16% fewer errors on noisy real-life data vs v5. Core deps include torch; the ONNX path avoids importing it |
| `webrtcvad-wheels` 2.0.14 | macOS arm64 wheels, Python 3.6–3.13 | ~0.1 ms/frame | 10/20/30 ms; 8/16/32/48 kHz; 16-bit mono | Fallback / A-B comparison. Energy+GMM based — false-fires on keyboard, HVAC, breath |
| TEN VAD | Apache-2.0 *with additional conditions*; git-only install | Vendor RTF 0.0086 vs Silero 0.0127; 306 KB | 160 or 256 samples | Faster offset detection claimed, but git install and Linux-first prebuilts. Avoid for a demo that must not break |
| Raw energy threshold | — | ~0 | any | Only as a pre-gate. Alone it is the "toy" signal an interviewer will probe |

Useful starting values, taken from Pipecat's defaults: `confidence=0.7`, `start_secs=0.2`,
`stop_secs=0.2`, `min_volume=0.6`. Silero's own `VADIterator` defaults are `threshold=0.5`,
`min_silence_duration_ms=100`, `speech_pad_ms=30`. **Do not ship these numbers as chosen — §8.3
sweeps the arming silence and the classifier threshold against labelled fixtures and picks an
operating point.**

### Endpointing

| Approach | Decides on | Added latency | Verdict |
|---|---|---|---|
| Fixed silence threshold | N ms of silence | = N (typically 500–800) | **Fatal here.** See above — and §8.3 measures exactly how fatal, on the same fixtures |
| Adaptive threshold | utterance length, trailing conjunctions, question vs statement | 200–1,500 | Better, but hand-tuned heuristics are hard to defend |
| **`smart-turn-v3.2`** | Whisper-Tiny encoder + linear head over the **last 8 s of waveform** → complete/incomplete at p>0.5 | ~12–30 ms, **overlapped with the STT finalization pass, so effectively 0** | **Recommended.** BSD-2, 8.68 MB int8 ONNX, ~8M params, 23 languages. v3.1 accuracy: English 94.7% (8 MB) / 95.6% (32 MB) — on its own evaluation set, not on interview pauses (§14) |
| LiveKit turn detector v1 | dual-branch semantic + acoustic | ~50–160 ms | **Disqualified.** Best benchmark available (9.9% false-cutoff at 300 ms latency) but its MODEL_LICENSE forbids use "on a standalone basis or with any frameworks other than LiveKit Agents." Cite it as the comparator; don't ship it |
| LLM-as-endpointer | extra Groq call on the partial transcript | +200–600 ms and burns TPM | Strictly worse than a 12 ms local classifier |
| TEN Turn Detection | Qwen2.5-7B, text-only | large | Not real-time on a Mac CPU |

**State machine:**

```
SPEAKING ──(VAD silence ≥ 200 ms)──► CANDIDATE_END

CANDIDATE_END:
    stt.force_update()                     ← MOONSHINE_FLAG_FORCE_UPDATE, §4.1
    p = smart_turn(last 8 s of mic audio, 16 kHz mono, FRONT-padded)
        ↑ these two run concurrently on stt_pool and turn_pool; the budget charges max(), not sum
    p > 0.5                      → END_TURN     (≈ 200 ms + max(258, ~25) ms)
    p ≤ 0.5                      → back to SPEAKING; re-check on next silence
    silence ≥ stop_secs (3.0 s)  → END_TURN regardless        ← hard fallback

END_TURN:
    flush protocol (§8.1) → transcript, t0 → TurnController.on_endpoint()
```

The 3.0 s fallback is Pipecat's `SmartTurnParams.stop_secs` default and is not optional: without
it, a model stuck on "incomplete" hangs the turn forever. Tune it *up* for this domain — 3.0–4.0 s.
A coach that waits half a second too long reads as polite; one that interrupts a nervous
candidate reads as broken.

**Log `p` for every candidate endpoint and render it live in the demo UI.** That single
visualisation is what converts "it works" into "I understand why it works" in an interview — and
§8.3 turns the same log into a measured operating point rather than a felt one.

### §8.1 End-of-turn STT flush protocol

§6's data-flow diagram draws `turn_end(transcript, t0)` and the budget charges 258 ms for it. Here
is where that transcript actually comes from. At the instant the endpointer fires END, Moonshine
has an open line whose text is a *partial*, and the final for that line has not been emitted yet —
even though the forced pass at CANDIDATE_END has been running for the 200 ms of arming silence
plus the smart-turn window.

**The primitive.** `moonshine-voice` exposes `create_stream()`, `add_audio()`, `stop()`,
`update_transcription()` and the `TranscriptEventListener` callbacks. The documented finalisation
behaviour is: *if `stop()` is called on a transcriber or stream, any active lines will have
`LineCompleted` called.* So: **one `create_stream()` per user turn, and `stream.stop()` is the
flush.** The long-lived `Transcriber` is never torn down, so model weights are loaded once at
startup.

**The two operations are different and the distinction is load-bearing.**
`update_transcription(FORCE_UPDATE)` at CANDIDATE_END forces a decode pass over audio already fed:
it does not close the line and no word timestamp moves, which is why §4.1 puts it *before* the
turn decision and why the budget gets smart-turn for free. `stop()` at END_TURN closes the line.
**Calling `stop()` early is what is rejected** — it splits the line before the candidate has
actually finished, and if smart-turn then says "incomplete" every subsequent word timestamp is
rebased. Word timings are what the evidence-span validator (§9.5) and the WPM metrics (§9.7 #6)
are built on. Not worth 20 ms.

**Protocol:**

```
END fired (t0 = last speech frame)
  ├─ mark("stt_flush_started_ms")
  ├─ FAST PATH: a final line already covers end-of-speech  → return, 0 ms
  ├─ else submit stream.stop() to stt_pool
  │     └─ await it under asyncio.timeout(STT_FLUSH_TIMEOUT_S = 0.25)
  │           ├─ final arrives  → mark("stt_final_ms")         → use it
  │           └─ TimeoutError   → mark("stt_flush_timeout_ms") → use newest partial
  ├─ seal(): every later on_line_completed goes to the turn log, never to the prompt
  └─ transcript empty → log empty_turn, return to LISTENING. Do NOT call the LLM.
```

**`STT_FLUSH_TIMEOUT_S = 0.25`.** Moonshine v2 Medium's published response latency on an Apple
MacBook M3 is **258 ms** (§4.1), and that figure measures precisely this operation — text produced
after the audio containing it. 250 ms is one decode interval. If a final has not landed after a
full interval the decoder is contended, and a second interval does not reliably fix it; it just
spends the STT budget twice.

**How this interacts with §7 row 4.** The budget charges 258 ms for finalization as a whole,
measured from CANDIDATE_END, and the forced pass has already been running for most of it by the
time END fires. On the fast path the flush therefore costs 0 ms and row 4's 258 ms is the honest
end-to-end figure. On a contended machine the flush adds up to 250 ms on top, which is why
`stt_final_ms` gets its own p95 column in the report. **If that p95 sits near the ceiling on more
than a few percent of turns, the headline is not 1,230 ms and you say so.**

#### Implementation

```python
# src/coach/stt/base.py
class FlushResult(NamedTuple):
    text: str
    timed_out: bool
    used_partial: bool

class STTProvider(Protocol):
    def start_turn(self, turn_id: int) -> None: ...
    def feed(self, frame: bytes) -> None: ...            # 512 samples; submits to stt_pool
    def force_update(self) -> None: ...                  # FORCE_UPDATE at CANDIDATE_END (§4.1)
    async def flush(self, timeout_s: float) -> FlushResult: ...
    def seal(self) -> None: ...                          # later finals -> log only
    def snapshot(self) -> str: ...                       # finals + newest partial, no wait
```

```python
# src/coach/stt/moonshine_local.py  (flush path only)
STT_FLUSH_TIMEOUT_S = 0.25

async def flush(self, timeout_s: float = STT_FLUSH_TIMEOUT_S) -> FlushResult:
    if self._last_final_end >= self._speech_end - 0.05:        # fast path, 0 ms
        return FlushResult(self._joined(), False, False)
    self._final_event.clear()
    fut = self._loop.run_in_executor(self._pool, self._stream.stop)
    try:
        async with asyncio.timeout(timeout_s):
            await self._final_event.wait()
        return FlushResult(self._joined(), False, False)
    except TimeoutError:
        self._orphan = fut                 # keep a reference; see "late finals" below
        return FlushResult(self._joined(partial=True), True, True)

def _on_line_completed(self, line) -> None:                    # Moonshine's own thread
    self._loop.call_soon_threadsafe(self._commit, line)

def _commit(self, line) -> None:                               # event loop
    if self._sealed:
        self._log.append("stt_late_finals", line.text)         # never reaches the prompt
        return
    self._lines.append(line)
    self._last_final_end = line.end_time
    self._log.set("stt_engine_latency_ms", line.last_transcription_latency_ms)
    self._final_event.set()
```

`_joined()` is the concatenation of every final line started since `start_turn()`, plus — when
`partial=True` — the newest partial text. Line splits within a turn are harmless because the
transcript is always the concatenation; that is why a mid-turn finalisation never costs anything,
and it is the same pattern as Deepgram Flux's `TurnResumed`.

**Late finals.** `run_in_executor` cancels the *future*; it cannot interrupt the thread. On
timeout, `stop()` is still running, will still complete, and will still deliver its final — after
`seal()`. That is the whole reason `seal()` exists, and it is set at flush-return rather than at
`llm_request_sent`: strictly earlier, and it removes a race for one line of code.

Two consequences:

- **The next turn's `add_audio()` queues behind the stale `stop()` on the single-worker pool.**
  Acceptable, because the bot is about to speak for 2–8 seconds, during which `stt_pool` is idle.
  The barge-in case is covered by the ring buffer already specified in `audio/frames.py`: frames
  go into a bounded deque (8 s = 250 frames) in front of the pool, so nothing is lost, only
  delayed. If the orphan future has not completed within 2 s, log `stt_pool_stalled` and recycle
  the pool.
- **Late finals are a free accuracy metric, not just debris.** They land in
  `turn_log["stt_late_finals"]`, so `bench/wer.py` can report how often the flush timeout fired
  and by how many words the used transcript differed from the complete one. "The 250 ms flush
  timeout fired on 4.2% of turns and changed the transcript by a median of one word" is a better
  README line than any latency figure in the document.

### §8.2 Barge-in cancellation

Three things break here: cancelled work that keeps writing to the audio device, audio that
keeps playing after cancel, and conversation history containing words the user never heard.

The obvious implementation has four defects. Two contradict §6's declared interfaces, one
mismeasures the headline metric, and one breaks the rule the section itself states.

| # | Defect | Consequence |
|---|---|---|
| 1 | `pcm = await self.tts.synthesize(sentence)` awaits a whole sentence, but §6.3 declares `synthesize(sentence) -> AsyncIterator[bytes]` | Kills the first-clause behaviour the 90–200 ms TTS row assumes. First audio waits for the *last* byte of the first sentence |
| 2 | `mark_once("tts_first_byte")` placed after that await | Marks the last byte. The headline number is wrong in the one direction that flatters it |
| 3 | `commit_assistant(" ".join(spoken))` commits every synthesised sentence | Violates the "commit only what the user heard" rule, which is point 3 of this section's own list of four |
| 4 | `cancel()` calls `player.abort()` *after* `await t` | Audio keeps playing through the task-cancellation round-trip, and `audio_abort_ms` measures cancellation time rather than audible-stop time |

Fix 4 is the cheapest real latency win in the section: **abort first, then unwind.** The device
is silenced before any await, so audible-stop latency is `abort()` plus device drain, not
`abort()` plus a scheduler round-trip.

#### `AudioPlayer`

`player.py` needs a played-frames counter before anything above is implementable.

```python
# src/coach/audio/player.py
import asyncio, collections
import sounddevice as sd

SR, BYTES_PER_FRAME = 24_000, 2          # Kokoro 24 kHz mono int16 (§6.1)
MAX_LOOKAHEAD_S = 2.0                     # §6.5 backpressure ceiling


class LocalAudioPlayer:
    def __init__(self, loop, on_playout_start=None, blocksize=480):     # 480 frames = 20 ms
        self._loop, self._on_start = loop, on_playout_start
        self._q: collections.deque[bytes] = collections.deque()
        self._written = self._played = 0   # _played written ONLY by the PortAudio thread
        self._started = False
        self._room = asyncio.Event(); self._room.set()
        self._stream = sd.RawOutputStream(samplerate=SR, channels=1, dtype="int16",
                                          blocksize=blocksize, latency="low",
                                          callback=self._cb)
        self._stream.start()

    def _cb(self, outdata, frames, time_info, status):      # PortAudio thread
        need, buf = frames * BYTES_PER_FRAME, bytearray()
        while self._q and len(buf) < need:                  # deque ops are atomic in CPython
            buf += self._q.popleft()
        if len(buf) > need:
            self._q.appendleft(bytes(buf[need:])); del buf[need:]
        outdata[:len(buf)] = bytes(buf)
        outdata[len(buf):] = b"\x00" * (need - len(buf))
        if buf:
            self._played += len(buf) // BYTES_PER_FRAME     # single writer -> no lock needed
            if not self._started:
                self._started = True
                if self._on_start:                          # marks audio_first_frame_out_ms
                    self._loop.call_soon_threadsafe(self._on_start)
        self._loop.call_soon_threadsafe(self._room.set)

    def frames_written(self) -> int: return self._written
    def played_frames(self)  -> int: return min(self._played, self._written)
    def queued_seconds(self) -> float:
        return (self._written - self.played_frames()) / SR

    def played_up_to(self) -> float:
        """Seconds of this turn's audio that actually reached the speaker."""
        return max(0.0, self.played_frames() / SR - (self._stream.latency or 0.0))

    async def enqueue(self, pcm: bytes) -> int:
        while self.queued_seconds() > MAX_LOOKAHEAD_S:      # check BEFORE appending:
            self._room.clear()                              # an oversized chunk must not
            await self._room.wait()                         # deadlock on itself
        self._q.append(pcm)                                 # <- the cancel point
        self._written += len(pcm) // BYTES_PER_FRAME
        return self._written

    async def drain(self) -> None:
        while self.played_frames() < self._written:
            await asyncio.sleep(0.02)

    def abort(self) -> None:
        """Sync, sub-ms, idempotent. Freezes (_written, _played) for the commit truncation."""
        self._stream.abort()          # does NOT wait for pending buffers; stop() would
        self._q.clear()
        self._room.set()

    def reset(self) -> None:
        """Called by the controller AFTER the commit, before the next turn."""
        self._q.clear(); self._written = self._played = 0; self._started = False
        self._room.set(); self._stream.start()
```

`abort()` deliberately does **not** zero `_written`. The gap between frozen `_written` and frozen
`_played` is exactly the truncation signal, and zeroing it would make every interrupted sentence
look fully heard.

**Browser build.** In the demo topology audio plays in the browser, so the server cannot see the
playout head and the browser must report it. `WebSocketAudioPlayer` implements the same
interface with two additions: the client sends `playback_ack {utt_id, frames_played, played_ms}`
every 250 ms and once immediately after handling an abort; `played_frames()` returns the last
reported value for the current generation, clamped to `_written`. The playhead is therefore
stale by up to 250 ms + RTT, which at 196 WPM is ~0.8 words — inside the ±1–2 word granularity
declared below, so it does not change the rule.

#### Commit truncation: granularity

**Whole sentences, plus a proportional word-level cut on the one sentence that was interrupted.**

1. Sentence fully played → kept verbatim.
2. The interrupted sentence → keep `floor(f × n_words)` words, where `f` is the fraction of that
   sentence's audio that reached the speaker, and append `—`. If that yields zero words, drop it.
3. Sentences synthesised but never started → dropped entirely, along with everything after.

The word-level cut is an approximation: Kokoro emits no word timings, so speech rate within a
sentence is assumed uniform. Error is ±1–2 words. That is immaterial, because the purpose is to
stop the LLM referencing content the candidate never heard — not to produce a verbatim record.
The verbatim record of what was *generated* goes to the turn log separately, which is what makes
"generated vs heard" a diffable metric.

**Truncation applies only on the interrupted path.** On a normal turn the LLM stream ends while
several seconds of audio are still queued; truncating there would commit only the first sentence
of every successful reply. The normal path therefore ends with `await player.drain()` — which is
also the correct definition of "the bot stopped speaking" for the endpointer.

#### `TurnController`

```python
# src/coach/turn/controller.py
import asyncio, contextlib
from dataclasses import dataclass


@dataclass
class Spoken:
    text: str
    start_frame: int
    end_frame: int = -1        # -1 while still being written


class TurnController:
    def __init__(self, llm, tts, player, stt, history, sentencizer, log):
        self.gen = 0
        self.task: asyncio.Task | None = None
        self.llm, self.tts, self.player, self.stt = llm, tts, player, stt
        self.history, self.sentencizer, self.log = history, sentencizer, log

    # ---- entry points -------------------------------------------------------
    async def on_endpoint(self, turn_log):
        """END fired. Obtain the transcript, then respond. See §8.1."""
        turn_log.mark("stt_flush_started_ms")
        r = await self.stt.flush()
        turn_log.mark("stt_flush_timeout_ms" if r.timed_out else "stt_final_ms")
        turn_log.set("stt_used_partial", r.used_partial)
        self.stt.seal()
        text = r.text.strip()
        if not text:                                  # VAD false-fire: a cough, a door
            turn_log.set("empty_turn", True)          # do NOT spend 8K-TPM budget on it
            return
        await self.on_user_turn_end(text, turn_log)

    async def on_user_turn_end(self, text: str, turn_log):
        await self.cancel()
        self.player.reset()
        self.sentencizer = type(self.sentencizer)()   # emitted counter drives the opening rule
        self.gen += 1
        self.task = asyncio.create_task(self._respond(self.gen, text, turn_log))

    async def on_barge_in(self, turn_log):
        turn_log.mark("barge_in_detected_ms")
        self.player.abort()                           # FIRST — no await between these two marks
        turn_log.mark("audio_abort_ms")               # true audible-stop latency
        await self.cancel()
        turn_log.mark("turn_unwound_ms")              # report separately; it is not user-visible

    async def cancel(self):
        self.player.abort()                           # idempotent; order matters (defect 4)
        t, self.task = self.task, None
        if t and not t.done():
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t                               # AWAIT it — the finally must have run

    # ---- response path ------------------------------------------------------
    async def _respond(self, gen: int, text: str, turn_log):
        spoken: list[Spoken] = []
        committed: str | None = None
        try:
            turn_log.mark("llm_request_sent_ms")
            # stream_chat is a coroutine returning a DeltaStream (§6.3) — it MUST be awaited;
            # the ladder awaits the first delta inside it. aclosing() then calls the stream's
            # aclose(), dropping the HTTP body on every exit path including cancellation.
            async with contextlib.aclosing(
                    await self.llm.stream_chat(self.history.messages(text))) as stream:
                async for delta in stream:
                    if gen != self.gen:
                        break
                    turn_log.mark_once("llm_first_token_ms")
                    turn_log.mark_once("llm_first_speakable_token_ms")
                    for sentence in self.sentencizer.push(delta):
                        await self._speak(gen, sentence, spoken, turn_log)
            for sentence in self.sentencizer.flush():     # MANDATORY — §6.6 rule 2
                await self._speak(gen, sentence, spoken, turn_log)
            await self.player.drain()                     # the bot has finished speaking
            committed = " ".join(s.text for s in spoken)  # normal path: commit everything
        except asyncio.CancelledError:
            raise                                         # NEVER swallow
        finally:
            if committed is None:                         # cancelled, or Groq 429 mid-reply
                committed = self._heard_prefix(
                    spoken, self.player.played_frames(), self.player.frames_written())
            self.history.commit_assistant(committed)
            turn_log.set("generated_text", " ".join(s.text for s in spoken))
            turn_log.set("committed_text", committed)     # diffable: generated vs heard
            turn_log.mark("turn_done_ms")

    async def _speak(self, gen: int, sentence: str, spoken: list[Spoken], turn_log):
        if gen != self.gen:
            return
        turn_log.mark_once("llm_first_sentence_ms")
        rec = Spoken(sentence, self.player.frames_written())
        spoken.append(rec)                                # append BEFORE synthesis, so an
        async with contextlib.aclosing(                   # interrupted sentence is still
                self.tts.synthesize(sentence)) as audio:  # visible to _heard_prefix
            async for chunk in audio:                     # AsyncIterator[bytes], per §6.3
                turn_log.mark_once("tts_first_byte_ms")   # FIRST chunk of the FIRST sentence
                if gen != self.gen:
                    break
                await self.player.enqueue(chunk)          # backpressure + cancel point
                turn_log.mark_once("ws_first_pcm_sent_ms")
        rec.end_frame = self.player.frames_written()

    @staticmethod
    def _heard_prefix(spoken: list[Spoken], played: int, written: int) -> str:
        out: list[str] = []
        for s in spoken:
            end = s.end_frame if s.end_frame >= 0 else written
            if played >= end:                             # fully heard
                out.append(s.text)
                continue
            f = (played - s.start_frame) / max(end - s.start_frame, 1)
            if f > 0:
                words = s.text.split()
                n = int(f * len(words))
                if n:
                    out.append(" ".join(words[:n]) + " —")
            break                                         # nothing after this was heard
        return " ".join(out)
```

**The seven details that separate this from the naive version:**

1. **`player.abort()`, not `stop()`, and abort *before* awaiting the cancelled task.**
   `sounddevice`'s `stop()` "waits until all pending audio buffers have been played before it
   returns"; `abort()` "does not wait for pending buffers to complete." Combined with
   `latency='low'` (the default is `'high'`, which the docs themselves flag as possibly "too large
   for interactive applications") and a ~20 ms blocksize, the un-cancellable tail stays around
   20–60 ms. Aborting after the await adds a whole scheduler round-trip of audio.
2. **`await` the cancelled task.** `Task.cancel()` "does not guarantee that the Task will be
   cancelled" — it schedules `CancelledError` for the next event-loop cycle. Awaiting is how you
   know the `finally` block ran before the next turn starts.
3. **Commit only spoken text to history**, computed from the player's played-frames counter.
   Otherwise the LLM later references feedback the candidate never heard, which looks exactly like
   hallucination. `_heard_prefix` runs on **every** abnormal exit, not just cancellation: a 429
   mid-reply (§11 failure #2) leaves the user having heard two sentences of three.
4. **`CancelledError` subclasses `BaseException`.** A bare `except Exception:` in a TTS or HTTP
   wrapper will not catch it (good), but a bare `except:` will swallow it and deadlock barge-in.
   Audit every handler on the response path.
5. **`async for chunk in self.tts.synthesize(sentence)`**, and `tts_first_byte_ms` marks the
   **first** chunk. This is why §5 requires `kokoro_local.py` to re-chunk into ~50 ms pieces: a
   generator that yields one array per sentence makes "first byte" and "last byte" the same
   instant and silently invalidates the 90–200 ms TTS row.
6. **`audio_first_frame_out_ms` is set from the PortAudio callback**, via `on_playout_start`, not
   at enqueue time. Only the callback knows when a frame actually left. On the LOCAL build this is
   `t_v2v`'s endpoint; marking it at enqueue understates by the full device latency, i.e. flatters
   the number by 20–50 ms.
7. **`contextlib.aclosing` over a bare `try/finally`.** §6.3 declares
   `stream_chat(messages) -> DeltaStream`: an awaitable returning an async-iterable object, so the
   call is awaited and the awaited result is what `aclosing()` wraps. `DeltaStream`, `_Prefixed`
   and `FakeLLM` therefore expose **`aclose()`**, not `close()` — `aclosing()` calls `aclose()` by
   name, and a stream exposing only `close()` raises `AttributeError` on exit. `aclose()` closes
   the `httpx` response body and exits the stream context on every path, cancellation included. Closing the body does not tell
   Groq to stop generating — see §14 item 6. And `self.sentencizer = type(self.sentencizer)()` per
   turn matters: the opening-clause rule is gated on `emitted == 0`, so a sentencizer reused across
   turns applies it once per session and every later turn silently loses the first-clause win.

**The TTS worker thread is not interruptible.** `run_in_executor` cancellation cancels the future,
not the thread, so on barge-in the Kokoro worker finishes its current ~50 ms chunk. This does
**not** delay audible stop — `player.abort()` already silenced the device — it only delays reuse
of `tts_pool`. `aclosing` on the TTS generator then closes the underlying sync generator so no
orphan chunk is enqueued.

**Trigger policy:** require ~100–200 ms of continuous VAD speech before cancelling, and
optionally 2–3 words from the partial transcript (Pipecat ships
`MinWordsUserTurnStartStrategy(min_words=3)` for exactly this). An interview coach hears
"mm-hmm" and "right" constantly; barging in on a backchannel is as bad as ignoring a real
interruption.

**The eight marks that form the spine.** §6.7's table has forty fields; these eight are the
critical path, and every one is a `t0`-relative float in `turnlog/v2`:

| # | Mark | Set by |
|---|---|---|
| 1 | `vad_silence_armed_ms` | endpointer, on CANDIDATE_END |
| 2 | `smart_turn_done_ms` | endpointer, when the classifier returns `p` |
| 3 | `stt_final_ms` **or** `stt_flush_timeout_ms` | `on_endpoint` (§8.1) |
| 4 | `llm_request_sent_ms` | `_respond` |
| 5 | `llm_first_speakable_token_ms` | first SSE content delta |
| 6 | `llm_first_sentence_ms` | first sentencizer emission |
| 7 | `tts_first_byte_ms` | first PCM chunk from the TTS generator |
| 8 | **`audio_first_frame_out_ms`** (LOCAL) / **`client_first_audio_played_ms`** (browser) | PortAudio callback / browser playhead — **the `t_v2v` endpoint** |

Barge-in marks (`barge_in_detected_ms`, `audio_abort_ms`, `turn_unwound_ms`,
`client_audio_silent_ms`) are a separate event class and are reported separately, per §11 failure
#5. The audible-stop gap you want is `client_audio_silent_ms − audio_abort_ms`.

### §8.3 Evaluating the endpointer

This section opens "this is where the project is won or lost," and §9.7 ranks interview-tuned
endpointing as differentiator #5. A claim with no measurement behind it is the exact thing this
report tells you not to publish. So `bench/endpoint.py` measures it.

#### What to measure

Four numbers, all against hand labels:

| Metric | Definition | Target |
|---|---|---|
| **FCR — false-cutoff rate** | % of clips where `END_TURN` fires at a time earlier than the labelled true end-of-turn | **≤ 5%** on the thinking-pause subset |
| **Added latency** | `t_fire − t_true_eot`, over clips that were *not* false cutoffs. Report **median and P90**, in ms | Median ≤ 400 ms |
| **SLA-fallback rate** | % of clips ended only by the `stop_secs` timer, i.e. smart-turn never crossed the threshold | ≤ 15% |
| **Words lost** | On a false cutoff, the number of words in the reference transcript after `t_fire`. **Median and max** | Max ≤ 4 words |

FCR and added latency trade against each other, and the whole point is to see the trade instead
of guessing at it. The best published comparator is LiveKit's turn detector v1 at **9.9%
false-cutoff at 300 ms latency**, on general conversational data. Interview answers are harder —
the 2–4 s mid-answer pause is the *normal* case here, not the tail — so the target is a lower FCR
bought with more latency, and saying so with a curve is the differentiator.

"Words lost" exists because FCR alone is misleading. Cutting someone off 200 ms before they
finish the word "and" is a rounding error. Cutting them off before the sentence that contains
the only number in their STAR answer destroys the rubric score. Report both.

#### The fixture set

**40 self-recorded answers, `data/fixtures/endpoint/`, 16 kHz mono WAV**, recorded on the demo
machine with the demo microphone. Composition:

| Subset | N | Content |
|---|---|---|
| `pause` | 24 | A real 45–90 s behavioural answer containing **1–2 deliberate mid-answer pauses of 2.0–4.0 s** — genuinely stop to recall a number or decide how to frame something. This is the subset FCR is scored on |
| `clean` | 10 | An answer ending with clear falling prosody, no interior pause > 800 ms. Measures added latency in the easy case |
| `trailing` | 6 | An answer ending on a filler or a rising tone — *"...so, yeah, um"* / *"...I think?"*. The genuinely ambiguous case. Expect most SLA fallbacks here |

Every clip carries **3.5 s of trailing room tone** after the last word, so the 3.0 s `stop_secs`
timer can actually fire inside the file. Without that the SLA-fallback rate is unmeasurable.

Label sidecar, `data/fixtures/endpoint/<id>.json`:

```json
{
  "id": "eo_014",
  "question_id": "q02",
  "subset": "pause",
  "true_eot_ms": 41230,
  "interior_pauses_ms": [[12850, 15600], [27310, 30020]],
  "reference": "so the deadline moved up by three weeks and I had to ... we shipped it on the 14th",
  "word_offsets_ms": [[0, 120], [130, 310]],
  "notes": "4.0 s pause at 27.3 s while recalling the revenue number"
}
```

Label `true_eot_ms` as the **offset of the final word's release**, to ±50 ms, in any waveform
editor. Budget ~2 minutes per clip. Label the interior pauses too — they are what lets the
harness report *which* pause caused a cut, which is the debugging signal.

**Commit the labels; gitignore the WAVs** (40 × ~60 s of 16 kHz PCM ≈ 75 MB), and publish two or
three clips plus their labels so the methodology is reproducible without the whole set. Run
`bench/scrub.py` over the labels before committing (§6.10).

#### `bench/endpoint.py`

The naive sweep re-runs every clip for every (threshold × arming-silence) pair: 40 clips ×
4 arming values × 13 thresholds = 2,080 ONNX passes over the fixtures. That is unnecessary.
**Whether a candidate endpoint is *accepted* depends only on the threshold; whether a candidate
*exists* depends only on the VAD arming silence.** A rejected candidate returns the machine to
`SPEAKING` either way, so the candidate set is threshold-independent. Run each clip once per
arming value, record every `(t_ms, p)`, and sweep thresholds offline — 160 passes instead of
2,080, and the sweep itself is instant.

```python
"""bench/endpoint.py — evaluate the two-stage endpointer against hand labels.

    python -m bench.endpoint --fixtures data/fixtures/endpoint --out runs/
    python -m bench.endpoint --fixtures ... --plot runs/endpoint-roc.png

Replays each labelled WAV through the real Silero VAD + smart-turn v3.2 path once per
arming-silence value, recording every candidate endpoint as (t_ms, p). Decision thresholds
are then swept offline, which is exact: a candidate's existence does not depend on tau.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from coach.turn.endpointer import SmartTurn        # p(complete) over the last 8 s
from coach.turn.vad import SileroVAD               # enforces the 512-sample contract

FRAME = 512                                        # samples @ 16 kHz = 32 ms
ARM_MS = (150, 200, 300, 400)
TAUS = tuple(round(0.30 + 0.05 * i, 2) for i in range(13))   # 0.30 .. 0.90
STOP_SECS = 3.0


@dataclass
class Candidate:
    t_ms: int          # when the candidate endpoint was evaluated, from clip start
    p: float           # smart-turn p(complete)


@dataclass
class ClipTrace:
    clip_id: str
    subset: str
    true_eot_ms: int
    last_speech_end_ms: int
    n_words: int
    word_offsets_ms: list[tuple[int, int]]
    candidates: list[Candidate]


def trace_clip(wav: Path, label: dict, vad: SileroVAD, st: SmartTurn, arm_ms: int) -> ClipTrace:
    audio, sr = sf.read(wav, dtype="int16")
    assert sr == 16_000 and audio.ndim == 1, f"{wav}: want 16 kHz mono, got {sr} Hz"

    candidates: list[Candidate] = []
    silence_ms = 0
    last_speech_end_ms = 0
    armed = False
    ring = np.zeros(16_000 * 8, dtype=np.int16)     # smart-turn's 8 s window

    for i in range(0, len(audio) - FRAME + 1, FRAME):
        frame = audio[i : i + FRAME]
        t_ms = (i + FRAME) * 1000 // sr
        ring = np.concatenate([ring[FRAME:], frame])

        if vad.is_speech(frame):
            silence_ms = 0
            armed = False
            last_speech_end_ms = t_ms
        else:
            silence_ms += FRAME * 1000 // sr
            if silence_ms >= arm_ms and not armed:
                armed = True                        # one candidate per silence run
                candidates.append(Candidate(t_ms=t_ms, p=float(st.p_complete(ring))))

    return ClipTrace(
        clip_id=label["id"],
        subset=label["subset"],
        true_eot_ms=label["true_eot_ms"],
        last_speech_end_ms=last_speech_end_ms,
        n_words=len(label.get("word_offsets_ms", [])),
        word_offsets_ms=[tuple(w) for w in label.get("word_offsets_ms", [])],
        candidates=candidates,
    )


def fire_time(trace: ClipTrace, tau: float) -> tuple[int, bool]:
    """(t_fire_ms, hit_sla). First candidate above tau wins; else the stop_secs timer."""
    sla_ms = trace.last_speech_end_ms + int(STOP_SECS * 1000)
    for c in trace.candidates:
        if c.t_ms > sla_ms:
            break
        if c.p > tau:
            return c.t_ms, False
    return sla_ms, True


def words_after(trace: ClipTrace, t_ms: int) -> int:
    return sum(1 for start, _end in trace.word_offsets_ms if start >= t_ms)


def score(traces: list[ClipTrace], tau: float) -> dict:
    cuts, latencies, slas, lost = [], [], 0, []
    for tr in traces:
        t_fire, hit_sla = fire_time(tr, tau)
        slas += hit_sla
        if t_fire < tr.true_eot_ms:
            cuts.append(tr.clip_id)
            lost.append(words_after(tr, t_fire))
        else:
            latencies.append(t_fire - tr.true_eot_ms)
    n = len(traces)
    return {
        "tau": tau,
        "n": n,
        "fcr_pct": round(100 * len(cuts) / n, 1),
        "added_latency_p50_ms": int(statistics.median(latencies)) if latencies else None,
        "added_latency_p90_ms": (
            int(statistics.quantiles(latencies, n=10)[8]) if len(latencies) >= 10 else None
        ),
        "sla_fallback_pct": round(100 * slas / n, 1),
        "words_lost_median": statistics.median(lost) if lost else 0,
        "words_lost_max": max(lost) if lost else 0,
        "cut_clips": cuts,
    }


def pick_operating_point(rows: list[dict], max_fcr_pct: float = 5.0) -> dict | None:
    """Lowest tau (fastest) whose FCR clears the bar. Ties broken on median latency."""
    ok = [r for r in rows if r["fcr_pct"] <= max_fcr_pct and r["added_latency_p50_ms"] is not None]
    return min(ok, key=lambda r: (r["added_latency_p50_ms"], r["tau"])) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--subset", default="pause", help="subset FCR is scored on")
    ap.add_argument("--max-fcr", type=float, default=5.0)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()

    labels = [json.loads(p.read_text()) for p in sorted(args.fixtures.glob("*.json"))]
    vad, st = SileroVAD(), SmartTurn()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "nogit"

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"endpoint-{sha}.jsonl"
    picks: list[dict] = []
    with out.open("w") as fh:
        for arm_ms in ARM_MS:
            traces = [trace_clip(args.fixtures / f"{lb['id']}.wav", lb, vad, st, arm_ms)
                      for lb in labels]
            scored = [tr for tr in traces if tr.subset == args.subset]
            rows = [score(scored, tau) | {"arm_ms": arm_ms, "subset": args.subset}
                    for tau in TAUS]
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            if pick := pick_operating_point(rows, args.max_fcr):
                picks.append(pick)

    if not picks:
        raise SystemExit(f"No (arm_ms, tau) reaches FCR <= {args.max_fcr}% on '{args.subset}'. "
                         f"Either the fixture set is too hard or smart-turn is the wrong tool "
                         f"for it — report that, do not relax the bar quietly.")
    # Total latency the user feels = arming silence + added latency after true EOT.
    best = min(picks, key=lambda r: (r["arm_ms"] + r["added_latency_p50_ms"], r["tau"]))
    print(f"operating point: arm_silence_ms={best['arm_ms']} threshold={best['tau']} "
          f"FCR={best['fcr_pct']}% p50_added={best['added_latency_p50_ms']}ms "
          f"p90={best['added_latency_p90_ms']}ms SLA={best['sla_fallback_pct']}% "
          f"words_lost_max={best['words_lost_max']}  (N={best['n']})")
    print(f"wrote {out}  ({len(ARM_MS) * len(TAUS)} rows)")


if __name__ == "__main__":
    main()
```

Two interface obligations this creates, both trivial and both already declared in §6.3:

- `turn/vad.py` exposes `SileroVAD.is_speech(frame: np.ndarray[int16]) -> bool` for exactly
  512 samples.
- `turn/endpointer.py` exposes `SmartTurn.p_complete(window: np.ndarray[int16]) -> float`
  for exactly 8 s at 16 kHz, front-padded — **separable from the state machine**, so the harness
  can call the classifier without driving a live session.

If the classifier is not separable from the state machine, it cannot be evaluated. Build it
separable on day one.

#### Reporting it

`bench/endpoint.py --plot` emits FCR (x) against median added latency (y), one line per
`arm_ms`, points labelled with τ — the shape §9.7 #5 is claiming and currently cannot show. The
README table is:

| arm_ms | τ | FCR % (pause subset) | added latency P50 / P90 ms | SLA-fallback % | words lost med / max |
|---|---|---|---|---|---|
| 200 | — | — | — | — | — |

**Leave those cells empty until the sweep runs. Do not pre-fill them from this document —
there is no measurement behind them yet.** The one number you may state before the sweep is the
decision rule: the operating point is the *lowest* τ whose FCR on the `pause` subset is ≤ 5%,
because latency you can feel is cheaper than a cutoff that destroys the answer, and the interview
domain is the reason the bar is 5% rather than LiveKit's 9.9%.

Also report the **degenerate baselines** on the same fixtures, in the same table — fixed 500 ms
silence, fixed 800 ms, fixed 1,500 ms. "Semantic endpointing beats a fixed threshold" is an
assertion in §7 optimisation #3 worth 300–600 ms; against these three rows it becomes a result.
The 500 ms row is the one that should look terrible on the `pause` subset, and if it does not,
that is a finding you publish rather than bury. If **no** (arm_ms, τ) pair clears 5%, that is also
a finding — a publishable negative result about smart-turn on interview-style pauses (§14) — and
the response is a longer `stop_secs`, not a quietly relaxed bar.

### §8.4 Echo cancellation

Without AEC, the bot's own TTS trips the VAD, fires a barge-in, cancels itself mid-sentence, and
loops. **This is the most likely way a live voice demo falls apart.**

| Option | Works on macOS ARM from Python? | Effort | Notes |
|---|---|---|---|
| **Browser `getUserMedia({audio:{echoCancellation:true, noiseSuppression:true, autoGainControl:true}})`** | Yes — AEC runs in Chrome; Python just receives PCM | Low | **Baseline across browsers since January 2020.** Also gives you the demo UI |
| Loopback `RTCPeerConnection` for TTS playback | Yes (browser-side) | ~half a day | Routes local TTS through a peer connection so the AEC reference includes the bot's voice. Coverage of plain Web Audio playback is genuinely ambiguous — a demo repo shows it failing, while Chrome's own blog describes an internal loopback that implies it works. **Test on your exact demo machine and browser** |
| `livekit` `rtc.AudioProcessingModule` | Yes (Apache-2.0, macOS arm64 wheel, livekit 1.1.18) | Medium | Real WebRTC APM. **Exactly 10 ms frames (160 samples @16 kHz)**; feed TTS to `process_reverse_stream()` and calibrate `set_stream_delay_ms()`. Budget a day for delay calibration. This is the +5 h contingency in §10 |
| `pywebrtc-audio` 0.2.0 | Yes (Apache-2.0, macOS arm64) | Medium | Simpler `EchoCanceller.process(near, far)`; **v0.2.0 beta — pin it** |
| macOS `VoiceProcessingIO` | **No Python path** | High | Swift/Obj-C Core Audio only; PortAudio cannot reach it |
| Headphones, no AEC | Yes | Zero | **A legitimate demo posture — say so in the README.** Not a substitute for implementing the AEC path |
| Half-duplex mic gating while the bot speaks | Yes | Zero | **Never ship this.** It kills barge-in, the headline feature |

**Decision: browser AEC + loopback-RTCPeerConnection playback, with headphones as the
demo-day guarantee.** Test with speakers at demo volume, not headphones — that is how you will
actually demo it. The 15-minute smoke test at the end of Milestone 1 is the detection point:
play 20 s of Kokoro output through the demo speakers at demo volume with the mic live, and count
VAD speech-onsets. Non-zero means it failed, and §10's contingency fires.

---

## 9. Interview-coach domain design

### The competitive gap

The niche is unusually weak. **Google Interview Warmup was retired around April 2026** — its URL
now serves a generic prep article. **Pramp** survives as Exponent Practice with ~5 free peer
credits/month. **Yoodli** owns the speech-metrics niche but caps its free tier at 5 *lifetime*
sessions and presents metrics without baselines ("your pace is fast" — versus what?). On GitHub
the bar is low: the most-starred dedicated open-source AI mock interviewer is
`IliaLarchenko/Interviewer` at **127 stars** (Gradio, no barge-in); every voice-based one found
is under ~60 stars, and the closest full-duplex Groq+Whisper clone has 0 stars.

The recurring complaints about the loudest commercial player are literally this project's
thesis: lag, failure to detect when the speaker finished, over-long responses, and feedback that
"sounds heavily AI-generated."

`PROMPT_VERSION = "coach-2026-09-13a"` is stamped on every score record. Calibration κ is a
property of a prompt version, not of "the project."

### §9.1 Division of labour

The product's behaviour is split across three layers, and the split is the design:

| Layer | Decides | Cost | Deterministic? |
|---|---|---|---|
| **Gap detectors** (`probe.py`, local regex/counting) | Which rubric slots the answer left empty | 0 tokens, <1 ms | Yes |
| **Session state machine** (`session.py`) | Which question, whether to probe, which probe, when to move on, when to wrap | 0 tokens | Yes |
| **Hot-path LLM** (`qwen/qwen3.6-27b`) | Only the *wording* of a probe it was handed, anchored to a phrase the candidate used | ~1,585 tok/probe | No |
| **Scoring LLM** (`openai/gpt-oss-20b`, off hot path) | The level and the evidence span, one competency per call | ~530 tok/call cached | Temperature 0 |

**The LLM does not choose questions or probes. The state machine does.** The LLM voices a
directive it is handed. That is why the demo is reproducible and why the token bill is small.

The gap detectors are deliberately **recall-biased**: they over-fire. A false probe costs one
follow-up question, which is in character for an interviewer. A missed gap costs the product's
only differentiator. They are a trigger, not a score — the score comes from the LLM pass, and the
two are allowed to disagree. `bench/calibrate.py` should report that disagreement rate; it is a
more interesting number than the detectors' own accuracy.

Two further consequences, both load-bearing elsewhere in this document. **All fixed speech is
pre-rendered Kokoro audio at startup** — the intro, all 14 questions, all 5 move-on lines, the
rung-1 filler and the 429 fallback: 21 clips. Only probes and the wrap go through the LLM on the
hot path. And **the model never emits timestamps**: it emits a verbatim quote, and the server
attaches `t_start`/`t_end` by matching that quote against the word-timed transcript. A quote that
does not match is a discarded score, not a fabricated timestamp.

### §9.2 The rubric

Anchor it in real I/O psychology rather than invented dimensions. **Structured interviews are
the strongest single predictor of job performance (ρ = .42, Sackett et al. 2022, above cognitive
ability tests at .31)**, and what makes them reliable is behaviourally anchored rating scales.

**On the external reliability benchmark, one correction that matters.** Earlier drafts of this
document cited "0.77–0.84 human interrater reliability" and attributed it to ETS RR-17-28. **That
figure is not in that paper.** Checked 2026-09-13 against the open-access copy: Kell et al. (2017)
report **ICC = .74** for three raters across 12 questions, per dimension .66–.82, over 652 usable
**written** responses on a **7-point** scale. Use .74 / .66–.82, call it an ICC, and treat it as
external context in a footnote — never as "the bar" — because it differs from this project on
every axis that moves the statistic. The quantity actually comparable to this project is the
second-rater measurement in §9.8.

Use 3–5 competencies with explicit behavioural anchors per level, not a 1–10 "quality" score.
The UK Civil Service Success Profiles behaviours framework is publicly published with
level-by-level behavioural indicators and is free to adapt in *format*.

The anchors below are original, written against the STAR structure and the Success Profiles
format. **They are not a validated instrument.** No published interrater reliability attaches to
these specific anchors; the ICC figure above is the benchmark you measure *against*, not a
property you inherit by copying a format. Say that in the README.

Levels 2 and 4 carry no written anchor by design: standard BARS practice is to anchor the poles
and the midpoint and let raters interpolate. The scoring prompt states the interpolation rule
explicitly so the model does not invent its own.

```yaml
# config/rubric/behavioral_v1.yaml
version: behavioral_v1
scale:
  min: 1
  max: 5
  anchored_levels: [1, 3, 5]
  interpolation: "2 = above the level-1 anchor, short of level 3. 4 = above level 3, short of level 5."
composite_score: false   # deliberate. See notes.composite below.

competencies:

  - id: situation_context
    name: "Situation and context"
    asks: "Did the candidate establish enough context to make the story legible to someone who was not there?"
    anchors:
      1: "No time, place, role or stakes. A listener cannot tell what kind of organisation this was, what the candidate's job was, or why any of it mattered. Example shape: 'We had a problem with the pipeline, so I fixed it.'"
      3: "Two of {when or where, the candidate's role, what was at risk} are stated outright. The third is inferable but never said."
      5: "When or where, the candidate's role, what was at risk, and the binding constraint (deadline, headcount, budget, a dependency they did not control) are all stated, and stated before the first action is described."
    not_scored:
      - "Length of the setup. A 20-second setup that carries all four elements scores 5; a 90-second one that carries two scores 3."

  - id: individual_ownership
    name: "Individual ownership"
    asks: "Is the candidate's own contribution distinguishable from the team's?"
    anchors:
      1: "No sentence attributes a specific decision, action or artefact to the candidate. Their contribution is inferable only from the fact that they are the one telling the story."
      3: "At least one concrete action is attributed to the candidate, but the boundary between their work and the team's is never drawn — a listener cannot say what would have been different had the candidate not been there."
      5: "At least two specific decisions or artefacts are attributed to the candidate, and at least one sentence explicitly marks the boundary, e.g. 'Priya owned the migration itself, I owned the rollback plan and the go/no-go call.'"
    not_scored:
      - "Pronoun choice. 'We' is NOT evidence of absent ownership and must never be cited as evidence against this competency. The recruiting convention that 'we' is a red flag has no peer-reviewed support and systematically penalises collectivist cultures and genuinely collaborative roles. Score the presence of attributed specifics, not the grammar of the pronoun."

  - id: action_specificity
    name: "Action specificity"
    asks: "Could a listener reconstruct what the candidate actually did, in order?"
    anchors:
      1: "Actions given only as categories or intentions — 'I aligned stakeholders', 'I improved the process', 'I made sure everyone was on the same page'. No named tool, artefact, sequence or decision point."
      3: "A sequence of at least two concrete actions with real objects ('I wrote a query against the events table, then took the numbers to the weekly review'), but at least one major step is still a category, or the order is not recoverable."
      5: "A recoverable sequence of three or more concrete actions, each with an object, plus at least one stated decision point that names the alternative the candidate rejected and why."
    not_scored:
      - "Technical depth or domain seniority. A specific account of scheduling three conversations scores the same as a specific account of a schema migration."

  - id: quantified_result
    name: "Quantified result"
    asks: "Is the outcome stated as a magnitude a listener could in principle check?"
    anchors:
      1: "No outcome, or the outcome given only as a valence — 'it went well', 'the team was happy', 'it was a big success'."
      3: "A magnitude with no baseline and no timeframe ('we cut the latency', 'saved about ten hours'), OR a baseline with no magnitude ('it used to take all week')."
      5: "Magnitude with a unit, plus a baseline or a timeframe, plus how it was measured or who else saw it: 'p95 went from 4.1 seconds to 900 milliseconds over six weeks, on the same dashboard the SRE team already watched.'"
    not_scored:
      - "How impressive the number is. A small honest number with a source scores 5. A large unsourced one scores 3. This competency measures verifiability, not impact."

  - id: reflection
    name: "Reflection"
    asks: "Does the candidate show what the experience cost them and what they changed as a result?"
    anchors:
      1: "No retrospective content, or retrospection used only as self-praise — 'it really showed I work well under pressure'."
      3: "A lesson is stated but is detachable from the story — 'I learned communication is important' would fit any answer to any question."
      5: "Names something specific that did not work, or a cost that was actually paid, AND states a concrete change in later behaviour, AND gives evidence the change happened ('so on the next migration I wrote the rollback before the forward script — which is why the December one took twenty minutes to undo')."
    not_scored:
      - "Whether the candidate sounds humble, apologetic or confident. Score the specificity of the named change, not the emotional register."

notes:
  composite: >
    No competency weights and no overall score. There is no validated weighting for these five
    against any job, a composite invites exactly the single-number "hireability" framing the
    project rejects, and averaging a 5 in quantified_result against a 1 in reflection destroys
    the only information the report has. The report names the strongest and weakest dimension
    and shows all five levels. It never adds them up.
  never_score:
    - accent, fluency, articulateness, vocabulary
    - pace, pauses, disfluency, filler words
    - confidence, enthusiasm, warmth, "culture fit", "hireability"
    - anything not present as words in the transcript
```

**Every score must cite the verbatim transcript span (with a word-level timestamp) that
justifies it, and the validator must reject any score whose evidence field is empty.** This is
what kills the "generic AI praise" complaint, and it makes the feedback clickable in the report.

Score each dimension in a **separate LLM call**. The literature on LLM-as-a-judge documents
position bias (>10% accuracy swings from reordering in pairwise judging) and verbosity bias
(longer, more fluent answers score higher regardless of substance). Verbosity bias is the exact
anti-pattern an interview coach must not reward — a waffly confident answer outscoring a concise
correct one.

### §9.3 The coach system prompt (hot path)

Paste as `COACH_SYSTEM_PROMPT` in `src/coach/coachlogic/prompts.py`. ~400 tokens. It is entirely
static so it sits at the front of R1 (§3.1); session state arrives as a separate user message.

```python
PROMPT_VERSION = "coach-2026-09-13a"

COACH_SYSTEM_PROMPT = """\
You are the interviewer in a practice behavioural interview. The candidate is speaking out loud,
and everything you say is turned into speech and heard immediately. Write for the ear.

YOUR ONLY JOB THIS TURN is to say one short follow-up question. You are not scoring, not
summarising, and not giving advice. Written feedback is produced separately after the session.
Anything you say now is time taken from the candidate.

HARD RULES
1. One or two sentences. Never more than 45 words.
2. Plain speech. No markdown, no bullet points, no headings, no emoji, no stage directions, no
   numbered steps, no "Great question".
3. Name something the candidate actually said. If you cannot point at a concrete noun from their
   answer, ask the shortest version of the directive question instead of inventing a detail.
4. Never invent facts about the candidate, the company, the numbers or the outcome. Never replay
   their answer back to them at length.
5. Do not evaluate. No "that's a strong example", no "good", no "I like that". A neutral two- or
   three-word acknowledgement is allowed, then the question.
6. Never say the words rubric, competency, score, dimension, criterion or evidence. The candidate
   must not hear the machinery.
7. Exactly one question. Never stack two questions into one turn.
8. Never comment on accent, fluency, pace, pauses, filler words, nervousness or tone of voice.
   You do not have that information and it is not what is being practised.
9. If the transcript is garbled, empty, or under about fifteen words, say plainly that you did
   not catch it and ask them to take the answer again from the top.

STYLE
Warm, brisk, faintly clipped. The register of a good hiring manager who is genuinely interested
and has another meeting at eleven. Contractions are fine.

You will be given a DIRECTIVE naming exactly what to ask. Follow it. If the directive's template
contains {anchor}, replace it with a phrase the candidate actually used, in their words.
"""
```

**Runtime assembly.** Note the ordering: static system prompt, then history, then the *variable*
directive last. That is §3.1's R1/R2/R3 layout applied to the hot path — correct for prefix
caching if the hot path ever moves onto a gpt-oss model, and the order that keeps the directive
nearest the generation point.

```python
# src/coach/coachlogic/prompts.py

def build_probe_messages(block, directive, history_budget_tokens: int = 900) -> list[dict]:
    """Messages for one PROBE turn. `history` is already token-capped by the caller."""
    return [
        {"role": "system", "content": COACH_SYSTEM_PROMPT},
        *block.history.as_messages(budget=history_budget_tokens),
        {"role": "user", "content": (
            f"QUESTION ASKED\n{block.question.text}\n\n"
            f"CANDIDATE'S ANSWER (automatic transcript, may contain errors)\n"
            f"{block.answer_text}\n\n"
            f"DIRECTIVE\n"
            f"  intent: PROBE\n"
            f"  ask_about: {directive.competency}\n"
            f"  template: {directive.template!r}\n"
            f"  probes_used: {block.probes_used} of {MAX_PROBES_PER_QUESTION}\n"
        )},
    ]

WRAP_DIRECTIVE = """\
DIRECTIVE
  intent: WRAP
  Say goodbye in at most 45 words. Name the one dimension that was strongest and the one that was
  weakest, in plain English without using the competency ids, quote at most eight words the
  candidate actually said, and name one concrete thing to change next time. Then say the full
  report is on screen. Do not apologise, do not thank them twice, do not list.
"""
```

Hot-path call settings, restated here so `prompts.py` is self-contained: `model` from config
(`qwen/qwen3.6-27b`), `reasoning_effort="none"`, `stream=True`, `max_completion_tokens=120`,
`temperature=0.6`. 120 tokens is ~90 words — comfortably above the 45-word rule, so the cap is a
runaway guard, not the thing enforcing brevity.

### §9.4 Scoring prompts

**Model decision, and why it differs from the hot path.** Verified on
`console.groq.com/docs/structured-outputs.md`, 2026-09-13: the models supporting `json_schema`
with `strict: true` are `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`.
**`qwen/qwen3.6-27b` — the hot-path model — is not on the list in either strict or best-effort
mode.** All models support the weaker `json_object` mode, which guarantees syntactic JSON but not
your fields.

So the scoring path runs on **`openai/gpt-oss-20b`, `reasoning_effort:"low"`,
`response_format={"type":"json_schema","json_schema":{...,"strict":true}}`, `temperature=0`**.
Four reasons, in order:

1. Strict schema means no JSON-repair code path and no malformed-output retries — the failure
   mode that survives is *wrong content*, which is exactly what calibration is for.
2. It is a Production model. The scoring path produces the artifact a recruiter reads; it is the
   wrong place to carry Preview risk.
3. **Prompt caching, gpt-oss only.** The static prefix (instructions + full rubric) is
   byte-identical across all 25 scoring calls in a session, so calls 2–25 pay only ~530 tokens
   against the 8,000 TPM cap. That is what makes per-dimension scoring affordable at all.
4. Its 3.55 s time-to-first-answer-token is a hot-path disqualifier and completely irrelevant here.

Fallback: `qwen/qwen3.8-27b`, strict mode, no caching — costs ~1,730 tokens/call instead of ~530.
Structured outputs **do not support streaming or tool use**; scoring needs neither.

**The one thin assumption in this design, named.** Caching has a documented minimum prefix length
that "varies by model, ranging from 128 to 1024 tokens"; the exact figure for gpt-oss-20b is not
published. **The prefix as written below is 1,100–1,260 tokens — estimated, not
tokenizer-measured** (828 words; 1.33 tokens/word gives 1,101, chars/4 gives 1,261). That clears
1,024 by only 8–23%, which is thinner than it should be for a load-bearing assumption. Two things
follow. First, count it properly and read `usage.prompt_tokens_details.cached_tokens` on scoring
call 2 — that is ground truth and it takes one request (§3.1's `confirm_caching`). Second, if it
does not cache, lengthen the prefix with content that is actually useful rather than padding:
append each competency's `not_scored` list in full (a further ~200 tokens, and it demonstrably
reduces off-competency scoring) before you consider anything else. Filed as §14 item 24.

#### The static prefix (shared by all 25 calls)

```python
SCORING_SYSTEM_PREFIX = """\
You are a rating instrument. You are not a coach and not a conversationalist. You apply one
behavioural rating scale to one transcript and return one JSON object.

METHOD
1. Read the answer once. Find the span of the candidate's own words that bears most directly on
   the named competency.
2. Copy that span into evidence_text VERBATIM. Verbatim means word for word from the transcript
   above: same words, same order, no paraphrase, no ellipsis, no repair of grammar or
   disfluency, no added punctuation. Between 4 and 40 words.
3. Choose the level whose anchor that span satisfies. Levels 1, 3 and 5 have written anchors.
   Use 2 for behaviour clearly above the level-1 anchor but short of level 3, and 4 for
   behaviour above level 3 but short of level 5.
4. Write rationale: one sentence, at most 30 words, naming which part of the anchor the evidence
   does or does not meet. Not advice. Not encouragement. Not a summary of the answer.

RULES THAT OVERRIDE YOUR PRIORS
- Length is not evidence. A 250-word answer with no measured outcome scores lower on
  quantified_result than a 40-word answer with one. Fluency, vocabulary and confidence are not
  evidence of anything on this scale.
- The transcript came from automatic speech recognition. It contains recognition errors, has
  little punctuation, and has had filler words removed by the recogniser. Never score grammar,
  articulacy, disfluency, pace or tone. You cannot hear the audio and you are not being asked to.
- Score only what was said. Do not credit what the candidate evidently knows but did not say, and
  do not penalise them for what a different question would have elicited.
- Score only the named competency. The other definitions are present so you can tell them apart,
  not so you can score them.
- If no span in the answer bears on this competency at all, set level to null and abstain_reason
  to "no_evidence". Do not stretch an unrelated span to fill the field. Abstaining is a correct
  answer and is recorded as one.
- Do NOT output timestamps. You do not have them. They are attached afterwards by matching your
  evidence_text against the word-timed transcript. If your quote is not verbatim, the match
  fails and your score is discarded.

Return exactly one JSON object and nothing else.

RATING SCALE: behavioral_v1
"""
```

The rubric block follows immediately, rendered from the YAML — all five competencies, every
anchor. Including all five is deliberate: it is what makes the prefix cacheable, and it improves
discriminant validity by telling the model where the boundaries are, so `quantified_result`
evidence does not get scored under `action_specificity`. The cost is a possible contamination
effect. Do not argue about it — **A/B it**: `bench/calibrate.py --prompt-variant full_rubric` vs
`--prompt-variant single_competency`, report κ_w for both, ship the winner. That is a half-day of
work and a far better answer to "how do you know your prompt is right" than a paragraph of
reasoning.

#### One scoring call in full — `quantified_result`

Everything above, then:

```
COMPETENCY DEFINITIONS (all five, for boundaries — score only the one named below)

situation_context — Did the candidate establish enough context to make the story legible to
  someone who was not there?
  1: No time, place, role or stakes. A listener cannot tell what kind of organisation this was,
     what the candidate's job was, or why any of it mattered.
  3: Two of {when or where, the candidate's role, what was at risk} are stated outright. The
     third is inferable but never said.
  5: When or where, the candidate's role, what was at risk, and the binding constraint are all
     stated, and stated before the first action is described.
  Do not score: length of the setup.

individual_ownership — Is the candidate's own contribution distinguishable from the team's?
  1: No sentence attributes a specific decision, action or artefact to the candidate.
  3: At least one concrete action is attributed to the candidate, but the boundary between their
     work and the team's is never drawn.
  5: At least two specific decisions or artefacts are attributed to the candidate, and at least
     one sentence explicitly marks the boundary.
  Do not score: pronoun choice. "We" is NOT evidence of absent ownership and must never be cited
     as evidence against this competency.

action_specificity — Could a listener reconstruct what the candidate actually did, in order?
  1: Actions given only as categories or intentions. No named tool, artefact, sequence or
     decision point.
  3: A sequence of at least two concrete actions with real objects, but at least one major step
     is still a category, or the order is not recoverable.
  5: Three or more concrete actions in recoverable order, each with an object, plus a stated
     decision point naming the rejected alternative.
  Do not score: technical depth or domain seniority.

quantified_result — Is the outcome stated as a magnitude a listener could in principle check?
  1: No outcome, or the outcome given only as a valence.
  3: A magnitude with no baseline and no timeframe, OR a baseline with no magnitude.
  5: Magnitude with a unit, plus a baseline or a timeframe, plus how it was measured or who else
     saw it.
  Do not score: how impressive the number is. This competency measures verifiability, not impact.

reflection — Does the candidate show what the experience cost them and what they changed?
  1: No retrospective content, or retrospection used only as self-praise.
  3: A lesson is stated but is detachable from the story.
  5: Names something specific that did not work or a cost actually paid, AND a concrete change in
     later behaviour, AND evidence the change happened.
  Do not score: whether the candidate sounds humble, apologetic or confident.

---
SCORE ONLY: quantified_result

QUESTION ASKED
Walk me through the change you made in the last two years that had the biggest measurable effect.
What was the number before, and what was it after?

ANSWER TRANSCRIPT
so the checkout page was really slow and people were complaining about it a lot in support
tickets I dug into it and found we were doing three separate calls to the pricing service on
every page load so I batched them into one and added a short cache and after that it was much
faster and the complaints basically stopped

---
```

Expected output for that answer:

```json
{"competency":"quantified_result","level":1,
 "evidence_text":"after that it was much faster and the complaints basically stopped",
 "rationale":"Outcome given only as a valence with no magnitude, baseline or timeframe, which is the level-1 anchor.",
 "abstain_reason":null}
```

#### The template for the other four

The prefix, the definitions block and the transcript are **byte-identical** across all five calls.
Only the `SCORE ONLY:` line changes. That is the entire template:

```python
def build_scoring_messages(competency_id: str, block, rubric_block: str) -> list[dict]:
    return [
        {"role": "system", "content": SCORING_SYSTEM_PREFIX + rubric_block},   # cached prefix
        {"role": "user", "content": (
            f"SCORE ONLY: {competency_id}\n\n"
            f"QUESTION ASKED\n{block.question.text}\n\n"
            f"ANSWER TRANSCRIPT\n{block.full_text()}\n"
        )},
    ]
```

`block.full_text()` is the candidate's original answer plus every probe answer for that question,
concatenated in order — one score per competency per *question block*, not per utterance.
Otherwise a probe that successfully extracts a number produces a separate, contradictory score.

The repair prompt, used exactly once when span matching fails:

```python
REPAIR_SUFFIX = """\
Your previous evidence_text did not appear in the transcript, so the score was discarded.
Copy the span again, character for character, from the ANSWER TRANSCRIPT above. Do not repair
grammar. Do not shorten with an ellipsis. Do not join two non-adjacent phrases. If no span of
four or more consecutive words supports any level, set level to null and abstain_reason to
"no_evidence".
"""
```

### §9.5 The score object: JSON Schema and the span-matching rule

Two schemas. The model produces the first; the server produces the second.

#### `schemas/score_response.schema.json` — what the model returns

Sent as `response_format.json_schema.schema` with `strict: true`, which requires every property
listed in `required` and `additionalProperties: false`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ScoreResponse",
  "type": "object",
  "additionalProperties": false,
  "required": ["competency", "level", "evidence_text", "rationale", "abstain_reason"],
  "properties": {
    "competency": {
      "type": "string",
      "enum": ["situation_context", "individual_ownership", "action_specificity",
               "quantified_result", "reflection"]
    },
    "level": {
      "type": ["integer", "null"], "minimum": 1, "maximum": 5,
      "description": "null if and only if abstain_reason is set"
    },
    "evidence_text": {
      "type": "string", "maxLength": 400,
      "description": "Verbatim contiguous span of the candidate's words, 4-40 words. Empty string only when abstaining."
    },
    "rationale": { "type": "string", "maxLength": 220 },
    "abstain_reason": { "type": ["string", "null"], "enum": ["no_evidence", "answer_too_short", null] }
  }
}
```

#### `schemas/score_record.schema.json` — what is stored and what the report reads

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ScoreRecord",
  "type": "object",
  "additionalProperties": false,
  "required": ["session_id", "block_id", "question_id", "competency", "status", "level",
               "evidence", "rationale", "model", "prompt_version", "rubric_version", "scored_at"],
  "properties": {
    "session_id":     { "type": "string" },
    "block_id":       { "type": "integer", "minimum": 0 },
    "question_id":    { "type": "string", "pattern": "^q[0-9]{2}$" },
    "competency":     { "type": "string" },
    "status":         { "type": "string", "enum": ["scored", "abstained", "rejected"] },
    "level":          { "type": ["integer", "null"], "minimum": 1, "maximum": 5 },
    "rationale":      { "type": "string" },
    "evidence": {
      "type": ["object", "null"],
      "additionalProperties": false,
      "required": ["text", "t_start", "t_end", "word_start", "word_end", "match_ratio", "match_mode"],
      "properties": {
        "text":       { "type": "string", "description": "The transcript span as transcribed, NOT as quoted by the model" },
        "t_start":    { "type": "number", "minimum": 0, "description": "seconds, session-relative" },
        "t_end":      { "type": "number", "minimum": 0 },
        "word_start": { "type": "integer", "minimum": 0 },
        "word_end":   { "type": "integer", "minimum": 0 },
        "match_ratio":{ "type": "number", "minimum": 0, "maximum": 1 },
        "match_mode": { "type": "string", "enum": ["exact", "fuzzy"] }
      }
    },
    "reject_reason":  { "type": ["string", "null"],
                        "enum": ["evidence_unmatched", "evidence_length", "level_null_with_evidence",
                                 "schema_invalid", null] },
    "attempts":       { "type": "integer", "minimum": 1, "maximum": 2 },
    "model":          { "type": "string" },
    "prompt_version": { "type": "string" },
    "rubric_version": { "type": "string" },
    "scored_at":      { "type": "string", "format": "date-time" }
  }
}
```

Three field choices worth defending out loud:

- **`evidence.text` is the transcript's words, not the model's quote.** After a fuzzy match they
  can differ. The report must show what the candidate actually said, or the first person to click
  a citation and hear something different has caught you fabricating.
- **`t_start`/`t_end` are session-relative seconds**, not answer-relative, because the report's
  audio element seeks one session recording.
- **`status: "rejected"` is kept, not deleted.** `rejected / total` is your **ungrounded rate** —
  publish it in the README next to κ_w. A judge that abstains honestly 8% of the time is a better
  artifact than one that never abstains, and no competing project publishes this number.

#### The span-matching rule

**Normalisation.** Per-token, and strictly token-preserving on the transcript side — NFKC, casefold,
curly apostrophes to `'`, strip everything that is not word-character/apostrophe/hyphen, strip
leading and trailing hyphens and apostrophes. Nothing is merged, split or dropped, so normalised
token index `k` still maps to `words[k]`'s timestamps.

**Do not reuse Whisper's `EnglishTextNormalizer` here**, even though `bench/wer.py` and
`bench/disagree.py` use it. It expands contractions and rewrites numbers, which changes the token
count — and a normaliser that changes token count destroys the index-to-timestamp mapping the
whole feature rests on. Different job, different normaliser. Say this in the docstring or you will
"simplify" it later and break every citation.

**Matching.** Exact contiguous subsequence first. On failure, fuzzy: anchor on the rarest token in
the quote, walk every occurrence of it, try window lengths from `len(q)-2` to `len(q)+6`, score
each window with `difflib.SequenceMatcher(None, q, window).ratio()` — order-sensitive, stdlib, no
new dependency — and accept the best window at **ratio ≥ 0.80**.

0.80 is a starting value, not a measured one. It is roughly "one word in five may differ", which
covers a model that drops a repeated word or normalises a number the recogniser spelled out, and
rejects a model that paraphrases. **Sweep it in `bench/calibrate.py`** over the 50 hand-scored
answers and report the chosen value with the rejected/accepted counts at each threshold.

**Bounds.** Quote must be 4–40 words after normalisation. Below 4, "that I did the" matches
everywhere and the citation means nothing. Above 40, the model is quoting the whole answer to
avoid committing, which is the verbosity-bias failure wearing a disguise.

**On no match:** retry once with `REPAIR_SUFFIX`. If the retry also fails, store
`status: "rejected"`, `level: null`, `reject_reason: "evidence_unmatched"`. The report renders
that cell as "not assessed — the model could not ground this judgement in the transcript", which
is an honest sentence, and the run counts it in `ungrounded_rate`.

```python
# src/coach/coachlogic/rubric.py
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

MIN_EVIDENCE_WORDS = 4
MAX_EVIDENCE_WORDS = 40
FUZZY_THRESHOLD = 0.80
WINDOW_SLACK = range(-2, 7)          # len(q)-2 .. len(q)+6

_APOST = re.compile(r"[‘’ʼ]")
_STRIP = re.compile(r"[^\w'\-]+", re.UNICODE)


def norm_token(tok: str) -> str:
    """Token-preserving normalisation. May return '' for the query side only."""
    t = unicodedata.normalize("NFKC", tok).casefold()
    t = _APOST.sub("'", t)
    t = _STRIP.sub("", t)
    return t.strip("-'")


@dataclass(frozen=True)
class Word:
    text: str      # exactly as Moonshine emitted it
    start: float   # seconds, session-relative
    end: float


@dataclass(frozen=True)
class SpanMatch:
    word_start: int
    word_end: int      # inclusive
    ratio: float
    mode: str          # "exact" | "fuzzy"


class SpanMatcher:
    """Maps a model's quote back onto the word-timed transcript, or refuses."""

    def __init__(self, words: list[Word]):
        self.words = words
        self.norm = [norm_token(w.text) for w in words]
        self.freq = Counter(self.norm)
        self.index: dict[str, list[int]] = {}
        for k, t in enumerate(self.norm):
            self.index.setdefault(t, []).append(k)

    def match(self, quote: str) -> tuple[SpanMatch | None, str | None]:
        q = [t for t in (norm_token(x) for x in quote.split()) if t]
        if not MIN_EVIDENCE_WORDS <= len(q) <= MAX_EVIDENCE_WORDS:
            return None, "evidence_length"
        m = self._exact(q)
        if m is None:
            m = self._fuzzy(q)
        if m is None:
            return None, "evidence_unmatched"
        return m, None

    # --- the two strategies -------------------------------------------------

    def _exact(self, q: list[str]) -> SpanMatch | None:
        n = len(q)
        for k in self.index.get(q[0], ()):
            if self.norm[k:k + n] == q:
                return SpanMatch(k, k + n - 1, 1.0, "exact")
        return None

    def _fuzzy(self, q: list[str]) -> SpanMatch | None:
        anchor = min(q, key=lambda t: self.freq.get(t, 0) or 10**6)
        if self.freq.get(anchor, 0) == 0:
            anchor = max(q, key=len)                       # every token is OOV; fall back
        offsets = [p for p, t in enumerate(q) if t == anchor] or [0]
        best: SpanMatch | None = None
        for k in self.index.get(anchor, ()):
            for p in offsets:
                start = k - p
                if start < 0:
                    continue
                for slack in WINDOW_SLACK:
                    end = start + len(q) + slack           # exclusive
                    if end <= start or end > len(self.norm):
                        continue
                    r = difflib.SequenceMatcher(None, q, self.norm[start:end]).ratio()
                    if r >= FUZZY_THRESHOLD and (best is None or r > best.ratio):
                        best = SpanMatch(start, end - 1, r, "fuzzy")
        return best

    def span_text(self, m: SpanMatch) -> str:
        return " ".join(w.text for w in self.words[m.word_start:m.word_end + 1])
```

```python
def attach_evidence(resp: dict, matcher: SpanMatcher, *, meta: dict) -> dict:
    """ScoreResponse -> ScoreRecord. Never raises on model misbehaviour; records it."""
    rec = {**meta, "competency": resp["competency"], "rationale": resp["rationale"],
           "level": resp["level"], "evidence": None, "reject_reason": None}

    if resp["abstain_reason"] is not None:
        return {**rec, "status": "abstained", "level": None}

    if resp["level"] is None:                       # abstained without saying so
        return {**rec, "status": "abstained"}

    m, err = matcher.match(resp["evidence_text"])
    if m is None:
        return {**rec, "status": "rejected", "level": None, "reject_reason": err}

    return {**rec, "status": "scored", "evidence": {
        "text": matcher.span_text(m),
        "t_start": matcher.words[m.word_start].start,
        "t_end": matcher.words[m.word_end].end,
        "word_start": m.word_start, "word_end": m.word_end,
        "match_ratio": round(m.ratio, 4), "match_mode": m.mode,
    }}
```

**Behaviour of the rule as written**, run against the `q03` transcript from §9.4 above
(0.42 s/word synthetic timings). These eight cases are the first unit test in `tests/`:

| Model quote | Result |
|---|---|
| exact span | `exact`, ratio 1.000, t = 21.00–25.58 s |
| same span, one word dropped | `fuzzy`, ratio 0.952 — accepted |
| same span, re-cased and re-punctuated | `exact`, ratio 1.000 — normalisation absorbs it |
| a mid-answer span ("three separate calls to the pricing service…") | `exact`, ratio 1.000, t = 11.34–15.92 s |
| a paraphrase of the span | **rejected**, `evidence_unmatched` |
| "much faster" (2 words) | **rejected**, `evidence_length` |
| the entire answer quoted back | **rejected**, `evidence_length` |
| a fabricated metric never said aloud | **rejected**, `evidence_unmatched` |

The last row is the whole point: a model that invents "p95 went from 4.1 s to 900 ms" when the
candidate said "it was much faster" gets its score discarded, not rendered as a citation.

### §9.6 Session shape, question bank, probes, report, token cost

#### Session shape

| | Default | Demo preset |
|---|---|---|
| Questions | **5** | **2**, pinned to `[q10, q03]` |
| Probes per question | max **2** | max 2 |
| Target duration | **16 min** | **~5 min** |
| Hard cap | **22 min** (`SESSION_HARD_CAP_S = 1320`) | 8 min |
| Question block cap | **240 s** (`QUESTION_BLOCK_CAP_S`) | 240 s |
| Wrap reserve | **90 s** (`WRAP_RESERVE_S`) | 90 s |

Arithmetic behind 16 minutes: ask ~12 s (pre-rendered, instant) + answer 75–120 s + on average
1.5 probes × (8 s ask + 40 s answer) ≈ 3 min per block. Five blocks ≈ 15 min, plus a 40 s intro
and a 90 s wrap ≈ 17 min. Five is chosen because it is the smallest number that puts each of the
five competencies in a question's primary slot exactly once — below that the report has a
competency scored only incidentally, above that the session outruns a practice session's
attention and the 200,000 TPD cap.

*(The hosted build caps sessions at 6 minutes — one question, up to three probes, one wrap — for
the quota reasons in §6.8, not for product reasons.)*

**The demo preset pins explicit question ids rather than seeding a shuffle.** A live demo must be
byte-reproducible; `q10` (incident) is chosen to open because it gets a nervous speaker talking
about mechanics rather than about themselves, and `q03` because its `quantified_result` gap fires
on almost every first-time answer, which means the probe — the thing you are demoing — actually
happens.

#### State machine

```
BOOT ──(models warm, audio pre-rendered, /healthz green)──► INTRO
INTRO ──(pre-rendered clip finishes)──► ASK

ASK      play pre-rendered question audio (0 LLM tokens, 0 TTS latency)
            └──► LISTEN

LISTEN   §8 two-stage endpointer. On END_TURN ──► DECIDE
            └── barge-in during ASK/PROBE playback ──► LISTEN (TurnController.cancel)

DECIDE   (local, 0 tokens, <1 ms)
            reason = should_advance(block)
            reason is None  ──► PROBE
            reason not None ──► ADVANCE

PROBE    probe.select(block) ──► hot-path LLM ──► sentencizer ──► TTS
            block.probes_used += 1
            └──► LISTEN

ADVANCE  play pre-rendered move-on line for `reason`
            enqueue block for scoring (background, token-bucketed)
            blocks_done < question_count ──► ASK
            otherwise                     ──► WRAP

WRAP     await outstanding scoring futures (≤ 45 s budget) ──► one LLM call with WRAP_DIRECTIVE
            ──► REPORT ──► ENDED
```

```python
# src/coach/coachlogic/session.py
MAX_PROBES_PER_QUESTION = 2
QUESTION_BLOCK_CAP_S    = 240
SESSION_HARD_CAP_S      = 1320
WRAP_RESERVE_S          = 90
STALL_WORD_THRESHOLD    = 15

def should_advance(self, block) -> str | None:
    """Returns the move-on reason, or None to probe again. Order is priority order."""
    if self.session_elapsed_s >= SESSION_HARD_CAP_S - WRAP_RESERVE_S:
        return "time_cap"
    if block.elapsed_s >= QUESTION_BLOCK_CAP_S:
        return "time_cap"
    if block.probes_used >= MAX_PROBES_PER_QUESTION:
        return "probes_exhausted"
    if block.last_probe_new_words is not None and block.last_probe_new_words < STALL_WORD_THRESHOLD:
        return "answer_stalled"
    if not probe.gaps(block):
        return "gap_filled" if block.probes_used else "no_gap"
    return None
```

`answer_stalled` is the humane one: two short answers in a row means the candidate has nothing
more on this story, and a third probe is an interrogation. It is also what stops a demo dying on
a shrug.

#### Pre-rendered audio (Kokoro, at startup, 21 clips, cached to `runs/tts_cache/`)

This is §7 optimisation #8 taken to its conclusion: **the only speech that costs an LLM call is a
probe and the wrap.** Everything else is a WAV.

**INTRO** (~55 words, ~20 s) — carries the consent and retention statement §9.9 requires:

> Practice behavioural interview. I'll ask five questions, and follow up where an answer leaves
> something out. Take your time — pauses are fine, I won't cut in on you. You can interrupt me
> any time. Your audio and transcript stay on this machine and are deleted when you close the
> tab, unless you download the report. Ready? First question.

**Move-on lines** — exact text, one per `should_advance` reason. None of them ends with "next
question", so the question clip can be concatenated straight onto them, or the wrap on the last
block:

| `reason` | Exact line |
|---|---|
| `no_gap` | "That's a complete answer — nothing I'd chase there." |
| `gap_filled` | "Got it, that's much clearer." |
| `probes_exhausted` | "Let's leave that there, I've noted it for the report." |
| `answer_stalled` | "No problem, we can come back to that one." |
| `time_cap` | "I'm going to move us on to keep us on time." |

**Failure fallbacks**, also pre-rendered: one played whenever a Groq call 429s or times out on the
hot path, and one covering the rung-1 switch (§11.1). Better than a spinner:

> "Give me one second there — let's keep going. "
> "Let me think about that for a second."

#### Question bank — `config/questions/behavioral_v1.yaml`

14 questions. `primary` is the competency the question most reliably *elicits*; it drives
selection, not scoring. **Every answer is scored on all five competencies regardless of tag.**
`family` prevents a session asking two questions of the same shape.

```yaml
version: behavioral_v1
questions:
  - {id: q01, family: failure,        primary: individual_ownership, secondary: reflection,
     text: "Tell me about a time a project you were responsible for missed its deadline or its target. What happened?"}
  - {id: q02, family: conflict,       primary: action_specificity,   secondary: individual_ownership,
     text: "Describe a disagreement with a colleague about a technical or strategic decision. How did it end?"}
  - {id: q03, family: impact,         primary: quantified_result,    secondary: action_specificity,
     text: "Walk me through the change you made in the last two years that had the biggest measurable effect. What was the number before, and what was it after?"}
  - {id: q04, family: ambiguity,      primary: situation_context,    secondary: action_specificity,
     text: "Tell me about a time you had to start work with incomplete requirements. What did you do first?"}
  - {id: q05, family: influence,      primary: action_specificity,   secondary: individual_ownership,
     text: "Describe a time you changed someone's mind about something that mattered, without having any authority over them."}
  - {id: q06, family: prioritisation, primary: quantified_result,    secondary: action_specificity,
     text: "Tell me about a week where you had more committed work than time. How did you decide what to drop, and what did dropping it cost?"}
  - {id: q07, family: learning,       primary: reflection,           secondary: quantified_result,
     text: "Describe something you had to learn quickly in order to ship. How did you learn it, and how did you know it had worked?"}
  - {id: q08, family: feedback,       primary: reflection,           secondary: individual_ownership,
     text: "Tell me about a piece of critical feedback you received that you initially disagreed with."}
  - {id: q09, family: conflict,       primary: individual_ownership, secondary: action_specificity,
     text: "Describe a time you pushed back on a request from someone more senior than you. What was your argument?"}
  - {id: q10, family: incident,       primary: action_specificity,   secondary: situation_context,
     text: "Walk me through an incident or an outage you helped resolve. Start from the moment you found out."}
  - {id: q11, family: crossfunc,      primary: situation_context,    secondary: individual_ownership,
     text: "Tell me about a project that needed you to work with a team whose priorities were different from yours."}
  - {id: q12, family: tradeoff,       primary: reflection,           secondary: quantified_result,
     text: "Describe a time you shipped something you knew was not as good as it should have been. Why, and what happened afterwards?"}
  - {id: q13, family: initiative,     primary: individual_ownership, secondary: quantified_result,
     text: "Tell me about something you built or changed that nobody asked you to."}
  - {id: q14, family: mentoring,      primary: action_specificity,   secondary: quantified_result,
     text: "Describe a time you helped someone else get better at their job. What specifically did you do?"}
```

**Selection rule.** Deterministic given `session_seed`:

1. Slot order is fixed and escalating in demand:
   `[situation_context, action_specificity, individual_ownership, quantified_result, reflection]`.
   Context first because it is the easiest thing to answer cold; reflection last because it is the
   one people answer best once they are warm.
2. For each slot, pick uniformly at random (seeded) from questions whose `primary` matches and
   whose `family` is unused this session.
3. `demo` preset ignores steps 1–2 and uses `demo_question_ids: [q10, q03]`.

Bank coverage: `situation_context` 2 candidates, `action_specificity` 4, `individual_ownership` 3,
`quantified_result` 2, `reflection` 3. The two-candidate slots are the thin ones — if you extend
the bank, extend those first.

#### Probe mapping table

Gap detectors run on the question block's accumulated transcript, locally, for zero tokens. All
five are **recall-biased**; the listed predicates over-fire and that is the intended trade.

| Competency | Gap condition (fires when) | Follow-up templates | Max probes |
|---|---|---|---|
| **quantified_result** | No numeral token (digit or number-word) anywhere after the last first-person action sentence; **or** a numeral is present but no unit, currency, percent or time token within ±3 tokens of it | 1. "What was the number before, and what was it after?"<br>2. "How did you measure {anchor}, and over what period?"<br>3. "Who else saw that result, and how did they see it?" | 2 |
| **individual_ownership** | Fewer than 2 sentences match a first-person-singular action verb (`I` + one of *built, wrote, led, decided, designed, shipped, ran, negotiated, chose, proposed, rewrote, owned, fixed, escalated, cut, hired, killed, rebuilt*) | 1. "What part of {anchor} was yours, specifically?"<br>2. "If you hadn't been on that team, what would have been different?"<br>3. "Which decision in that story was yours to make?" | 2 |
| **action_specificity** | Fewer than 3 concrete-action sentences, where concrete = first-person verb with a direct object **not** in `ABSTRACT_OBJECTS = {process, communication, alignment, stakeholders, strategy, efficiency, collaboration, visibility, buy-in, culture}` | 1. "Take me through {anchor} step by step — what did you do first?"<br>2. "What did you actually do to {anchor}? I want the mechanics."<br>3. "What was the alternative you decided against, and why?" | 2 |
| **situation_context** | The first 60 words contain no role marker (`I was`, `my role`, `I worked`, `as a`, `I led`, `on the … team`) **and** no time/place marker (`last year`, `in 20\d\d`, `at the time`, `when I was`, `a few months`, `at <Proper>`) | 1. "Before we go further — where were you working, and what was your role on it?"<br>2. "What was actually at stake if {anchor} hadn't worked?"<br>3. "Who else was involved, and what was your job in it specifically?" | 1 |
| **reflection** | No retrospective marker (`learned`, `next time`, `in hindsight`, `looking back`, `I'd do`, `I would do`, `differently`, `what I'd change`) | 1. "What would you do differently if you ran {anchor} again?"<br>2. "What did that cost you that you weren't expecting?" | 1 |

**`{anchor}`** is not filled by the state machine. The template goes to the LLM verbatim and rule 3
of the system prompt makes the model substitute a phrase the candidate actually used — which is the
one thing an LLM is genuinely better at than a regex, and the reason a probe does not sound canned.
If the model cannot find an anchor, rule 3 tells it to drop the clause rather than invent one.

**Priority when several gaps fire** (only one probe is asked at a time):

```python
PROBE_PRIORITY = ["quantified_result", "individual_ownership",
                  "action_specificity", "situation_context", "reflection"]
```

Ordered by how much one follow-up sentence can repair, not by story order.
`quantified_result` first because it is both the most commonly missing and the most completely
fixable in one sentence. `situation_context` is deliberately fourth despite being first in the
story — probing it early sounds pedantic, and the missing context usually arrives on its own.
`reflection` is last because it is the one gap a written report can prompt on afterwards without
needing the live turn.

Within a competency, templates are used in listed order, never repeated in a session.

```python
# src/coach/coachlogic/probe.py
from dataclasses import dataclass

MAX_PROBES_PER_COMPETENCY = {"quantified_result": 2, "individual_ownership": 2,
                             "action_specificity": 2, "situation_context": 1, "reflection": 1}

@dataclass(frozen=True)
class Directive:
    competency: str
    template: str

def gaps(block) -> list[str]:
    """Recall-biased. Returns competency ids in PROBE_PRIORITY order."""
    text = block.full_text()
    fired = {c for c, detect in DETECTORS.items() if detect(text)}
    return [c for c in PROBE_PRIORITY if c in fired]

def select(block) -> Directive | None:
    for c in gaps(block):
        if block.probes_by_competency[c] >= MAX_PROBES_PER_COMPETENCY[c]:
            continue
        used = block.templates_used.get(c, 0)
        if used >= len(TEMPLATES[c]):
            continue
        return Directive(c, TEMPLATES[c][used])
    return None
```

`select()` returning `None` while `gaps()` is non-empty means every fired gap is exhausted —
`should_advance` treats that as `probes_exhausted`, not as `no_gap`. The move-on lines differ, and
so does what the report says.

The same `probe.select()` machinery drives rung 2 of the degradation ladder (§11.1): with no LLM
reachable at all, the state machine still picks a competency and plays the pre-rendered generic
version of that template. The interview keeps running; it stops adapting.

#### The wrap-up report — `report.py` output

Single self-contained HTML file with the session's `.wav` inlined or referenced, written to
`runs/<session_id>/report.html`. Seven blocks:

1. **Header** — date, duration, questions asked, and the full provenance line: hot-path model id,
   scoring model id, `prompt_version`, `rubric_version`, git sha, `runs/<file>.jsonl`.
2. **Per question block** — question text, the probes that were asked and why (competency and gap
   condition, in plain English), and the full transcript with word timings, each word clickable to
   seek the audio.
3. **Score grid** — 5 competencies × 5 blocks = 25 cells. Each scored cell shows the level, the
   quoted span as a button that seeks to `t_start`, and the one-sentence rationale. Abstained and
   rejected cells render as "—" with the reason spelled out, never silently blank.
4. **Across the session** — level distribution per competency, strongest and weakest named in a
   sentence. **No composite score**, with the one-line reason from `rubric.yaml: notes.composite`
   printed right there so its absence reads as a decision rather than an omission.
5. **Delivery metrics, opt-in and collapsed by default** — words per minute against the 196 WPM
   conversational baseline (Yuan et al. 2006, range 111–291), pause-length distribution,
   time-to-first-word per question. Descriptive sentences only, no levels, no colour coding.
6. **Instrumentation** — the per-stage latency waterfall for this session (`t_endpoint`,
   `t_server_v2v`, `t_v2v` with P50/P95, N turns, and the clock-offset uncertainty), barge-in
   decision vs audible-stop latency as separate numbers, which ladder rung served each turn, Groq
   `x-ratelimit-remaining-tokens` at its session minimum, `cache_hit_ratio_session`, and
   `ungrounded_rate` = rejected / total scores.
7. **What this does not measure** — the §9.9 ethics block, verbatim, plus the ASR-bias statement
   and the "correct the transcript and rescore" control.

Block 6 is the one no competitor ships and the one that makes the report a portfolio artifact
rather than a printout. Block 3's abstentions are the second.

#### Token arithmetic for one default session

Assumes a shared 8,000 TPM pool across model IDs, which is the conservative reading of Groq's docs
(§14 item 23).

| Path | Calls | Tokens counted against TPM | Note |
|---|---|---|---|
| INTRO, ASK ×5, move-on ×5 | 0 | **0** | pre-rendered WAV |
| Probes | ~8 | 8 × ~1,585 = **12,680** | 420 system + 60 directive + ≤900 history + ~160 answer + ~45 out; no caching on qwen3.6 |
| Scoring | 25 | 1,730 + 24 × 530 = **14,450** | ~1,200-token prefix cached after call 1 |
| Wrap | 1 | **~920** | |
| **Total** | **~34** | **≈ 28,050** | |

Consequences, all of which the README can state:

- **7 full sessions/day** against the 200,000 TPD cap. TPD binds first — 1,000 RPD would allow 29.
- **Average 1,650 tokens/min** over a 17-minute session, against 8,000 TPM. Comfortable.
- **The peak is what bites**: a probe turn (1,585) landing in the same minute as a five-call
  scoring batch (2,650) is 4,235. Still under, but a burst of short probe answers plus two
  overlapping scoring batches is not. Hence the reserved floor:

```python
# hot path is never starved by the scorer
HOT_PATH_RESERVE_TPM = 3_000
scorer_budget = min(remaining_from_header, 8_000 - HOT_PATH_RESERVE_TPM)
```

**If Groq's limits turn out to be per-model rather than org-wide** — the docs present them in a
per-model table but state them at organisation level, and do not say which (§14 item 23) — then
running the hot path on qwen3.6 and scoring on gpt-oss-20b doubles this headroom for free and the
reserved floor becomes belt-and-braces. Design for the conservative case; check
`console.groq.com/settings/limits` and confirm by deliberately saturating one model.

**Scoring is scheduled, not batched at the end.** A block's five scoring calls are enqueued at
`ADVANCE` and run while the candidate answers the *next* question — roughly 90 seconds of budget
against a 2,650-token batch. By `WRAP` only the last block is outstanding: five calls, ~40 seconds,
which is what `WRAP_RESERVE_S = 90` pays for. Doing all 25 at the end would take four minutes
against the TPM cap and turn the best moment of the demo into a progress bar.

### §9.7 Differentiating features, ranked by impressiveness ÷ effort

| # | Feature | Why it's impressive | Effort |
|---|---|---|---|
| **1** | **Calibration set with a published agreement ceiling.** 50 hand-scored answers; report **quadratically-weighted** Cohen's κ per rubric dimension with bootstrap 95% CIs, N, score marginals and full confusion matrices — plus two references the judge is measured against: my own test–retest on 20 answers, and a **second human rater on the same 20**, which is the human–human ceiling *on this rubric*. ETS RR-17-28's ICC of .74 (dimension range .66–.82) is cited as external context only, not as the bar | **Almost no portfolio project measures its own evaluator, and fewer measure the noise in their own labels.** Converts "I prompted an LLM" into "I built an evaluator and bounded what it can claim" — and the up-front statement that a ±0.17 interval at N = 50 cannot separate 0.63 from 0.75 is the part that reads as a scientist rather than a marketer | ~5.5 h mine + 1 h second rater |
| **2** | **Evidence-linked rubric scoring** with a validator that refuses to score without a quoted span, **and a published ungrounded rate** — the fraction of scores discarded because the model could not quote the transcript | Kills the #1 complaint about every competitor. Shows you know LLM-judge verbosity/position bias and designed against it. The ungrounded rate is one extra column in the README and no competing project reports anything like it | ~2–3 days |
| **3** | **Per-stage latency waterfall with three named terms** (`t_endpoint` / `t_server_v2v` / `t_v2v`), P50/P95 with bootstrap CIs, rendered live in the UI, **and validated against an acoustic loopback recording** | Proves the claim on stage. "My software timing agrees with an acoustic measurement to within 12 ms (N=20, IQR 31 ms)" is a sentence almost no portfolio project can write | ~1.5 days |
| **4** | **Rubric-gap-driven follow-up probing** | *The* product differentiator, and it is prompt + state machine, not new infrastructure | ~3 days |
| **5** | **Interview-tuned endpointing + barge-in, with a measured FCR/latency curve** (§8.3) against 40 hand-labelled answers containing deliberate thinking pauses, plus three fixed-threshold baselines | Domain insight a generic voice bot lacks, visibly better in the demo, fixes the exact failure users report in the commercial leader — and it is *measured*, against LiveKit's published 9.9% at 300 ms, rather than asserted | ~4–5 days (partly mandatory) |
| **6** | **Delivery metrics with published baselines** — WPM against the **196 WPM** conversational average (Yuan et al., Interspeech 2006; range 111–291), pause-length distribution, time-to-first-word | Cheap once word timestamps exist, and honest in a way Yoodli isn't: "you spoke at 210 WPM; conversational English averages 196 [citation]" | ~1 day |
| **7** | **Timestamped session report / replay** | Makes the project *linkable* — a recruiter sees output without running it — and is the Tier 0 public artifact (§6.8), not a fallback | ~2 days |
| **8** | **Résumé-grounded adversarial probing** — pick a claimed achievement and demand evidence and individual contribution | Table stakes now (Yoodli, Final Round all ingest JD/CV), so it differentiates only in the adversarial framing. Do it last | ~3 days |
| — | ~~**Filler-word counting as a headline feature**~~ | **Cut from v1.** Whisper — including Groq's endpoints — is *trained to delete* "um"/"uh", so counts are fabricated. The verbatim fix (CrisperWhisper, disfluency F1 87.8) ships **non-commercial research weights**. And the evidence that fillers hurt hiring outcomes is weak and contested. If you ship it, report descriptively against the **5.97 disfluencies per 100 words** baseline (Bortfeld et al. 2001), never as a score | Cut / stretch |
| — | ~~**Confidence / emotion / "hireability" scoring**~~ | **Never build this.** It is the HireVue failure mode — it dropped facial analysis in January 2021 after an EPIC FTC complaint, conceding the technology "wasn't worth the concern." Naming *why* you didn't build it is worth more than building it | — |

### §9.8 How the judge is calibrated, and what that number can and cannot say

**Read this limitation before the numbers.** The calibration set is 50 answers. A 95% bootstrap
CI on a quadratically-weighted kappa at N = 50 is roughly **±0.17**, which means a measured 0.63
and a measured 0.75 are the same result. Every kappa in this project is reported with its
interval and its N, and no claim is made that a difference smaller than the interval is real.
The set is 50 because hand-scoring costs about 3 minutes an answer and 50 answers is 2.5 hours;
100 answers would cut the half-width from ±0.17 to ±0.12 for another 2.5 hours. That trade is
stated, not hidden.

**Statistic: quadratically-weighted Cohen's kappa (κ_w), weights w_ij = (i−j)² / (k−1)², k = 5.**

Unweighted kappa is wrong here and is the default most portfolio projects reach for. On a 1–5
behaviourally anchored scale it treats "I said 4, the judge said 5" as exactly as bad as "I said
1, the judge said 5". A judge that is almost always within one level — the realistic good
outcome — scores near zero under unweighted kappa. Quadratic weights penalise by squared
distance, so adjacent disagreements cost 1/16 of a maximal one.

Quadratic specifically, not linear, for two reasons. First, Fleiss & Cohen (1973) show weighted
kappa with quadratic weights is equivalent to the intraclass correlation coefficient under stated
assumptions — which matters because the structured-interview literature reports ICC, and this is
the only weighting that makes the two even the same *family* of quantity. Second, it is the
standard metric in ordinal automated-scoring work, so it is the number a reader can compare
against something.

Known cost of quadratic weighting, disclosed alongside the result: it is sensitive to the
marginal distribution and can be inflated when one rater uses a wider spread than the other.
That is why the per-dimension table prints both raters' score marginals next to every kappa, and
why the full confusion matrices are printed underneath. A reader who prefers linear weights or
Gwet's AC2 can recompute from the matrices.

**One annotator is a real problem. Here is what is done about it.** A single annotator conflates
judge error with annotator noise. Three measurements, not one, reported together:

| # | Comparison | N | What it bounds | Cost |
|---|---|---|---|---|
| 1 | **κ_w(me pass 1, me pass 2)** — test–retest, ≥7 days apart, answers reshuffled, my first-pass labels hidden | 20 | My own label noise. This is the ceiling on anything measured against me. If this is 0.70, a judge scoring 0.70 against me is at the noise floor, not "moderately good" | ~1 h, mine |
| 2 | **κ_w(me, rater 2)** — a second human, blind to my labels and to the judge's, same rubric card, 3 warm-up answers not counted, per-rater randomised order | 20 | **The human–human ceiling on *this* rubric.** This is the bar the judge is reported against | ~1 h theirs, 0.5 h mine |
| 3 | **κ_w(me, judge)** and **κ_w(rater 2, judge)** | 50 / 20 | The headline. Reporting both matters: rater 2 is independent of me, so if the judge agrees with rater 2 about as well as I do, the judge is inside the human band rather than merely imitating me | ~1.5 h (code + runs) |

Rater 2 is one hour of a friend's or partner's time, not an expert panel. Say that in the README.
The honest claim the design supports is *"the judge sits inside the interval of two humans
scoring the same rubric"*, never *"the judge is as reliable as a trained interviewer"*.

**The external comparison, corrected.** ETS RR-17-28 (Kell, Martin-Raugh, Carney, Inglese, Chen &
Feng, 2017), *Exploring Methods for Developing Behaviorally Anchored Rating Scales for Evaluating
Structured Interview Performance*: **ICC = .74** for three raters across 12 questions; per
dimension Communication .75, Leadership .66, Persuasion and Negotiation .82, Teamwork .72;
N = 652 usable **written** responses from 68 Mechanical Turk participants on a **7-point**
effectiveness scale. Frame it as external context in a footnote, never as "the bar", because it
differs from this project on every axis that moves the statistic: ICC not kappa, 7 levels not 5,
written text not spoken transcripts, three trained SMEs not one hobbyist and a friend, N = 652 not
50, and a different construct set. The comparable quantity is measurement #2 above.

(Conway, Jako & Goodman 1995 is the usual source for meta-analytic interview reliability. Its
abstract gives upper limits of **validity** — .67 highly structured, .34 unstructured — not
interrater reliability; the paper is paywalled and the reliability figures could not be verified
on 2026-09-13. Do not cite a number from it that you have not read.)

**Per-dimension, never pooled.** One answer's five scores are not five independent draws — they
share the answer, the transcript, and its ASR errors. A pooled kappa over 250 cells claims
N = 250 and has the precision of N = 50. `bench/calibrate.py` refuses to print one.

**The kappa paradox.** If a dimension's marginal collapses — say 80% of answers score 3 on
`reflection` — kappa goes to zero at high raw agreement because the chance-agreement term
explodes. The tool flags any dimension where either rater puts >80% of mass in one level and
prints raw agreement and within-one agreement beside every kappa, so a collapsed dimension reads
as "uninformative here" rather than "the judge failed".

**What N buys you.** Simulated: 5 ordinal levels, marginal [.10 .20 .35 .25 .10], Gaussian rater
noise, 400 synthetic datasets per cell, 4,000 percentile-bootstrap resamples each, median across
datasets. `bench/power_sim.py` reproduces it.

| N (answers per dimension) | true κ_w | median 95% CI | median half-width |
|---|---|---|---|
| 20 | 0.63 | [0.30, 0.83] | ±0.26 |
| 20 | 0.71 | [0.41, 0.86] | ±0.22 |
| 20 | 0.80 | [0.54, 0.91] | ±0.18 |
| **50** | **0.63** | **[0.43, 0.77]** | **±0.17** |
| 50 | 0.71 | [0.54, 0.82] | ±0.14 |
| 50 | 0.80 | [0.66, 0.88] | ±0.11 |
| 100 | 0.63 | [0.50, 0.74] | ±0.12 |
| 100 | 0.71 | [0.60, 0.80] | ±0.10 |
| 100 | 0.80 | [0.71, 0.86] | ±0.07 |

Two consequences to state in the README. At N = 50 the judge-vs-me interval is about ±0.17, so
only a difference of ~0.3 in κ_w is detectable. At N = 20 the human–human ceiling is ±0.22 wide,
so it is a **sanity floor, not a precise ceiling** — its job is to catch the case where two
humans only reach 0.45 on this rubric, which would mean the rubric is underspecified and no judge
score means anything.

#### Data layout

One JSONL per rating pass under `data/calibration/`:

```
data/calibration/
├── answers.jsonl                      # {"answer_id","question_id","audio_path","transcript"}
├── labels_human1_pass1.jsonl          # me, first pass, all 50
├── labels_human1_pass2.jsonl          # me, ≥7 days later, 20-answer subset, blind
├── labels_human2.jsonl                # second rater, same 20
└── judge_<model>_<prompt_sha>.jsonl   # one file per judge configuration
```

Label rows: `{"answer_id": "a017", "scores": {"situation_context": 3, "individual_ownership": 4,
"action_specificity": 2, "quantified_result": 1, "reflection": 3}}`. The judge files carry the
same shape plus `evidence` spans, so a rejected-for-no-evidence score never enters the kappa.

```python
#!/usr/bin/env python3
"""bench/calibrate.py - rater agreement over data/calibration/.

Reports QUADRATICALLY-WEIGHTED Cohen's kappa PER RUBRIC DIMENSION with a
percentile bootstrap 95% CI over answers, next to the raw agreement, adjacent
agreement and score marginals that a kappa alone hides.

It never prints a pooled kappa across dimensions: one answer's five scores are
not five independent draws, so a pooled figure claims an N it does not have.

    python bench/calibrate.py --a labels_human1_pass1.jsonl \
                              --b judge_gpt-oss-20b_9f2c1ab.jsonl --label "human vs judge"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import numpy as np

LEVELS = np.arange(1, 6)
K = len(LEVELS)
W = ((LEVELS[:, None] - LEVELS[None, :]) ** 2) / (K - 1) ** 2  # quadratic weights
MIN_N = 15          # below this, print N and refuse to print a kappa
B = 10_000          # bootstrap resamples
CAL = pathlib.Path("data/calibration")


def load(path: pathlib.Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["answer_id"]] = {k: int(v) for k, v in row["scores"].items()}
    return out


def qwk(a: np.ndarray, b: np.ndarray) -> float:
    """Quadratically-weighted Cohen's kappa on a fixed 1-5 label set."""
    o = np.zeros((K, K))
    np.add.at(o, (a - 1, b - 1), 1.0)
    o /= o.sum()
    e = o.sum(1)[:, None] * o.sum(0)[None, :]
    den = (W * e).sum()
    return float("nan") if den == 0 else 1.0 - (W * o).sum() / den


def boot_ci(a: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[float, float, int]:
    """Percentile bootstrap over ANSWERS, not over cells."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    vals = np.array([qwk(a[i], b[i]) for i in idx])
    ok = vals[~np.isnan(vals)]
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return float(lo), float(hi), int(B - ok.size)


def report(a_lab: dict, b_lab: dict, label: str) -> None:
    ids = sorted(set(a_lab) & set(b_lab))
    dims = sorted({d for i in ids for d in a_lab[i]} & {d for i in ids for d in b_lab[i]})
    print(f"\n## Agreement: {label}   (N = {len(ids)} answers, {len(dims)} dimensions)\n")
    print("| dimension | N | kw | 95% CI | exact | within 1 | A marginal | B marginal |")
    print("|---|---|---|---|---|---|---|---|")
    for d in dims:
        pairs = [(a_lab[i][d], b_lab[i][d]) for i in ids if d in a_lab[i] and d in b_lab[i]]
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        n = len(a)
        exact = float((a == b).mean()); adj = float((np.abs(a - b) <= 1).mean())
        ca, cb = np.bincount(a, minlength=6)[1:6], np.bincount(b, minlength=6)[1:6]
        ma = "/".join(str(int(c)) for c in ca); mb = "/".join(str(int(c)) for c in cb)
        if n < MIN_N:
            print(f"| {d} | {n} | — (N<{MIN_N}) | — | {exact:.0%} | {adj:.0%} | {ma} | {mb} |")
            continue
        k = qwk(a, b); lo, hi, drops = boot_ci(a, b)
        # kappa paradox guard: a collapsed marginal makes kappa uninformative
        flag = " ⚠skew" if max(ca.max(), cb.max()) / n > 0.8 else ""
        print(f"| {d} | {n} | {k:.2f}{flag} | [{lo:.2f}, {hi:.2f}] | {exact:.0%} | {adj:.0%} "
              f"| {ma} | {mb} |")
        if drops:
            print(f"|   ↳ {drops}/{B} resamples degenerate (a rater was constant) ||||||||")
    print("\nConfusion matrices (rows = A score 1-5, cols = B score 1-5):\n")
    for d in dims:
        m = np.zeros((K, K), dtype=int)
        for i in ids:
            if d in a_lab[i] and d in b_lab[i]:
                m[a_lab[i][d] - 1, b_lab[i][d] - 1] += 1
        print(f"{d}:\n{m}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--dir", default=str(CAL))
    args = p.parse_args()
    root = pathlib.Path(args.dir)
    report(load(root / args.a), load(root / args.b), args.label or f"{args.a} vs {args.b}")


if __name__ == "__main__":
    main()
```

`bench/power_sim.py` — the table above, reproducible, ~30 s:

```python
"""bench/power_sim.py - what a 95% bootstrap CI on quadratically-weighted kappa
costs at N=20/50/100. Run this BEFORE hand-labelling anything."""
import numpy as np
from calibrate import qwk   # same weights, same label set

rng = np.random.default_rng(11)
P = np.array([.10, .20, .35, .25, .10])          # realistic 1-5 rubric marginal

def pair(n, sigma):
    t = rng.choice(np.arange(1, 6), size=n, p=P)
    f = lambda: np.clip(np.rint(t + rng.normal(0, sigma, n)), 1, 5).astype(int)
    return f(), f()

for n in (20, 50, 100):
    for sigma, true in ((0.80, 0.63), (0.65, 0.71), (0.50, 0.80)):
        hw = []
        for _ in range(400):
            a, b = pair(n, sigma)
            idx = rng.integers(0, n, size=(4000, n))
            v = np.array([qwk(a[i], b[i]) for i in idx]); v = v[~np.isnan(v)]
            lo, hi = np.percentile(v, [2.5, 97.5]); hw.append((hi - lo) / 2)
        print(f"N={n:<4} true kw={true:.2f}  median 95% CI half-width = ±{np.median(hw):.2f}")
```

### §9.9 Ethics and legal boundary (put this in the README)

As a **candidate-facing self-practice tool the user scores themselves**, which very likely sits
outside NYC Local Law 144 and the Illinois AI Video Interview Act. State the boundary plainly:

- **NYC LL144** would require an annual independent bias audit, public posting, and 10 business
  days' notice — if an *employer* used this to screen.
- **Illinois 820 ILCS 42** (effective 2020-01-01) requires pre-interview notice, an explanation
  of what the AI evaluates, explicit consent to AI analysis, and deletion within 30 days on
  request.
- **EU AI Act** keeps employment/recruitment as Annex III high-risk, now applying from
  **2 December 2027** (deferred from 2026-08-02 by Regulation (EU) 2026/1744).
- **Groq's own AUP** forbids automated decisions with material detrimental impact in employment
  without human supervision. A self-practice coach is fine; repositioning it as candidate
  screening is not.
- Note that the **EEOC's 2023 Title VII AI technical assistance and its ADA/algorithms guidance
  were removed from the agency website in early 2025 and now 404.** Title VII and the ADA still
  apply — but do not cite those documents as live guidance.

**The hosted build changes the analysis and the README must say so.** Everything above reasons
about the *user* scoring *themselves* on their own machine. The moment a stranger's voice transits
two US clouds on someone else's API keys, that visitor is a third party to a contract they did not
sign. That is the second, independent reason the live link is invite-gated rather than posted, and
it is why §6.10's consent screen exists, why the hosted build constructs no audio writer at all,
and why nothing but an aggregate timing row outlives a hosted session.

And name the bias inherited from the ASR layer: **Koenecke et al. (PNAS 2020) measured average
WER of 0.35 for Black speakers versus 0.19 for white speakers** across five major commercial ASR
systems, with over 40 errors per 100 words for Black men. If the transcript is wrong the rubric
score is wrong. Mitigations to implement *and* state: always show the transcript, let the user
correct it before scoring, make delivery metrics opt-in and descriptive, and never score
"clarity" or "articulateness" from transcription quality. The rubric enforces the last of these
structurally — `notes.never_score` in `behavioral_v1.yaml` lists accent, fluency and articulateness
by name, and the scoring prompt repeats the prohibition.

---

## 10. Three-milestone build plan

**Planned work ~81–95 h, plus 8 h of named contingency, so 89–103 h all-in.** At 15 h/week that is
**6–7 weeks**; at 20 h/week, **4.5–5 weeks**. Earlier drafts said "2–3 weeks, 45–55 hours", and
that estimate omitted the fallback that makes §11 #2/#3/#4 true, the endpointer evaluation that
makes §9.7 #5 a claim rather than an assertion, the hosted build and its wire protocol, the
acoustic calibration that validates the headline number, and any tests at all. **Decide before
hour 1 which rate you are working at — and if the answer is 15 h/week, the cut list below fires on
schedule, not in a panic at hour 80.** The shortest honest shippable scope is cut item 1 alone:
Tier 0 only, ~73.5–87.5 h planned.

| Block | Earlier estimate | Now | What moved |
|---|---|---|---|
| Milestone 1 — "It talks back" | 14–18 | **22–26** | Day-0 spike ×3 candidates, clock sync, local-LLM rung, sentencizer goldens, re-chunker properties, AEC smoke test, pre-rendered clips |
| Milestone 2 — "It's a conversation, and it's an interviewer" | 16–20 | **23–27** | Ladder + `provider_switch` + HUD, rung-2 scripted probes, scoring moved to gpt-oss with cache-safe prompts, cancellation tests |
| Milestone 3 — "It's measured" | 12–16 | **36–42** | Wire protocol v1, session resume, guard + ledgers, hosted config, consent + scrub, endpointer evaluation, calibration test–retest + second rater, acoustic loopback, CI |
| **Planned subtotal** | **45–55** | **81–95** | |
| **Contingency (named triggers, below)** | — | **8** | |
| **Total** | — | **89–103** | |

### Milestone 1 — "It talks back" (~22–26 h)

**Working software at the end:** a browser page where you press start, speak a behavioural
interview question's answer out loud, and hear a spoken coach response. Half-duplex (no
barge-in yet), fixed silence endpointing, single question. A latency HUD shows per-stage
timings live, on a reconciled clock. **This is demoable.**

| Task | h |
|---|---|
| **Day-0 spike (do this before anything else):** `bench/spike_llm.py` measuring time-to-first-**content**-delta with the real `COACH_SYSTEM_PROMPT` + a `build_probe_messages` payload with a 60-word canned answer (~1,585 tokens in, ~45 out — the real shape of the request), N=20 per candidate, P50/P90, against **three** candidates: (a) Groq `qwen/qwen3.6-27b` `reasoning_effort:"none"`, (b) Groq `openai/gpt-oss-20b` `"low"`, (c) **local `gemma-3n-e4b-it-text` via LM Studio**, warm. Also: assert `reasoning_content` deltas are zero on (a); record `x-ratelimit-remaining-tokens`; confirm `sw_vers -productVersion` ≥ 15.0; confirm Render signup needs no card; install LM Studio and pre-download the 4 GB weights | 3.5 |
| Browser: `getUserMedia` + AudioWorklet → 16 kHz Int16 → 512-sample frames + `capture_ctx_s`, over WS. **Ends with the 15-minute AEC smoke test** (§8.4) | 3.25 |
| Server: FastAPI `/ws/v1/<sid>`, Silero VAD via our own ORT session, fixed 800 ms silence endpointer, the three executors and thread-count settings (§6.5) | 2.5 |
| `stt/moonshine_local.py` behind `STTProvider`, including `force_update()` and the §8.1 flush protocol | 2.5 |
| `llm/groq_llm.py` + `llm/stream.py` (`DeltaStream`, first-token and stall deadlines) + `llm/local_llm.py` with `health()`/`warm()` wired into `/healthz`; model ID from config; startup `GET /v1/models` check | 4 |
| `sentence.py` (+`flush()`, golden files) + `tts/kokoro_local.py` (float32→int16, ~50 ms re-chunk) + `audio/player.py` (`enqueue`/`abort`/`drain`/`played_frames`) | 4 |
| Pre-render the 21 fixed clips at startup — removes TTS latency from every ASK and INTRO turn and is the cheapest win in the build | 0.5 |
| `metrics/clock.py` + `web/clock.js`: NTP-style offset on WS open and per turn; `turn_log.py` with the §6.7 field table, client-mark merge, and the three derived headline numbers | 3 |
| `tests/test_frames.py`: Hypothesis properties for the 512-sample re-chunker, including split-invariance | 0.75 |

**Build the instrumentation in Milestone 1, not later.** It is the differentiating artifact and
it is the first thing that gets cut when time runs short. The extra hour on the clock row is what
makes the headline number quotable at all.

### Milestone 2 — "It's a conversation, and it's an interviewer" (~23–27 h)

**Working software at the end:** you can interrupt the bot mid-sentence and it stops within
~50 ms. It waits through a three-second thinking pause without cutting you off. It asks
rubric-gap follow-up questions. When Groq 429s it visibly drops to a local model and keeps going.
After the session it produces a timestamped HTML report where every feedback item quotes the
transcript span that justifies it.

| Task | h |
|---|---|
| `turn/endpointer.py`: smart-turn v3.2 ONNX, two-stage state machine, `stop_secs` fallback, `p` in the HUD, `p_complete()` separable from the FSM so §8.3 can sweep it | 4 |
| `turn/controller.py`: generation-id guard, abort-before-await, `_heard_prefix` truncation, spoken-only history commit | 3 |
| AEC: loopback-RTCPeerConnection playback; test with speakers at demo volume | 3 |
| `coachlogic/rubric.py`: `SpanMatcher` + `attach_evidence` + the strict `json_schema`; then the retry/abstain path | 2 |
| Scoring path: the 25-call scheduler, the token bucket and `HOT_PATH_RESERVE_TPM`, the gpt-oss-20b client, `llm/prompt.py`'s three-region layout and `CacheMeter` | 3 |
| `coachlogic/probe.py` + the question bank + the deterministic selection rule | 3 |
| `coachlogic/report.py`: timestamped HTML with audio-linked feedback, all seven blocks | 2 |
| `llm/ladder.py`, the `provider_switch` frame, HUD badge + toast in `web/app.js` | 2 |
| Rung 2: generate the 6 generic probe WAVs + the filler WAV offline; wire to `coachlogic/probe.py` | 1 |
| `tests/conftest.py` fakes + `test_turn_controller.py` (4 tests) + `test_ladder.py` (3 tests) | 2 |

### Milestone 3 — "It's measured" (~36–42 h)

**Working software at the end:** `bench/replay.py` feeds recorded WAVs through the real pipeline
and writes `runs/<date>-<sha>.jsonl`; `bench/latency_report.py` emits the three-column table and
CDF plots; `bench/wer.py` reports pooled WER on the five ungated ESB splits; `bench/disagree.py`
reports local-vs-cloud disagreement; `bench/endpoint.py` reports the FCR/latency curve;
`bench/calibrate.py` reports κ_w per dimension against three label sets. Provider swap works
across STT and LLM backends. The README carries a methodology block and measured numbers with N
and confidence intervals. Tier 0 is published; Tier 1 is deployed and gated.

| Task | h |
|---|---|
| `bench/replay.py`: fake transport, deterministic fixtures, temperature 0. **Two modes** — recorded-SSE stub at N=200, live-LLM at N=60/day (see below) | 3 |
| `bench/latency_report.py`: three columns, P50/P95/P99, bootstrap 95% CIs, CDF plots, split by `llm_rung`, the `output_latency_implied_ms` assertion | 2 |
| `bench/wer.py` (pooled, five ungated splits) + `bench/disagree.py` (reference-free + adjudication CSV) | 3 |
| Record 40 endpointing fixtures (scripted prompts, one take each, 3.5 s trailing room tone) | 1.5 |
| Hand-label `true_eot_ms` + interior pauses to ±50 ms, ~2 min/clip | 1.5 |
| `bench/endpoint.py` + the offline threshold sweep | 2.5 |
| Sweep, ROC plot, three fixed-threshold baselines, README table | 1 |
| Hand-label the 50-answer calibration set; test–retest pass ≥7 days later on 20; brief and coordinate rater 2; `bench/calibrate.py` + `bench/power_sim.py` | 5.5 |
| Second STT backend (`deepgram_flux.py`) + NIM LLM backend; provider comparison table | 2 |
| Wire protocol v1: `wire.py`, `protocol.js`, the message table in the README, a round-trip test over every frame type | 2 |
| `sessions.py` + resume: session store, replay ring, `playback_ack` reconciliation, reconnect backoff in `app.js` | 3 |
| `guard.py`: invite key, origin check, IP-hash limiter, frame-rate cap, two daily ledgers wired into `groq_llm.py` and the Deepgram clients | 3 |
| `config/hosted.yaml`: Flux STT backend, Aura-2 TTS backend, browser-side VAD path, mu-law downlink; verify `mulaw@24000` is accepted | 2 |
| Consent screen + `text_sha256` pinning; `bench/scrub.py` + pre-commit hook | 2 |
| Tier 0 publish (session report + recording + measured artifacts on GitHub Pages), Render deploy, `/healthz`, README methodology + "what this does not measure" + the free-tier arithmetic | 2 |
| **Acoustic ground-truth calibration — REQUIRED, both parts (below)** | 1.5 |
| `bench/resources.py` to fill §6.5's RSS column | 0.5 |
| `pytest.ini` markers, `make test`, GitHub Actions on `macos-latest` running unit-only | 0.75 |

**`--runs 200` through the real pipeline is arithmetically impossible against the free tier.** At
~1,870 tokens per replayed turn that is 374,000 tokens against a 200,000 TPD cap. So the harness
has two modes, reported as two configurations with two Ns, never averaged: pipeline runs — VAD,
STT, sentencizer, TTS, plumbing — use a **recorded-SSE stub** that replays captured token timings
and go to **N=200**; the **live-LLM** row runs at **N=60/day**, about a third of the daily budget,
accumulated across three days to N=180. Two honest numbers beat one averaged one.

### Acoustic ground-truth calibration — required, ~1.5 h, two parts

**This is not a stretch goal.** Every number in the README depends on a clock offset estimated
over a WebSocket and a browser's own estimate of its output latency, and **nothing in the software
can check either of those against reality.** An instrumentation stack that has never been compared
to a physical measurement is a hypothesis, and the first competent interviewer will say so.

| Task | h |
|---|---|
| **Calibration A — digital loopback (N=20 turns).** Install BlackHole 2ch (free, open-source, Apple Silicon, no kernel extension). Build an Aggregate Device mixing mic input and system output, record both onto one 48 kHz stereo timeline (L = mic, R = system output). For each turn, measure last-speech-sample in L → first sample in R above −45 dBFS over a 10 ms window. Publish `acoustic_v2v_ms` against `t_v2v_ms` per turn: median delta, IQR, and a scatter plot | 1 |
| **Calibration B — room recording (N=5 turns).** Second recorder (a phone, 48 kHz) placed between your mouth and the laptop speaker. Same measurement, one file, one clock, no software involved at any point. Publish the constant | 0.5 |

**What each one validates — state this in the README, because it is the interesting part.**
Calibration A puts every software mark on a single physical timeline and so tests four things the
software cannot test itself: (1) the **clock offset** — a systematic error in `offset_ms` shows up
as a constant delta; (2) the **`getOutputTimestamp` mapping** — the ~25 ms double-count described
in §6.7 shows up as a constant bias of roughly `output_latency_ms`, in a direction that tells you
which mistake you made; (3) the **jitter-buffer accounting**, since A's delta includes real
scheduling and underruns; (4) **`t0` itself** — if the software's end-of-speech instant is late or
early relative to the acoustic one, `acoustic_v2v_ms − t_v2v_ms` is offset by exactly that amount.

What Calibration A **cannot** see: it records the *digital* output stream, so the DAC → amplifier →
speaker → air path is absent, and so is the microphone's own hardware capture delay. Both are
excluded by construction, not by oversight. Calibration B supplies exactly those two missing terms.
It is a single acoustic recording containing both the human voice and the bot's voice, so it has no
clocks, no drivers, and nothing to reconcile. **`B_median − A_median` is the hardware constant**,
typically a few tens of ms. Add it to every software-derived `t_v2v_ms` in the README, or state it
as a separate line item — either is honest, but silently omitting it is not.

**The pass criterion:** `median(acoustic_v2v_ms − t_v2v_ms)` from Calibration A should sit within
**±25 ms** of **zero**. This is only a meaningful bar because `_derive()` adds `pre_t0_ms` into
`t_v2v_ms` (§6.7): both sides now measure from the same acoustic instant, so the expected delta is
0, not 28 ms. Comparing the acoustic figure against a t0-anchored `t_v2v` would offset the median
by the whole of budget row 1 and fail this check by construction — if you ever see a median near
28 ms, `pre_t0_ms` is not being added. The IQR bar is, and its IQR under **±40 ms**. Outside that, the instrumentation is wrong and the
published latency numbers are not yet publishable. Put the measured agreement in the README next
to the CDF.

### The test suite

`tests/` is the gap a senior engineer finds in ninety seconds if it is empty, in a project whose
thesis is cancellation correctness and frame-contract discipline.

**Hard constraint: the whole unit suite runs in under 5 seconds with no microphone, no network,
no API key and no model weights.** Everything on the hot path is behind a Protocol already
(§6.3) — that was the point. Anything needing real weights or a real socket is
`@pytest.mark.integration` and excluded from the default run.

#### 1. Cancellation — fake clock + fake player

The §8.2 `TurnController` has four invariants that are all silently breakable and none of which a
manual demo reliably catches. Drive them with fakes gated on `asyncio.Event`, not on sleeps —
sleeps make a slow test *and* a flaky one.

```python
# tests/conftest.py
import asyncio
from dataclasses import dataclass, field


class FakeClock:
    """Monotonic clock under test control. Injected into TurnLog and LLMLadder."""
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@dataclass
class FakePlayer:
    enqueued: list[bytes] = field(default_factory=list)
    aborts: int = 0
    _aborted: bool = False

    async def enqueue(self, pcm: bytes) -> None:
        assert not self._aborted, "enqueue() after abort() — cancelled work still reached the device"
        self.enqueued.append(pcm)

    def abort(self) -> None:          # sync, exactly like sounddevice's abort()
        self.aborts += 1
        self._aborted = True

    def rearm(self) -> None:
        self._aborted = False


class FakeLLM:
    """Yields `deltas`, blocking on `gate` before the delta at index `block_at`."""
    def __init__(self, deltas: list[str], block_at: int | None = None) -> None:
        self.deltas, self.block_at = deltas, block_at
        self.gate = asyncio.Event()
        self.closed = False
        self.cancelled_cleanly = False

    async def stream_chat(self, messages):
        return self

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        try:
            for i, d in enumerate(self.deltas):
                if i == self.block_at:
                    await self.gate.wait()
                yield d
        except asyncio.CancelledError:
            self.cancelled_cleanly = True
            raise

    async def aclose(self):
        self.closed = True
```

```python
# tests/test_turn_controller.py
import asyncio
import pytest

pytestmark = pytest.mark.asyncio


async def test_barge_in_stops_audio_and_commits_only_spoken(controller, llm, player, history):
    # Two sentences will stream; the second is gated so we can interrupt between them.
    await controller.on_user_turn_end("I led the migration.", turn_log=...)
    await _until(lambda: len(player.enqueued) == 1)     # first sentence is out the door

    await controller.on_barge_in(turn_log=...)

    assert player.aborts == 1
    assert llm.closed, "finally: did not close the HTTP body — Groq keeps generating and billing TPM"
    assert llm.cancelled_cleanly, "CancelledError was swallowed somewhere on the response path"
    assert history.last_assistant() == "I led the migration."   # NOT the ungated second sentence


async def test_no_audio_from_a_stale_generation(controller, tts, player):
    """FakeTTS stamps each PCM blob with the generation that requested it."""
    await controller.on_user_turn_end("first answer", turn_log=...)
    await controller.on_user_turn_end("second answer", turn_log=...)   # cancels gen 1
    player.rearm()
    await _drain(controller)
    assert player.enqueued, "second turn produced no audio at all"
    assert {tts.generation_of(pcm) for pcm in player.enqueued} == {2}


async def test_cancel_is_idempotent(controller):
    await controller.cancel()
    await controller.cancel()          # must not raise; on_user_turn_end calls it unconditionally


async def test_cancel_awaits_the_task(controller, llm):
    """Task.cancel() only *schedules* CancelledError. If cancel() doesn't await,
    the finally block races the next turn and history commits out of order."""
    await controller.on_user_turn_end("x", turn_log=...)
    await controller.cancel()
    assert llm.closed, "cancel() returned before the finally block ran"
```

`_until` polls with `asyncio.sleep(0)` up to a bounded number of loop turns. No wall-clock sleeps
anywhere in the unit suite.

#### 2. Sentencizer — golden files

Each golden file is one JSONL: the delta stream as produced by a real SSE capture, then the
expected sentence list. Required cases, one file each:

| File | Case | Why it bites |
|---|---|---|
| `abbrev.jsonl` | "Dr. Patel", "e.g.", "i.e.", "vs." | Naive `.` splitting speaks half a word |
| `decimals.jsonl` | "3.5 seconds", "$1.2M", "a 12.5% lift" | The most common content in a quantified-result answer |
| `split_midword.jsonl` | The `.` and the following space arrive in **different deltas** | The actual streaming failure; a non-incremental sentencizer passes every offline test and fails here |
| `first_clause.jsonl` | §7 optimisation #2: emit at the first comma past 40 chars for the **opening** clause only | Verifies the fast-start rule fires once per turn, not on every comma |
| `no_terminator.jsonl` | Stream ends mid-sentence (barge-in, `max_tokens`, stall) | `flush()` must return the remainder or the last clause is never spoken |
| `question.jsonl` | `He asked "what was the outcome?" and I said no.` | Terminator inside quotes |

Regenerate with `pytest tests/test_sentencizer.py --update-golden`, and **review the diff** —
a golden file you regenerate without reading is a test that asserts whatever the bug does.

#### 3. Frame re-chunker — property test

§11 #6 is "Silero `ValueError` on the first frame under load", mitigated by "unit-test the
re-chunker." This is that test. Hypothesis, because the bug lives in an arbitrary split pattern
you would not think to write by hand.

```python
# tests/test_frames.py
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from coach.audio.frames import ReChunker

blocks = st.lists(
    arrays(np.int16, st.integers(min_value=0, max_value=2048)),
    min_size=0, max_size=40,
)


@given(blocks)
def test_every_emitted_frame_is_exactly_512_int16(bs):
    rc, out = ReChunker(512), []
    for b in bs:
        out += rc.push(b)
    assert all(f.shape == (512,) and f.dtype == np.int16 for f in out)


@given(blocks)
def test_samples_are_preserved_in_order_with_no_loss_or_duplication(bs):
    rc, out = ReChunker(512), []
    for b in bs:
        out += rc.push(b)
    src = np.concatenate(bs) if bs else np.zeros(0, np.int16)
    emitted = np.concatenate(out) if out else np.zeros(0, np.int16)
    n = len(src) - len(src) % 512
    assert np.array_equal(emitted, src[:n])
    assert len(rc.pending) == len(src) % 512      # remainder stays buffered, never dropped


@given(blocks, st.integers(min_value=1, max_value=997))
def test_output_is_invariant_to_how_the_input_was_split(bs, chunk):
    """The real bug: an audio callback delivers variable block sizes. The frame
    boundary the VAD sees must depend only on the sample stream, never on the blocking."""
    src = np.concatenate(bs) if bs else np.zeros(0, np.int16)
    a, b = ReChunker(512), ReChunker(512)
    out_a = [f for blk in bs for f in a.push(blk)]
    out_b = [f for i in range(0, len(src), chunk) for f in b.push(src[i : i + chunk])]
    assert len(out_a) == len(out_b)
    assert all(np.array_equal(x, y) for x, y in zip(out_a, out_b))
```

The third property is the one worth having. It is the exact §11 #6 failure, it is not expressible
as an example-based test without writing out dozens of split patterns, and it catches
off-by-one errors in the pending buffer that the first two properties miss. It declares one
interface obligation: `ReChunker.pending` is public (a `np.ndarray` of the unemitted tail).

#### 4. What is deliberately not unit-tested

Say this in the README, because "we test everything" is not credible and the boundary is the
interesting part:

- **Model output quality.** Measured statistically by `bench/calibrate.py` (κ_w with CIs), not
  asserted per-case. An assertion on an LLM's exact words is a test that fails on temperature.
- **The ONNX classifier itself.** Measured by `bench/endpoint.py` on labelled fixtures, not
  mocked into a unit test where it would assert only that the mock works.
- **Echo cancellation.** §14 #4 says outright that browser AEC behaviour is contested and
  hardware-dependent. It gets a manual pre-demo checklist item — speakers at demo volume, the
  actual machine — not a green CI badge implying a guarantee nobody can make.

### Contingency: 8 hours, against named triggers

Every trigger below is a §14 unverified claim or a §11 failure mode that is *already known* to be
unresolved. None is a surprise; each has a detection point early enough to act on, and a fixed
response. Total exposure is 14 h against an 8 h budget — that is deliberate. The budget assumes at
most two fire. If three fire, the cut list fires with them.

| Trigger | Detected | Cost | Response |
|---|---|---|---|
| **Browser AEC does not cancel the bot's own TTS on the demo machine** (§14 #4) | End of M1, 15-min smoke test: play 20 s of Kokoro output through the demo speakers at demo volume with the mic live, and count VAD speech-onsets. Non-zero = it failed | **+5 h** | `livekit` `rtc.AudioProcessingModule`: re-chunk to exactly 10 ms / 160-sample frames, feed TTS to `process_reverse_stream()`, calibrate `set_stream_delay_ms()`. Headphones remain the demo-day guarantee either way |
| **No Groq model reaches first *speakable* token under ~800 ms** (§11 #3, §14 #1) | Day-0 spike, hour 2 | **+3 h** | The §11.1 ladder already names the answer: rung 1 is local (`gemma-3n-e4b-it-text` in LM Studio). The 3 h buys keeping the weights warm, tuning `max_tokens` down to 70–90, and re-running the spike. The headline stays ~1,230 ms only if the local rung's P90 first token is under ~700 ms; otherwise the README says "sub-1.3 s on Groq, degrades to scripted on failure" |
| **Render free requires a card at signup** (§14 #13) | Hour 0. Do the signup on Day 0, before writing code | **+2 h** | Drop Tier 1 entirely. Tier 0 was already the public artifact (§6.8), so this costs a paragraph in the README rather than a feature — and saves the 7.5 h in cut item 1 |
| **`sw_vers -productVersion` < 15.0** (§14 #8) | Hour 0, one command | **+2 h** | `parakeet-mlx` becomes primary: different streaming API (`transcribe_stream` with `finalized_tokens`/`draft_tokens`), adds an `ffmpeg` dependency, and §4's word-timing story needs re-checking against it — which the evidence-span validator and the delivery metrics both depend on |
| **Groq 429s during a full 10-minute rehearsal at the demo hour** (§11 #2) | M3, at the first full rehearsal | **+2 h** | Tighten `budget_tokens`, lower `keep_verbatim_turns`, and verify the rung-1 switch actually fires and is visible in the HUD. A 429 must be a visible design decision, not a traceback on the projector |

### Cut list, in order, if the hours are not there

Cutting is a decision, not a failure — but cut in this order and say in the README what was cut
and why.

| # | Cut | Saves | What is lost |
|---|---|---|---|
| **1** | **Tier 1, the live hosted link** — `guard.py`, `config/hosted.yaml`, the consent screen, the Render deploy | **−7.5 h** | Nothing a recruiter reads. Tier 0 (static session report + recorded walkthrough + measured artifacts) was already the public artifact. Keep §6.8's arithmetic in the README as a costed analysis — "here is why I did not ship a public live link" is itself a good answer. **`wire.py` and `sessions.py` stay: they are the local build's transport** |
| **2** | `stt/deepgram_flux.py` + `llm/nim_llm.py` second backends | **−2 h** | The provider-swap story degrades from a comparison table to "the interface exists; one backend is measured". §12 #8 already says Deepgram is the thing you would do with money, and NVIDIA's ToS §1.4 forbids production use anyway |
| **3** | `report.py` audio-linked feedback → plain timestamped HTML, no audio seek | **−1 h** | The clickable citation, which is the most demo-able consequence of the span matcher. Cut reluctantly |
| **4** | Résumé-grounded adversarial probing (§9.7 #8) | **−3 days** | Already ranked last, already table stakes elsewhere |

**Never cut, in any scenario:** `metrics/clock.py` + `metrics/turn_log.py`, `bench/replay.py`,
`bench/endpoint.py`, `bench/calibrate.py` and the calibration set, `tests/test_turn_controller.py`,
the acoustic loopback calibration, or the local LLM rung. Those are §9.7 features #1, #3 and #5 —
the entire argument that this project is engineering rather than a prompt — and they are the first
things that feel cuttable at hour 80. That is precisely why §10 puts the instrumentation in
Milestone 1. (Earlier drafts listed the acoustic calibration as optional and as a cut candidate;
it is neither. It is the only thing that can validate the clock offset the headline number rests
on.)

---

## 11. Failure modes

Ordered by likelihood of biting you during a live demo.

| # | Failure | Mitigation |
|---|---|---|
| **1** | **Echo loop.** Without AEC the bot's TTS trips the VAD, fires a barge-in, cancels itself, and loops | Browser AEC + loopback playback. **Test with speakers at demo volume, not headphones.** Keep headphones as the guaranteed fallback and say so in the README. The M1 smoke test is the detection point and §10's +5 h contingency is the response |
| **2** | **Groq 8,000 TPM exhaustion mid-demo.** Not 30 RPM — TPM. A probe turn is ~1,585 tokens and a five-call scoring batch is ~2,650; a session's worst minute is ~4,235 | Cap history in *tokens* with §3.1's three-region layout; roll over at question boundaries only; reserve `HOT_PATH_RESERVE_TPM = 3000` for the conversation; read `x-ratelimit-remaining-tokens` off **every** response (not just 429s) into the turn log and the HUD. On 429, **§11.1 rung 1 fires**: `provider_switch` frame, pre-rendered filler, local `gemma-3n-e4b-it-text`, sticky for `max(120 s, retry-after)`. Rubric scoring runs on `gpt-oss-20b`, which is *probably* a separate rate-limit bucket (§14 #23) — the arithmetic assumes it is not. **Rehearse a full 10-minute session at the demo hour.** Note the free plan gets only the `on_demand` service tier, documented as having "occasional queue latency during peak times" |
| **3** | **The LLM row of the latency budget is wrong.** If your measured time-to-first-speakable-token is ~800 ms+, the ~1,230 ms plan is arithmetically unreachable with a cloud LLM | The Day-0 spike measures **three** candidates including the local one, so the fallback is exercised on day zero rather than discovered on demo day (3.5 h, §10 M1). Ladder order: `qwen3.6-27b` → `qwen3.8-27b` (availability only) → **local** → scripted probes. **gpt-oss-20b is never a hot-path fallback**: at 3.55 s to first answer token it does not degrade the product, it breaks it |
| **4** | **Preview model disappears.** `qwen/qwen3.6-27b` is a Beta Service that "may be discontinued at short notice." Groq has shut down 8 models in 12 months, including both Llama chat models on 2026-08-16 | Model IDs in config, never literals. Startup `GET /openai/v1/models` check that fails loudly. **Successor is `qwen/qwen3.8-27b`** — the only other free model with `reasoning_effort:"none"`, and also a strict-JSON model so it doubles as the scoring fallback. Re-verify the week of the demo |
| **5** | **Barge-in has a long tail.** `sounddevice.stop()` drains buffers, `latency` defaults to `'high'`, and aborting *after* awaiting the cancelled task adds a whole scheduler round-trip — the bot keeps talking for hundreds of ms after interruption and the demo looks broken | `abort()` first, then unwind; `latency='low'`; small blocksize. Report `barge_in_detected_ms`, `audio_abort_ms` and `client_audio_silent_ms` as **separate** metrics; the gap between the last two is your jitter buffer and it is the honest number |
| **6** | **Silero `ValueError` on the first frame under load.** It requires exactly 512 samples at 16 kHz; audio callbacks deliver variable block sizes | Ring-buffer and re-chunk in `audio/frames.py`; have the browser worklet emit 512-sample frames so no re-chunking is needed on the happy path. The Hypothesis split-invariance property in §10 is the test that actually catches this |
| **7** | **`CancelledError` swallowed.** It subclasses `BaseException`; a bare `except:` in a TTS or HTTP wrapper deadlocks barge-in | Audit every handler on the response path. Always `raise` after cleanup. `test_cancel_awaits_the_task` asserts it |
| **8** | **Copy-pasted tutorial code fails immediately.** Essentially every Groq voice-agent blog post and repo hardcodes `llama-3.1-8b-instant` or `llama-3.3-70b-versatile`. LiveKit's own Groq example README still references `playai-tts`, shut down 2025-12-31. `faster-whisper` tutorials assume GPU acceleration that does not exist on a Mac | Treat any tutorial older than a few months as wrong about model IDs. Validate against `/v1/models` |
| **9** | **Network variance on demo-day wifi** dominates everything you optimised | Rehearse on the actual demo network. Keep the fully-local path one config flag away. Record a backup walkthrough video — which Tier 0 (§6.8) already requires |
| **10** | **The hosted link is a different product than the one you built, and it can be drained by a stranger.** Render Free is 0.1 CPU / 512 MB: `moonshine-voice` has no Linux wheel at all, and Kokoro at RTF 0.08 on an M5 Max lands around RTF 2.4 on a 10% CPU quota. So the hosted build is Deepgram Flux + Aura-2, and it spends real credit — Flux bills on **stream duration**, so five idle sockets burn **$55/day** and the $200 pot is gone in 3.6 days with nobody speaking | Ship the static session report + recorded walkthrough as the public artifact (Tier 0); gate the live link behind a rotating invite key in the URL **fragment**. Global concurrency 1, 6-minute hard cap, 5 sessions/day, 3 per IP hash per 24 h, 40 mic frames/s. Two daily ledgers (120,000 Groq tokens, 200 Deepgram cents) that **reserve before the call and fail closed** with a friendly message |
| **11** | **Render cold start.** A recruiter clicking the link after 15 idle minutes waits ~1 minute | `/healthz` warm-up on page load with a determinate "waking the free instance — about 60 s" state; `/healthz` touches **no** paid provider. No external cron pingers: 730 of your 750 monthly instance-hours would go to staying awake. Warm it yourself before you send the link. Free sockets die on every deploy (30 s SIGTERM grace) — broadcast `error{code:"server_restarting"}` and reconnect with session resume |
| **12** | **Statistics theatre.** A P95 from 20 hand-run turns is noise | The replay harness is not optional. N=200 in stub mode and N=180 accumulated in live-LLM mode, reported separately; bootstrap CIs; always print N; never publish a mean latency; never average cold and warm runs, and never average two ladder rungs, into one distribution |
| **13** | **Licence landmine in a public repo.** Coqui XTTS's licence page and whole domain 404. Piper's VOICES.md says it is "intended for personal use and text to speech research only." LiveKit's turn detector forbids standalone use. CrisperWhisper weights are non-commercial | Use only Apache-2.0/MIT/BSD components on the shipped path, and state each licence in the README |
| **14** | **Render egress cliff.** The Hobby workspace includes **5 GB/month outbound**, and with no payment method on file Render **spins down every service in the workspace until the 1st of next month.** Five concurrent `linear16` downlinks at 48 KB/s exhaust it in **under 6.2 hours** | `mulaw` @ 24 kHz downlink (24 KB/s → ~2.6 MB per 6-minute session → ~1,970 sessions/month of headroom). Never send `linear16` over the public link. Track cumulative egress in the daily ledger and alert yourself at 3 GB |
| **15** | **Resume commits audio the user never heard.** The socket drops mid-utterance; the server already sent the PCM, so history records feedback that died in a jitter buffer — indistinguishable from hallucination on the next turn | `playback_ack {utt_id, frames_played}` every 250 ms; assistant sentences sit in `pending_spoken` until acked and are dropped on resume if `frames_played < frames_sent`. This is §8.2's "commit only what the user heard" extended across a dead socket |

### §11.1 The degradation ladder

Rows #2, #3 and #4 all end with "fall back". This is the fallback, specified.

#### Rungs

| Rung | LLM | Answer budget | Triggers entry | What the user **sees** | What the user **hears** |
|---|---|---|---|---|---|
| **0** | Groq `qwen/qwen3.6-27b`, `effort:"none"` | `max_completion_tokens=120` | Default | HUD badge green: `groq · qwen3.6-27b`, live `x-ratelimit-remaining-tokens` bar | Nothing unusual. ~1,230 ms `t_v2v` |
| **0b** | Groq `qwen/qwen3.8-27b`, `effort:"none"` | 120 | Startup `/v1/models` miss on 0, **or** two 404 `model_not_found` in one session | Same badge, different model ID | Nothing. Same class of model |
| **1** | **Local `gemma-3n-e4b-it-text`** | `max_tokens=90` | 429, first-token timeout > 850 ms, connect error, or 5xx — **before any content delta** | Badge turns amber: `local · gemma-3n-e4b-it-text`. One toast, 6 s: *"Groq rate limit — running the model on this Mac."* | A **pre-rendered** 0.9 s filler — *"Let me think about that for a second."* — played the instant the switch fires, covering the local model's prefill. Then a slightly shorter answer |
| **2** | **None.** Scripted probes | — | Rung 1 `health()` false at startup, or rung 1 also first-token-times-out | Badge red: `scripted`. Toast: *"No language model reachable — running the scripted question set."* | Pre-rendered audio for the 6 generic competency probes from §9.6's table ("What was the measurable outcome?"). The interview continues; it stops adapting |
| **3** | Hard stop | — | Rung 2 exhausts its probe list, or the WS drops and reconnect fails 3× | Transcript stays on screen with a **Download report** button | One pre-rendered line: *"I've lost the language model. Your transcript and timings are saved."* Then silence |

Rung 2 is the one that actually saves a demo, and it costs about an hour: six WAVs generated
offline with the same Kokoro voice, plus the gap-detector tracker in `coachlogic/probe.py` that
already exists to pick which probe to ask. A coach that keeps asking sensible interview questions
with no LLM at all is a better failure story than a spinner.

#### The local rung, and what is and is not known about it

**Named model: `gemma-3n-e4b-it-text`, ~4 GB resident, served by LM Studio.** That exact model ID
and server are what `kwindla/macos-local-voice-agents` — the Pipecat-author repo cited in §13 for
the "<800 ms fully local voice-to-voice on an M-series Mac" claim — configures in `server/bot.py`
at `base_url="http://127.0.0.1:1234/v1"`. Using the same model means the one public latency anchor
for this configuration actually applies to it.

**Expected time to first token: 250–600 ms for a ~1,800-token prompt on an M-series Mac.
⚠ UNVERIFIED — no published benchmark measures prefill latency for a 4B-class 4-bit MLX model at
this prompt length on this hardware.** The Apple-Silicon inference benchmarks that exist report
prompt-processing *throughput* at synthetic prompt lengths, not end-to-end TTFT for a chat request,
and they disagree with each other. What *is* defensible is an upper bound by subtraction: that repo
reports total voice-to-voice under 800 ms with Silero VAD + MLX Whisper large-v3-turbo-q4 + this
model + Kokoro in the loop, so the LLM's contribution must be well under 800 ms. Treat 250–600 ms
as a hypothesis to be killed or confirmed by the Day-0 spike (§14 #25).

Alternative if Gemma 3n disappoints on instruction-following:
`mlx-community/Qwen3-4B-Instruct-2507-4bit` (verified present on Hugging Face, 2026-09-13) — the
`-Instruct-` variant specifically, because the reasoning variants reintroduce exactly the
chain-of-thought problem that disqualified gpt-oss-20b.

#### Thresholds, and why these numbers

```yaml
# config/default.yaml
llm:
  rungs:
    - {id: groq_qwen,  provider: groq,  model: "qwen/qwen3.6-27b", reasoning_effort: none, max_tokens: 120}
    - {id: groq_qwen8, provider: groq,  model: "qwen/qwen3.8-27b", reasoning_effort: none, max_tokens: 120, availability_only: true}
    - {id: local,      provider: local, model: "gemma-3n-e4b-it-text", base_url: "http://127.0.0.1:1234/v1", max_tokens: 90}
    - {id: scripted,   provider: scripted}
  thresholds:
    first_token_timeout_ms: 850     # rung 0; rung 1 overrides to 1500
    stall_timeout_ms: 1200          # rung 0; rung 1 overrides to 2500
    switch_only_before_first_token: true
    sticky_seconds: 120
    probe_interval_seconds: 60
  scoring:                          # off the hot path
    provider: groq
    model: "openai/gpt-oss-20b"
    reasoning_effort: low
    max_tokens: 200
    response_format: json_schema
```

- **`first_token_timeout_ms: 850`.** §7 budgets 300–800 ms for the LLM row, plan 500. At 850 ms the
  turn is already lost; waiting longer only makes it worse. Switching costs 850 ms + local TTFT,
  which the pre-rendered filler covers because the filler starts playing at the switch, not after
  the local model responds.
- **`stall_timeout_ms: 1200`, and `switch_only_before_first_token: true`.** If the user has
  already *heard* half a sentence, switching providers mid-answer changes the model's voice and
  register inside one utterance, which reads as a bug. So a mid-stream stall is not a switch: it
  ends the turn with whatever was spoken, commits only that to history (§8.2 detail 3), and the
  coach moves on. The switch decision belongs entirely before the first audible word.
- **`sticky_seconds: 120`.** After dropping to rung 1, do not re-try rung 0 on the next live
  turn. Groq's 429 response carries `retry-after` in seconds and every response carries
  `x-ratelimit-reset-tokens`; hold the lower rung for `max(120 s, retry-after, reset_tokens)`.
  Then probe rung 0 **in the background** with a 1-token request every `probe_interval_seconds`,
  and promote only after a clean probe. Never gamble a live turn on recovery.

#### The control frame

`provider_switch` joins the JSON control frames of §6.2:

```json
{
  "v": 1, "t": "provider_switch", "seq": 87, "ts": 8321,
  "turn_id": 14,
  "from": {"rung": "groq_qwen", "provider": "groq", "model": "qwen/qwen3.6-27b"},
  "to":   {"rung": "local", "provider": "local", "model": "gemma-3n-e4b-it-text"},
  "reason": "first_token_timeout",
  "detail": {"waited_ms": 851, "threshold_ms": 850,
             "ratelimit_remaining_tokens": "412", "retry_after_s": null},
  "ui": {"badge": "amber", "label": "local · gemma-3n-e4b-it-text",
         "toast": "Groq rate limit — running the model on this Mac."},
  "filler_audio": "prerendered/think_about_that.wav"
}
```

`reason` is a closed enum: `rate_limited_429`, `first_token_timeout`, `connect_error`,
`http_5xx`, `model_not_found`, `startup_health_fail`, `manual`, `recovered`. Every frame is also a
row in the turn log via `llm_rung`, so `bench/latency_report.py` can split the latency CDF by rung
— which is the honest way to publish the number. **Never average rung-0 and rung-1 turns into one
P95.**

#### `src/coach/llm/stream.py`

Shared by every provider, so the first-token timeout that drives the ladder is implemented once
and measured identically on Groq, local, and NIM.

```python
"""Provider-agnostic SSE → text-delta stream with first-token and stall deadlines.

Satisfies the LLMProvider contract used by TurnController (§8.2):
async-iterable of `str`, plus an async `close()` that drops the HTTP body.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import AsyncIterator

import httpx


class FirstTokenTimeout(asyncio.TimeoutError):
    """No *content* delta arrived inside first_token_timeout_ms. Ladder-switchable."""


class StreamStalled(asyncio.TimeoutError):
    """Stream went quiet mid-answer. NOT ladder-switchable — the user has already heard words."""


class DeltaStream:
    def __init__(
        self,
        ctx,                              # the un-entered httpx stream context manager
        response: httpx.Response,
        *,
        provider: str,
        model: str,
        first_token_timeout_ms: int,
        stall_timeout_ms: int,
    ) -> None:
        self._ctx, self._r = ctx, response
        self.provider, self.model = provider, model
        self._first_ms, self._stall_ms = first_token_timeout_ms, stall_timeout_ms
        self.saw_first_token = False
        self.reasoning_chars = 0          # must stay 0; see assertion below
        self.usage: dict | None = None
        # Rate-limit telemetry, read on EVERY response, not only on 429 (§11 #2).
        h = response.headers
        self.ratelimit = {
            "remaining_tokens": h.get("x-ratelimit-remaining-tokens"),
            "remaining_requests": h.get("x-ratelimit-remaining-requests"),
            "reset_tokens": h.get("x-ratelimit-reset-tokens"),
        }

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        try:
            async with asyncio.timeout(self._first_ms / 1000) as deadline:
                async for line in self._r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    chunk = json.loads(payload)
                    if u := chunk.get("usage"):
                        self.usage = u
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    # Qwen/NIM leak chain-of-thought here when the kill switch is misconfigured.
                    # Count it, never speak it, and fail the startup check if it is ever non-zero.
                    if rc := delta.get("reasoning_content"):
                        self.reasoning_chars += len(rc)
                        continue
                    text = delta.get("content")
                    if not text:
                        continue
                    self.saw_first_token = True
                    deadline.reschedule(loop.time() + self._stall_ms / 1000)
                    yield text
        except asyncio.TimeoutError as exc:
            raise (StreamStalled if self.saw_first_token else FirstTokenTimeout)(
                f"{self.provider}:{self.model} "
                f"{'stalled' if self.saw_first_token else 'no first token'}"
            ) from exc

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._r.aclose()
        with contextlib.suppress(Exception):
            await self._ctx.__aexit__(None, None, None)
```

`asyncio.timeout(...).reschedule()` is the 3.11+ API; it is how one context manager enforces two
different deadlines — a tight one until the first token, a looser one between tokens — without a
second task.

#### `src/coach/llm/local_llm.py`

```python
"""Rung 1 of the degradation ladder: a local OpenAI-compatible LLM server.

Tested against LM Studio (default) and Ollama. Both expose /v1/chat/completions,
/v1/models and SSE streaming; neither validates the API key.

    LM Studio : base_url http://127.0.0.1:1234/v1   model "gemma-3n-e4b-it-text"
    Ollama    : base_url http://127.0.0.1:11434/v1  model "gemma3n:e4b"

Keep the weights resident or the first request after idle pays a cold load:
    LM Studio -> Developer tab, set JIT auto-unload TTL to 0 (never unload)
    Ollama    -> OLLAMA_KEEP_ALIVE=-1
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from .stream import DeltaStream


@dataclass(frozen=True)
class LocalLLMConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "gemma-3n-e4b-it-text"
    api_key: str = "local"                 # ignored by both servers; the SDK contract wants one
    max_tokens: int = 90                   # not 120: local decode is ~5x slower than Groq's 437 t/s
    temperature: float = 0.6
    # Looser than rung 0's 850 ms because there is no faster rung below this one.
    first_token_timeout_ms: int = 1_500
    stall_timeout_ms: int = 2_500
    connect_timeout_s: float = 0.5         # localhost: if it doesn't connect fast, it isn't running


class LocalLLM:
    provider = "local"
    rung_id = "local"

    def __init__(self, cfg: LocalLLMConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=httpx.Timeout(30.0, connect=cfg.connect_timeout_s),
            limits=httpx.Limits(max_keepalive_connections=2, keepalive_expiry=600),
        )

    async def health(self) -> bool:
        """Startup check. Mirrors the Groq GET /v1/models check (§3)."""
        try:
            r = await self._client.get("/models", timeout=2.0)
            r.raise_for_status()
        except Exception:
            return False
        return any(m.get("id") == self.cfg.model for m in r.json().get("data", []))

    async def warm(self) -> float:
        """Force weight load + KV allocation. Returns seconds. Called from lifespan and /healthz.

        LM Studio JIT-loads on first request; a cold 4B load is seconds, not milliseconds,
        and it would otherwise land on the first turn the ladder ever needs. Never skip this.
        """
        import time
        t0 = time.perf_counter()
        await self._client.post(
            "/chat/completions",
            json={
                "model": self.cfg.model,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 1,
                "stream": False,
            },
            timeout=120.0,
        )
        return time.perf_counter() - t0

    async def stream_chat(self, messages: list[dict]) -> DeltaStream:
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "stream": True,
        }
        ctx = self._client.stream("POST", "/chat/completions", json=body)
        response = await ctx.__aenter__()
        response.raise_for_status()
        return DeltaStream(
            ctx,
            response,
            provider=self.provider,
            model=self.model,
            first_token_timeout_ms=self.cfg.first_token_timeout_ms,
            stall_timeout_ms=self.cfg.stall_timeout_ms,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
```

#### `src/coach/llm/ladder.py`

```python
"""Rung-switching LLMProvider. Presents the same interface as any single provider,
so TurnController (§8.2) is unchanged."""
from __future__ import annotations

import contextlib
import time
from typing import AsyncIterator, Awaitable, Callable, Sequence

import httpx

from .stream import DeltaStream, FirstTokenTimeout

SWITCHABLE = (FirstTokenTimeout, httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)

_REASONS = {
    FirstTokenTimeout: "first_token_timeout",
    httpx.ConnectError: "connect_error",
    httpx.ConnectTimeout: "connect_error",
    httpx.ReadTimeout: "first_token_timeout",
}


def _reason_for(exc: Exception) -> str:
    return _REASONS.get(type(exc), "connect_error")


def _rung_desc(rung) -> dict | None:
    if rung is None:
        return None
    return {"rung": rung.rung_id, "provider": rung.provider, "model": getattr(rung, "model", None)}


def _frame(frm, to, reason: str, retry_after: float, stream: DeltaStream | None = None) -> dict:
    return {
        "type": "provider_switch",
        "from": _rung_desc(frm),
        "to": _rung_desc(to),
        "reason": reason,
        "detail": {
            "retry_after_s": retry_after or None,
            "ratelimit_remaining_tokens": (stream.ratelimit["remaining_tokens"] if stream else None),
        },
        "ui": _UI.get(getattr(to, "rung_id", "scripted"), _UI["scripted"]),
        "filler_audio": "prerendered/think_about_that.wav",
    }


_UI = {
    "groq_qwen":  {"badge": "green", "label": "groq · qwen3.6-27b", "toast": None},
    "groq_qwen8": {"badge": "green", "label": "groq · qwen3.8-27b", "toast": None},
    "local":      {"badge": "amber", "label": "local · gemma-3n-e4b-it-text",
                   "toast": "Groq rate limit — running the model on this Mac."},
    "scripted":   {"badge": "red", "label": "scripted",
                   "toast": "No language model reachable — running the scripted question set."},
}


async def _one_token_ok(rung) -> bool:
    """Background health probe. Cheapest possible request against the rung."""
    try:
        stream = await rung.stream_chat([{"role": "user", "content": "ok"}])
    except Exception:
        return False
    try:
        async for _ in stream:
            return True
    except Exception:
        return False
    finally:
        await stream.aclose()
    return False


class _Prefixed:
    """Replays the delta the ladder already pulled, then defers to the real stream.

    The ladder must consume the first delta itself — that is the only way a
    FirstTokenTimeout becomes the ladder's problem instead of the caller's.
    """

    def __init__(self, first: str, inner: AsyncIterator[str], stream: DeltaStream) -> None:
        self._first, self._inner, self._stream = first, inner, stream
        self.provider, self.model = stream.provider, stream.model
        self.ratelimit, self.usage = stream.ratelimit, stream.usage

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        yield self._first
        async for delta in self._inner:
            yield delta

    async def aclose(self) -> None:
        await self._stream.aclose()


class LLMLadder:
    def __init__(
        self,
        rungs: Sequence[object],                       # providers, best-first
        *,
        emit: Callable[[dict], Awaitable[None]],       # sends the provider_switch WS frame
        sticky_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,   # injectable for tests
    ) -> None:
        self.rungs, self.emit = list(rungs), emit
        self.sticky_seconds, self.clock = sticky_seconds, clock
        self.i = 0
        self.sticky_until = 0.0

    async def stream_chat(self, messages: list[dict], turn_log=None):
        last: Exception | None = None
        while self.i < len(self.rungs):
            rung = self.rungs[self.i]
            stream = None
            try:
                stream = await rung.stream_chat(messages)
                # Pull the first delta HERE. This is the whole design: a first-token
                # timeout has to be the ladder's problem, and it stops being switchable
                # the instant the caller has a token it can hand to TTS.
                inner = stream.__aiter__()
                first = await anext(inner)
                if turn_log is not None:
                    turn_log.set("llm_rung", rung.rung_id)
                return _Prefixed(first, inner, stream)
            except StopAsyncIteration:                  # empty completion == a dead rung
                await self._demote(rung, "http_5xx", 0.0, turn_log, stream)
                last = RuntimeError("empty completion")
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code == 429:
                    retry_after = float(exc.response.headers.get("retry-after", 0) or 0)
                    await self._demote(rung, "rate_limited_429", retry_after, turn_log, stream)
                elif code == 404:
                    await self._demote(rung, "model_not_found", 0.0, turn_log, stream)
                elif code >= 500:
                    await self._demote(rung, "http_5xx", 0.0, turn_log, stream)
                else:
                    raise
                last = exc
            except SWITCHABLE as exc:
                await self._demote(rung, _reason_for(exc), 0.0, turn_log, stream)
                last = exc
        raise RuntimeError("ladder exhausted") from last

    async def _demote(self, rung, reason: str, retry_after: float, turn_log, stream=None) -> None:
        if stream is not None:                          # never leak the abandoned HTTP body
            with contextlib.suppress(Exception):
                await stream.aclose()
        nxt = self.rungs[self.i + 1] if self.i + 1 < len(self.rungs) else None
        self.i += 1
        self.sticky_until = self.clock() + max(self.sticky_seconds, retry_after)
        if turn_log is not None:
            turn_log.mark(f"provider_switch:{reason}")
        await self.emit(_frame(rung, nxt, reason, retry_after, stream))

    async def maybe_promote(self) -> None:
        """Background task, every probe_interval_seconds. Never called on the hot path."""
        if self.i == 0 or self.clock() < self.sticky_until:
            return
        best = self.rungs[0]
        if await _one_token_ok(best):
            self.i = 0
            await self.emit(_frame(None, best, "recovered", 0.0))
```

Every rung object supplies three attributes the ladder reads: `rung_id`, `provider`, `model`.
`clock` is injected so `tests/test_ladder.py` can advance the sticky window without sleeping.
`anext()` is 3.10+; the project already requires 3.11+ for `asyncio.timeout`.

---

## 12. Interview answers

**1. "Why is this cascaded rather than a single speech-to-speech model?"**
Partly constraint, partly merit. Groq has no realtime or speech-to-speech endpoint at all — I
verified the API surface — so unified S2S wasn't reachable inside the $0 + Groq/NIM constraint.
But it's also the right call: this product has to follow a scoring rubric and produce structured
feedback with evidence citations, and instruction-following is exactly where S2S models are
weakest, because the text LLM is bypassed. The trade-off I accept is losing prosody at the STT
boundary — I don't score tone, deliberately.

**2. "Why does speech recognition run locally when you're already calling a cloud API?"**
Two reasons, and I want to be careful not to overstate the first one. The structural reason is that
Groq's Whisper endpoint is batch-only — no `stream` parameter, no WebSocket, no interim transcripts —
and capped at 20 requests per minute on the free tier, which is one request every three seconds.
Chunked pseudo-streaming needs 60–120 RPM, so the ceiling alone rules it out, and without partials
there is nothing to gate barge-in on. The latency reason is smaller than people expect. Moonshine v2
Medium's published finalization latency — end-of-speech to final transcript — is 258 ms in the paper
on an M3, against 300–400 ms for a Groq batch round-trip on a three-second clip. That is a 50–150 ms
win, not a 300 ms one, and I only get it because the encoder has already consumed the audio while the
candidate was speaking; the decode still costs what it costs. Interestingly the vendor's own repo
publishes 107 ms for that same metric on a Mac, and its benchmark notes mention runs at 56–83 ms, so
the published figures span roughly 3×. Rather than pick one, I log
`TranscriptLine.last_transcription_latency_ms` every turn and publish the distribution — the library
measures this itself, and one field settled what three tables could not.

**3. "Walk me through where the time goes in one turn."**
I'll use one term consistently: `t_v2v`, the acoustic end of my speech to the first audible sample of
the reply. Plan is about 1.23 seconds. Roughly 28 ms of mic capture, a 32 ms frame quantum and the
socket hop; 200 ms of VAD silence to arm a candidate endpoint; then 258 ms for the streaming STT to
finalize, and I trigger that pass the instant the silence arms rather than waiting for the turn
decision, which buys me the 25 ms smart-turn classifier for free underneath it. Then the LLM's time
to first *speakable* token, about 500 ms, which is the term that can blow the budget and is why I
measured it before writing any pipeline code. Then 35 ms to assemble the first sentence, 140 ms for
Kokoro to synthesise the first clause, and 65 ms of last mile — WebSocket hop, jitter buffer, and
device output latency. The last mile is the part I originally got wrong: every mark was a
`time.monotonic()` reading in Python, but "first audible word" happens in the browser on
`performance.now()`, and those two clocks have unrelated epochs. So I estimate the offset NTP-style
over the same socket, keep the minimum-RTT sample, and report the residual uncertainty as ±rtt/2 —
sub-millisecond locally, ±40 ms over the hosted deployment. Then I checked the whole thing against a
BlackHole loopback recording and a phone recording of the room, because an instrumentation stack
that has never met a microphone is a hypothesis. Here are my P50 and P95 with N and confidence
intervals, and the measured agreement with the acoustic ground truth — [show the CDF].

**4. "What's the hardest bug you hit?"**
The reasoning-token trap. Artificial Analysis publishes `gpt-oss-20b` at 0.81 s time-to-first-token
on Groq, which looks fine — but its time to first *answer* token is 3.55 s, because the model
emits chain-of-thought first and `reasoning_effort` has no "off" setting; the floor is "low", and
"low" is barely faster than "high". For a voice agent, TTFT is a meaningless metric — the only
number that matters is time to the first token you can hand to a speech synthesiser. I switched
to `qwen/qwen3.6-27b` with `reasoning_effort: "none"`, which is the only model on Groq's free
tier with a documented reasoning kill switch. The part I got wrong for longer than I'd like is
what I did with gpt-oss afterwards: I'd written it into three places as the fallback, without
noticing that a fallback taking 3.55 seconds to its first speakable token doesn't degrade the
product, it breaks it. The fix was to separate the two axes — the latency and quota fallback goes
*local*, and the availability fallback, for when Groq retires a Preview model, goes to
`qwen3.8-27b`. And gpt-oss moved to the post-turn rubric scoring path, where 3.55 seconds costs
nothing and where it's actually the only free Groq model that supports strict JSON schemas, which
the scoring path needs and qwen3.6 doesn't have at all.

**5. "How does barge-in actually work?"**
Three things have to happen together, and the ordering matters more than any of them. A monotonic
generation id invalidates any in-flight work from the previous turn. `asyncio` cancellation
propagates into the LLM stream and the TTS generator — and I `await` the cancelled task rather
than fire-and-forget, because `Task.cancel()` only schedules a `CancelledError` for the next
event-loop cycle and doesn't guarantee cancellation. And critically, the audio device gets
`abort()`, not `stop()` — `stop()` waits for pending buffers to play out — and the `abort()`
happens *before* the await, not after. That one ordering bug was costing a whole scheduler
round-trip of audio, and it meant my "audio stopped" mark was measuring cancellation time instead
of audible-stop time. Then I commit only the sentences that were actually spoken to conversation
history, so the model never references feedback the user didn't hear — and the sentence it was cut
off mid-way through is committed as the words that actually reached the speaker, computed from the
player's played-frames counter. It's accurate to a word or two, because Kokoro doesn't give word
timings, and that's fine: the point is not a verbatim record, it's that the model can't reference
something nobody heard.

**6. "How do you know the feedback is any good — not just plausible-sounding?"**
I hand-scored 50 recorded answers against the rubric and measured the judge's agreement with
them — quadratically-weighted Cohen's kappa, per dimension, with bootstrap confidence intervals,
because unweighted kappa on a 1-to-5 anchored scale treats a one-level disagreement as identical
to a four-level one, and because weighted kappa with quadratic weights is the version that maps
onto the ICCs the interview literature reports. The important part is what I measured it
*against*. I'm one annotator, so agreement with me conflates judge error with my own noise. So I
re-scored 20 answers a week later blind — that's my test–retest, the ceiling on anything measured
against me — and I had a second person score the same 20, which gives a human–human ceiling on
*my* rubric. The judge is reported relative to that, not relative to a published figure from a
different instrument. For external context, the ETS BARS study reports an ICC of .74 across three
raters, dimension range .66 to .82 — different statistic, 7-point scale, written responses, 652
of them, so I cite it and explicitly don't compare to it. And I say the limitation first: at
N = 50 the interval on kappa is about ±0.17, so I can't distinguish 0.63 from 0.75, and I don't
pretend to. One more number I publish that nobody else does: the ungrounded rate — the fraction
of scores I threw away because the model couldn't quote a span of the transcript that actually
existed.

**7. "What did you deliberately not build?"**
Confidence, emotion, and "hireability" scoring. That's the HireVue failure mode — it dropped
facial analysis in January 2021 after an FTC complaint, conceding the technology "wasn't worth
the concern." I also cut filler-word counting from v1, for a technical reason: Whisper is trained
to delete "um" and "uh," so any count built on it is fabricated, and the verbatim alternative
ships non-commercial research weights. And I made delivery metrics opt-in and purely
descriptive, because pace and disfluency metrics systematically penalise disabled and
second-language speakers — that's an ADA screen-out problem, not just a UX one. The rubric
enforces that structurally rather than by convention: the YAML has a `never_score` list naming
accent, fluency, pace and articulateness, and the scoring prompt repeats the prohibition.

**8. "What would you do differently with a budget, or more time?"**
Three things. First, Deepgram Flux for STT — it's the only provider that makes turn-taking an
API primitive, with eager end-of-turn events that let you start generating on a
medium-confidence transcript while the user is still finishing, and a guarantee that the final
transcript matches the eager one. I have it wired in behind the provider interface and measured
against local; it's also the STT layer of the hosted build, because Moonshine has no Linux wheel.
Second, colocating the LLM with everything else, which the literature puts at 50–200 ms of
savings. Third, a much larger calibration set — 50 labelled answers gives a ±0.17 interval on
kappa, and I report the confidence interval rather than pretending otherwise. A hundred answers
would take it to ±0.12 for another two and a half hours of hand-scoring, and beyond that you need
a second trained rater, not more of my own labels.

**9. "What happens when it fails?"**
Four rungs, and I have run all of them. Rung zero is Groq qwen3.6-27b. On a 429, a connection
error, or no first token inside 850 ms, the server emits a `provider_switch` frame, the browser
plays a pre-rendered "let me think about that for a second," and the turn completes on a local
Gemma 3n 4B in LM Studio — about 4 GB resident on this machine. The badge in the HUD turns amber
and names the model, because hiding a degradation from the person watching is how you lose
their trust. If there is no local server either, rung two drops to scripted STAR probes with
pre-rendered audio: it stops adapting, but it still runs a coherent interview, because the gap
detectors that choose which probe to ask are local regex, not the LLM. Rung three saves the
transcript and says so out loud. Two design rules make this work. First, 850 ms, not 3
seconds — by 850 ms the turn is already lost, so waiting costs more than switching. Second, the
switch only happens *before* the first audible word; if the user has already heard half a
sentence, a stall ends the turn with what was spoken rather than changing voices mid-utterance.
And every turn logs which rung served it, so the latency CDF is split by rung — averaging a
cloud turn and a local turn into one P95 would be the same mistake as averaging cold and warm runs.

**10. "How did you test this?"**
The unit suite runs in under five seconds with no microphone, no network, no API key and no
model weights, because every stage is behind a Protocol. Three things carry the weight. The
cancellation tests drive `TurnController` with a fake player and a fake LLM gated on an
`asyncio.Event` — no sleeps, so they're fast and not flaky — and they assert the four invariants
that are silently breakable: `abort()` called exactly once, nothing enqueued to the device after
abort, the HTTP body actually closed in the `finally`, and only the sentences the user *heard*
committed to history. The sentencizer has golden files for the cases that actually break it:
"Dr. Patel", "$1.2M", a terminator inside quotes, and — the real one — a period and its following
space arriving in two different SSE deltas, which a non-incremental sentencizer passes offline
and fails in production. I found two genuine bugs that way, including one where "no" was in my
abbreviation list and silently swallowed the boundary in "Dr. Chen said no. What did you do?"
And the 512-sample re-chunker has a Hypothesis property test whose key property is
split-invariance: the frame boundaries Silero sees must depend only on the sample stream, never
on how the audio callback happened to block it. That's the exact bug that throws `ValueError` on
the first frame under load. Then there's a layer above unit tests: `bench/` measures latency
percentiles with bootstrap CIs, pooled WER, weighted kappa for the judge, and false-cutoff rate
for the endpointer. What I deliberately don't unit-test is model output quality, the ONNX
classifier, and echo cancellation — the first two are measured statistically and the third is
hardware-dependent enough that a green CI badge would be a lie. That boundary is in the README.

**11. "How many people can use it at once?"**
One, locally, and the interesting part is that the Mac isn't what limits it. Arithmetic: a probe
turn costs about 1,585 tokens against Groq's published 8,000 per minute, and a five-call scoring
batch is another 2,650, so one session's worst minute is roughly 4,235 tokens. Two of those
colliding is 8,470 — over the cap. On average a session runs at about 1,650 tokens a minute, so
two fit comfortably and the third starts 429ing; at the peak, two collide. That's why the hosted
build enforces global concurrency of one rather than two: I'd rather turn the second visitor away
with a clear message than rate-limit both of them. So the free tier caps concurrency before
Moonshine, Kokoro or the CPU get anywhere near saturated — the resource table says the local
stack is about 1.15 GB and six runnable threads on eight cores. The architecture is per-session
objects around a shared model pool with one single-worker executor per model, and the server is
async throughout, so scaling is a worker pool with bounded queues and backpressure rather than a
rewrite — but I haven't built it, because building for concurrency I can't provision would be
theatre. The hosted Render build is a different topology entirely: 0.1 CPU and 512 MB can't do
inference at all — Moonshine doesn't even publish a Linux wheel — so it's a thin relay with
browser-side VAD and Deepgram doing STT and TTS, and I keep the two configs behind one interface
rather than pretending they're the same system.

**12. "What does a session cost you, and how many demos can you run in a day?"**
A full 17-minute session is about 34 Groq calls and 28,050 tokens against a published 200,000 per
day. That's seven full sessions daily, and requests-per-day never binds at 34 against 1,000. The
shape matters: twenty of the roughly thirty-four utterances in a session cost zero tokens, because
the intro, all five questions and all five move-on lines are pre-rendered Kokoro audio generated at
startup. Only the probes and the wrap go through the model. Scoring is 25 calls, five dimensions
across five answers, and it's the most cacheable text in the project — the system prompt and the
full rubric are byte-identical across all 25, so calls two onward pay about 530 tokens instead of
1,730, and Groq's cached tokens don't count toward rate limits at all. There's one sharp exception
I got wrong at first. My replay harness was specified as `--runs 200` through the real pipeline.
That's 374,000 tokens: it cannot execute against the free tier in a single day, full stop. So the
harness has two modes. Pipeline runs — VAD, STT, sentencizer, TTS, plumbing — use a recorded-SSE
stub that replays captured token timings, and those go to N=200. The live-LLM row runs at N=60
per day, accumulated across three days to N=180, and it's reported as its own configuration with
its own N. I'd rather publish two honest numbers than one averaged one. Every rate-limit header
goes into the turn log, so the budget is measured rather than estimated — and I'd re-verify the
caps themselves before quoting them, because Groq's own rate-limit page now labels its table as
Developer-plan base limits, which means the one authoritative source for my free-tier numbers is
the limits page on my own logged-in account.

**13. "What would you do differently if you started over?"**
Three things, and they're all the same mistake in different clothes: I specified before I
measured. First, I designed the fallback path around a second cloud model, and only noticed when
I laid the numbers side by side that its time to first *answer* token was 3.55 seconds — the
fallback would have been slower than the failure it was catching. I should have caught it by
writing the degradation ladder as an explicit table of rungs and thresholds on day one instead of
writing "fall back to X" in three separate places and never reconciling them. Second, I wrote
that endpointing was where the project was won or lost, and then built a benchmark harness that
measured latency, word error rate and judge agreement — and nothing about endpointing. I now
have 40 hand-labelled answers with deliberate thinking pauses and a threshold sweep, but I built
them in week four, which means the state machine was tuned by feel until then. Fixture sets
should come before the thing they measure. Third, I'd separate the classifier from the state
machine on day one. I nearly built `p_complete()` inside the endpointer, which would have made
it untestable and unsweepable without driving a live session — the same interface discipline
that made every other stage mockable, applied one layer too shallow. And a fourth, if you'll
allow it: my first latency budget claimed 900 milliseconds, which was 135 milliseconds below what
its own rows summed to, and it booked the transcription step at a number I could not source
anywhere. The version I publish is 1,230 and it reconciles. Publishing the larger, checkable
number is strictly better than publishing the smaller one and waiting for someone to add it up.

---

## 13. Sources

All URLs verified **2026-09-13**. This document says in its own first paragraph that these numbers
rot on a scale of weeks, so a live URL alone is not a citation — by the time a hiring manager
clicks it, Groq may have retired the model, Render may have changed the free tier, and
Artificial Analysis (a *live rolling benchmark*) will certainly show a different number than the
one quoted here. **Run `docs/archive_sources.sh` before publishing and again the week of the
demo**, and read the diff: a changed snapshot is your early warning that a number moved. See
§13.1.

### Groq
- https://console.groq.com/docs/rate-limits.md — free-tier limits. ⚠ **now self-labelled "base limits for the Developer plan"** (re-verified 2026-09-13); per-model-ID table; `retry-after` only on 429, `x-ratelimit-*` on every response
- https://console.groq.com/docs/models.md — model list, context, "SPEED (T/SEC)" column. ⚠ its rate-limit column is the **Developer** plan
- https://console.groq.com/docs/deprecations.md — Llama shutdown 2026-08-16; full shutdown history
- https://console.groq.com/docs/reasoning.md — `reasoning_effort` options per model
- https://console.groq.com/docs/structured-outputs.md — `json_schema` + `strict:true` on `gpt-oss-20b`, `gpt-oss-120b`, `qwen3.8-27b` only; **no streaming, no tool use**; `qwen3.6-27b` absent
- https://console.groq.com/docs/api-reference.md — `stream`, `stream_options`, audio endpoints
- https://console.groq.com/docs/speech-to-text.md — batch-only STT, WER, 25 MB free cap
- https://console.groq.com/docs/text-to-speech/orpheus — 200-char cap, WAV only
- https://console.groq.com/docs/prompt-caching — gpt-oss only; 2 h expiry; 50% discount; cached tokens don't count to limits; **minimum prefix "128 to 1024 tokens depending on the specific model"**; `usage.prompt_tokens_details.cached_tokens`
- https://console.groq.com/docs/service-tiers / .../flex-processing.md — free = `on_demand` only
- https://console.groq.com/docs/legal/services-agreement — §4.2 no training; §6.3
- https://console.groq.com/docs/legal/ai-policy.md — AUP; employment clause; multi-account ban
- https://console.groq.com/docs/your-data — default no retention
- https://console.groq.com/settings/billing/plans — free plan, no card
- https://console.groq.com/settings/limits — **your actual org limits; now the only authority for the free tier. Check this**
- https://console.groq.com/docs/changelog.md — ⚠ lags the models page; not authoritative
- https://artificialanalysis.ai/providers/groq — TTFT / first-answer-token / throughput (**live rolling benchmark, ±5%**)

### NVIDIA NIM
- https://integrate.api.nvidia.com/v1/models — 82 models, responds unauthenticated
- https://build.nvidia.com/llms.txt — "no credit card required"; base URL; OpenAI compat
- https://build.nvidia.com/models.md — official catalog listing
- https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b — `enable_thinking` default; streaming via body param
- https://build.nvidia.com/nvidia/nemotron-asr-streaming/api and /modelcard — gRPC, function ID, architecture
- https://build.nvidia.com/nvidia/magpie-tts-multilingual/api and /modelcard — function ID, 22.05 kHz, streaming
- https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html — offline vs streaming
- https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html — GPU requirement (CC 8.0+, 16 GB)
- https://docs.nvidia.com/nim/speech/latest/reference/api-references/asr/protos.html — `interim_results`, `EndpointingConfig`
- https://docs.nvidia.com/nim/speech/latest/reference/api-references/tts/protos.html — `SynthesizeOnline`, 2,000-char cap
- https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf — §1.4 non-production, §4.12
- https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA_Technology_Access_TOU.pdf — §2, §5(e), §7
- https://forums.developer.nvidia.com/t/api-credits-for-build-nvidia-com/306633 — credits (staff, 2024-09-16)
- https://forums.developer.nvidia.com/t/your-free-nvidia-api-credits-expire-in-2-days/318141 — 30-day expiry (staff, 2025-01-03)

### Local LLM
- https://github.com/kwindla/macos-local-voice-agents + `server/bot.py` — `gemma-3n-e4b-it-text` via LM Studio at `http://127.0.0.1:1234/v1`, "~4GB of RAM", <800 ms local voice-to-voice
- https://lmstudio.ai/docs/developer/openai-compat/chat-completions — base URL `http://localhost:1234/v1`, `/chat/completions`, supported fields, `stream`
- https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit — alternative local model (HTTP 200, 2026-09-13)
- https://huggingface.co/mlx-community/gemma-3n-E4B-it-lm-4bit — MLX weights (HTTP 200, 2026-09-13)

### STT
- https://arxiv.org/html/2602.12241v1 — Moonshine v2. Table 2 response latency (Medium 258 ms, Small 148, Tiny 50, Apple M3); WER by size (6.65 / 7.84 / 12.01%); the metric's definition; "algorithmic lookahead of 4 × 20 ms = 80 ms"; compute-load = inverse RTF
- https://github.com/moonshine-ai/moonshine/blob/main/docs/using/benchmarks.md — **the same metric defined verbatim**, the README table's provenance (`scripts/test-mobile-latency.sh`, `scripts/run-benchmarks.py`), the "56 to 83 ms across three runs" note, the <200 ms design goal, and `--transcription-interval` default 0.5 s
- https://github.com/moonshine-ai/moonshine/blob/main/language-bindings/python/src/moonshine_voice/transcriber.py — `MOONSHINE_FLAG_FORCE_UPDATE`, `Stream.update_transcription()` (synchronous FFI), the adaptive update-interval floor (`_MAX_UPDATE_INTERVAL_FACTOR = 10`), the Tiny cost model (102 ms fixed + 269 ms per second of audio), `TranscriptEventListener` callbacks, `stop()` completes active lines
- https://github.com/moonshine-ai/moonshine/blob/main/language-bindings/python/src/moonshine_voice/moonshine_api.py — `TranscriptLine.last_transcription_latency_ms`, `WordTiming(word,start,end,confidence)`, `TranscriptLine(text,start_time,duration,line_id,is_complete,...)`
- https://pypi.org/pypi/moonshine-voice/json — 0.1.5, 2026-08-24, MIT, `macosx_15_0_arm64` wheel **only** (no Linux wheel), API
- https://github.com/moonshine-ai/moonshine-v2 — licence nuance for non-English
- https://github.com/senstella/parakeet-mlx — `transcribe_stream`, `finalized_tokens`/`draft_tokens`
- https://pypi.org/pypi/parakeet-mlx/json — 0.5.2, 2026-06-05, deps
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 — 6.34% WER, CC-BY-4.0
- https://deepgram.com/pricing — $200 credit, no card, no expiration, Flux $0.0077/min (promo $0.0065), Aura-2 $0.030/1k chars
- https://developers.deepgram.com/docs/flux/feature-overview and /configuration — events, `eot_threshold` 0.7 (0.5–1.0), `eager_eot_threshold` unset (0.3–0.9), `eot_timeout_ms` 5000 (500–60000)
- https://deepgram.com/learn/coval-validates-flux-no-tradeoff-between-latency-and-interruption — Coval median EOT <300 ms (**third-party benchmark published by the vendor**)
- https://developers.deepgram.com/reference/api-rate-limits — concurrency
- https://www.assemblyai.com/pricing — $50 credit, rates, session-duration billing
- https://www.assemblyai.com/docs/faq/how-does-automatically-scaling-concurrency-for-streaming-stt-work — 5 new sessions/min
- https://opennmt.net/CTranslate2/hardware_support.html — **CPU + CUDA only, no Metal**
- https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/README.md — `--step 0` sliding+VAD
- https://github.com/ufal/whisper_streaming — 3.3 s headline latency
- https://github.com/argmaxinc/argmax-oss-swift — OSS vs Pro split
- https://developer.apple.com/documentation/speech/speechanalyzer — Swift-only, macOS 26
- https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b — GPU-only
- https://arxiv.org/html/2510.06961v1 — Open ASR Leaderboard; Whisper large-v3 7.44%, RTFx
- https://huggingface.co/datasets/hf-audio/esb-datasets-test-only — test splits; **`common_voice`, `gigaspeech`, `spgispeech` are gated** (verified 2026-09-13); the other five are open
- https://github.com/anvanvan/mac-whisper-speedtest — M4 comparison (secondary)
- https://docs.pipecat.ai/server/services/stt/groq — `SegmentedSTTService`, no interim results
- https://www.rev.com/pricing — human transcription **$1.99/minute**, 99%+ accuracy, 12 h turnaround

### TTS
- https://huggingface.co/hexgrad/Kokoro-82M — Apache-2.0, 82M, 54 voices
- https://github.com/hexgrad/kokoro — `KPipeline` generator; espeak-ng dependency
- https://api.github.com/repos/Blaizzy/mlx-audio — MIT, pushed 2026-09-11, `--stream`
- https://api.github.com/repos/thewh1teagle/kokoro-onnx/releases/tags/model-files-v1.0 — `kokoro-v1.0.onnx` 325,532,387 B; `fp16` 177,464,787 B; `int8` **92,361,271 B**; `voices-v1.0.bin` **28,214,398 B**
- https://contracollective.com/blog/kokoro-vs-piper-vs-xtts-local-text-to-speech-m5-max-2026 — M5 Max first-audio, RTF 0.08 (third-party blog)
- https://api.github.com/repos/OHF-Voice/piper1-gpl — GPL-3.0, pushed 2026-09-09
- https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md — "personal use and … research only"
- https://huggingface.co/coqui/XTTS-v2 — CPML; **`coqui.ai` returns 404**
- https://github.com/resemble-ai/chatterbox + https://www.resemble.ai/learn/models/chatterbox-turbo — MIT; GPU-only latency figures
- https://github.com/canopyai/Orpheus-TTS — Apache-2.0, Llama-3b backbone, vLLM/GPU
- https://elevenlabs.io/pricing, /terms-of-use, /docs/models — quota, §1(c) non-commercial, Flash v2.5 ~75 ms
- https://cartesia.ai/pricing + https://docs.cartesia.ai/api-reference/tts/tts — Pro for commercial; `max_buffer_delay_ms` default 3000
- https://www.rime.ai/pricing/ — **3,000 free minutes on Starter**
- https://developers.deepgram.com/reference/text-to-speech-api/speak-streaming — WS `encoding` ∈ {`linear16`,`mulaw`,`alaw`} (**no `opus`**); `sample_rate` ∈ {8000,16000,24000,32000,48000}; client `Speak`/`Flush`/`Clear`/`Close`; server `SpeakV1Audio`/`Metadata`/`Flushed`/`Cleared`/`Warning`
- https://deepgram.com/learn/introducing-aura-2-enterprise-text-to-speech — sub-200 ms TTFB claim
- https://deepgram.com/product/text-to-speech/flux — Flux TTS promo ended 2026-09-12/13
- https://api.github.com/repos/KittenML/KittenTTS — Apache-2.0, <25 MB
- https://raw.githubusercontent.com/rany2/edge-tts/master/LICENSE — LGPLv3

### Turn-taking, VAD, AEC, audio clocks
- https://pypi.org/project/silero-vad/ — 6.2.1, 2026-02-24, MIT; **core deps include `torch`/`torchaudio`; `onnxruntime` is the `onnx-cpu` extra**
- https://github.com/snakers4/silero-vad + `src/silero_vad/utils_vad.py` — <1 ms, 512-sample contract, `VADIterator` defaults, bundled `.onnx`
- https://pypi.org/project/webrtcvad-wheels/ — 2.0.14, macOS arm64
- https://huggingface.co/pipecat-ai/smart-turn-v3 and /tree/main — BSD-2, 8.68 MB v3.2-cpu
- https://github.com/pipecat-ai/smart-turn + `inference.py` — 16 kHz, 8 s, front-pad, p>0.5
- https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/ — 12.6 ms, 23 languages, pair with VAD
- https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/ — 94.7% / 95.6% English (**on its own eval set**)
- https://reference-server.pipecat.ai/en/stable/_modules/pipecat/audio/vad/vad_analyzer.html — VAD defaults
- https://reference-server.pipecat.ai/en/stable/_modules/pipecat/audio/turn/smart_turn/base_smart_turn.html — `stop_secs=3.0`
- https://reference-server.pipecat.ai/en/stable/_modules/pipecat/audio/turn/smart_turn/local_smart_turn_v3.html — numpy log-mel, no `transformers`
- https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-turn-strategies — `MinWordsUserTurnStartStrategy`
- https://raw.githubusercontent.com/livekit/agents/main/MODEL_LICENSE — **standalone use forbidden**
- https://livekit.com/blog/solving-end-of-turn-detection — 9.9% false-cutoff @300 ms
- https://python-sounddevice.readthedocs.io/en/0.5.2/api/streams.html — `stop()` vs `abort()`; `latency`
- https://docs.python.org/3/library/asyncio-task.html — `Task.cancel()` guarantee; `asyncio.timeout` and `.reschedule()` (3.11)
- https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints/echoCancellation — Baseline since Jan 2020
- https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/getOutputTimestamp — {contextTime, performanceTime}; performanceTime is when that frame reached the output device
- https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/outputLatency — device output delay
- https://developer.mozilla.org/en-US/docs/Web/API/BaseAudioContext/baseLatency — graph latency
- https://raw.githubusercontent.com/livekit/python-sdks/main/livekit-rtc/livekit/rtc/apm.py — APM API, 10 ms frames
- https://pypi.org/project/pywebrtc-audio/ — 0.2.0, Apache-2.0, macOS arm64
- https://github.com/nguyenvulebinh/browser-aec — Web Audio AEC failure demo (**contested**)
- https://www.pnas.org/doi/10.1073/pnas.0903616106 — Stivers et al. 2009, 0–200 ms modal turn gap

### Architecture, frameworks, deployment, evaluation
- https://pypi.org/pypi/pipecat-ai/json — 1.10.0, 2026-09-12, Python ≥3.11
- https://pypi.org/pypi/livekit-agents/json — 1.8.1, 2026-09-10
- https://pypi.org/pypi/vocode/json — **last stable 2024-06-17**; https://api.github.com/repos/vocodedev/vocode-core — last commit 2024-11-15
- https://docs.pipecat.ai/server/services/transport/small-webrtc — no API keys; TURN needed on macOS
- https://www.daily.co/blog/advice-on-building-voice-ai-in-june-2025/ — 800 ms budget split
- https://www.fullstackml.dev/p/15-where-does-the-time-go-measuring — 570 ms summed vs 1,048 ms wall-clock
- https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection — cost of an 800 ms threshold
- https://docs.livekit.io/agents/multimodality/audio/ — preemptive generation; issue #4219
- https://render.com/docs/compute-plans — Free = **0.1 CPU / 512 MB**; Starter = 0.5 CPU / 512 MB
- https://render.com/docs/free — 750 instance-hours/month, 15-minute idle spin-down, ~1 minute spin-up
- https://render.com/docs/outbound-bandwidth — **5 GB/month** on Hobby; $0.15/GB overage; **no payment method → all workspace services spun down until the start of the next month**
- https://render.com/changelog/free-web-services-now-remain-active-while-receiving-websocket-messages — **2026-02-24**
- https://render.com/docs/websocket — WS behaviour
- https://huggingface.co/docs/hub/en/spaces-overview and /spaces-zerogpu — paid plan required; ZeroGPU carve-out
- https://fly.io/docs/about/pricing/ — card required; https://docs.railway.com/pricing/plans — $1/mo
- https://modal.com/docs/guide/billing — payment method mandatory
- https://vercel.com/docs/functions/limitations — Hobby 300 s max duration
- https://developers.cloudflare.com/workers/platform/limits/ — 10 ms CPU/request
- https://jitsi.github.io/jiwer/ — jiwer 4.x API, `process_words` alignments
- https://github.com/kurianbenoy/whisper_normalizer — `EnglishTextNormalizer`
- https://github.com/pipecat-ai/stt-benchmark — TTFS + pooled WER methodology
- https://github.com/ExistentialAudio/BlackHole — loopback for acoustic ground truth
- https://docs.pipecat.ai/pipecat/fundamentals/metrics — TTFB / TTFA / TTFAT vocabulary

### Domain
- https://www.trustpilot.com/review/finalroundai.com — 2.9/5, 276 reviews
- https://grow.google/interview-warmup/ — tool absent (retired ~April 2026)
- https://github.com/IliaLarchenko/Interviewer — 127 stars
- https://www.isca-archive.org/interspeech_2006/yuan06_interspeech.pdf — 196 WPM, range 111–291
- https://heatherbortfeld.com/wp-content/uploads/2016/09/bortfeld_etal_ls2001.pdf — 5.97 disfluencies/100 words
- https://pubmed.ncbi.nlm.nih.gov/34968080/ — Sackett et al. 2022, structured interviews ρ=.42
- https://files.eric.ed.gov/fulltext/EJ1168380.pdf — **ETS RR-17-28, Kell et al. 2017. ICC = .74 across three raters, 12 questions; per dimension .66–.82; N = 652 written responses, 7-point scale.** Open-access copy; the Wiley DOI returns HTTP 403 to non-subscribers. **This does not contain the "0.77–0.84" figure earlier drafts attributed to it**
- Conway, Jako & Goodman (1995), meta-analytic interview reliability — **paywalled; abstract reports upper limits of *validity* (.67 structured / .34 unstructured), not interrater reliability.** Do not quote a reliability figure from it unread
- Fleiss & Cohen (1973) — quadratically-weighted kappa ≡ ICC under stated assumptions
- https://www.pnas.org/doi/10.1073/pnas.1915768117 — Koenecke et al. 2020, ASR racial disparity
- https://arxiv.org/html/2606.19544v1 — LLM-judge position and verbosity bias
- https://artificialintelligenceact.eu/implementation-timeline/ — Annex III now 2027-12-02
- https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page — LL144
- https://law.justia.com/codes/illinois/chapter-820/act-820-ilcs-42/ — AI Video Interview Act
- https://epic.org/hirevue-facing-ftc-complaint-from-epic-halts-use-of-facial-recognition/ — Jan 2021
- https://www.cooley.com/news/insight/2025/2025-02-21-gone-but-not-forgotten-federal-laws-still-apply-despite-guidance-disappearance-act — EEOC guidance removed
- https://github.com/SYSTRAN/faster-whisper/discussions/569 — Whisper strips fillers
- https://github.com/nyrahealth/CrisperWhisper — non-commercial weights

### §13.1 Durable snapshots

Durability has three tiers. Use the strongest one available for each source:

| Tier | Applies to | How |
|---|---|---|
| **1 — commit pin** | Anything on GitHub or Hugging Face | Cite a permalink at a commit SHA (press `y` on GitHub) or a HF revision hash, never `main`. Strictly better than any snapshot, and free. Applies to `livekit/agents` MODEL_LICENSE, `piper1-gpl` VOICES.md, Pipecat source links, smart-turn weights, `mlx-audio`, the Moonshine Python bindings |
| **2 — Wayback snapshot** | Vendor docs, pricing pages, live benchmarks | `web.archive.org/save/<url>`, snapshot URL printed **beside** the live one |
| **3 — committed copy** | Groq's docs, which are served as plain `.md` | `curl` into `docs/snapshots/`. Diffable, greppable, and survives Wayback refusing a JS-rendered console page |
| **— auth-walled** | `console.groq.com/settings/limits` | Cannot be archived. Commit a dated screenshot to `docs/snapshots/` and label it as one |

The pages below are load-bearing: each carries a number that a decision in §1–§9 rests on,
and each is vendor-controlled or a rolling benchmark. Everything else in §13 is either a stable
paper, a pinnable repo, or context.

| # | Page | Number the report takes from it | Why it rots |
|---|---|---|---|
| 1 | `console.groq.com/docs/rate-limits.md` | 30 RPM / 1K RPD / **8K TPM** / 200K TPD | The binding constraint of the entire architecture — and the page has already changed how it describes itself |
| 2 | `console.groq.com/docs/models.md` | Model list, context windows, Preview vs Production | 8 shutdowns in 12 months |
| 3 | `console.groq.com/docs/reasoning.md` | `reasoning_effort: "none"` on qwen only | The decisive reason in §1 |
| 4 | `console.groq.com/docs/deprecations.md` | Llama shutdown 2026-08-16 | Grows by definition |
| 5 | `console.groq.com/docs/prompt-caching` | gpt-oss only, 2 h, 50%, exempt from rate limits, 128–1024 min prefix | §3.1 and §9.4's token budget rest entirely on it |
| 6 | `console.groq.com/docs/structured-outputs.md` | Which models accept `strict:true` — and that qwen3.6 does not | The reason scoring runs on a different model than the hot path |
| 7 | `console.groq.com/docs/speech-to-text.md` | Batch-only, 20 RPM, 12% WER, 25 MB | The reason STT is local |
| 8 | `console.groq.com/docs/text-to-speech/orpheus` | 200 chars/request, 100 RPD | The reason TTS is local |
| 9 | `artificialanalysis.ai/providers/groq` | TTFT, time-to-first-answer-token, throughput | **A live rolling benchmark — the snapshot IS the citation.** Highest-value entry on this list |
| 10 | `render.com/docs/free` and `/docs/outbound-bandwidth` | 750 h/month, 15-min spin-down, **5 GB egress, spin-down with no card on file** | Free tiers change quietly, and this one changed since the previous draft |
| 11 | `render.com/changelog/free-web-services-now-remain-active-while-receiving-websocket-messages` | The 2026-02-24 entry | The single changelog line the whole deployment choice rests on |
| 12 | `pypi.org/pypi/moonshine-voice/json` | 0.1.5, MIT, `macosx_15_0_arm64` wheel only | The macOS 15.0 gate *and* the reason the hosted build is a relay; a Linux wheel would change both |
| 13 | `arxiv.org/html/2602.12241v1` | 6.65% WER, 258 ms on M3 | arXiv is durable, but `/html/` renders track the version — pin `v1` and snapshot |
| 14 | `deepgram.com/pricing` | $200 credit, no card, "No expiration", Flux and Aura-2 rates | §14 #14 already flags the expiry claim as contested, and the whole hosted burn rate depends on the rates |
| 15 | `www.rime.ai/pricing/` | 3,000 free minutes on Starter | The most generous and most volatile allowance in §5 |

```bash
#!/usr/bin/env bash
# docs/archive_sources.sh - snapshot the load-bearing pages, print citation lines.
# Run before publishing and again the week of the demo; diff docs/snapshots/.
set -uo pipefail
STAMP=$(date -u +%Y-%m-%d)
OUT="docs/snapshots"; mkdir -p "$OUT"

URLS=(
  "https://console.groq.com/docs/rate-limits.md"
  "https://console.groq.com/docs/models.md"
  "https://console.groq.com/docs/reasoning.md"
  "https://console.groq.com/docs/deprecations.md"
  "https://console.groq.com/docs/prompt-caching"
  "https://console.groq.com/docs/structured-outputs.md"
  "https://console.groq.com/docs/speech-to-text.md"
  "https://console.groq.com/docs/text-to-speech/orpheus"
  "https://artificialanalysis.ai/providers/groq"
  "https://render.com/docs/free"
  "https://render.com/docs/outbound-bandwidth"
  "https://render.com/changelog/free-web-services-now-remain-active-while-receiving-websocket-messages"
  "https://pypi.org/pypi/moonshine-voice/json"
  "https://arxiv.org/html/2602.12241v1"
  "https://deepgram.com/pricing"
  "https://www.rime.ai/pricing/"
)

echo "## Snapshots taken ${STAMP}"
for u in "${URLS[@]}"; do
  snap=$(curl -sL --max-time 180 -o /dev/null -w '%{url_effective}' "https://web.archive.org/save/${u}")
  case "$snap" in
    https://web.archive.org/web/*)
      echo "- ${u} — [archived ${STAMP}](${snap})" ;;
    *)
      echo "- ${u} — ⚠ ARCHIVE FAILED — commit a local copy and say so in §13" ;;
  esac
  sleep 20   # Save Page Now throttles anonymous callers; the exact anonymous
             # limit is not published. On HTTP 429, raise this and re-run.
done

# Tier 3: Groq serves its docs as plain markdown. Mirror them - diffable, and
# immune to Wayback refusing a JS-rendered console page.
for p in rate-limits models deprecations reasoning speech-to-text structured-outputs; do
  curl -sS --fail "https://console.groq.com/docs/${p}.md" \
       -o "${OUT}/groq-${p}-${STAMP}.md" || echo "⚠ groq/${p}.md fetch failed"
done

echo
echo "Not archivable (auth-walled): https://console.groq.com/settings/limits"
echo "  -> commit a dated screenshot to ${OUT}/groq-org-limits-${STAMP}.png"
```

**Citation format.** Each of those pages gets its snapshot beside it:

```markdown
- https://console.groq.com/docs/rate-limits.md — free-tier limits.
  [archived 2026-09-13](https://web.archive.org/web/20260913000000/https://console.groq.com/docs/rate-limits.md)
  · local copy `docs/snapshots/groq-rate-limits-2026-09-13.md`
```

The other ~90 sources above are live URLs verified 2026-09-13 and are not snapshotted. Stable
papers and pinnable repositories do not need it; the rest are context, not load-bearing. **If a
number in §1–§9 depends on a page, it is in the list above.**

---

## 14. Unverified claims

**Spot-check every item in this list before you design around it.** Nothing here should appear
as a fact in the README.

### Must be settled by your own measurement (no source can settle them)

1. **Time to first *speakable* token from Groq, from your machine, with your prompt.** The whole
   latency budget hangs on this. Artificial Analysis is a live rolling benchmark using its own
   prompts; its 2.44 s "time to first answer token" for qwen3.6 with reasoning off is hard to
   reconcile with its own 1.29 s TTFT for the same model. **Day-0 spike.**

   **1a. Moonshine v2 Medium Streaming's finalization latency on your Mac.** Three first-party
   figures for the same metric: **258 ms** (arXiv:2602.12241 Table 2, Apple M3, Python harness),
   **107 ms** (repo README cell, "MacBook Pro", native C++ harness), and **56–83 ms across three
   runs** (the repo's own benchmarks doc, describing the runs behind that cell). The two tables
   share their Whisper rows byte-for-byte while every Moonshine row differs, so they are different
   harnesses stitched together. The budget uses 258 ms. Log
   `TranscriptLine.last_transcription_latency_ms` from turn one.

   **1b. Whether `update_interval=0.25` actually shortens finalization on your Mac.** The floor
   adapts upward when a pass costs more than the interval, so asking for 0.25 is safe but may be
   ignored. Medium's per-pass cost constants are published nowhere — only Tiny's (102 ms fixed +
   269 ms per second of audio looked at). Run 20 turns at 0.5 and 20 at 0.25 and compare
   `stt_engine_latency_ms`.

   **1c. Whether `AudioContext.outputLatency` and `getOutputTimestamp()` are present and correct
   on the demo browser.** Chrome and Firefox expose both; Safari support was not verified. The
   loopback calibration in §10 is how you find out whether the mapping is right — a constant bias
   of roughly `output_latency_ms` in the A-vs-software comparison means it is double-counted or
   missing.

   **1d. The residual mic-side capture latency.** There is no `inputLatency` API. The
   AudioWorklet's `currentTime` stamp bounds everything above the hardware; what is underneath it
   is only reachable through Calibration B (§10), and it is the single term no software mark can
   ever see.
2. **Kokoro's actual time-to-first-audio on your specific Mac.** Circulating figures disagree by
   an order of magnitude (40 ms to 3,658 ms). The most-cited cross-engine benchmark is published
   by a vendor whose own engine wins, was run on an AMD Ryzen 7 5700X, and appears not to chunk
   by sentence.
3. **parakeet-mlx speed on Apple Silicon.** No published benchmark exists from any source — the
   repo cited for a "24× real-time" figure contains no benchmark numbers at all.
4. **Whether browser AEC cancels your TTS playback.** A demo repo shows it failing for Web Audio
   playback; Chrome's own blog describes an internal loopback implying it works. Behaviour varies
   by browser, OS, and hardware AEC path. Test on the exact demo machine. **This is the §10
   contingency with the largest exposure (+5 h).**
5. **Whether a sub-10-second clip consumes 10 audio-seconds against Groq's free ASH/ASD quota.**
   The 10-second minimum is documented only as a *billing* minimum. If quota accounting also
   rounds up, you get ~720 short clips/hour instead of the 1,200 that 20 RPM implies.
6. **Whether tokens generated before you close a Groq stream still count against TPM/TPD.** Not
   documented anywhere. Assume they do; frequent barge-in could burn quota faster than expected.
   This is also why §8.2 closes the HTTP body promptly via `aclosing` rather than letting it drain.
7. **Whether your Groq org gets a split ITPM/OTPM limit** instead of one 8,000 TPM number.
   `rate-limits.md` says some do. Check `console.groq.com/settings/limits` while logged in.
8. **Whether your Mac runs macOS 15.0+.** If not, there is no `moonshine-voice` wheel and the
   primary STT choice changes.

### Secondary sources only — not on an official page

9. **NVIDIA's free-tier credits: 1,000 on signup, +4,000 with a business email, 30-day expiry.**
   All from NVIDIA staff forum posts dated Sept 2024 and Jan 2025. Not restated in current docs.
10. **NVIDIA's per-call credit cost.** Published nowhere. ToS says only that deduction is "as
    stated with the relevant API Service"; no model page states a rate.
11. **NVIDIA's "Up to 40 rpm / 10,000 requests per day."** This *is* first-party — it ships in
    `content/copy.yaml`, server-rendered into build.nvidia.com pages — but NVIDIA hedges it
    ("Up to", "may vary by model", "traffic from other users may cause throttling"). It is a soft
    ceiling, not an SLA. **Nobody has tested whether it enforces as documented.**
12. **Whether `POST /v1/audio/synthesize_online` is exposed on the hosted NVCF invocation URL.**
    Documented for the self-hosted container; the hosted Magpie page shows only the
    non-streaming `/v1/audio/synthesize`. If not exposed, streaming TTS requires the gRPC client
    — a meaningfully larger lift. Day-1 test if you go that route.
13. **Render's "no credit card required."** Sourced from Render's own comparison *article*, not
    from `render.com/docs`. It is the weakest load-bearing claim in the deployment section — and
    it is now load-bearing in a **second** place: the 5 GB egress policy explicitly distinguishes
    workspaces *with* a payment method (billed $0.15/GB) from those *without* (every service in
    the workspace suspended until the 1st of next month). Confirm at signup which side you are on.
    **Do the signup on Day 0, before writing code** (§10 contingency).
14. **Deepgram credit expiry.** Deepgram's Terms say credits expire when the account or
    subscription ends, and that "certain Credits may also expire earlier as stated in the
    Pricing List." The widely-quoted "12 months from issue" figure does not appear on the Terms
    page. The pricing page states "No expiration" for the $200 signup credit.
15. **Apple Silicon speed figures for whisper.cpp / mlx-whisper / faster-whisper.** All from
    third-party blogs, mutually inconsistent, and in one case contradicted by another source in
    this same research set. Measure if you cite.
16. **Chatterbox Turbo on MPS.** The "RTX 4090" attribution and the "sub-150 ms first sound"
    figure appear on no official page; the repo's device options are `cuda` and `cpu` only. A
    reported "Cannot convert MPS Tensor to float64" crash and float32 patch could not be
    corroborated.
17. **The "~600–800 ms naturalness threshold"** widely cited in voice-AI blogs is
    marketing-adjacent, not peer-reviewed. Cite Stivers et al. (PNAS 2009) instead. This matters
    more now that the honest headline is ~1,230 ms: do not quote a threshold you cannot source in
    order to say you beat it, and do not quote one you can source in order to say you missed it.
18. **Google Interview Warmup's retirement date.** Confirmed by fetch that the tool is gone; no
    official shutdown notice found, so "around April 2026" is inferred.
19. **The claim that filler words measurably harm interview outcomes**, and the recruiting
    convention that "we" statements are a red flag versus "I" statements. Neither traces to a
    peer-reviewed hiring-outcome study. Do not put either in the README — and note that
    `behavioral_v1.yaml` encodes the second one as an explicit `not_scored` prohibition.
20. **The "~73% of employers use behavioral interviews" statistic.** Widely repeated; original
    survey untraceable. Do not cite.
21. **Moonshine's non-English licensing.** The `moonshine-voice` docs and the `moonshine-v2`
    README contradict each other. **English-only use is unambiguously MIT** — scope the project
    to English and the ambiguity disappears.
22. **"Word-level timestamps incur additional latency" on Groq STT.** This appears nowhere in
    Groq's speech-to-text docs. Treat as folklore.

### Added by the 2026-09-13 revision

Each is tagged with how it gets settled.

23. **Whether Groq's 8,000 TPM is a single org-wide pool or a per-model pool.** *(measure)*
    `rate-limits.md` states limits at organisation rather than user level but does not say
    whether two different model ids draw on one bucket or two, while presenting the limits in a
    per-model table. **The token arithmetic in §9.6 and §6.8 assumes one shared pool**, which is
    the conservative reading; a per-model table is evidence about presentation, not about
    accounting. If limits are per-model, running the hot path on qwen3.6 and scoring on
    gpt-oss-20b doubles the available headroom for free and the `HOT_PATH_RESERVE_TPM` floor
    becomes belt-and-braces. Check `console.groq.com/settings/limits` and confirm by deliberately
    saturating one model while watching the other's `x-ratelimit-remaining-tokens`.
24. **gpt-oss-20b's exact minimum cacheable prefix length, and the true size of the scoring
    prefix.** *(measure, one request)* Groq documents only that the minimum "varies by model,
    ranging from 128 to 1024 tokens". The §9.4 prefix is 1,100–1,260 tokens by two crude
    estimators — never tokenized. It clears 1,024 by 8–23%, which is not comfortable. Confirm both
    numbers by reading `usage.prompt_tokens` and `usage.prompt_tokens_details.cached_tokens` on
    scoring call 2 before relying on the TPM saving. **The whole §9.6 token budget assumes this
    cache hits.**
25. **Local LLM time to first token on your Mac.** *(measure — Day-0 spike, candidate (c))* The
    250–600 ms figure for a 4B-class 4-bit model at a ~1,800-token prompt is an inference, not a
    measurement; no published benchmark covers this configuration on Apple Silicon. The only
    public anchor is `kwindla/macos-local-voice-agents` reporting **<800 ms voice-to-voice** with
    `gemma-3n-e4b-it-text` in LM Studio plus Silero, MLX Whisper and Kokoro in the same loop,
    which bounds the LLM's share well under 800 ms by subtraction.
26. **Whether smart-turn v3.2 can reach ≤5% false-cutoff on interview-style thinking pauses at
    all.** *(measure — §8.3)* The 94.7% English accuracy figure is on the model's own evaluation
    set, not on 2–4 s mid-answer pauses. If the sweep cannot reach the bar at any (arm_ms, τ),
    that is a publishable negative result about the tool — report it, and fall back to a longer
    `stop_secs`. Do not quietly relax the bar.
27. **Resident memory for the four local models on Apple Silicon.** *(measure —
    `bench/resources.py`)* §6.5's RSS column is weights plus typical overhead, not a measurement.
    No source measures RSS for any of these four on this hardware. Fill the table before the
    README quotes the ~1.15 GB total — which is also the arithmetic behind "the hosted build must
    be a relay."
28. **Kokoro's chunk-level generator granularity from `mlx-audio` 0.5.3.** *(check, then measure)*
    Not documented. If it yields one array per sentence rather than per segment, the ~50 ms
    re-chunking in §5 is mandatory rather than an optimisation — and without it `tts_first_byte_ms`
    silently marks the last byte, which flatters the headline.
29. **That `silero_vad`'s wheel ships a `.onnx` alongside the `.jit`.** *(verify on install)*
    §6.5 asserts it in order to keep PyTorch off the shipped path. Verify with
    `python -c "import importlib.resources as r; print(list(r.files('silero_vad').rglob('*.onnx')))"`.
    The torch fallback is one line, plus two thread-count calls at the top of `app.py`.
30. **Whether Deepgram's TTS WebSocket accepts `encoding=mulaw` together with
    `sample_rate=24000`.** *(Day-1 test)* The parameter reference lists both values as valid
    independently and documents no matrix of accepted combinations; mu-law is conventionally an
    8 kHz telephony codec. The whole hosted egress budget halves on this. Fallbacks: `linear16` @
    24 kHz (double egress) or `mulaw` @ 8000 (telephone-grade audio).
31. **Whether Deepgram Flux bills on WebSocket connection duration or on audio bytes streamed.**
    *(secondary)* Assumed connection-duration here, by analogy with AssemblyAI (§4) and because it
    is the pessimistic case. If it bills on audio, browser VAD gating cuts the STT line of the burn
    rate by roughly 40% and the $200 pot goes further. It does not change the gating decision,
    which is driven by Groq's TPD.
32. **Whether Render appends to `X-Forwarded-For` or replaces it.** *(verify by logging the full
    header once on deploy)* The per-IP limiter takes the right-most entry on the assumption Render
    appends its observed client IP. If Render instead sets a single value, right-most and left-most
    coincide and the code is still correct; if it passes the client header through untouched,
    **the per-IP cap is trivially bypassable** and you must fall back to the raw peer address.
33. **The 1:3 Render-vCPU-to-M5-Max-performance-core ratio** used to derive Kokoro's RTF ≈ 2.4 on
    Render Free. *(estimate)* Not a measurement. The conclusion — local TTS is an order of
    magnitude too slow on a 10% CPU quota — holds across any plausible ratio, but do not print the
    2.4 as a measured figure.
34. **Hosted relay RSS (~70 MB base, ~3 MB/session).** *(measure)* Estimated from component sizes.
    Log `resource.getrusage(RUSAGE_SELF).ru_maxrss` on `/healthz` from the first deploy and replace
    the estimate with the real number before the README quotes it.

### Corrections applied to earlier drafts of this research (noted so you don't re-derive them)

**Latency and instrumentation**

- The §7 STT row previously read "30–120 ms (local streaming, already overlapped with speech)".
  That figure appears in no source and contradicts §1 and §4 of the same document. Moonshine's
  published metric is end-of-speech → final transcript and is fully on the critical path; the
  tuned total moved from **~900 ms to ~1,230 ms** as a result.
- The old ~900 ms plan figure was **135 ms below what its own table's row midpoints summed to**
  (~1,035 ms). That gap was never sourced to anything. The STT correction is only the second
  largest cause of the headline moving.
- The §7 "playback enqueue + device latency: 20–50 ms" row described a server-side `enqueue()`
  call that returns before anything is audible. It is replaced by an explicit WS-transit row, a
  jitter-buffer row and a device-output row, measured on the client clock.
- The 6.65% WER and the 258 ms latency both belong to **Medium Streaming (245M)**. Earlier drafts
  printed the WER without attribution beside a latency table listing three model sizes.
- Earlier drafts mixed a server-side event with a browser-side one in one phrase ("end of speech →
  first audible word") and measured both on the server. The three named terms `t_endpoint`,
  `t_server_v2v` and `t_v2v` (§7) exist so no claim can span undefined events again.
- `AudioContext.getOutputTimestamp()`'s mapping **already includes** device output latency; adding
  `ctx.outputLatency` on top is a ~25 ms double-count that flatters the number.
- The acoustic loopback calibration was listed as "optional, ~1 h". It is **required, ~1.5 h, two
  parts** — it is the only thing that can validate the clock offset, the `t0` anchor, and the
  mic-side hardware delay that no browser API exposes.

**Models and providers**

- `openai/gpt-oss-20b` was named in three places as the hot-path fallback while the same document
  proved it takes **3.55 s to its first answer token**. It is removed from the hot path entirely
  and reassigned to post-turn scoring, where latency is free and its prompt caching and strict
  `json_schema` support are genuine assets. The hot-path fallback is **local**.
- `qwen/qwen3.6-27b` is **absent from Groq's structured-outputs support list** in both strict and
  best-effort mode, so the scoring path cannot run on the hot-path model. That is why §3 has three
  decision lines rather than two.
- **`console.groq.com/docs/rate-limits.md` now carries the line "the limits shown below are the
  base limits for the Developer plan."** Earlier drafts stated that this page is the authoritative
  free-tier source; as of 2026-09-13 it labels itself as Developer. The 8,000 TPM / 200,000 TPD
  figures underpinning §3, §9.6, §11 #2 and §12 are therefore **unconfirmed for the free plan**,
  and `console.groq.com/settings/limits` on the actual account is now the only authority. All the
  arithmetic is stated as "against a published 8,000 TPM" precisely so it can be redone with one
  substitution. **Check this before writing the README.**
- `retry-after` is returned **only** on a 429; `x-ratelimit-remaining-tokens` and
  `x-ratelimit-reset-tokens` are on every response. Read them always, not just on failure.
- Groq's "SPEED (T/SEC)" table is on `/docs/models.md`, **not** on the optimizing-latency page.
- Groq has **three** audio endpoints (`transcriptions`, `translations`, `speech`), not two.
- `meta/llama-3.3-70b-instruct`'s NVIDIA page does **not** 404 — it carries a deprecation notice
  dated 08/25/2026.
- NVIDIA LLM streaming uses `"stream": true` in the **body** with `Accept: application/json` —
  **not** `Accept: text/event-stream`.
- NVIDIA ToU §5(e) ("publicly display, offer as a service") restricts redistributing *NVIDIA's*
  technology, not publishing your own app. The operative constraint is **ToS §1.4**
  (internal testing and evaluation only). Do not tell anyone you legally cannot show the project.
- Deepgram Flux's event values are `Update`, `StartOfTurn`, `EagerEndOfTurn`, **`TurnResumed`**,
  `EndOfTurn` — `TurnInfo` is the message envelope, not an event.
- Deepgram's TTS WebSocket supports only `linear16`, `mulaw` and `alaw`. **`opus` is REST-only**,
  so there is no cheap compressed downlink from Aura-2.
- AssemblyAI's free "5" is **new sessions per minute**, not concurrent sessions.
- Rime's free tier is **3,000 minutes** (~3 M characters), not 10,000 characters.
- ElevenLabs' free 10k credits buys ~20k characters on Flash v2.5 (50% lower per-char), not ~10k.
- No attribution/"credit elevenlabs.io" clause exists in the live Terms of Use, despite many
  blogs asserting one.

**Measurement and methodology**

- **The "0.77–0.84 human interrater reliability" figure is not in the cited source.** ETS RR-17-28
  (Kell et al. 2017) reports **ICC = .74** across three raters and 12 questions, per-dimension
  .66–.82, over 652 written responses on a 7-point scale. Use .74 / .66–.82, call it an ICC, and
  cite the open-access copy at `files.eric.ed.gov/fulltext/EJ1168380.pdf` rather than the
  paywalled Wiley page (which returns HTTP 403 to non-subscribers). Conway, Jako & Goodman (1995)
  is paywalled and its abstract reports upper limits of **validity** (.67 structured / .34
  unstructured), not interrater reliability — do not quote a reliability number from it unread.
- Cohen's κ should be **quadratically weighted** on a 1–5 anchored scale. Unweighted κ treats a
  one-level disagreement as identical to a four-level one and scores a realistically good judge
  near zero.
- **"WER delta against Groq Whisper" was not a valid measurement** and is renamed to *transcript
  disagreement rate and latency delta* (§4.2). Real WER is confined to the five ungated
  `hf-audio/esb-datasets-test-only` splits — `librispeech`, `voxpopuli`, `tedlium`, `earnings22`,
  `ami`; `common_voice`, `gigaspeech` and `spgispeech` are gated (verified 2026-09-13).
- **`bench/replay.py --runs 200` through the real pipeline is arithmetically impossible against
  the free tier** — 374,000 tokens against a 200,000 TPD cap. Split into a recorded-SSE stub mode
  at N=200 and a live-LLM mode at N=60/day.
- **Groq prompt caching has a minimum cacheable prefix of "128 to 1024 tokens depending on the
  specific model"**, and reports hits as `usage.prompt_tokens_details.cached_tokens`. Caching
  applies to `gpt-oss-20b`, `gpt-oss-120b` and `gpt-oss-safeguard-20b` only; **the hot-path model
  `qwen/qwen3.6-27b` has none**, so caching is a property of the scoring path, not of the
  recommended hot path.
- **Rev human transcription is $1.99/minute** (verified 2026-09-13), i.e. ~$124 for 62 minutes of
  in-domain audio — outside a $0 budget, which is why in-domain references are self-produced and
  scoped to 20 answers.

**Deployment and runtime**

- **Render's free workspace includes 5 GB/month outbound bandwidth, not 100 GB**, and with no
  payment method on file Render **spins down every service in the workspace until the 1st of next
  month**. That is why the hosted downlink is mu-law.
- `moonshine-voice` publishes **no Linux wheel at all** — the hosted build cannot run the local
  stack even in principle, which is a stronger argument than the CPU and memory ones.
- `silero-vad` 6.2.1 hard-depends on `torch`, so the project loads its bundled `.onnx` into its own
  ORT session rather than importing the package. PyTorch is off the shipped path entirely.
- **`asyncio.to_thread` must never be used for inference**: its default executor is
  `min(32, os.cpu_count() + 4)` = 12 threads on an 8-core Mac, and twelve concurrent ONNX sessions
  on eight cores turn a 12 ms classifier into a 200 ms one.
- The `Kokoro → player` format is **24 kHz mono int16 LE in ~50 ms chunks**, not float32. A
  per-sentence generator makes "first byte" and "last byte" the same instant.
- `player.abort()` must be called **before** awaiting the cancelled task, not after. Aborting after
  the await plays audio through the whole cancellation round-trip and makes the audible-stop mark
  measure the wrong thing.
- Earlier drafts estimated the whole project at **45–55 h over 2–3 weeks**. With the degradation
  ladder, the endpointer evaluation, the hosted build, the acoustic calibration and a test suite,
  the planned figure is **81–95 h plus 8 h contingency**. The 2–3 week framing does not survive;
  §10's cut list is how you get back to a shippable scope.
