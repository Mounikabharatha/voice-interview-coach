# Open defects — voice-interview-coach-research.md

Audit of 2026-09-13. Three blocking defects (1, 2, 3) and the last-mile range (15) were fixed
by hand after the agent quota ran out. The items below are still open. Each is a precision fix,
not a design change — the underlying stack decisions are sound and settled.

## Closed

- **[1] Latency anchor** — `_derive()` now adds `pre_t0_ms` so the tooling produces the acoustic
  ~1,230 ms it claims; Calibration A's ±25 ms bar is now centred on zero and passable.
- **[2] Hot path crashed on turn one** — `stream_chat` is now awaited; `DeltaStream`, `_Prefixed`
  and `FakeLLM` expose `aclose()` so `contextlib.aclosing()` works. §8.2 detail 7 rewritten.
- **[3] False privacy claim** — `turn_log.write()` now splits rows by value type: numbers and an
  allow-listed set of identifier strings go to the committed `runs/*.jsonl`; every other string
  goes to a gitignored `runs/*.text.jsonl`. §6.10 item 4 rewritten to match the code.
- **[15] Last-mile range** — corrected to 31–103 ms (plan 67), matching budget rows 8–10.

## Open

### [4] MEDIUM — §7 optimisation #8 / 'get under 1 s' table / §12 answer #12

'20 of ~34 utterances per session' is wrong twice, and §12 states it in a form that self-refutes: 'twenty of the roughly thirty-four utterances in a session cost zero tokens, because the intro, all five questions and all five move-on lines are pre-rendered' — that enumerates 11, not 20. A default 5-question session has ~20 spoken utterances total (1 intro + 5 questions + 5 move-ons + ~8 probes + 1 wrap), of which 11 are pre-rendered. '34' is the Groq CALL count from §9.6 (8 probes + 25 scoring + 1 wrap), and 25 of those produce no speech at all. The correct claim is '11 of ~20 utterances'. This appears in three places and is exactly the kind of arithmetic a hiring manager checks.

### [5] MEDIUM — §9.6 / §9.1 / §7 #5,#8 / §6.5 lifespan / §6.3 / §11.1

The pre-rendered clip count does not reconcile. §9.1 and §7 optimisation #8 enumerate intro + 14 questions + 5 move-on lines + rung-1 filler + 429 fallback = 22, then both state '21 clips'. §6.5's lifespan comment and §9.6's section heading also say 21. §6.3's config/prerendered/ lists 'intro, 14 questions, 5 move-on lines, filler, 6 probes' = 27, adding six probe WAVs that appear in no other list of the startup set. §11.1's rung 3 adds a further pre-rendered line ('I've lost the language model...') counted nowhere. §9.6 itself specifies two failure fallbacks, not one.

### [6] MEDIUM — §1 / §2 / §7 optimisation #6 / §10 Milestone 2 vs §7 budget table

Barge-in is claimed as 'sub-60 ms' (§1), 'stops it inside 60 ms' (§2), 'Barge-in 300 ms → ~50 ms' (§7 #6) and 'stops within ~50 ms' (§10 M2). §7's own budget gives two separate rows: decision → abort() = 20–60 ms, plan 40; and decision → LAST AUDIBLE SAMPLE = 50–160 ms, plan 105 (LOCAL), and instructs 'report it separately'. The 40–60 ms figure is the device abort, not the audible stop the user experiences. The document is scrupulous about exactly this distinction for t_v2v and abandons it for the barge-in claim — the one number §9.7 #5 sells as a differentiator.

### [7] MEDIUM — §3.1 prompt layout vs §9.3 build_probe_messages vs §9.6 token arithmetic

§3.1 mandates R1 STATIC = 'system prompt + the full rubric YAML verbatim + the question bank', 'byte-identical for the life of the process', target ≥1,200 tokens, and says the layout 'costs nothing on qwen... so it is built regardless'. But §9.3's build_probe_messages sends only COACH_SYSTEM_PROMPT, and §9.6 books '420 system' tokens per probe. Building R1 as §3.1 specifies would add ~800–1,200 UNCACHED tokens to every hot-path call on a model the document repeatedly says has no caching — pushing a probe turn from ~1,585 to ~2,400 tokens. That breaks the 28,050-tokens/session figure, the 7-sessions/day figure, the 4,235-token worst minute, and therefore the global-concurrency-of-1 decision. 'Costs nothing on qwen' is false as stated.

### [8] MEDIUM — §6.8 hosted token block / §9.6 vs session.py

Both §6.8 and §9.6 describe the hosted session as 'one question, up to three probes', and §6.8's token arithmetic charges 3 × ~1,585 = 4,755 on that basis to reach ~9,525 tokens/session. But session.py sets MAX_PROBES_PER_QUESTION = 2, and no hosted override is specified anywhere. With the stated cap the hosted session is ~7,940 tokens, which moves '200,000 TPD ÷ ~9,500 ≈ 21 hosted sessions' to ~25 and undermines the derivation of the 5-sessions/day public ceiling. Separately, that ceiling's stated derivation ('reserve ~60%' of 21) yields 8.4, not 5.

### [9] MEDIUM — §6.5 lifespan vs §8.3 / §6.3 / §8.1 declared interfaces

The warm-up calls methods that no declared Protocol contains, with mismatched signatures. `smart_turn.infer(np.zeros(128_000, 'f4'))` vs the interface §6.3 and §8.3 both declare as `SmartTurn.p_complete(window: np.ndarray[int16]) -> float` — different method name AND different dtype (float32 vs int16; bench/endpoint.py passes an int16 ring). Likewise `stt.warmup()` is absent from §8.1's STTProvider Protocol, and `local_llm.warm()` / `health()` are absent from §6.3's llm/base.py Protocol. §8.3 makes a point of 'two interface obligations this creates, both trivial and both already declared in §6.3' — one of them is contradicted 2,000 lines earlier.

### [10] MEDIUM — §6.2 wire protocol v1 vs §6.7 clock.py vs §11.1 ladder.py

§6.2 states the contract for every JSON control frame in both directions: {"v":1, "t":<type>, "seq":<uint32>, "ts":<int ms>, ...}, with 'v != 1 is fatal'. Two reference implementations violate it. metrics/clock.py emits {"type":"clock_ping"} / {"type":"clock_pong"} / {"type":"clock_offset"} with no v/t/seq/ts envelope, and llm/ladder.py::_frame builds {"type": "provider_switch", ...} likewise — while §11.1's own hand-written example of that same frame correctly uses {"v":1, "t":"provider_switch", "seq":87, "ts":8321}. The document contradicts its own protocol in the two places it implements it.

### [11] MEDIUM — §6.8 build table / §7 LOCAL column vs §6.7 mark table / §8.2 AudioPlayer

The LOCAL build's playback topology is undecided and the wrong player is the one specified in full. §6.8's build table puts browser playback in BOTH builds (AEC row: 'Identical'), and §7's LOCAL column charges rows 8–10 = server→browser WS transit + browser jitter buffer + AudioContext.outputLatency. But §6.7's mark table says audio_first_frame_out_ms is 'LOCAL build only... t_v2v's endpoint when audio plays on the server', and §8.2's spine table says '#8 audio_first_frame_out_ms (LOCAL) / client_first_audio_played_ms (browser)' — treating LOCAL/HOSTED and server-playback/browser-playback as the same axis when they are not. §8.2 then gives ~60 lines of LocalAudioPlayer (sounddevice) and one paragraph for WebSocketAudioPlayer, which is the one the demo topology actually needs. A builder must invent this decision, and the ≈1,230 ms figure only holds for one of the two answers.

### [12] MEDIUM — §11.1 rung 2 / §10 M2 / §9.6 failure fallbacks

Rung 2 specifies 'pre-rendered audio for the 6 generic competency probes from §9.6's table' and §10 M2 budgets an hour to 'generate the 6 generic probe WAVs'. §9.6's table has 5 competencies and 14 templates; there is no set of 6, and the example line quoted for rung 2 ('What was the measurable outcome?') is not any of the 14. A builder cannot know which six lines to record. Related: §9.6 specifies one fallback clip played 'whenever a Groq call 429s or times out on the hot path' and a second 'covering the rung-1 switch' — but §11.1 makes a 429 or first-token timeout BE the rung-1 switch, so the first clip has no reachable trigger.

### [13] MEDIUM — §9.5 score_response.schema.json

The strict json_schema uses keywords that OpenAI-compatible Structured Outputs in strict mode characteristically reject: minimum/maximum on `level`, maxLength on `evidence_text` (400) and `rationale` (220). Groq's structured-outputs implementation is OpenAI-compatible, so this schema plausibly 400s at request time — which would take out the entire scoring path and falsify §9.4's claim that 'strict schema means no JSON-repair code path and no malformed-output retries'. §14 flags the caching-prefix length and the mulaw/24000 combination as things to verify but says nothing about supported schema keywords, which is the same class of risk with a larger blast radius.

### [14] MEDIUM — §6.5 resource table / §12 answer #11

The resource table's 'Peak total ≈ 1.15 GB' and §12's 'the resource table says the local stack is about 1.15 GB' exclude the rung-1 local LLM — but §6.5's own lifespan calls `await local_llm.warm()` at startup specifically to keep 'rung 1 weights resident before it is ever needed', and §11.1 puts that at ~4 GB resident. Real machine footprint at peak is ~5.1 GB, not 1.15 GB. The table is the stated arithmetic behind the hosted-build decision and is quoted verbatim in an interview answer.

### [16] LOW — §9.6 session shape table

'Target duration: 16 min' contradicts the arithmetic printed three lines below it ('Five blocks ≈ 15 min, plus a 40 s intro and a 90 s wrap ≈ 17 min') and the rest of the document, which says '17-minute session' in §3, §3.1, §6.8, §9.6's own token table (28,050/17 = 1,650 tok/min) and §12 #12. Separately, the INTRO clip is specified as '~55 words, ~20 s' but the duration arithmetic budgets 40 s for it, and the quoted intro text is ~70 words.

### [17] LOW — §12 answer #11

'On average a session runs at about 1,650 tokens a minute, so two fit comfortably and the third starts 429ing' is arithmetically false by the document's own figures: 3 × 1,650 = 4,950 against an 8,000 TPM cap, and four fit. The real argument for global concurrency 1 — which the document makes correctly in §6.8, §6.10 and §11 #2 — is the 4,235-token PEAK minute, not the average. As written this is a checkable error inside a prepared interview answer.

### [18] LOW — §6.8 burn-rate table vs §6.8 egress bullet vs §11 #14

Egress arithmetic is inconsistent. The cost table books '~0.57 MB/min (mulaw downlink + control frames)', which is 3.4 MB per 6-minute session and 5 GB ÷ 3.4 MB ≈ 1,500 sessions/month. Both the bullet below it and §11 #14 use '2.6 MB per six-minute session ≈ 1,970 sessions/month' — a figure that corresponds to 0.43 MB/min, i.e. the audio without the control frames the 0.57 figure explicitly includes.

### [19] LOW — §5 TTS decision

'the 3,600 TPD cap probably binds sooner' (Groq Orpheus) is an unsourced number appearing exactly once. §5's own limits column for that model lists 10 RPM / 100 RPD / 200 chars / WAV only — no TPD. It is not in §13's sources or §14's unverified list, which is precisely the kind of orphan figure §14 exists to catch.

### [20] LOW — §1 stack table vs §14 #2

§1 states Kokoro is 'fast on Apple Silicon (~90 ms to first audio measured on an M5 Max)' and §6.8 says 'Kokoro's measured RTF is 0.08 on an M5 Max' — 'measured' in both cases. §5 labels the same figure '(M5 Max, third-party)', §13 sources it to a single blog post, and §14 #2 says circulating Kokoro TTFA figures 'disagree by an order of magnitude (40 ms to 3,658 ms)'. §14's own opening rule is 'Nothing here should appear as a fact in the README'; the executive summary breaks it for the number the entire TTS decision rests on.

### [21] LOW — §7 row 2 vs §8.3 sweep

Row 2 of the budget fixes the VAD arming silence at 200 ms and the headline depends on it, but §8.3 sweeps arm_ms over (150, 200, 300, 400), says 'Do not ship these numbers as chosen', and explicitly instructs 'Leave those cells empty until the sweep runs' — while pre-filling 200 in the README template row. Nothing states what happens to the ≈1,230 ms headline if the sweep selects 300 or 400 ms, which would move it by 100–200 ms. Relatedly, `stop_secs` denotes 0.2 s in §7 row 2 and 3.0 s in §8's state machine and the §6 data-flow diagram (two different Pipecat parameters sharing a name), with no disambiguation at the point of use.

### [22] LOW — §4.2 vs §1/§4 WER claims

The headline 6.65% WER is an eight-dataset Open ASR average, but §4.2 establishes that only five of the eight splits are ungated and scopes bench/wer.py to those five. The document never says that the project's own pooled five-split number will not be comparable to the published 6.65%, nor to Whisper large-v3's 7.44% from the same leaderboard — leaving the 'Moonshine beats Whisper large-v3' line resting on numbers the project cannot itself reproduce.

### [23] LOW — §6.9 sessions.py

reconcile_playback(played_through, frames_per_sentence) takes a frames_per_sentence map that is never populated or defined anywhere — SessionState carries only pending_spoken: dict[int, str] (utt_id → text). The resume path that §11 #15 calls out as closing a hallucination-shaped bug is missing its denominator.
