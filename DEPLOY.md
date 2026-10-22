# Deploying

Two options, both free. **Netlify is not one of them** — it does not host or proxy persistent
WebSocket connections, and its own guidance is to run the WebSocket server elsewhere. This app
is a WebSocket server, so there is nothing for Netlify to host.

| | Vercel (Hobby) | Render (Free) |
|---|---|---|
| WebSockets | Yes — **public beta**, shipped June 2026 | Yes, generally available |
| Session length | **Capped at 5 min** by function duration | Unlimited |
| First visit after idle | Fast | **~50 s** while the container wakes |
| Spend limits | Weaker — counters are per-instance | Exact — one long-running process |
| Config in this repo | `vercel.json` + `pyproject.toml` | `render.yaml` |

**Pick Vercel if** the demo matters more than the limits — a recruiter clicking a cold Render
link waits most of a minute, and that is the first impression.

**Pick Render if** you want sessions to never be cut off and the spend caps to mean exactly what
they say.

---

## Vercel

Vercel auto-detects FastAPI from `requirements.txt` and loads the app named in
`pyproject.toml` under `[tool.vercel]`. No code changes needed.

```bash
npm i -g vercel
cd ~/Documents/voice-interview-coach
vercel          # first run links the project
vercel --prod
```

Or through the website: **vercel.com → Add New → Project → import the GitHub repo → Deploy**.

### Environment variables to set

In the Vercel dashboard under **Settings → Environment Variables**:

| Variable | Value | Why |
|---|---|---|
| `GROQ_API_KEY` | from your `.env` | |
| `NVIDIA_API_KEY` | from your `.env` | |
| `COACH_MAX_SESSION_SECONDS` | `240` | **Important.** Hobby kills a function at 300 s. Ending at 240 s means the app closes the session politely instead of the connection dying mid-sentence |
| `COACH_DAILY_SESSIONS` | `15` | Lower than Render's 40, because the counter is per-instance and several instances may run |
| `COACH_MAX_CONCURRENT` | `2` | |
| `COACH_PLAYBACK_HZ` | `22050` | Halves outbound audio |

### Things to know

- **WebSocket support is in public beta.** It works, but beta means it can change. For a link
  that has to keep working for months unattended, that is a real risk — check it occasionally.
- **The 5-minute cap is hard on Hobby** and cannot be raised. A 3-question interview with
  follow-ups can exceed it, which is why the session cap is set below it.
- **Spend limits are weaker here.** `limits.py` counts per process; serverless runs many. The
  real daily ceiling is the budget times however many instances spin up.

---

## Render

Config is already committed in `render.yaml`.

1. **render.com** → sign up with GitHub
2. **New → Blueprint** → pick `voice-interview-coach` → it reads `render.yaml`
3. Paste `GROQ_API_KEY` and `NVIDIA_API_KEY` when prompted
4. **Apply**. First build takes 3–5 minutes

Defaults in `render.yaml` are already right: 40 sessions/day, 2 concurrent, 7-minute cap.

### Things to know

- **Free services sleep when idle.** The first visitor after a quiet spell waits ~50 s. Put the
  recorded demo above the live link in your README so no first impression depends on it
- **5 GB/month outbound.** Exceed it with no card on file and Render suspends the service until
  the 1st of the next month. At 22.05 kHz that is roughly 2,000 sessions

---

## Either way: check it worked

```bash
curl https://<your-url>/healthz
```

Expect `{"ok": true, "playback_rate_hz": 22050, "limits": {...}}`. Then open the URL and press
Start — microphone access needs HTTPS, which both platforms provide.

## Spend limits

Your keys sit behind a public URL, so the app refuses work rather than draining your quota.
Groq allows 1,000 requests/day and a session uses roughly 20.

| Variable | What it does |
|---|---|
| `COACH_DAILY_SESSIONS` | Sessions per day, then a polite refusal until tomorrow |
| `COACH_MAX_CONCURRENT` | Simultaneous conversations |
| `COACH_MAX_SESSION_SECONDS` | Hard stop, so an abandoned tab cannot hold a slot |
| `COACH_PLAYBACK_HZ` | Audio rate out |

A refused visitor sees a plain message pointing at the README and the recorded demo.

## If the link dies

1. **Daily budget** — resets at midnight UTC
2. **Render bandwidth cap** — suspended until the 1st; lower `COACH_DAILY_SESSIONS`
3. **Groq retired a model** — the app resolves models at startup and logs the substitution; it
   should recover on its own
4. **NVIDIA credits exhausted** — the logs will say so

`python scripts/check_providers.py` locally tells you which.
