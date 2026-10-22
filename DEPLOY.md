# Deploying

Five minutes on Render's free plan. You need a Render account; everything else is committed.

## Before you start — what the free plan actually gives you

| | |
|---|---|
| WebSockets | Supported, and a service stays awake while receiving them, so a session in progress is never cut off |
| Cold start | **A service sleeps after inactivity.** The first visitor after a quiet spell waits ~50 seconds for it to wake |
| Bandwidth | 5 GB/month outbound. Exceed it with no card on file and Render **suspends the service until the 1st of next month** |
| Cost | £0, no card required |

The cold start is the one to know about. A recruiter clicking a cold link sees a blank page for
most of a minute. Put the **demo video first** in your README and the live link second, so
nobody's first impression depends on a container waking up.

## Steps

1. **render.com** → sign up with GitHub
2. **New → Blueprint**
3. Pick `Mounikabharatha/voice-interview-coach`. Render reads `render.yaml` and fills in the rest
4. It will ask for two secrets. Paste the same values that are in your local `.env`:
   - `GROQ_API_KEY`
   - `NVIDIA_API_KEY`
5. **Apply**. First build takes 3–5 minutes

Your URL will be `https://voice-interview-coach.onrender.com` (Render may add a suffix if the
name is taken).

## Check it worked

```bash
curl https://<your-url>/healthz
```

Expect `{"ok": true, ...}` with the current limits. Then open the URL in a browser and press
Start — HTTPS is required for microphone access and Render provides it automatically.

## Spend limits

Your API keys sit behind a public URL, so the app refuses work rather than draining your quota.
All four are environment variables in the Render dashboard; changing them takes effect on the
next deploy and needs no code change.

| Variable | Default | What it does |
|---|---|---|
| `COACH_DAILY_SESSIONS` | 40 | Sessions per day, then a polite refusal until tomorrow |
| `COACH_MAX_CONCURRENT` | 2 | Simultaneous conversations |
| `COACH_MAX_SESSION_SECONDS` | 420 | Hard stop per session, so an abandoned tab cannot hold a slot |
| `COACH_PLAYBACK_HZ` | 22050 | Audio rate out. Halving it doubles the sessions that fit in 5 GB |

Groq's free tier is 1,000 requests/day and a session uses roughly 20, so 40 sessions/day leaves
comfortable headroom for your own testing. Raise it only if you are watching the quota.

A refused visitor sees a plain message pointing at the README and the recorded demo — not an
error, and not a wall of stack trace.

## Watching it

- **Logs** — Render dashboard → your service → Logs
- **Usage** — `curl https://<your-url>/healthz` shows sessions used today
- **Bandwidth** — Render dashboard → service → Metrics. Watch this one; hitting 5 GB suspends
  the service until the 1st

## If the link dies

In order of likelihood:

1. **Bandwidth cap hit** — suspended until the 1st of next month. Lower `COACH_DAILY_SESSIONS`
2. **Daily session budget** — resets at midnight UTC
3. **Groq retired a model** — the app resolves models at startup and logs a warning naming the
   substitution, so check the logs. It should recover on its own
4. **Free credits exhausted** on NVIDIA — the logs will say so

Run `python scripts/check_providers.py` locally to see which of these it is.
