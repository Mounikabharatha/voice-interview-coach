# Recording the demo

The demo video is the single highest-value thing in this repository. Most people who open it
will watch that and read the first screen of the README, and nothing else. Budget an hour.

**Wear headphones.** Not optional. Without them the coach's voice goes back into the microphone
and can trigger barge-in mid-demo.

---

## What the video has to show

Three things, in this order. Each one is a claim the README makes, and the video is the
evidence.

1. **Words appear while you are still speaking.** Proves the pipeline streams.
2. **A pause mid-answer does not cut you off**, and a vague answer gets probed. This is the
   hard part of the project and the thing nobody else's voice-bot demo shows.
3. **You interrupt it mid-sentence and it stops instantly.**

Then the feedback report at the end.

Target: **45 to 70 seconds.** Shorter than a minute is hard; longer and people stop watching.

---

## Setup

```bash
cd ~/Documents/voice-interview-coach
python -m coach
```

Wait for `ready on http://127.0.0.1:8000` in the terminal — that line means prewarming has
finished. **Do not start recording before it appears**, or the first reply pays the cold-start
cost and the demo looks a second slower than the project actually is.

Open <http://localhost:8000>. Make the browser window about 1280×800 — big enough to read, small
enough that the text is not tiny when scaled down.

---

## Recording on macOS

Built in, no install:

1. **Cmd + Shift + 5**
2. Choose **Record Selected Portion**, drag around the browser window
3. Under **Options**, set **Microphone → MacBook Pro Microphone**. This is the step people
   forget, and a silent demo of a voice app is worthless
4. Click **Record**

Stop from the menu bar. It saves to the Desktop as `.mov`.

> Screen recording records system audio poorly on macOS. With headphones on, the microphone
> captures your voice but **not the coach's replies**. If the coach is inaudible in your
> recording, use [OBS](https://obsproject.com) (free) with a Desktop Audio source, or add
> subtitles for the coach's lines — the on-screen transcript already shows them.

---

## The script

Read this through once before recording. Do not read it aloud word for word — it should sound
like someone actually being interviewed.

**[Press Start. The coach greets you and asks its first question.]**

**You:**
> "I led the migration of our billing system to a new payments provider…"

**[Now stop. Look away. Count two full seconds.]** ← *this is the money shot*

> "…and it took about four months, with three teams involved. We cut failed payments by about
> twelve percent in the first quarter."

**[Coach acknowledges and asks the next question.]**

**You** — answer this one deliberately vaguely:
> "We worked on it as a team and it went pretty well in the end."

**[The coach will probe — something like "What was the measurable outcome?"]**

**You** — start answering, then cut across it while it is still talking:
> "Actually, sorry — let me redo that answer."

**[It stops mid-word.]**

Then either answer a final question, or stop the recording and cut to the report screen.

---

## The three moments to make sure you caught

Watch it back and check:

- [ ] Text visibly appears **while** you are still speaking
- [ ] The two-second pause did **not** trigger a reply
- [ ] The coach **stopped mid-word** when you talked over it
- [ ] The `reply ___ ms` counter in the top right is readable
- [ ] The feedback report is on screen at the end

If the pause did trigger a reply, pause a bit longer next take — the threshold is 900 ms after a
sentence that sounds finished.

---

## Turning it into a GIF for the README

A GIF autoplays in the README; a video file does not. Keep it under about 10 MB.

```bash
brew install ffmpeg gifski

ffmpeg -i ~/Desktop/demo.mov -vf "fps=12,scale=900:-1:flags=lanczos" -f yuv4mpegpipe - \
  | gifski -o docs/demo.gif -
```

If it comes out too large, drop to `fps=10` or `scale=760`.

Commit it and put it at the top of the README:

```markdown
![Demo](docs/demo.gif)
```

A GIF has no sound, so also upload the `.mov` — drag it into a GitHub issue or release and use
the URL GitHub gives back — and link it underneath:

```markdown
[Watch with audio](https://github.com/.../demo.mov)
```

---

## If you want a second take

Restart the server between takes. The session picks its three questions at random, so a fresh
run gives you different questions, and the conversation history starts clean.
