# Recording the demo

The video is the highest-value thing in this repository. Most people who open it will watch
that and read the first screen, and nothing else. Budget an hour.

---

## The problem to solve first: capturing the coach's voice

macOS screen recording captures **your microphone**, not what comes out of your speakers. So a
plain `Cmd+Shift+5` recording has your voice and silence where the coach replies — in a video
about a talking app.

And you cannot simply use speakers instead of headphones, because then the microphone hears the
coach too, the speech recogniser transcribes it, and it pollutes your answer.

**Use Loom's Chrome extension.** It records the browser tab *including tab audio*, plus your
microphone, in one file. Free, no audio routing to set up, and the app already runs in Chrome.

1. Install the **Loom** extension from the Chrome Web Store, sign in free
2. Click the Loom icon → **Screen + Cam** or **Screen only**
3. Recording type: **Current Tab**
4. Turn **ON**: microphone, and **"Share tab audio"** (this is the setting that matters)
5. Record

Loom's free plan caps videos at 5 minutes, which is far more than you need.

> **Alternative without an account:** install [BlackHole 2ch](https://existential.audio/blackhole/)
> (free), create an Aggregate Device in Audio MIDI Setup combining your mic and BlackHole, and
> record that with QuickTime. More setup, no sign-up.

---

## Before you hit record

- [ ] **Headphones in** — keeps the coach's voice out of the microphone
- [ ] Open <https://voice-interview-coach.onrender.com> and **wait for the page to load fully**
      (if it has been idle it takes ~50 s to wake; do not record that)
- [ ] Press Start once and say a word to confirm the mic works, then **reload** for a clean start
- [ ] "Allow interrupting" left **unchecked**
- [ ] Browser window about **1280×800** — big enough to read when scaled down
- [ ] Close other tabs — the tab bar is in shot
- [ ] Your answers open on your **phone**, not on screen

---

## What the video must show

Three moments, in this order. Each is a claim the README makes.

| Moment | Why it matters |
|---|---|
| Words appearing **while you speak** | Proves it streams rather than waiting |
| **A 2-second pause mid-answer** that does not cut you off | The hard part, and nobody else's demo shows it |
| The **feedback report** at the end | Proves it is a product, not a chatbot |

Target **60–90 seconds**.

---

## The script

**1.** Press **Start**, allow the microphone. The coach asks a question.

**2.** Answer with **ANSWER A**, pausing where marked:

> I built a real-time voice interview coach. You speak an answer out loud and it listens while
> you talk, replies out loud and scores you at the end. **⏸ (stop, count one-two)** I chose the
> providers, wrote the pipeline, and measured every stage. I cut the reply time from about
> three seconds to one and a half seconds, and I learned to test with a real microphone much
> earlier.

> If the question does not fit, open with: *"The clearest example is a project I built recently."*

**3.** It asks the next question. Answer deliberately vaguely and **stop**:

> We worked on it as a team and it went pretty well in the end.

It will come back with **"What was the measurable outcome?"** — that is the moment worth having
on film.

**4.** Answer properly with **ANSWER C** from `demo-answers.md`, then **ANSWER D** for the last
question. The report appears and the session **stops by itself**.

**5.** Let the report sit on screen for 3 seconds, then stop recording.

---

## Compress it

GitHub accepts up to **10 MB** per file in a README. A 90-second recording is usually 30–60 MB.

```bash
brew install ffmpeg
```

```bash
ffmpeg -i demo.mp4 -vf "scale=1280:-2" -c:v libx264 -crf 30 -preset slow -c:a aac -b:a 96k demo-small.mp4
ls -lh demo-small.mp4
```

Still over 10 MB? Raise `-crf 30` to `34`, or drop `scale=1280` to `scale=1024`.

---

## Put it in the README

1. Open the repo on **github.com**
2. Click **README.md** → the **pencil** icon to edit
3. Find this line near the top:

   ```
   _Demo video: recording in progress._
   ```
4. **Delete that line** and **drag `demo-small.mp4` onto the empty line.** GitHub uploads it and
   inserts a link that plays inline
5. Scroll down → **Commit changes**

Then view the README and check it plays.

---

## After it is up

- Post it on **LinkedIn** — the draft is in `demo-answers.md`
- Add the same video to your other project READMEs if you record those too

---

## If you re-record

Reload the page first so the transcript starts empty. The session picks its three questions at
random, so a second take gives different questions — which is why `demo-answers.md` has an
answer prepared for all fourteen.

---

## Why `docs/demo.gif` is still in the repo

The README no longer uses it — the uploaded mp4 plays with sound, which a GIF cannot do, so
keeping both at the top was redundant.

It is kept because **GIFs autoplay in a LinkedIn feed and videos do not**. A post with a moving
image stops a scroll; a post with a play button usually does not. Use `docs/demo.gif` there and
the mp4 here.
