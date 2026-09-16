# Deploying to Vercel

The app ships as a single Python serverless function. It deploys and works
with **no environment variables at all** — `DEMO_MODE` defaults to true, so a
fresh deploy runs the seeded story immediately. Real AI is one variable away.

---

## 1 · Push to GitHub

```bash
gh repo create hrin-vidhi --public --source=. --remote=origin --push
```

Or by hand:

```bash
git remote add origin https://github.com/<you>/hrin-vidhi.git
git branch -M main
git push -u origin main
```

## 2 · Import into Vercel

At **vercel.com/new**, pick the repo and deploy. Leave every build setting
empty — `vercel.json` already declares the function, and Vercel installs
`requirements.txt` on its own.

That is the whole deployment. The site comes up on fixtures and every feature
works.

## 3 · Optional — switch on real AI

Add these under **Settings → Environment Variables**, then redeploy. No code
changes.

| Variable | Effect |
|---|---|
| `DEMO_MODE` = `false` | Stop forcing mocks; adapters resolve per the keys below. |
| `ANTHROPIC_API_KEY` | Real Claude vision on photographed loan papers (F2), and prose tightening on the letter (F6). |
| `GOOGLE_API_KEY` | The same two jobs on Gemini instead. Either provider alone is enough — see the note below. |
| `SARVAM_API_KEY` | Real Indic speech-to-text, translation and text-to-speech. |

Each capability resolves independently, so one key is enough to upgrade one
stage. A missing key or a failed call degrades that stage to its mock and says
so in `/health`'s `warnings`, rather than breaking the page.

**Anthropic or Google, not both.** The two cover identical ground and share
the same extraction prompt, so pick whichever you have credit on and add only
that SDK to `requirements.txt` — `anthropic` or `google-genai`. Adding both
works and lets you mix them (`DOCPARSER_PROVIDER=gemini`,
`LLM_PROVIDER=anthropic`), but it is two dependencies in the function bundle
for no gain.

**Check what is live** at `https://<your-app>.vercel.app/health` — it names
the active provider for each of the five capabilities.

---

## What changes between local and deployed

Three things differ on a serverless host, and each is handled rather than
hoped about. `tests/test_serverless.py` holds them in place.

**No system speech voices.** Linux has no `say`. The TTS mock returns the
script with `audio_base64: null` and the provider `mock-unavailable`, and the
page speaks it with the browser's Web Speech API. Marathi and Bhojpuri have no
browser voice on any common platform, so they fall back to a Hindi voice —
same script, accented but intelligible, and the same compromise the offline
voices make.

It previously returned a *silent* WAV here, which was wrong twice over: it
looked like a real voice note to the UI, and a hundred-second script is ~6 MB
of base64.

**No shared memory between invocations.** A server-side session would vanish
on a cold start mid-demo. The browser holds the conversation and passes facts
explicitly; every backend route already worked statelessly. The session routes
still exist and still work within one warm instance — the UI simply does not
depend on them. The letter file is assembled in the browser from the drafted
letter for the same reason.

**A 4.5 MB response cap.** Because no audio crosses the wire, the largest
response is the compliance report. Measured on the full pipeline:

| Response | Size |
|---|---|
| `/app` | 39 KB |
| `/analysis/compliance` | 37 KB |
| `/grievance/draft` | 31 KB |
| `/calc/debt-trap` | 16 KB |
| `/explain/rights` | 11 KB |

A test fails the build if any of these approaches the limit.

---

## Notes

- **Cold starts** run about a second. The corpus is 31 JSON chunks and BM25
  indexing is pure Python, so there is no model to load.
- **Do not add `sentence-transformers`** to `requirements.txt` for a deployment.
  It pulls in torch, blows past the function size limit, and tries to reach
  HuggingFace at import. BM25 is the default for exactly this reason.
- **The microphone needs HTTPS**, which Vercel provides. On `http://` (other
  than localhost) the browser blocks `getUserMedia` and the app falls back to
  its sample recording.
- **`maxDuration` is 30s** in `vercel.json`. Ample for fixtures; raise it if
  you enable real vision parsing on large photographs.
