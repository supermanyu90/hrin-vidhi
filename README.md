# Hrin Vidhi · ऋण विधि

A voice-first, vernacular legal and debt-rights assistant for rural and semi-urban India.

A borrower who took an informal, micro-NBFC or BNPL loan speaks their problem in their own
language, photographs the loan paper or the harassment notice, and gets back a plain-language
explanation of their rights as a voice note plus a ready-to-send grievance letter to the
lender's Nodal Officer and the RBI Ombudsman.

> **This tool gives information and drafts. It is not legal advice.** Every borrower-facing
> output carries a disclaimer, points at RBI's official NBFC register, and tells the borrower
> to have the draft reviewed before filing.

---

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m backend.main                      # → http://127.0.0.1:8000 (no keys needed)
pytest -q                                   # 594 tests
```

**No API keys. No network.** That is the point — see below.

Deploying it? **[`docs/DEPLOY.md`](docs/DEPLOY.md)** — push to GitHub, import to Vercel,
done. It runs on fixtures with no environment variables; real AI is one variable away.

Presenting it? **[`docs/DEMO.md`](docs/DEMO.md)** is a scripted three-minute walkthrough:
pre-flight checks, minute-by-minute with the exact buttons and figures, what to do when
something breaks on stage, and honest answers to the questions judges ask.

| Route | |
|---|---|
| `GET /` | the landing page |
| `GET /app` | the assistant (WhatsApp-style chat) |
| `GET /visualizer` | standalone debt-trap visualizer |
| `GET /health` | adapter status |
| `GET /corpus/status` | retrieval backend and corpus provenance |
| `POST /calc/debt-trap` | money math (F5) |
| `POST /intake/voice` | transcribe a voice note (F1) |
| `POST /intake/document` | parse a loan paper or notice (F2) |
| `POST /session` · `GET`/`DELETE /session/{id}` | open, read, erase a conversation |
| `POST /analysis/compliance` | run the compliance shield (F3) |
| `POST /ask` | grounded question answering (F3) |
| `POST /explain/rights` | rights script + voice note (F4) |
| `POST /grievance/draft` · `GET /grievance/{id}/download` | the letter (F6) |

---

## Fallback, and how you can tell

The application reaches for a real provider whenever a key is present, and falls back to an
offline implementation only when one is not. **A fresh clone with no keys still demos** — that
guarantee now comes from the fallback rather than from a flag, because every `*_PROVIDER` is
`auto`, which resolves to the offline implementation when nothing else can be constructed.

`DEMO_MODE=true` is the explicit override: force every capability offline even where a key is
set, for a venue with no WiFi or a deliberately deterministic run. It defaults to **false**.

Nothing about this is silent. Each capability resolves to one of three states — `live` (a real
provider is answering), `fallback` (none was available) or `forced` (`DEMO_MODE`) — and all
three are reported at `/health` and shown on a badge in the interface, which opens to say which
capability is which and why:

```
GenAI · 3/5          Hearing you          sarvam     sarvam key found
                     Reading your document fallback  [anthropic] ANTHROPIC_API_KEY is not set
```

A key that is set but whose SDK is missing resolves to `fallback`, not `live`: an adapter that
claimed to be live and then degraded on the first real call would make the badge a lie.

Provider contracts are pinned to versions **verified against the live APIs**, not to what the
SDK docs said when the adapter was written: Sarvam retired `saarika:v2`, `bulbul:v1` and the
speaker `meera`, and all three returned a flat 400 until they were checked against a real key.

**Every provider call has a wall-clock ceiling** (`PROVIDER_TIMEOUT_SECONDS`, 25s). Provider
SDKs retry internally with backoff, so a transient upstream error does not fail fast — an
observed Google 503 turned one document upload into three retries over 110 seconds and was
still climbing, and Vercel kills the function at 30s regardless. Past the ceiling the provider
is abandoned, the offline implementation answers, and the response reports `mock` rather than
pretending the model produced it.

The switch is enforced in one place, `backend/adapters/registry.py`, and the registry raises if
`DEMO_MODE` is on while any real adapter has been selected. `/health` reports exactly what is
wired up, which is the first thing to check at a demo:

```jsonc
{ "status": "ok", "demo_mode": true,
  "adapters": { "stt": "mock", "translate": "mock", "tts": "mock",
                "docparser": "mock", "llm": "mock" },
  "corpus_chunks": 31, "warnings": [] }
```

**The mocks are honest about what they are.** None of them fabricates content:

| Adapter | What the mock actually does |
|---|---|
| `stt` | Returns the seeded borrower transcript for the requested language. Discards the audio — it does not pretend to have heard anything. |
| `translate` | Looks the text up in the fixture corpus and a hand-written phrasebook. On a miss it **passes the text through unchanged** and logs it, rather than inventing a translation and putting words in the borrower's mouth. |
| `tts` | Resolves fixture → macOS `say` → silent WAV. `say` ships offline Hindi, Tamil, Telugu and Indian-English voices, so the demo voice note is **real speech**, offline. The provider string (`mock-say` vs `mock-silent`) says which branch ran. |
| `docparser` | Serves the seeded parse for the uploaded document. |
| `llm` | Returns the caller's deterministic draft verbatim. Given no draft it emits a visible `[could not be drafted offline]` placeholder instead of improvising. |

---

## Going live

Set keys in `.env` (copy `.env.example`). Each capability resolves independently, so you can
run a live document parser against mocked speech:

```bash
DEMO_MODE=false
ANTHROPIC_API_KEY=sk-ant-...     # docparser (F2) + prose tightening (F6)
GOOGLE_API_KEY=...               # the same two jobs on Gemini instead
SARVAM_API_KEY=...               # stt, translate, tts
BHASHINI_API_KEY=... BHASHINI_USER_ID=...   # STT only — see the note below
```

**Anthropic and Google are interchangeable.** They cover exactly the same two
capabilities, share the same extraction prompt, and return the same typed
`DocumentExtraction`, so one key is a complete answer:

| Keys set | docparser | llm |
|---|---|---|
| `ANTHROPIC_API_KEY` | anthropic | anthropic |
| `GOOGLE_API_KEY` | gemini | gemini |
| both, `auto` | anthropic | anthropic |
| both, `DOCPARSER_PROVIDER=gemini` | gemini | anthropic |

Mixing is fine — Gemini for vision and Claude for prose, or the reverse.
Install only the SDK you use: `anthropic`, or `google-genai`.

Resolution order per capability:

1. `DEMO_MODE=true` → always the offline implementation. No exceptions.
2. `<NAME>_PROVIDER=<x>` → that provider, and startup **fails loudly** if its key is missing.
   Naming a provider is a claim that it is configured.
3. `<NAME>_PROVIDER=auto` (the default) → the best provider whose credentials **and SDK** are
   present, else the offline implementation. Why each candidate was passed over is recorded and
   shown, so "no key" and "package not installed" are distinguishable at a glance.

**Swapping one mock for the real thing** means adding its key — no code change. To add a *new*
provider, implement the ABC in `backend/adapters/base.py` and register it in the `builders`
dict in `registry.py`.

Two provider notes worth knowing before you rely on them: Bhashini's real flow is two-step
(pipeline config, then compute) and only the compute call is implemented; Sarvam has no
Bhojpuri model, so Bhojpuri requests fall back to Hindi rather than failing.

---

## Architecture

```
Borrower (voice note + photo, WhatsApp-style web UI)
      │
      ▼
FastAPI gateway ──► Intake pipeline
      ├─ 1. SpeechToText → transcript (native language, retained)
      ├─ 2. Translate    → English working copy (original kept)
      ├─ 3. DocParser    → LoanFacts / NoticeFacts
      ├─ 4. Analysis
      │     ├─ finance/  flat→reducing, effective APR, fee impact
      │     └─ analysis/ deterministic rules + RAG over the seeded corpus
      ├─ 5. Rights explainer → plain-language script → TTS voice note
      └─ 6. Grievance drafter → formal letter + vernacular summary
```

Every stage has typed inputs and outputs (`backend/schemas.py`) and can be tested or demoed
alone. Three design rules hold throughout:

- **Unknown is `None`, never a plausible default.** `LoanFacts()` yields `principal=None`,
  `rate_type=UNKNOWN`. A guessed principal would poison the money math and the letter.
- **Every legal claim carries a `Citation` pointing into the corpus by `chunk_id`.** The
  drafter cannot assert a violation without one, so "never fabricate a citation" is enforced
  structurally rather than by prompt.
- **The frontend contains no loan mathematics.** It posts to `/calc/debt-trap` and plots what
  comes back, so the chart, the summary strip and any figure quoted in a letter cannot disagree.

### Layout

```
backend/
  schemas.py            every typed contract
  config.py             DEMO_MODE + adapter selection (the only env reader)
  privacy.py            log redaction
  main.py               app, /health, static mount
  adapters/             base ABCs + mock + real, one module per capability
  finance/              emi, solver, schedules, analysis   ← standalone, importable
  analysis/             corpus loader, retriever, rules engine, citations, RAG
  routes/               intake.py, calc.py, analysis.py, output.py
  explainer/            rights script + the six-language phrasebook (F4)
  grievance/            letter templates + drafter (F6)
  sessions.py           in-memory, TTL-bounded; nothing touches disk
  corpus/               seeded legal chunks
fixtures/               canned transcripts, parses, audio for DEMO_MODE
frontend/
  home.html             the landing page
  index.html            the chat app
  visualizer.html       standalone debt-trap visualizer
  luminous.css          design tokens, shared
  i18n.js               every borrower-facing string, six languages
  format.js             INR/percent formatting — the only maths in the browser
  debt-chart.js         the chart, shared by both pages
tests/
```

`backend/finance/` and `backend/adapters/` are deliberately free of FastAPI and project
imports beyond `schemas`, so either can be lifted into another project unchanged.

---

## The money math

This is the credibility core, in `backend/finance/` with 60+ dedicated tests.

A flat quote charges interest on the *original* principal for the whole term, ignoring every
rupee already repaid. At the same headline percentage it costs roughly twice a real
reducing-balance loan. The module converts one into the other properly:

```python
from backend.finance import analyse_debt

a = analyse_debt(80_000, 12.0, 24, processing_fee=3_200, other_fees_total=2_150)
a.emi              # 4133.33   what the paper says
a.effective_apr    # 21.57 %   the same cash flows as a reducing-balance rate
a.all_in_apr       # 28.94 %   once fees reduce what was actually disbursed
a.honest_emi       # 3765.88   what a real 12% loan would have charged
a.hidden_cost_gap  # ₹14,169   advertised vs actual
```

- **EMI from a flat quote:** `total = P + P·r·years`, `EMI = total / n`.
- **True rate:** solve `P = EMI·(1 − (1+i)^−n)/i` by Newton–Raphson with an unconditional
  bisection fallback. Bisection cannot fail here — the present-value factor is continuous and
  strictly decreasing in `i`, so a sign change brackets exactly one root.
- **Three rates are reported and labelled distinctly**, because they differ and a banker will
  notice: `effective_apr` (nominal, `i × 12` — comparable to an advertised rate),
  `effective_annual_rate` (compounded, always larger, *not* what lenders advertise), and
  `all_in_apr` (fees treated as a reduction in net disbursal — the honest headline).
- **The hidden-cost gap** is measured against what the loan would have cost if the quoted
  percentage had been a genuine reducing-balance rate. That is the comparison the borrower was
  implicitly promised, and it is exactly the vertical gap the chart shades.

Validation worth noting: the solver is checked by **round-trip against an independent closed
form** — build an EMI from a known reducing rate, then require the solver to recover that rate —
across 56 rate/term combinations, plus a 132-case sweep asserting convergence everywhere. Edge
cases covered: 0%, one-month terms, 10%/month predatory rates, and "0% EMI" plus a processing
fee (which correctly reports a non-zero all-in APR).

`RateSolution` carries `converged`, `method`, `iterations` and `residual` so any figure on
screen can be audited rather than taken on trust.

> Stretch, not in the MVP: `finance/` is pure-Python and dependency-free. If the IRR solver ever
> became a hot path it could be swapped for a Rust/PyO3 implementation behind the same
> `solve_monthly_rate` signature.

---

## The compliance shield (F3)

Two layers, and the split between them is load-bearing.

**Deterministic rules** (`backend/analysis/rules.py`) are pure functions from extracted facts
to findings — no LLM, no retrieval, no I/O. Same facts in, same findings out, which is what
makes the output auditable and safe to put in a letter to a regulator. Ten rules currently:
recovery contact outside 08:00–19:00, seven categories of threat, a private agency's letter
presented as court process, undisclosed charges, a missing Key Fact Statement, the rate
illusion, an excessive penalty rate, unverified NBFC registration, unverifiable claims, and a
deadline shorter than the statutory cheque-notice period.

Findings come in two kinds and the distinction is deliberate:

- **`Violation`** — the facts on their face breach a rule we can cite. Assertable in the letter.
- **`Flag`** — something to check, or something we cannot confirm from a photograph. Never
  asserted as a breach.

The rate illusion is a *flag*, not a violation: the arithmetic is certain but whether it
breaches a disclosure rule depends on documents we have not seen.

**Rules refuse to assert what they cannot cite.** Each rule declares its supporting corpus
chunks by id. If any id fails to resolve, the finding is demoted into `report.unconfirmed`
instead of being asserted — so a corpus edit that breaks a reference downgrades a claim rather
than silently shipping an uncited one. A test exercises this path directly.

Three more places the engine declines to overstate:

- `sender_is_court=None` means the parser could not tell, so the court-notice
  misrepresentation rule stays silent. Only a positive `False` triggers it.
- A fee with `disclosed_in_sanction_letter=None` produces no accusation — only a positive
  `False` does.
- A lender that quoted reducing-balance honestly raises no APR flag.

**Grounded Q&A** (`backend/analysis/rag.py`) answers free-form questions from the corpus, and
the answer is composed extractively from retrieved text. Three gates must all pass before
anything is cited:

| Gate | Guards against |
|---|---|
| `RAG_MIN_SCORE` | weak matches generally |
| `RAG_MIN_COVERAGE` | one rare term carrying a whole answer — "night" in *"who won the cricket match last night"* scored well against the recovery-hours rule until this was added |
| `RAG_RELATIVE_FLOOR` | padding a borrower-facing answer with tangentially-related clauses |

Below the gates it returns `grounded=false` and says it could not confirm, rather than letting
a model fill the gap. The LLM is offered the composed draft and may tighten the prose; it is
given the retrieved chunks as its only permitted source, and any failure falls back to the
deterministic draft.

### Retrieval

`GET /corpus/status` reports which backend is live. **BM25 is the default** — pure Python, no
dependencies, no model download, deterministic. The spec asked for sentence-transformers "so it
works offline", but that package fetches its weights from HuggingFace on first use, which at a
demo means depending on the venue's WiFi. So embeddings are an opt-in upgrade: the MiniLM path
is constructed with `local_files_only=True` and is **never** allowed to trigger a download at
request time. A cache miss degrades to BM25 silently.

### Asking in the borrower's own language

The corpus is English. The borrower is not — and a question typed into the ask box in
Devanagari, Tamil or Telugu used to be refused every single time, in all five languages,
because the tokenizer matched `[a-z0-9]+` and Indic text tokenised to nothing.

Two things were wrong, and the second is the subtle one. The pattern was ASCII-only; widening
it to `\w` was not enough either, because a Devanagari vowel sign is a combining mark, which
`\w` excludes — so `सकाळी` was torn into the fragments `सक` and `ळ`. Tokens now include the
marks that belong to them, and the nukta is folded so that `फ़ीस` and `फीस` are one word rather
than two spellings, one of which costs the borrower their answer.

Bridging a vernacular question to an English corpus is the other half. Machine-translating the
question needs a translator that DEMO_MODE does not have; translating the corpus would mean a
model writing legal text in five languages that nobody can check. So `analysis/vernacular.py`
takes the road the explainer already takes — **assembled from pre-translated sentences, not
generated**. The phrasebook's hand-written sentences are indexed in their own language, and
each is bound to the corpus chunks that back it. The borrower reads a sentence a human wrote
in their language; the citation under it points at the English source it came from.

The gates are not relaxed for vernacular queries. Both paths share one scorer
(`analysis/bm25.py`) precisely so `RAG_MIN_SCORE` and `RAG_MIN_COVERAGE` cannot come to mean
different things depending on which language was spoken — *"कल क्रिकेट कौन जीता?"* is refused
exactly as its English twin is, and the refusal is written in the borrower's language too.

**Coverage is at parity with English: all 31 corpus chunks are reachable in all six
languages**, across 20 concepts — the recovery rules, the money rules, the cheque-bounce
limits, the consumer forum, and the whole Ombudsman route. `test_every_corpus_chunk_is_
reachable_in_the_borrowers_language` enforces it: adding a chunk without a phrasebook sentence
fails the build rather than quietly making that material English-only. Widening the corpus
means writing the sentence too — a translation review task, not a model task.

One wrinkle worth knowing about. Indexing the answer sentences alone conflated two different
things: what the borrower *reads* and what the borrower *types*. Closely-related concepts share
their answer vocabulary almost entirely — "how do I complain to the Ombudsman" and "what can
the Ombudsman do" are nearly the same bag of words — so the index picked between them at
random. `_HINTS` fixes that the way the corpus already does with `topics`: a small,
hand-written list of the words a borrower actually reaches for, indexed alongside the sentence
at the same field boost. It is populated only where a concept was measurably missed or
confused; most concepts need none.

### The corpus

31 chunks in `backend/corpus/`, versioned JSON, covering the five sources §5 names. Every chunk
carries `source`, `citation`, `topics` and `is_summary`.

**These are plain-language summaries for retrieval, not statutory text**, and the schema says so
on every chunk. Section numbers appear only where we are confident — Section 138 of the
Negotiable Instruments Act, and two definitions in the Consumer Protection Act, 2019. Everywhere
else the `citation` field carries a descriptive locator rather than a number we would be
guessing at. A test enforces this: a new chunk asserting an unvouched "Section N" fails CI
rather than reaching a letter.

Each chunk has empty `verified_by` and `last_reviewed` fields so a lawyer can replace the text
in place and mark it vetted without any code change. `reviewed_by_lawyer` on `/corpus/status`
currently reports 0, honestly.

---

## Voice intake and the chat (F1)

`GET /` is the borrower-facing app: a WhatsApp-style thread with a language picker, a mic
button, a photo tile, and a question box. It carries no loan mathematics and no legal
reasoning — every number and every finding on screen came from the backend.

The flow: pick a language → record → the transcript comes back in **both** the borrower's
language and English → photograph the loan paper and the notice → the cost analysis, the
violations and the flags appear as cards, each citation shown → ask follow-up questions.

**Language is declared, not detected.** The spec listed auto-detect as a nice-to-have; a
half-working detector that silently picks the wrong language would show a borrower an answer
they cannot read, so the picker is the source of truth and auto-detect is simply absent.

**The mic never blocks the demo.** If `getUserMedia` is unavailable or permission is denied —
a borrowed laptop, a locked-down venue machine — the app substitutes a valid silent WAV and
says so in the bubble. §1 says the flow must complete; it does, and it does not pretend it
heard anything.

Sessions (`backend/sessions.py`) join the stages together. They are **in-memory only**, TTL
bounded, capped, and keyed by `secrets.token_urlsafe` because a session id is effectively a
bearer token for someone's loan documents. Uploaded image and audio bytes are never stored at
all — they are parsed in the request that carried them and dropped, so even a session dump
cannot leak a borrower's paperwork. The 🗑 button erases the conversation immediately; every
route also works statelessly.

All six languages carry the full UI string set and the §9 disclaimer, and a test fails the
build if one language drifts from another or from the backend's phrasebook.

---

## The spoken rights explainer (F4)

`POST /explain/rights` turns a `ComplianceReport` into a plain-language script and speaks it.

**The script is assembled, not generated.** Each finding maps to one hand-written sentence that
already exists in all six languages (`backend/explainer/phrasebook.json`), and the script is the
sentences whose rules actually fired, in a fixed spoken order, de-duplicated. Money first
(the borrower's own question), harassment rights next (what stops tonight's phone call), what
to do last.

Why not generate English and translate it? Because offline there is no translator, and the mock
deliberately refuses to invent one. Assembling from pre-translated sentences means the borrower
hears real sentences in their own language whether or not a provider key is present — and it
means no model can put a legal claim in the script that the rules engine did not make.

The only sentence carrying numbers is the true-cost one, and every figure in it is computed. If
the money math did not run, **the sentence is dropped** rather than spoken with a figure we do
not have. The §9 disclaimer is appended in every case, spoken *and* on screen.

Three tests guard the phrasebook: every message exists in all six languages, no vernacular
string contains Latin letters (which would mean an untranslated English string was pasted in),
and `{placeholder}` sets match across languages so no figure goes missing from one of them.

> **Format note:** the spec asked for `.mp3`; the voice note is `.wav`. No encoder (`lame`,
> `ffmpeg`) is available offline, and browsers play WAV natively, so the acceptance intent —
> inline, downloadable, in the selected language — is met. Setting `SARVAM_API_KEY` yields real
> Indic TTS instead.

## The grievance letter (F6)

`POST /grievance/draft` builds the letter; `GET /grievance/{id}/download` serves it as a text
file with both addressee blocks and the disclaimer attached.

Deterministic skeleton first, LLM second and only to tighten prose. The drafter walks
`report.violations` and `report.flags` and nothing else, so a finding the rules engine demoted
to `unconfirmed` cannot reach the letter.

**The LLM pass is verified, not trusted.** Its output is rejected — and the deterministic draft
sent instead — if it introduces a section reference that was not in the draft, drops a citation,
drops a placeholder, or comes back drastically shorter. Two tests drive this with deliberately
misbehaving adapters: one that inserts *"an offence under Section 420 of the Indian Penal Code"*,
one that strips every `Reference relied on:` line. Neither reaches the letter.

The letter separates what it asserts from what it asks:

- **Section 3** states violations, each with what happened, why it is a problem, and its
  citation — followed by a caveat that the references are plain-language summaries and that the
  borrower is not a lawyer.
- **Section 4** puts the flags as questions, prefaced *"I do not allege wrongdoing on these
  points."* Flags carry a separate `lender_ask` for this, because `Flag.action` is written for
  the borrower — telling a lender to *"verify this lender on RBI's list"* would be nonsense.
- **Section 5** lists relief, de-duplicated: three separate threat violations share one relief,
  and asking a regulator for the same thing three times reads as careless.

Every blank is a self-describing `[BRACKETED INSTRUCTION]`, and `placeholders` enumerates them
for the UI — that is the F6 acceptance criterion about no unlabelled placeholder.

---

## Running it hosted

The app deploys to Vercel as one Python serverless function — see
[`docs/DEPLOY.md`](docs/DEPLOY.md). Three things differ from local, and each is
handled rather than hoped about:

| Serverless reality | What the app does |
|---|---|
| No system speech voices on Linux | Returns the script with **no audio** and the provider `mock-unavailable`; the page speaks it with the Web Speech API. Marathi and Bhojpuri fall back to a Hindi voice, which reads Devanagari. |
| No shared memory between invocations | The browser holds the conversation and passes facts explicitly. Every route already worked statelessly; the UI now relies on that rather than on a session surviving a cold start. |
| A 4.5 MB response cap | No audio crosses the wire, so the largest response is 39 KB. |

That middle row replaced a silent WAV that was wrong twice over: it looked like
a real voice note to the UI, and a hundred-second script is ~6 MB of base64 —
which the platform would have rejected outright.

`tests/test_serverless.py` holds all three in place, including a test that
fails if any language loses its speech fallback.

---

## Privacy

- Loan papers are treated as sensitive. Uploads and transcripts live in memory for the session
  only unless `PERSIST_UPLOADS=true` is explicitly set.
- A redacting filter is installed on the **root** logger, so every record — including ones from
  uvicorn and third-party SDKs — is scrubbed of phone numbers, account numbers, Aadhaar, PAN and
  email before formatting. Loan amounts and rates survive: they are the analysis.
- CORS is restricted to localhost, never a wildcard.

---

## Build status

| | Milestone | State |
|---|---|---|
| 1 | Scaffold, schemas, DEMO_MODE, adapter interfaces + mocks, `/health` | done |
| 2 | `finance/` module + tests | done |
| 3 | F5 visualizer wired to `/calc/debt-trap` | done |
| 4 | F2 doc parser → F3 rules engine → F3 RAG over the seeded corpus | done |
| 5 | F1 voice intake end-to-end + the WhatsApp-style chat UI | done |
| 6 | F4 rights explainer (TTS) + F6 grievance drafter | done |
| 7 | `docs/DEMO.md` — scripted 3-minute judge walkthrough | done |

594 tests passing; the full flow completes in `DEMO_MODE` with no keys set.
