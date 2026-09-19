# Hrin Vidhi — 3-minute demo script

Everything below runs **offline, with no API keys**. If the venue WiFi is dead, the demo is
unaffected.

---

## Before you present (60 seconds)

```bash
cd hrin-vidhi
source .venv/bin/activate
python -m backend.main   # no keys needed; the badge will read "Fallback"
```

Open **http://127.0.0.1:8000** and check three things:

| Check | Expect |
|---|---|
| Header pill | `DEMO MODE · OFFLINE` |
| `curl -s localhost:8000/health` | `"demo_mode": true`, all five adapters `"mock"`, `"warnings": []` |
| `curl -s localhost:8000/corpus/status` | `"chunk_count": 31` |

Then **turn the volume up** — one of the deliverables is a voice note, and it plays out loud.

Have two images in your Downloads folder to upload. Any JPEG works; the parser is mocked and
keys off the menu item you pick, not the file. Name them `loan-paper.jpg` and `notice.jpg` so
the file picker reads clearly on the projector.

> **If you have a real microphone**, use it — press the mic and speak for a few seconds. If you
> don't, or permission is denied, press it anyway: the app substitutes a sample recording and
> says so in the bubble. The demo completes either way. This is deliberate, not a fallback you
> need to hide.

---

## The story

**Ramesh Kumbhar**, Latur district, Maharashtra. He borrowed **₹80,000** for a second-hand
two-wheeler from *Sahyadri Finserv*, a micro-NBFC. He was told **"only 12% interest"**, over
24 months.

What the papers actually show:

- The 12% is a **flat** rate, so it is charged on the whole ₹80,000 for two years, ignoring
  every instalment he has already paid.
- **₹5,350** in processing and "loan protection" fees that appear nowhere in the sanction letter.
- A **3% per month** late-payment penalty.
- **No RBI registration number** anywhere on the document.
- A recovery agent called at **06:30**, threatened to tell his wife's family and to shame him
  in front of his village, and posted him a page headed **"COURT NOTICE"** — on a private
  agency's letterhead.

One story, and it hits every feature.

*(Ramesh and Sahyadri Finserv are invented. Nothing here is a claim about a real company.)*

---

## The script

### 0:00 — 0:20 · The problem

> "In rural India, someone takes a small loan for a phone or a two-wheeler and is told
> 'only twelve percent'. They cannot read the paper they signed, it is not in their language,
> and when the calls start at half past six in the morning they have no idea that is against
> the rules. Hrin Vidhi lets them just talk."

Point at the language picker. It's set to **मराठी**.

### 0:20 — 0:50 · He speaks (F1)

Press the **mic**. Speak for a few seconds, or let it substitute the sample.

Three bubbles come back:

1. His own voice note, playable.
2. **The transcript in Marathi**, with the **English underneath in italics**.
3. A prompt, in Marathi: *send a photo of your loan paper or the notice.*

> "Both halves are kept. The Marathi is what he reads and what we speak back to him. The English
> is the working copy the rules engine reasons over. He never sees English unless he wants to."

### 0:50 — 1:40 · The papers, and the money (F2, F5)

Press **📎 → कर्जाचा कागद**, pick `loan-paper.jpg`. Then **📎 → वसुलीची नोटीस**, pick `notice.jpg`.

Cards appear. Land on the **खरी किंमत** (true cost) card:

> "He was told twelve percent. He is paying **28.94%**. Over the loan that is
> **₹14,169** more than the same loan at an honest twelve."

**This is the number to dwell on.** The shaded region between the two curves *is* that ₹14,169.

If a judge looks like a finance person, say this:

> "We solve the annuity equation for the rate implied by his actual instalments —
> Newton–Raphson with a bisection fallback. Flat 12% over 24 months comes out at **21.57%**
> nominal; **28.94%** once the undisclosed fees are treated as reducing what he actually
> received. The browser does no arithmetic at all — it plots what the backend returns, so the
> chart and the letter cannot disagree."

Click **संपूर्ण हिशोब पहा →** if you have a spare ten seconds — it opens the full visualizer
with sliders, carrying his figures across.

### 1:40 — 2:20 · What they are not allowed to do (F3)

Scroll to **हे नियम मोडले गेले आहेत** (rules broken).

> "Seven violations. Every one carries the fact that triggered it and a citation into our legal
> corpus — the RBI Fair Practices Code, the digital lending guidelines, the Consumer Protection
> Act."

Point at the 06:30 one, then the family-threat one with the quote reproduced verbatim.

Then scroll to **या गोष्टी तपासा** (things to check):

> "And these are flags, not accusations. The arithmetic is certain but whether it breaches a
> disclosure rule depends on documents we have not seen — so we say 'check this', not 'they
> broke the law'. The tool refuses to assert more than it can cite."

### 2:20 — 2:50 · He hears his rights (F4)

Scroll to **तुमचे हक्क**. **Press play.** Let four or five seconds of Marathi play aloud.

> "That is a hundred-second voice note, in Marathi, generated offline. He does not have to read
> anything."

Then read one line off the screen:

> "'They cannot call you before eight in the morning or after seven at night.'
> 'A recovery agent is not a court. The paper they sent you is not a court notice.'"

### 2:50 — 3:00 · The letter (F6)

Scroll to **तुमचं तक्रारीचं पत्र**. Click **पत्र डाउनलोड करा** and let the file open.

> "A formal complaint to the lender's Nodal Officer — his chronology, every violation with its
> citation, the relief sought, and the escalation route to the RBI Ombudsman through the CMS
> portal if they don't reply in thirty days. Ten blanks, each one labelled with what goes in it.
> The summary above it is in Marathi so he knows what he is sending."

**Close on the disclaimer**, which is on every screen and in the letter:

> "And it says, every time, in his language: this is information and a draft, not legal advice.
> Check the lender on RBI's register. Show this to a lawyer before you file."

---

## If something goes wrong

| Symptom | Do this |
|---|---|
| Mic does nothing / permission denied | Press it anyway. The sample recording is substituted and the bubble says so. Keep going. |
| Voice note doesn't play | Say "the script is on screen" and read a line. The script is always returned even when synthesis fails. |
| A card is missing | `curl -s localhost:8000/health` on the second screen. `warnings: []` means the pipeline is fine. |
| Page looks stale after an edit | It shouldn't — asset URLs carry a build stamp. If it does: hard reload (⌘⇧R). |
| The WiFi dies and a live provider starts timing out | `DEMO_MODE=true python -m backend.main` — forces every capability offline. The badge switches to “Fallback” and the whole flow still works. |
| Everything is broken | `pytest -q` — 592 tests, ~10 seconds. It's a strong recovery move in front of judges. |

**Never** run with `DEMO_MODE=false` on stage. There are no keys, so adapters resolve to mocks
anyway, but `/health` would show a warning and a judge may reasonably ask about it.

---

## Questions judges ask

**"Is the legal content real?"**
The corpus is 31 plain-language summaries of real RBI and statutory provisions, marked
`is_summary: true` on every chunk. **Section numbers appear only where we are confident** —
Section 138 of the Negotiable Instruments Act, and two Consumer Protection Act definitions.
Everywhere else the citation is a descriptive locator rather than a number we would be guessing
at, and a test fails the build if a new chunk invents one. Every chunk has empty `verified_by`
and `last_reviewed` fields so a lawyer can replace the text in place without touching code.
`/corpus/status` reports `reviewed_by_lawyer: 0`, honestly.

**"What is actually AI here, and what is mocked?"**
In DEMO_MODE the speech, translation and document parsing are fixtures. The parts that make the
claims — **the money math and the compliance rules — are real code running live**, and the
retrieval behind the Q&A box is real BM25 over the corpus. With `ANTHROPIC_API_KEY` set, the
document parser becomes Claude vision with structured outputs; with `SARVAM_API_KEY`, the speech
stack becomes real Indic STT/TTS. One key each, no code change.

**"Could it invent a law?"**
Structurally, no. Every rule declares its supporting corpus chunks by id; if an id fails to
resolve the finding is demoted to `unconfirmed` rather than asserted. The letter drafter walks
only the report's violations. And the LLM's prose pass is **verified** — if it introduces a
section reference that was not in the draft, or drops a citation, its output is thrown away and
the deterministic draft is sent. There's a test that feeds it an adapter which inserts
*"an offence under Section 420 of the Indian Penal Code"*; it never reaches the letter.

**"Why should I trust 28.94%?"**
`RateSolution` carries `converged`, `method`, `iterations` and `residual` — visible under
*"How these numbers were worked out"* in the visualizer. The solver is validated by round-trip
against an independent closed form across 56 rate/term pairs, plus a 132-case convergence sweep.
92 tests on the finance module alone.

**"What about the borrower's privacy?"**
Uploaded photos and audio are never written to disk — parsed in the request that carried them
and dropped. Sessions are in-memory, TTL-bounded, and keyed with `secrets.token_urlsafe`. A
redacting filter sits on the root logger, so phone numbers, account numbers, Aadhaar and PAN are
scrubbed from every log line — including ones from uvicorn. Loan amounts and rates survive,
because they are the analysis. There's a 🗑 button that erases the conversation immediately.

**"Ask it something."**
Use the question box. *"Can they call me at 6 in the morning?"* → grounded answer with the Fair
Practices Code citation. Then ask it something off-topic — *"who won the cricket last night?"* —
and it says it could not confirm and cites nothing. **That refusal is the feature**; it is worth
demonstrating deliberately.

---

## What is honestly not finished

Say this before a judge finds it:

- **Auto-detect of spoken language** is not implemented. The spec listed it as a nice-to-have,
  and a detector that silently picks the wrong language would show a borrower an answer they
  cannot read. The picker is the source of truth.
- **The voice note is WAV, not MP3.** No encoder is available offline; browsers play WAV
  natively.
- **Bhojpuri and Marathi TTS** route through a Devanagari voice, so they are accented. The
  provider string says which path ran (`mock-say` vs `mock-fixture`).
- **No lawyer has reviewed the corpus.** The schema is built for that review; it has not
  happened.
- **Bhashini's real adapter** implements the compute call but not the two-step pipeline-config
  call it needs in production.
