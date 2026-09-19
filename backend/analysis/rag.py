"""F3 — grounded question answering over the seeded corpus.

The rule: **every legal statement returned here cites a retrieved chunk.** If
retrieval scores below `RAG_MIN_SCORE`, we return `grounded=False` and say we
could not confirm it, rather than letting a model fill the gap from memory.
That is the difference between a tool a borrower can take to an office and one
that invents a section number.

The answer itself is assembled from the retrieved text by a deterministic
composer. The LLM adapter is offered the composed draft and may tighten the
prose, but it is given the retrieved chunks as the only permitted source and
its output is discarded if it comes back empty. In DEMO_MODE the mock returns
the deterministic draft unchanged, so the demo answer is the real answer,
just less fluent.
"""

from __future__ import annotations

import logging

from backend.adapters.base import LLM, AdapterError
from backend.analysis.citations import to_citation
from backend.analysis.retriever import get_retriever, tokenize
from backend.analysis.vernacular import chunks_for, detect_language, looks_vernacular
from backend.analysis.vernacular import search as search_vernacular
from backend.config import get_settings
from backend.explainer.script import disclaimer_for, phrase
from backend.schemas import Citation, Language, RagAnswer

log = logging.getLogger(__name__)

COULD_NOT_CONFIRM = (
    "I could not find anything in my reference material that answers this reliably, so I "
    "will not guess. Please ask a lawyer or a free legal aid clinic."
)

#: A different failure from the one above, and it deserves a different answer.
#: "What should I do?" is every stopword and no subject: retrieval has nothing
#: to work with, so saying "I could not confirm that" is misleading — there was
#: no claim to confirm. Ask for the missing noun instead, and show what this
#: corpus actually covers so the next question can land.
TOO_VAGUE = (
    "I need a little more to go on. Try naming the thing you are worried about — a phone "
    "call, a threat, a court notice, a fee you were not told about, someone taking your "
    "vehicle, or who the lender really is."
)

RAG_SYSTEM_PROMPT = """\
You rewrite a draft answer for a borrower in rural India who is dealing with a
debt problem. You are an editor, not a source.

Rules:
- Use ONLY the facts in the draft and the reference extracts given. Add nothing.
- Never state a section number, statute name, date, or figure that is not
  already in the draft or the extracts.
- If the draft says something could not be confirmed, keep that.
- Write short sentences at roughly a 6th-grade reading level. No legalese.
- Do not add a greeting, a sign-off, or an offer to help further.
- Return only the rewritten answer.
"""


#: Answers already produced, keyed by the question and the language it was
#: asked in.
#:
#: The English path pays for a model call to tighten its draft, and that call
#: is the whole cost of the request: 8.5 seconds against 415 milliseconds for
#: the vernacular path, which uses no model at all. The questions that arrive
#: are not evenly distributed — the interface offers a fixed set as buttons,
#: and a demo asks the same handful repeatedly — so the second person to ask
#: gets the first person's answer.
#:
#: Bounded, and per instance: a cold start begins empty, which is correct.
#: Nothing here is stored beyond the process.
_ANSWER_CACHE: dict[tuple[str, str], RagAnswer] = {}
_ANSWER_CACHE_MAX = 128


def _cache_key(question: str, language: Language) -> tuple[str, str]:
    return (" ".join(question.lower().split()), language.value)


def clear_answer_cache() -> None:
    _ANSWER_CACHE.clear()


async def answer_question(
    question: str,
    llm: LLM | None = None,
    language: Language = Language.ENGLISH,
) -> RagAnswer:
    """Answer a free-form borrower question, grounded in the corpus.

    Two routes, one standard of proof. A question in an Indic script cannot be
    matched against an English corpus, so it goes through the phrasebook bridge
    in `vernacular.py`; everything else retrieves directly. Both apply the same
    score and coverage gates, and both cite real corpus chunks or refuse.
    """
    settings = get_settings()

    cached = _ANSWER_CACHE.get(_cache_key(question, language))
    if cached is not None:
        return cached

    if looks_vernacular(question) and language is Language.ENGLISH:
        # The ask box posts the borrower's selected language, but a question in
        # Devanagari with "English" selected is a real case — someone switched
        # the picker but kept typing. Trust the script over the dropdown.
        language = detect_language(question)

    if not tokenize(question):
        # No usable terms at all. Not the same thing as "nothing matched":
        # there was no claim to confirm, so asking for a noun beats refusing.
        log.info("RAG had no searchable terms in %r", question[:80])
        return RagAnswer(
            question=question,
            answer=phrase("ask_too_vague", language) or TOO_VAGUE,
            citations=[],
            grounded=False,
        )

    if looks_vernacular(question):
        return _answer_vernacular(question, language)

    retriever = get_retriever()
    hits = retriever.search(question, settings.rag_top_k)

    # Two independent gates, and both must pass. Score alone is not enough:
    # a single rare term ("night") can score well against a chunk that does
    # not answer the question at all. Coverage asks whether the chunk is
    # actually about what was asked.
    strong = [
        h
        for h in hits
        if h.score >= settings.rag_min_score and h.coverage >= settings.rag_min_coverage
    ]

    if not strong:
        if hits:
            best = max(hits, key=lambda h: h.score)
            detail = f"best score {best.score:.3f}, coverage {best.coverage:.2f}"
        else:
            detail = "no match"
        log.info(
            "RAG declined to answer %r (%s; needs score>=%s and coverage>=%s)",
            question[:80],
            detail,
            settings.rag_min_score,
            settings.rag_min_coverage,
        )
        return RagAnswer(question=question, answer=COULD_NOT_CONFIRM, citations=[], grounded=False)

    # Third gate, relative rather than absolute: drop anything well below the
    # best hit. A borrower reading this out loud should hear the rule that
    # answers their question, not three tangentially-related clauses after it.
    top = strong[0].score
    strong = [h for h in strong if h.score >= top * settings.rag_relative_floor]

    # And relative on coverage too: a chunk that answers a fraction of what
    # the best one answers is padding, however close its score.
    best_coverage = max(h.coverage for h in strong)
    strong = [
        h for h in strong if h.coverage >= best_coverage * settings.rag_relative_coverage
    ]

    citations = [to_citation(h.chunk, score=h.score) for h in strong]
    draft = _compose(question, citations)

    if llm is None:
        return RagAnswer(question=question, answer=draft, citations=citations, grounded=True)

    polished = await _polish(llm, question, draft, citations)
    answer = RagAnswer(question=question, answer=polished, citations=citations, grounded=True)

    # Only grounded answers are worth keeping. A refusal costs nothing to
    # produce again, and caching one would outlive a corpus that grew to
    # answer it.
    if len(_ANSWER_CACHE) >= _ANSWER_CACHE_MAX:
        _ANSWER_CACHE.pop(next(iter(_ANSWER_CACHE)))
    _ANSWER_CACHE[_cache_key(question, language)] = answer
    return answer


def _answer_vernacular(question: str, language: Language) -> RagAnswer:
    """Answer an Indic-script question from the phrasebook, with citations.

    The LLM is deliberately not invited to polish this. The sentence returned
    is one a human wrote and reviewed in the borrower's language; handing it to
    a model with an English editing prompt could only degrade it, and the whole
    reason this text is pre-written is that no model output in these languages
    can be checked at request time.
    """
    settings = get_settings()
    matches = search_vernacular(question, language, settings.rag_top_k)
    if not matches:
        log.info("RAG (vernacular, %s) declined to answer %r", language.value, question[:80])
        return RagAnswer(
            question=question,
            answer=phrase("ask_not_found", language) or COULD_NOT_CONFIRM,
            citations=[],
            grounded=False,
        )

    # Same relative floor as the English path: answer the question asked, not
    # every right that shares a word with it.
    top = matches[0][2]
    matches = [m for m in matches if m[2] >= top * settings.rag_relative_floor]

    lines: list[str] = []
    citations: list[Citation] = []
    for key, sentence, score, _coverage in matches:
        lines.append(sentence)
        citations.extend(to_citation(chunk, score=score) for chunk in chunks_for(key))

    lines.append(disclaimer_for(language))

    return RagAnswer(
        question=question,
        answer=" ".join(lines),
        citations=citations,
        grounded=True,
    )


def _compose(question: str, citations: list[Citation]) -> str:
    """Deterministic answer: the retrieved summaries, attributed.

    Deliberately extractive. Anything more would be this function inventing
    legal content, which is the failure mode the whole design exists to avoid.
    """
    lines = ["Here is what my reference material says about this:", ""]
    for citation in citations:
        lines.append(f"- {citation.quote}")
        lines.append(f"  (Source: {citation.source} — {citation.citation})")
    lines.append("")
    lines.append(
        "These are plain-language summaries, not the exact words of the law. Please have "
        "them checked before you rely on them."
    )
    return "\n".join(lines)


async def _polish(llm: LLM, question: str, draft: str, citations: list[Citation]) -> str:
    """Let the LLM tighten the prose. Any failure falls back to the draft."""
    extracts = "\n\n".join(
        f"[{c.chunk_id}] {c.source} — {c.citation}\n{c.quote}" for c in citations
    )
    prompt = (
        "TASK: rag_answer\n\n"
        f"The borrower asked: {question}\n\n"
        "Reference extracts you may use, and nothing else:\n"
        f"{extracts}\n\n"
        "DRAFT:\n"
        f"{draft}"
    )
    try:
        # 1024 was not enough. Current models spend part of the output budget
        # reasoning before they emit anything, so every English answer came
        # back cut mid-sentence — and lost the closing caveat that says these
        # are summaries to be checked, which is the one line that must survive.
        # Measured on the real prompt: 1024 truncates at 374 characters, 3072
        # completes at 541.
        text = await llm.complete(prompt, system=RAG_SYSTEM_PROMPT, max_tokens=3072, effort="low")
    except AdapterError as exc:
        log.warning("RAG polish failed (%s); returning the deterministic draft", exc)
        return draft
    return text.strip() or draft
