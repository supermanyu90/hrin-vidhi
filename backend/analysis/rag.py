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
from backend.analysis.retriever import get_retriever
from backend.config import get_settings
from backend.schemas import Citation, RagAnswer

log = logging.getLogger(__name__)

COULD_NOT_CONFIRM = (
    "I could not find anything in my reference material that answers this reliably, so I "
    "will not guess. Please ask a lawyer or a free legal aid clinic."
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


async def answer_question(question: str, llm: LLM | None = None) -> RagAnswer:
    """Answer a free-form borrower question, grounded in the corpus."""
    settings = get_settings()
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

    citations = [to_citation(h.chunk, score=h.score) for h in strong]
    draft = _compose(question, citations)

    if llm is None:
        return RagAnswer(question=question, answer=draft, citations=citations, grounded=True)

    polished = await _polish(llm, question, draft, citations)
    return RagAnswer(question=question, answer=polished, citations=citations, grounded=True)


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
        text = await llm.complete(prompt, system=RAG_SYSTEM_PROMPT, max_tokens=1024, effort="low")
    except AdapterError as exc:
        log.warning("RAG polish failed (%s); returning the deterministic draft", exc)
        return draft
    return text.strip() or draft
