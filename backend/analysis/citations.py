"""Turning corpus chunk ids into `Citation` objects.

This is the chokepoint that makes §9 structural rather than aspirational: a
rule declares which chunk ids support it, and if a chunk id does not resolve,
the finding it belongs to is dropped into `unconfirmed` instead of being
asserted. Nothing downstream can assert a legal claim without a real chunk
behind it, because nothing downstream is given a way to build a `Citation`
from thin air.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from backend.analysis.corpus_store import CorpusChunk, load_chunks
from backend.schemas import Citation

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _index() -> dict[str, CorpusChunk]:
    return {chunk.chunk_id: chunk for chunk in load_chunks()}


def to_citation(chunk: CorpusChunk, score: float | None = None) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        source=chunk.source,
        citation=chunk.citation,
        quote=chunk.text,
        url=chunk.url,
        is_summary=chunk.is_summary,
        retrieval_score=score,
    )


def resolve(chunk_ids: list[str]) -> tuple[list[Citation], list[str]]:
    """Resolve chunk ids to citations.

    Returns (citations, missing_ids). A caller that gets a non-empty
    `missing_ids` must not assert the claim those ids were supporting.
    """
    index = _index()
    citations: list[Citation] = []
    missing: list[str] = []
    for chunk_id in chunk_ids:
        chunk = index.get(chunk_id)
        if chunk is None:
            log.error("Rule referenced unknown corpus chunk %r", chunk_id)
            missing.append(chunk_id)
            continue
        citations.append(to_citation(chunk))
    return citations, missing


def dedupe(citations: list[Citation]) -> list[Citation]:
    """De-duplicate by chunk_id, keeping the best retrieval score seen."""
    best: dict[str, Citation] = {}
    for citation in citations:
        existing = best.get(citation.chunk_id)
        if existing is None:
            best[citation.chunk_id] = citation
            continue
        if (citation.retrieval_score or 0) > (existing.retrieval_score or 0):
            best[citation.chunk_id] = citation
    return sorted(best.values(), key=lambda c: c.chunk_id)


def reset_citation_cache() -> None:
    _index.cache_clear()
