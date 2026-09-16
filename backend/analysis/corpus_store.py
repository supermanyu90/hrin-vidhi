"""Loader for the seeded legal corpus.

Milestone 1 provides the loader and the schema contract only; the chunks
themselves and the retrieval layer land in milestone 4. `/health` reports the
chunk count so a judge can see at a glance whether retrieval has anything to
ground against.

Corpus files are versioned JSON under `backend/corpus/`, one file per source.
Every chunk carries its own provenance so a lawyer can replace the text
in place without touching any code.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from backend.config import CORPUS_DIR

log = logging.getLogger(__name__)


class CorpusChunk(BaseModel):
    """One retrievable unit of legal reference material.

    `is_summary` defaults to True and every seeded chunk keeps it that way:
    this corpus is plain-language summarisation for retrieval, not statutory
    text. A vetted replacement may set it False.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(description="Stable id, e.g. 'rbi-fpc-recovery-hours'.")
    title: str
    source: str = Field(description="Human-readable provenance, e.g. 'RBI Fair Practices Code'.")
    citation: str = Field(
        description="Clause/section reference. Left as a descriptive locator where an "
        "exact section number is not certain — never invented."
    )
    text: str = Field(description="The plain-language summary that gets retrieved.")
    topics: list[str] = Field(default_factory=list, description="Retrieval hints.")
    url: str | None = None
    is_summary: bool = True
    verified_by: str | None = Field(
        default=None, description="Set once a qualified reviewer has checked this chunk."
    )
    last_reviewed: str | None = None


class CorpusFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    source_note: str = Field(description="What this file covers and how it was written.")
    chunks: list[CorpusChunk]


@lru_cache(maxsize=1)
def load_chunks(corpus_dir: str = str(CORPUS_DIR)) -> tuple[CorpusChunk, ...]:
    """Read every corpus JSON file. Returns empty if the corpus is not seeded yet."""
    directory = Path(corpus_dir)
    if not directory.is_dir():
        log.warning("Corpus directory %s does not exist", directory)
        return ()

    chunks: list[CorpusChunk] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            corpus_file = CorpusFile.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill startup
            log.error("Skipping malformed corpus file %s: %s", path.name, exc)
            continue
        for chunk in corpus_file.chunks:
            if chunk.chunk_id in seen:
                log.error("Duplicate chunk_id %r in %s — skipping", chunk.chunk_id, path.name)
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)

    log.info("Loaded %d corpus chunks from %s", len(chunks), directory)
    return tuple(chunks)


def corpus_chunk_count() -> int:
    return len(load_chunks())


def reset_corpus_cache() -> None:
    load_chunks.cache_clear()
