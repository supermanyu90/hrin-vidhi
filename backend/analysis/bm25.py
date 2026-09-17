"""Okapi BM25, as a generic index over any set of tokenised documents.

Lifted out of `retriever.py` so it can serve two indexes with different
documents but identical scoring discipline:

  * the English legal corpus (`retriever.BM25Retriever`), and
  * the per-language phrasebook sentences (`vernacular.py`).

That matters because `RAG_MIN_SCORE` and `RAG_MIN_COVERAGE` are tuned numbers.
If the vernacular path had its own hand-rolled scorer, those thresholds would
quietly mean something different depending on which language the borrower
spoke. Sharing the scorer keeps one set of gates honest across both.

Nothing here knows about legal text, chunks or languages — it takes documents
as `(doc_id, tokens)` and returns scored ids.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Generic, TypeVar

ID = TypeVar("ID")


@dataclass(frozen=True, slots=True)
class Scored(Generic[ID]):
    """One hit.

    `score` is normalised to [0, 1] against the best the query could possibly
    have scored, not against the best it did score — so a weak query's top hit
    stays weak instead of being flattered to 1.0 by normalisation.

    `coverage` is the share of the query's distinct terms this document
    contains. Score and coverage answer different questions: score asks "how
    strong is the match", coverage asks "is this document about the whole
    question, or did one rare word happen to land".
    """

    doc_id: ID
    score: float
    coverage: float


class BM25Index(Generic[ID]):
    """Okapi BM25 with the standard +1 IDF guard."""

    K1 = 1.5
    B = 0.75

    def __init__(self, documents: list[tuple[ID, list[str]]]) -> None:
        self._ids: list[ID] = []
        self._freqs: list[Counter[str]] = []
        self._lengths: list[int] = []
        doc_frequency: Counter[str] = Counter()

        for doc_id, tokens in documents:
            counts = Counter(tokens)
            self._ids.append(doc_id)
            self._freqs.append(counts)
            self._lengths.append(len(tokens))
            doc_frequency.update(counts.keys())

        total = max(1, len(self._ids))
        self._avg_len = (sum(self._lengths) / total) if self._lengths else 0.0
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_frequency.items()
        }

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, terms: list[str], top_k: int) -> list[Scored[ID]]:
        """Score every document against pre-tokenised query terms."""
        if not terms or not self._ids:
            return []

        unique_terms = set(terms)
        scored: list[tuple[float, float, int]] = []

        for index, counts in enumerate(self._freqs):
            length = self._lengths[index] or 1
            score = 0.0
            matched = 0
            for term in unique_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                matched += 1
                idf = self._idf.get(term, 0.0)
                denominator = tf + self.K1 * (1 - self.B + self.B * length / (self._avg_len or 1))
                score += idf * (tf * (self.K1 + 1)) / denominator
            if score > 0:
                scored.append((score, matched / len(unique_terms), index))

        if not scored:
            return []

        ceiling = sum(self._idf.get(t, 0.0) for t in unique_terms) * (self.K1 + 1) / self.K1
        ceiling = max(ceiling, 1e-9)

        # Tie-break on insertion order so results are deterministic run to run.
        scored.sort(key=lambda row: (-row[0], row[2]))
        return [
            Scored(doc_id=self._ids[i], score=min(1.0, raw / ceiling), coverage=coverage)
            for raw, coverage, i in scored[:top_k]
        ]
