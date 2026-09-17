"""Retrieval over the seeded legal corpus.

Two backends behind one interface:

  * `BM25Retriever` — the default. Pure Python, no dependencies, no model
    download, deterministic. Always available.
  * `EmbeddingRetriever` — all-MiniLM-L6-v2 via sentence-transformers, used
    only when the package is installed AND the model is already in the local
    HuggingFace cache.

The spec asked for embeddings "so it works offline", but sentence-transformers
fetches its weights from HuggingFace on first use — on a venue's WiFi, at a
demo, that is exactly the dependency we were told to avoid. So the default is
lexical and the embedding path is an opt-in upgrade that is *never* allowed to
trigger a download at request time. `describe()` reports which is live.

Both return scores normalised to [0, 1] so `RAG_MIN_SCORE` means the same thing
either way: below it we say "couldn't confirm" rather than cite something weak.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from backend.analysis.bm25 import BM25Index
from backend.analysis.corpus_store import CorpusChunk, load_chunks
from backend.config import get_settings

log = logging.getLogger(__name__)

#: Unicode blocks for the scripts this corpus does NOT contain. A query in one
#: of these cannot be matched lexically against English chunks, so it takes the
#: vernacular path instead (see `analysis/vernacular.py`).
_INDIC_RANGES = (
    (0x0900, 0x097F),  # Devanagari — Hindi, Marathi, Bhojpuri
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
)

#: Combining marks — the vowel signs, viramas and nukta of the Indic scripts.
#:
#: These have to be spelled out because `\w` excludes them: a matra is Unicode
#: category Mn/Mc, which is not alphanumeric, so `\w+` tears a word apart at
#: every vowel sign. "सकाळी" became the two fragments "सक" and "ळ", which match
#: nothing and quietly corrupt every Indic token. Built from the ranges above
#: rather than hard-coded so the two cannot drift apart.
_MARKS = "".join(
    chr(cp)
    for low, high in _INDIC_RANGES
    for cp in range(low, high + 1)
    if unicodedata.category(chr(cp)) in ("Mn", "Mc")
)

#: A token is a run of word characters *plus* the marks that belong to them.
#: The previous `[a-z0-9]+` matched ASCII only, so a question asked in the
#: borrower's own language tokenised to nothing and was refused every time.
_TOKEN_RE = re.compile(rf"[\w{re.escape(_MARKS)}]+", re.UNICODE)


def is_indic(text: str) -> bool:
    """True when the text is written in a script the English corpus lacks."""
    return any(any(low <= ord(ch) <= high for low, high in _INDIC_RANGES) for ch in text)


# Words carrying no retrieval signal in a corpus that is entirely about lending
# and complaints. Deliberately short: over-pruning hurts a corpus this small.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "cannot",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "she",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


# (suffix, replacement, minimum token length to apply it). Ordered longest
# first so "agencies" takes the -ies rule before the -es one.
_SUFFIX_RULES: tuple[tuple[str, str, int], ...] = (
    ("ies", "y", 5),
    ("ing", "", 6),
    ("ed", "", 5),
    ("es", "", 5),
    ("s", "", 4),
)


def _stem(token: str) -> str:
    """Suffix stripping for English. Indic tokens are returned unchanged —
    the rules below encode English morphology and would mangle them."""
    if is_indic(token):
        return token
    return _stem_english(token)


def _stem_english(token: str) -> str:
    """Crude suffix stripping so call/calls/called and charge/charges match.

    Not a linguistically correct stemmer and it does not try to be: the only
    requirement is that related forms collapse to the *same* token, real word
    or not. Hence the trailing-'e' step — without it "charges" (-es -> "charg")
    and "charge" (unchanged) would stay apart, which is the bug this exists to
    prevent. Minimum lengths keep it from mangling short words.
    """
    for suffix, replacement, minimum in _SUFFIX_RULES:
        if len(token) >= minimum and token.endswith(suffix):
            token = token[: -len(suffix)] + replacement
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


#: The nukta, U+093C. Devanagari writes several sounds two ways — क़/क, फ़/फ,
#: ज़/ज — and borrowers type whichever their keyboard offers. Folding the nukta
#: away means "फ़ीस" in the phrasebook and "फीस" as typed are the same token,
#: instead of one costing the borrower their answer.
_NUKTA = "\u093c"


def normalize(text: str) -> str:
    """Canonical form for matching: decompose, drop the nukta, recompose."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", decomposed.replace(_NUKTA, ""))


def tokenize(text: str) -> list[str]:
    return [
        _stem(t)
        for t in _TOKEN_RE.findall(normalize(text).lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


@dataclass(frozen=True, slots=True)
class Retrieved:
    chunk: CorpusChunk
    score: float = 0.0
    #: Fraction of the query's distinct terms that appear in this chunk.
    #: A high BM25 score on a single rare term is not enough to ground a
    #: legal answer — see `rag.answer_question`.
    coverage: float = 0.0


class BM25Retriever:
    """Okapi BM25 over chunk text plus title and topic hints.

    Title and topics are repeated into the indexed document because they are
    hand-written retrieval signals — a chunk tagged `["night calls"]` should
    match a question about night calls even if the body never uses that phrase.
    """

    name = "bm25"

    #: Title and topic terms count this many times a body term.
    FIELD_BOOST = 3

    def __init__(self, chunks: tuple[CorpusChunk, ...]) -> None:
        self.chunks = chunks
        self._index: BM25Index[int] = BM25Index(
            [
                (
                    i,
                    tokenize(
                        " ".join([chunk.title, " ".join(chunk.topics)] * self.FIELD_BOOST)
                        + " "
                        + chunk.text
                    ),
                )
                for i, chunk in enumerate(chunks)
            ]
        )

    def search(self, query: str, top_k: int) -> list[Retrieved]:
        return [
            Retrieved(chunk=self.chunks[hit.doc_id], score=hit.score, coverage=hit.coverage)
            for hit in self._index.search(tokenize(query), top_k)
        ]

    def describe(self) -> str:
        return f"bm25 ({len(self.chunks)} chunks)"


class EmbeddingRetriever:
    """all-MiniLM-L6-v2 cosine similarity. Only constructed if already cached."""

    name = "minilm"

    def __init__(self, chunks: tuple[CorpusChunk, ...], model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        self.chunks = chunks
        self._model = SentenceTransformer(model_name, local_files_only=True)
        corpus = [f"{c.title}. {' '.join(c.topics)}. {c.text}" for c in chunks]
        self._matrix = self._model.encode(corpus, normalize_embeddings=True)
        self._model_name = model_name
        # Lexical term sets, kept alongside the vectors purely so `coverage`
        # means the same thing on both backends and the RAG gate is uniform.
        self._terms = [set(tokenize(text)) for text in corpus]

    def search(self, query: str, top_k: int) -> list[Retrieved]:
        if not self.chunks:
            return []
        query_terms = set(tokenize(query))
        vector = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._matrix @ vector
        ranked = sorted(range(len(self.chunks)), key=lambda i: -float(scores[i]))[:top_k]
        # Cosine over normalised vectors is in [-1, 1]; clamp to [0, 1] so the
        # configured threshold means the same thing as it does for BM25.
        return [
            Retrieved(
                chunk=self.chunks[i],
                score=max(0.0, min(1.0, float(scores[i]))),
                coverage=(
                    len(query_terms & self._terms[i]) / len(query_terms) if query_terms else 0.0
                ),
            )
            for i in ranked
        ]

    def describe(self) -> str:
        return f"minilm:{self._model_name} ({len(self.chunks)} chunks)"


def _embedding_retriever_if_cached(
    chunks: tuple[CorpusChunk, ...], model_name: str
) -> EmbeddingRetriever | None:
    """Build the embedding retriever only if it needs no network.

    `local_files_only=True` makes sentence-transformers raise rather than
    download, which is exactly the behaviour we want: a cache miss must
    degrade to BM25, never stall the demo on a model fetch.
    """
    try:
        import sentence_transformers  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        log.info("sentence-transformers not installed; using BM25")
        return None
    try:
        retriever = EmbeddingRetriever(chunks, model_name)
    except Exception as exc:  # noqa: BLE001 - any failure means "use BM25"
        log.info("Embedding model %r not available offline (%s); using BM25", model_name, exc)
        return None
    log.info("Embedding retriever active: %s", model_name)
    return retriever


@lru_cache(maxsize=1)
def get_retriever():
    """The process-wide retriever. Built once; never downloads at request time."""
    settings = get_settings()
    chunks = load_chunks()
    if not chunks:
        log.warning(
            "Corpus is empty — retrieval will return nothing and every legal "
            "claim will be reported as unconfirmed"
        )
    embedding = _embedding_retriever_if_cached(chunks, settings.embedding_model)
    return embedding or BM25Retriever(chunks)


def reset_retriever_cache() -> None:
    get_retriever.cache_clear()
