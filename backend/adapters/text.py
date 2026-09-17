"""Splitting text for providers that cap their input length.

Both the translator and the speech synthesiser take a long script and must send
it in pieces. They had a splitter each: one sliced every N characters, cutting
words and syllables in half so the rejoined audio had audible seams; the other
could emit a piece larger than the limit when a single sentence exceeded it,
which the provider then rejects.

One splitter, used by both.
"""

from __future__ import annotations

#: Sentence terminators across the scripts we speak: the Devanagari danda and
#: double danda, and the Latin stops Tamil and Telugu use in practice.
_SENTENCE_END = tuple("।॥.!?")


def split_on_sentences(text: str, limit: int) -> list[str]:
    """Break text into provider-sized pieces at sentence boundaries.

    Slicing every `limit` characters cuts words, and often syllables, in half;
    joined back together the seams are audible. Splitting on sentence ends
    means each clip begins and finishes on a natural pause.

    A single sentence longer than the limit is still hard-sliced — rare in
    hand-written phrasebook prose, but it must not silently vanish.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    sentences: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in _SENTENCE_END:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        while len(sentence) > limit:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= limit:
            buffer = candidate
        else:
            if buffer:
                pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return pieces


