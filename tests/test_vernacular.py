"""The vernacular ask path.

This file exists because of a bug that made the product's central promise
untrue: every question asked in an Indic script was refused, in all five
languages, because the tokenizer matched `[a-z0-9]+`. The ask box invited a
question in the borrower's language that could never be answered.

Two directions are tested, and both matter equally:

  * a real question in each language must be answered, cited, and written in
    that language; and
  * an off-topic question must still be refused, in every language.

The second set is the guard on the first. Fixing vernacular retrieval by
loosening the grounding gates would have made these pass while turning the
assistant into something that answers cricket questions with legal citations.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.analysis.corpus_store import load_chunks
from backend.analysis.rag import COULD_NOT_CONFIRM, answer_question
from backend.analysis.retriever import is_indic, normalize, tokenize
from backend.analysis.vernacular import (
    _HINTS,
    MESSAGE_TO_CHUNKS,
    chunks_for,
    detect_language,
)
from backend.explainer.script import phrase
from backend.schemas import Language

INDIC = [Language.HINDI, Language.MARATHI, Language.BHOJPURI, Language.TAMIL, Language.TELUGU]

#: "Can they call me at 6 in the morning?" — the recovery-hours question, the
#: single most common thing a harassed borrower asks.
RECOVERY_HOURS_QUESTION = {
    Language.HINDI: "क्या वे सुबह 6 बजे फोन कर सकते हैं?",
    Language.MARATHI: "ते सकाळी सहा वाजता फोन करू शकतात का?",
    Language.BHOJPURI: "का ऊ लोग भोरे 6 बजे फोन कर सकेला?",
    Language.TAMIL: "காலை 6 மணிக்கு அழைக்க முடியுமா?",
    Language.TELUGU: "వాళ్ళు ఉదయం 6 గంటలకు ఫోన్ చేయవచ్చా?",
}

#: "Who won the cricket yesterday?" — nothing to do with debt, in each script.
OFF_TOPIC = {
    Language.HINDI: "कल क्रिकेट कौन जीता?",
    Language.MARATHI: "काल क्रिकेट कोण जिंकलं?",
    Language.BHOJPURI: "काल्ह क्रिकेट में के जीतल?",
    Language.TAMIL: "நேற்று கிரிக்கெட்டில் யார் வென்றார்கள்?",
    Language.TELUGU: "నిన్న క్రికెట్‌లో ఎవరు గెలిచారు?",
}


# ---------------------------------------------------------------------------
# Tokenization — the layer the bug actually lived in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("सकाळी", "सकाळी"),  # Devanagari with two matras
        ("முடியுமா", "முடியுமா"),  # Tamil
        ("గంటలకు", "గంటలకు"),  # Telugu
    ],
)
def test_indic_words_survive_tokenization_whole(text: str, expected: str) -> None:
    """A vowel sign must not split a word.

    `\\w` excludes combining marks, so the original pattern tore "सकाळी" into
    "सक" and "ळ" — fragments that match nothing.
    """
    assert tokenize(text) == [expected]


def test_ascii_tokenization_is_unchanged() -> None:
    """The English path must be byte-for-byte what it was before the fix."""
    assert tokenize("Recovery agents calling at night") == ["recovery", "agent", "call", "night"]


def test_nukta_spelling_variants_collapse() -> None:
    """फ़ीस and फीस are the same word to a borrower; they must be one token."""
    assert tokenize("फ़ीस") == tokenize("फीस")
    assert normalize("क़र्ज़") == normalize("करज".replace("रज", "र्ज"))


def test_english_stemmer_is_not_applied_to_indic() -> None:
    """The suffix rules encode English morphology and would mangle Devanagari."""
    for word in ("बताया", "शकतात", "செலுத்த"):
        assert tokenize(word) == [word]


@pytest.mark.parametrize("language", INDIC)
def test_is_indic_detects_every_supported_script(language: Language) -> None:
    assert is_indic(RECOVERY_HOURS_QUESTION[language])


def test_is_indic_is_false_for_english() -> None:
    assert not is_indic("Can they call me at 6 in the morning?")


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (Language.HINDI, Language.HINDI),
        (Language.TAMIL, Language.TAMIL),
        (Language.TELUGU, Language.TELUGU),
    ],
)
def test_language_is_detected_from_script(language: Language, expected: Language) -> None:
    """Devanagari resolves to Hindi; Tamil and Telugu are unambiguous."""
    assert detect_language(RECOVERY_HOURS_QUESTION[language]) is expected


def test_detect_language_falls_back_to_english() -> None:
    assert detect_language("Can they call me at 6?") is Language.ENGLISH


# ---------------------------------------------------------------------------
# The mapping that keeps a vernacular answer grounded
# ---------------------------------------------------------------------------


def test_every_mapped_chunk_id_exists_in_the_corpus() -> None:
    """A citation that cannot be resolved is a fabricated citation."""
    known = {c.chunk_id for c in load_chunks()}
    referenced = {cid for ids in MESSAGE_TO_CHUNKS.values() for cid in ids}
    assert referenced <= known, sorted(referenced - known)


@pytest.mark.parametrize("key", sorted(MESSAGE_TO_CHUNKS))
@pytest.mark.parametrize("language", [*INDIC, Language.ENGLISH])
def test_every_answerable_concept_is_written_in_every_language(
    key: str, language: Language
) -> None:
    """An unanswerable language is a silently broken promise, not a fallback."""
    assert phrase(key, language).strip()


@pytest.mark.parametrize("key", sorted(MESSAGE_TO_CHUNKS))
def test_chunks_for_resolves_to_real_chunks(key: str) -> None:
    chunks = chunks_for(key)
    assert chunks
    assert all(c.text.strip() for c in chunks)


def test_true_cost_is_not_answerable_standalone() -> None:
    """It carries computed figures from a specific loan and means nothing alone."""
    assert "true_cost" not in MESSAGE_TO_CHUNKS


def test_every_corpus_chunk_is_reachable_in_the_borrowers_language() -> None:
    """Parity: anything answerable in English is answerable in every language.

    This is the guard that keeps the vernacular path from quietly falling
    behind. Adding a chunk to the corpus without a phrasebook sentence and a
    `MESSAGE_TO_CHUNKS` entry makes it English-only — a borrower asking in
    Hindi would be refused on material the assistant demonstrably holds. That
    is invisible in manual testing and obvious here.

    Widening the corpus therefore means writing the sentence too. That is the
    intended cost, and it is a translation review task, not a model task.
    """
    reachable = {chunk_id for ids in MESSAGE_TO_CHUNKS.values() for chunk_id in ids}
    orphaned = sorted({c.chunk_id for c in load_chunks()} - reachable)
    assert not orphaned, (
        "these corpus chunks can be retrieved in English but never in the borrower's "
        f"language: {orphaned}. Add a phrasebook sentence and map it in MESSAGE_TO_CHUNKS."
    )


@pytest.mark.parametrize("key", sorted(_HINTS))
def test_hints_are_written_for_every_language(key: str) -> None:
    """A hint in Hindi only would silently privilege Hindi speakers."""
    languages = {*(lang.value for lang in INDIC), Language.ENGLISH.value}
    assert set(_HINTS[key]) == languages, f"{key} is missing {languages - set(_HINTS[key])}"


@pytest.mark.parametrize("key", sorted(_HINTS))
def test_hints_only_exist_for_real_concepts(key: str) -> None:
    assert key in MESSAGE_TO_CHUNKS


# ---------------------------------------------------------------------------
# Answering — both directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", INDIC)
def test_a_real_question_is_answered_in_every_language(language: Language) -> None:
    answer = asyncio.run(answer_question(RECOVERY_HOURS_QUESTION[language], language=language))

    assert answer.grounded, f"{language.value} question was refused"
    assert answer.citations, "a grounded answer must cite something"
    assert answer.answer != COULD_NOT_CONFIRM


@pytest.mark.parametrize("language", INDIC)
def test_the_answer_is_written_in_the_borrowers_script(language: Language) -> None:
    """An English answer to a Hindi question is not an answer."""
    answer = asyncio.run(answer_question(RECOVERY_HOURS_QUESTION[language], language=language))
    assert is_indic(answer.answer)


@pytest.mark.parametrize("language", INDIC)
def test_the_answer_cites_the_recovery_hours_rule(language: Language) -> None:
    """The right answer, not merely an answer."""
    answer = asyncio.run(answer_question(RECOVERY_HOURS_QUESTION[language], language=language))
    assert any(c.chunk_id == "rbi-fpc-recovery-hours" for c in answer.citations)


@pytest.mark.parametrize("language", INDIC)
def test_off_topic_questions_are_still_refused_in_every_language(
    language: Language,
) -> None:
    """The guard on everything above.

    If a future change to the gates makes this pass a cricket question, the
    vernacular path has stopped being grounded retrieval.
    """
    answer = asyncio.run(answer_question(OFF_TOPIC[language], language=language))
    assert not answer.grounded
    assert answer.citations == []


@pytest.mark.parametrize("language", INDIC)
def test_a_refusal_is_written_in_the_borrowers_language(language: Language) -> None:
    """Being told "no" in a language you cannot read is its own failure."""
    answer = asyncio.run(answer_question(OFF_TOPIC[language], language=language))
    assert is_indic(answer.answer)


def test_the_script_overrides_a_stale_language_picker() -> None:
    """Someone switches the dropdown but keeps typing Hindi. Answer anyway."""
    answer = asyncio.run(
        answer_question(RECOVERY_HOURS_QUESTION[Language.HINDI], language=Language.ENGLISH)
    )
    assert answer.grounded
    assert is_indic(answer.answer)


def test_a_question_with_no_searchable_terms_asks_for_more() -> None:
    """ "What should I do?" is not a failed lookup — there was nothing to look up."""
    answer = asyncio.run(answer_question("What should I do?"))
    assert not answer.grounded
    assert answer.answer != COULD_NOT_CONFIRM


def test_the_english_path_is_unaffected() -> None:
    answer = asyncio.run(answer_question("Can they call me at 6 in the morning?"))
    assert answer.grounded
    assert any(c.chunk_id == "rbi-fpc-recovery-hours" for c in answer.citations)


#: One borrower question per concept, in Hindi, and the chunk that must be
#: cited back. Covers the whole answerable surface rather than one rule, so a
#: retrieval change that helps one topic and breaks another shows up here.
BY_CONCEPT: list[tuple[str, str]] = [
    ("क्या वे सुबह 6 बजे फोन कर सकते हैं?", "rbi-fpc-recovery-hours"),
    ("क्या वे मेरे पड़ोसियों को बता सकते हैं?", "rbi-fpc-third-party-contact"),
    ("वे मुझे धमका रहे हैं", "rbi-fpc-no-harassment"),
    ("क्या वसूली एजेंट मुझे गिरफ्तार करवा सकता है?", "ni-138-no-arrest-for-debt"),
    ("क्या वे मेरी बाइक ले जा सकते हैं?", "rbi-fpc-possession-due-process"),
    ("फोन करने वाला कौन है, कैसे पता करूँ?", "rbi-fpc-agent-identification"),
    ("मुझे कोर्ट नोटिस मिला है, क्या यह असली है?", "ni-138-court-notice-vs-private-letter"),
    ("मेरा चेक बाउंस हो गया", "ni-138-what-it-covers"),
    ("फीस के बारे में पहले क्यों नहीं बताया?", "rbi-dl-no-hidden-charges"),
    ("असली ब्याज दर क्या है?", "rbi-dl-apr-disclosure"),
    ("देर से भरने पर इतना जुर्माना क्यों?", "rbi-dl-penal-charges-reasonable"),
    ("क्या मैं कर्ज़ वापस कर सकता हूँ बिना जुर्माने के?", "rbi-dl-cooling-off"),
    ("मेरी क्रेडिट सीमा अपने आप बढ़ गई", "rbi-dl-no-unsolicited-increase"),
    ("क्या यह कंपनी आरबीआई में पंजीकृत है?", "rbi-fpc-nbfc-registration"),
    ("मैंने ऐप से कर्ज़ लिया, अब कौन जिम्मेदार है?", "rbi-dl-lsp-accountability"),
    ("मैं आरबीआई लोकपाल को शिकायत कैसे करूँ?", "rbios-how-to-file"),
    ("लोकपाल क्या कर सकता है?", "rbios-what-it-can-address"),
    ("उपभोक्ता आयोग में शिकायत कर सकता हूँ?", "cpa-redress-forums"),
    ("मेरे अधिकार क्या हैं?", "cpa-consumer-rights"),
]


@pytest.mark.parametrize(("question", "expected_chunk"), BY_CONCEPT)
def test_each_concept_answers_with_its_own_citation(question: str, expected_chunk: str) -> None:
    """The right answer, across the whole answerable surface — not just one rule."""
    answer = asyncio.run(answer_question(question, language=Language.HINDI))
    assert answer.grounded, f"refused: {question}"
    assert is_indic(answer.answer)
    assert any(c.chunk_id == expected_chunk for c in answer.citations), (
        f"{question!r} cited {[c.chunk_id for c in answer.citations]}, wanted {expected_chunk}"
    )


@pytest.mark.parametrize(
    "question",
    [
        "कल क्रिकेट कौन जीता?",
        "मौसम कैसा रहेगा?",
        "मेरी ट्रेन कितने बजे है?",
    ],
)
def test_more_off_topic_hindi_questions_are_refused(question: str) -> None:
    """Widening coverage must not widen what counts as a legal question."""
    answer = asyncio.run(answer_question(question, language=Language.HINDI))
    assert not answer.grounded
    assert answer.citations == []


def test_off_topic_english_is_still_refused() -> None:
    answer = asyncio.run(answer_question("Who won the cricket last night?"))
    assert not answer.grounded
    assert answer.answer == COULD_NOT_CONFIRM
