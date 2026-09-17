"""Answering a question asked in the borrower's own language.

The corpus is English. The borrower is not. Before this module, a question
typed into the ask box in Hindi, Marathi, Bhojpuri, Tamil or Telugu tokenised
to nothing against an English index and was refused every single time — the
product's central promise, broken in the one place it was most visible.

**How this bridges the gap, and what it refuses to do.**

The tempting fix is to machine-translate the question into English and retrieve
with that. Offline there is no translator, and the mock deliberately refuses to
invent one (see `adapters/translate.py`), so that path cannot work in DEMO_MODE
— which is the path the demo actually runs on. The second tempting fix is to
translate the corpus, which would mean a model generating legal text in five
languages with nobody able to check it. Section 9 of the brief rules that out.

So this takes the same road `explainer/script.py` already takes: it is
**assembled from pre-translated sentences, not generated**. The phrasebook
already holds hand-written sentences for the rights that matter, in all six
languages. Each of those sentences is:

  * indexed in its own language, so a Hindi question matches Hindi words, and
  * bound to the corpus chunks that back it, so the answer still carries a real
    citation — the same chunk an English question would have retrieved.

The borrower reads a sentence a human wrote in their language; the citation
underneath points at the English source it came from. Nothing is translated at
request time, and no legal claim appears that the corpus does not support.

The cost of this honesty is coverage: only concepts with a written sentence can
be answered vernacularly. A question outside them is refused, exactly as an
off-topic English question is. That is the correct failure — narrower, not
looser. The gates are not relaxed for vernacular queries; they are the same
`RAG_MIN_SCORE` and `RAG_MIN_COVERAGE` values, applied to the same shared
scorer, which is why `bm25.BM25Index` was factored out.

Widening coverage means writing more sentences into `phrasebook.json` and
mapping them here — a translation review task, not a model task.
"""

from __future__ import annotations

import logging
from functools import cache

from backend.analysis.bm25 import BM25Index
from backend.analysis.corpus_store import CorpusChunk, load_chunks
from backend.analysis.retriever import is_indic, tokenize
from backend.config import get_settings
from backend.schemas import Language

log = logging.getLogger(__name__)

#: Phrasebook key -> the corpus chunks that back that sentence.
#:
#: This is the join that keeps a vernacular answer grounded. The sentence is
#: what the borrower reads; these chunks are what it is answerable from, and
#: they are the citations attached to the answer. Every id is checked against
#: the loaded corpus at index build time — a typo here fails loudly rather than
#: silently dropping a citation (see `_validate`).
#:
#: `true_cost` is deliberately absent: that sentence carries computed figures
#: from a specific loan and means nothing as a standalone answer. A vernacular
#: question about interest or fees lands on `hidden_charges`, which is the
#: right answer to "what was I supposed to be told, and when".
MESSAGE_TO_CHUNKS: dict[str, tuple[str, ...]] = {
    # -- what is happening to me right now -------------------------------
    "recovery_hours": ("rbi-fpc-recovery-hours", "rbi-fpc-no-harassment"),
    "third_party": ("rbi-fpc-third-party-contact", "rbi-dl-data-and-recovery-conduct"),
    "no_threats": ("rbi-fpc-no-harassment", "rbi-fpc-third-party-contact"),
    "no_arrest": ("ni-138-no-arrest-for-debt", "ni-138-what-is-not-a-138-case"),
    "no_seizure": ("rbi-fpc-possession-due-process",),
    "agent_identity": ("rbi-fpc-agent-identification",),
    # -- the paper in my hand --------------------------------------------
    "fake_court_notice": ("ni-138-court-notice-vs-private-letter", "ni-138-notice-and-time"),
    "cheque_bounce": ("ni-138-what-it-covers", "ni-138-notice-and-time"),
    # -- what the loan actually costs ------------------------------------
    "hidden_charges": ("rbi-dl-no-hidden-charges", "rbi-dl-kfs-required"),
    "apr_disclosure": ("rbi-dl-apr-disclosure", "cpa-unfair-trade-practice"),
    "penalty_charges": ("rbi-dl-penal-charges-reasonable",),
    "cooling_off": ("rbi-dl-cooling-off",),
    "credit_limit": ("rbi-dl-no-unsolicited-increase",),
    # -- who am I actually dealing with ----------------------------------
    "check_nbfc": ("rbi-fpc-nbfc-registration", "rbi-fpc-grievance-redressal"),
    "app_accountability": ("rbi-dl-lsp-accountability",),
    # -- what I can do about it ------------------------------------------
    "ombudsman_how": (
        "rbios-how-to-file",
        "rbios-when-eligible",
        "rbios-one-nation-one-ombudsman",
        "rbios-lender-must-display",
    ),
    "ombudsman_scope": ("rbios-what-it-can-address", "rbios-limits-and-exclusions"),
    "consumer_forum": ("cpa-redress-forums", "cpa-relief-available"),
    "consumer_rights": ("cpa-consumer-rights", "cpa-deficiency-in-service"),
}

#: Question words and framing that carry no retrieval signal, per script. A
#: borrower asks "can they...", "what if they...", "is it allowed" — those
#: words appear in every question and must not decide which right is returned.
#: Kept short on purpose: this is an 8-document index, and over-pruning a short
#: question leaves nothing to match on.
#: Devanagari — Hindi, Marathi and Bhojpuri share a script and much framing.
_STOPWORDS_DEVANAGARI = (
    "क्या",
    "का",
    "की",
    "के",
    "को",
    "है",
    "हैं",
    "हूँ",
    "हो",
    "होता",
    "होती",
    "सकते",
    "सकता",
    "सकती",
    "सकत",
    "सकेला",
    "और",
    "या",
    "पर",
    "मैं",
    "मेरा",
    "मेरी",
    "मुझे",
    "मुझको",
    "हम",
    "आप",
    "आपको",
    "आपका",
    "वे",
    "वो",
    "ऊ",
    "उनका",
    "यह",
    "ये",
    "कोई",
    "क्यों",
    "कैसे",
    "कब",
    "कहाँ",
    "अगर",
    "तो",
    "नहीं",
    "ना",
    "ने",
    "से",
    "में",
    "भी",
    "बहुत",
    "काय",
    "आहे",
    "आहेत",
    "मी",
    "माझा",
    "माझी",
    "मला",
    "तुम्ही",
    "तुम्हाला",
    "ते",
    "तो",
    "ती",
    "शकतात",
    "शकते",
    "करू",
    "कर",
    "का",
    "नाही",
    "आणि",
    "किंवा",
    "वर",
    "मध्ये",
)

#: Tamil.
_STOPWORDS_TAMIL = (
    "என்ன",
    "எப்படி",
    "எப்போது",
    "எங்கே",
    "ஏன்",
    "அவர்கள்",
    "அவர்",
    "நான்",
    "எனக்கு",
    "என்",
    "நீங்கள்",
    "உங்கள்",
    "உங்களை",
    "இது",
    "அது",
    "முடியுமா",
    "முடியும்",
    "முடியாது",
    "வேண்டும்",
    "இல்லை",
    "ஆம்",
    "மற்றும்",
    "அல்லது",
)

#: Telugu.
_STOPWORDS_TELUGU = (
    "ఏమి",
    "ఏమిటి",
    "ఎలా",
    "ఎప్పుడు",
    "ఎక్కడ",
    "ఎందుకు",
    "వాళ్ళు",
    "వారు",
    "నేను",
    "నాకు",
    "నా",
    "మీరు",
    "మీ",
    "మిమ్మల్ని",
    "ఇది",
    "అది",
    "వచ్చా",
    "వచ్చు",
    "లేదు",
    "కాదు",
    "ఉంది",
    "ఉన్నాయి",
    "మరియు",
    "లేదా",
    "చేయవచ్చా",
    "చేయగలరా",
)

_STOPWORDS_INDIC = frozenset(_STOPWORDS_DEVANAGARI + _STOPWORDS_TAMIL + _STOPWORDS_TELUGU)


#: Extra query vocabulary per concept, in each language — the words a borrower
#: reaches for that the answer sentence itself never uses.
#:
#: The corpus solves this with `topics`: a chunk tagged "night calls" matches a
#: question about night calls even though the circular says "permitted contact
#: hours". The phrasebook had no equivalent, so the answer text was doing
#: double duty as the query index — and closely-related concepts, which by
#: definition share their answer vocabulary, became impossible to tell apart.
#: "How do I complain to the Ombudsman" and "what can the Ombudsman do" are
#: nearly the same bag of words; what separates them is *how* and *what*, which
#: only appear in the question.
#:
#: Populated only where a concept was actually missed or confused. An empty
#: entry means the sentence's own words were enough, which is the common case.
_HINTS: dict[str, dict[str, str]] = {
    "agent_identity": {
        "en": "who is calling name identity card proof which agency authorisation",
        "hi": "कौन बोल रहा पहचान नाम पहचान पत्र सबूत कैसे पता एजेंसी",
        "mr": "कोण बोलत ओळख नाव ओळखपत्र पुरावा कसं कळेल एजन्सी",
        "bho": "के बोलत पहचान नाम पहचान पत्र सबूत कइसे पता एजेंसी",
        "ta": "யார் அழைக்கிறார் அடையாளம் பெயர் அடையாள அட்டை நிறுவனம்",
        "te": "ఎవరు ఫోన్ గుర్తింపు పేరు గుర్తింపు కార్డు ఏజెన్సీ",
    },
    "check_nbfc": {
        "en": "registered company real genuine licence check verify NBFC list",
        "hi": "पंजीकृत कंपनी असली सही लाइसेंस जाँच पता करें एनबीएफसी सूची",
        "mr": "नोंदणीकृत कंपनी खरी परवाना तपासा यादी एनबीएफसी",
        "bho": "पंजीकृत कंपनी असली लाइसेंस जांच पता करीं एनबीएफसी सूची",
        "ta": "பதிவு நிறுவனம் உண்மையான உரிமம் சரிபார்க்க பட்டியல்",
        "te": "నమోదైన కంపెనీ నిజమైన లైసెన్స్ తనిఖీ జాబితా",
    },
    "consumer_rights": {
        "en": "my rights what rights do I have entitled right to know be heard",
        "hi": "मेरे अधिकार क्या अधिकार हक़ हक जानने का अधिकार सुनवाई",
        "mr": "माझे हक्क कोणते हक्क अधिकार जाणून घेण्याचा हक्क सुनावणी",
        "bho": "हमार अधिकार कवन अधिकार हक जाने के अधिकार सुनवाई",
        "ta": "என் உரிமைகள் என்ன உரிமை அறியும் உரிமை கேட்கும் உரிமை",
        "te": "నా హక్కులు ఏ హక్కులు తెలుసుకునే హక్కు వినిపించే హక్కు",
    },
    "cooling_off": {
        "en": "return the loan give it back cancel exit early without penalty change my mind",
        "hi": "कर्ज़ वापस लौटाना रद्द करना बाहर निकलना बिना जुर्माने मन बदल गया",
        "mr": "कर्ज परत करणे रद्द करणे बाहेर पडणे दंडाशिवाय विचार बदलला",
        "bho": "कर्जा वापस लवटावल रद्द कइल बाहर निकलल बिना जुर्माना मन बदल गइल",
        "ta": "கடனைத் திருப்பித் தர ரத்து செய்ய வெளியேற அபராதம் இல்லாமல் மனம் மாறியது",
        "te": "రుణం తిరిగి ఇవ్వడం రద్దు చేయడం బయటకు రావడం జరిమానా లేకుండా మనసు మారింది",
    },
    "ombudsman_how": {
        "en": "how where to complain file apply steps process cost free fee",
        "hi": "कैसे कहाँ शिकायत दर्ज करूँ आवेदन तरीक़ा प्रक्रिया ख़र्च मुफ़्त शुल्क पैसा",
        "mr": "कसं कुठे तक्रार नोंदवू अर्ज पद्धत प्रक्रिया खर्च मोफत शुल्क पैसे",
        "bho": "कइसे कहाँ शिकायत दर्ज करीं आवेदन तरीका प्रक्रिया खरचा मुफ़्त शुल्क पइसा",
        "ta": "எப்படி எங்கே புகார் அளிக்க விண்ணப்பம் நடைமுறை செலவு இலவசம் கட்டணம்",
        "te": "ఎలా ఎక్కడ ఫిర్యాదు చేయాలి దరఖాస్తు విధానం ఖర్చు ఉచితం రుసుము డబ్బు",
    },
    "ombudsman_scope": {
        "en": "what can the ombudsman do help power decide which complaints accepted rejected",
        "hi": "लोकपाल क्या कर सकता मदद अधिकार कौन सी शिकायत लेता नहीं लेता",
        "mr": "लोकपाल काय करू शकतो मदत अधिकार कोणती तक्रार घेतो घेत नाही",
        "bho": "लोकपाल का कर सकेला मदद अधिकार कवन शिकायत लेला ना लेला",
        "ta": "ஒம்புட்ஸ்மேன் என்ன செய்ய முடியும் உதவி அதிகாரம் எந்தப் புகார் ஏற்பார்",
        "te": "అంబుడ్స్‌మన్ ఏమి చేయగలరు సహాయం అధికారం ఏ ఫిర్యాదు స్వీకరిస్తారు",
    },
}

#: Hint terms count this many times an answer-sentence term, mirroring
#: `BM25Retriever.FIELD_BOOST` on the corpus side.
_HINT_BOOST = 3


def tokenize_vernacular(text: str) -> list[str]:
    """Tokenise a vernacular string, dropping question-framing words.

    Reuses the corpus tokenizer so both indexes agree on what a token is, then
    prunes the Indic stopwords — which the English stopword list obviously
    does not cover.
    """
    return [t for t in tokenize(text) if t not in _STOPWORDS_INDIC]


def _validate(chunks: tuple[CorpusChunk, ...]) -> None:
    """Fail loudly if this module cites a chunk the corpus does not have."""
    known = {c.chunk_id for c in chunks}
    missing = {
        chunk_id for ids in MESSAGE_TO_CHUNKS.values() for chunk_id in ids if chunk_id not in known
    }
    if missing:
        raise ValueError(
            "vernacular.MESSAGE_TO_CHUNKS refers to chunk ids that are not in the "
            f"corpus: {sorted(missing)}"
        )


@cache
def _index_for(language: Language) -> tuple[BM25Index[str], dict[str, str]]:
    """The phrasebook index for one language, plus key -> sentence.

    Built per language rather than one pooled index: pooling would let a Tamil
    question score against a Hindi sentence on shared digits and punctuation,
    and would wreck the IDF that tells a distinctive word from a common one.
    """
    # Imported here rather than at module scope: `explainer.script` imports the
    # analysis layer, and a top-level import would close the cycle.
    from backend.explainer.script import phrase

    sentences: dict[str, str] = {}
    documents: list[tuple[str, list[str]]] = []

    for key in MESSAGE_TO_CHUNKS:
        text = phrase(key, language)
        if not text:
            log.warning("No %s phrasebook sentence for %r; it cannot be asked about", language, key)
            continue
        sentences[key] = text
        hint = _HINTS.get(key, {}).get(language.value, "")
        documents.append(
            (key, tokenize_vernacular(f"{(hint + ' ') * _HINT_BOOST}{text}"))
        )

    return BM25Index(documents), sentences


def search(
    question: str, language: Language, top_k: int = 2
) -> list[tuple[str, str, float, float]]:
    """Match a vernacular question to phrasebook concepts.

    Returns `(key, sentence, score, coverage)`, best first, already filtered by
    the same two gates the English path uses. An empty list means refuse.
    """
    settings = get_settings()
    index, sentences = _index_for(language)

    hits = index.search(tokenize_vernacular(question), top_k)
    return [
        (h.doc_id, sentences[h.doc_id], h.score, h.coverage)
        for h in hits
        if h.score >= settings.rag_min_score and h.coverage >= settings.rag_min_coverage
    ]


def chunks_for(key: str) -> list[CorpusChunk]:
    """The corpus chunks backing a phrasebook sentence, in declared order."""
    chunks = load_chunks()
    _validate(chunks)
    by_id = {c.chunk_id: c for c in chunks}
    return [by_id[chunk_id] for chunk_id in MESSAGE_TO_CHUNKS.get(key, ())]


def looks_vernacular(text: str) -> bool:
    """Whether this question needs the vernacular path at all."""
    return is_indic(text)


#: Script -> the language whose phrasebook to search. Devanagari carries Hindi,
#: Marathi and Bhojpuri, and no amount of character inspection separates them,
#: so Devanagari resolves to Hindi — the same fallback the speech layer already
#: uses for Bhojpuri. This is only a guess of last resort: the ask box sends
#: the borrower's chosen language, and that choice always wins.
_SCRIPT_TO_LANGUAGE: tuple[tuple[int, int, Language], ...] = (
    (0x0900, 0x097F, Language.HINDI),
    (0x0B80, 0x0BFF, Language.TAMIL),
    (0x0C00, 0x0C7F, Language.TELUGU),
)


def detect_language(text: str) -> Language:
    """Best-effort language from the script alone, for when no choice was sent."""
    counts: dict[Language, int] = {}
    for ch in text:
        for low, high, language in _SCRIPT_TO_LANGUAGE:
            if low <= ord(ch) <= high:
                counts[language] = counts.get(language, 0) + 1
                break
    if not counts:
        return Language.ENGLISH
    return max(counts.items(), key=lambda kv: kv[1])[0]


def reset_cache() -> None:
    _index_for.cache_clear()
