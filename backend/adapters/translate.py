"""Translation: fixture/passthrough mock + Sarvam Mayura real implementation.

The mock is *not* a translator. It looks the text up in the fixture corpus and
returns the paired string when it recognises it; otherwise it passes the text
through unchanged and labels the result. Pretending to translate arbitrary text
offline would put words in the borrower's mouth, which is worse than not
translating at all.
"""

from __future__ import annotations

import logging

from backend.adapters.base import AdapterError, Translator
from backend.adapters.stt import _load_fixtures
from backend.config import Settings
from backend.schemas import Language, TranslationResult

log = logging.getLogger(__name__)

# Short phrases the rights explainer needs in every language. Keyed by an
# English canonical string. This is a lookup table, not machine translation —
# each entry was written as a translation, not generated at runtime.
PHRASEBOOK: dict[str, dict[Language, str]] = {
    "DISCLAIMER": {
        Language.ENGLISH: (
            "This is information and a draft, not legal advice. Please check your lender on "
            "RBI's official list of registered NBFCs, and show this to a lawyer or a free "
            "legal aid clinic before you file."
        ),
        Language.HINDI: (
            "यह जानकारी और एक मसौदा है, कानूनी सलाह नहीं। कृपया अपने ऋणदाता को आरबीआई की "
            "पंजीकृत एनबीएफसी सूची में जाँच लें, और दाखिल करने से पहले इसे किसी वकील या "
            "मुफ़्त कानूनी सहायता केंद्र को दिखाएँ।"
        ),
        Language.MARATHI: (
            "ही माहिती आणि एक मसुदा आहे, कायदेशीर सल्ला नाही. कृपया आपल्या सावकाराची आरबीआयच्या "
            "नोंदणीकृत एनबीएफसी यादीत तपासणी करा, आणि दाखल करण्यापूर्वी हे वकिलाला किंवा "
            "मोफत कायदेशीर मदत केंद्राला दाखवा."
        ),
        Language.TAMIL: (
            "இது தகவலும் ஒரு வரைவும் மட்டுமே, சட்ட ஆலோசனை அல்ல. உங்கள் கடன் வழங்குநரை "
            "ரிசர்வ் வங்கியின் பதிவு செய்யப்பட்ட NBFC பட்டியலில் சரிபார்க்கவும், தாக்கல் "
            "செய்வதற்கு முன் இதை ஒரு வழக்கறிஞரிடமோ இலவச சட்ட உதவி மையத்திலோ காட்டவும்."
        ),
        Language.TELUGU: (
            "ఇది సమాచారం మరియు ఒక ముసాయిదా మాత్రమే, న్యాయ సలహా కాదు. దయచేసి మీ రుణదాతను "
            "ఆర్‌బీఐ నమోదిత NBFC జాబితాలో తనిఖీ చేయండి, మరియు దాఖలు చేసే ముందు దీన్ని "
            "ఒక న్యాయవాదికి లేదా ఉచిత న్యాయ సహాయ కేంద్రానికి చూపించండి."
        ),
        Language.BHOJPURI: (
            "ई जानकारी आ एगो मसौदा बा, कानूनी सलाह ना ह। कृपया आपन करजा देवे वाला के "
            "आरबीआई के पंजीकृत एनबीएफसी सूची में जाँच लीं, आ दाखिल करे से पहिले एकरा के "
            "कवनो वकील भा मुफ्त कानूनी मदद केंद्र के देखाईं।"
        ),
    }
}


class MockTranslator(Translator):
    """Fixture lookup, then phrasebook, then labelled passthrough."""

    provider = "mock"

    async def translate(
        self, text: str, source: Language, target: Language
    ) -> TranslationResult:
        if source == target:
            return TranslationResult(
                text=text, source_language=source, target_language=target, provider=self.provider
            )

        resolved = self._lookup(text, source, target)
        if resolved is None:
            log.info(
                "MockTranslator passthrough %s->%s (%d chars, no fixture match)",
                source.value,
                target.value,
                len(text),
            )
            resolved = text

        return TranslationResult(
            text=resolved,
            source_language=source,
            target_language=target,
            provider=self.provider,
        )

    def _lookup(self, text: str, source: Language, target: Language) -> str | None:
        stripped = text.strip()

        # 1. The seeded voice-intake story: native <-> english, both directions.
        fixtures = _load_fixtures()
        src = fixtures.get(source.value)
        if src and target is Language.ENGLISH and stripped == src["native_text"].strip():
            return src["english_text"]

        if target is not Language.ENGLISH:
            tgt = fixtures.get(target.value)
            if tgt and source is Language.ENGLISH and stripped == tgt["english_text"].strip():
                return tgt["native_text"]

        # 2. The phrasebook, matched against any language's rendering.
        for renderings in PHRASEBOOK.values():
            if stripped in {v.strip() for v in renderings.values()}:
                return renderings.get(target)
        return None


_SARVAM_CODES: dict[Language, str] = {
    Language.HINDI: "hi-IN",
    Language.MARATHI: "mr-IN",
    Language.TAMIL: "ta-IN",
    Language.TELUGU: "te-IN",
    Language.BHOJPURI: "hi-IN",  # no Bhojpuri model; Hindi is the closest available
    Language.ENGLISH: "en-IN",
}


class SarvamTranslator(Translator):
    """Sarvam AI Mayura translation. Requires SARVAM_API_KEY."""

    provider = "sarvam"
    endpoint = "https://api.sarvam.ai/translate"
    #: Sarvam rejects inputs above this; we chunk on sentence boundaries.
    max_chars = 1000

    def __init__(self, settings: Settings) -> None:
        if not settings.sarvam_api_key:
            raise AdapterError(self.provider, "SARVAM_API_KEY is not set", recoverable=False)
        self._key = settings.sarvam_api_key

    async def translate(
        self, text: str, source: Language, target: Language
    ) -> TranslationResult:
        import httpx

        chunks = _chunk(text, self.max_chars)
        out: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for chunk in chunks:
                    response = await client.post(
                        self.endpoint,
                        headers={"api-subscription-key": self._key},
                        json={
                            "input": chunk,
                            "source_language_code": _SARVAM_CODES.get(source, "hi-IN"),
                            "target_language_code": _SARVAM_CODES.get(target, "en-IN"),
                            "mode": "formal",
                        },
                    )
                    response.raise_for_status()
                    out.append(response.json()["translated_text"])
        except Exception as exc:
            raise AdapterError(self.provider, f"translation failed: {exc}") from exc

        return TranslationResult(
            text=" ".join(out),
            source_language=source,
            target_language=target,
            provider=self.provider,
        )


def _chunk(text: str, limit: int) -> list[str]:
    """Split on sentence ends, never mid-word, keeping each piece under `limit`."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for sentence in text.replace("। ", "।|").replace(". ", ".|").split("|"):
        if len(current) + len(sentence) > limit and current:
            chunks.append(current.strip())
            current = ""
        current += sentence + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks
