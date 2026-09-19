"""Speech-to-text: fixture-backed mock + Sarvam AI / Bhashini real implementations."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from backend.adapters.base import AdapterError, SpeechToText
from backend.adapters.http import shared_client
from backend.config import FIXTURES_DIR, Settings
from backend.schemas import Language, Transcript

log = logging.getLogger(__name__)

_FIXTURE_PATH = FIXTURES_DIR / "transcripts" / "voice_intake.json"


@lru_cache(maxsize=1)
def _load_fixtures(path: str = str(_FIXTURE_PATH)) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if not k.startswith("_")}


class MockSpeechToText(SpeechToText):
    """Returns the canned borrower story for the declared language.

    The audio bytes are accepted and discarded — the mock exists so the demo
    runs with no key and no network, not to pretend it heard anything.

    It is reached ONLY when it is the configured adapter, never as a rescue
    for a provider that failed. The story it returns is detailed and
    convincing, which is what makes it useful offline and dangerous anywhere
    else: shown under "this is what I heard", it is the application inventing
    a borrower. The interface checks `provider` and labels this as a sample.
    """

    provider = "mock"

    async def transcribe(
        self,
        audio: bytes,
        language: Language,
        *,
        mime_type: str = "audio/webm",
    ) -> Transcript:
        fixtures = _load_fixtures()
        entry = fixtures.get(language.value)
        if entry is None:
            # Every declared language ships a fixture; English is the safety net.
            log.warning("No STT fixture for %s, falling back to English", language.value)
            entry = fixtures[Language.ENGLISH.value]
            language = Language.ENGLISH
        return Transcript(
            language=language,
            native_text=entry["native_text"],
            english_text=entry["english_text"],
            confidence=entry.get("confidence"),
            duration_seconds=entry.get("duration_seconds"),
            provider=self.provider,
        )


# Sarvam's Saarika model takes BCP-47-ish codes. Bhojpuri has no Sarvam model,
# so we send Hindi and accept the degradation rather than silently failing.
_SARVAM_CODES: dict[Language, str] = {
    Language.HINDI: "hi-IN",
    Language.MARATHI: "mr-IN",
    Language.TAMIL: "ta-IN",
    Language.TELUGU: "te-IN",
    Language.BHOJPURI: "hi-IN",
    Language.ENGLISH: "en-IN",
}


class SarvamSpeechToText(SpeechToText):
    """Sarvam AI Saarika ASR. Requires SARVAM_API_KEY."""

    provider = "sarvam"
    endpoint = "https://api.sarvam.ai/speech-to-text"
    #: saarika:v1 and v2 are retired. Verified live (September 2026): the API
    #: rejects them outright and names saaras:v3 as the replacement.
    model = "saaras:v3"

    def __init__(self, settings: Settings) -> None:
        if not settings.sarvam_api_key:
            raise AdapterError(self.provider, "SARVAM_API_KEY is not set", recoverable=False)
        self._key = settings.sarvam_api_key

    async def transcribe(
        self,
        audio: bytes,
        language: Language,
        *,
        mime_type: str = "audio/webm",
    ) -> Transcript:

        code = _SARVAM_CODES.get(language, "hi-IN")
        try:
            client = await shared_client()
            response = await client.post(
                self.endpoint,
                headers={"api-subscription-key": self._key},
                files={"file": ("audio.webm", audio, mime_type)},
                data={"language_code": code, "model": self.model},
                timeout=30.0,
            )
            if response.status_code >= 400:
                # The body names the cause — "Failed to read the file,
                # please check the audio format" is what a WebM upload
                # gets — and a bare status code hid that for a whole
                # debugging session.
                raise AdapterError(
                    self.provider,
                    f"transcription failed ({response.status_code}): {response.text[:300]}",
                )
            payload = response.json()
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(self.provider, f"transcription failed: {exc}") from exc

        return Transcript(
            language=language,
            native_text=payload.get("transcript", ""),
            english_text="",  # Translator stage fills this in.
            confidence=None,
            duration_seconds=None,
            provider=self.provider,
        )


class BhashiniSpeechToText(SpeechToText):
    """Bhashini ULCA ASR. Requires BHASHINI_API_KEY and BHASHINI_USER_ID.

    Bhashini's real flow is two-step (config call to resolve a pipeline, then a
    compute call). Only the compute call is sketched here; wire the config step
    before relying on this in production.
    """

    provider = "bhashini"
    endpoint = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

    def __init__(self, settings: Settings) -> None:
        if not settings.has_bhashini:
            raise AdapterError(
                self.provider, "BHASHINI_API_KEY / BHASHINI_USER_ID are not set", recoverable=False
            )
        self._key = settings.bhashini_api_key
        self._user_id = settings.bhashini_user_id

    async def transcribe(
        self,
        audio: bytes,
        language: Language,
        *,
        mime_type: str = "audio/webm",
    ) -> Transcript:
        import base64

        payload = {
            "pipelineTasks": [
                {"taskType": "asr", "config": {"language": {"sourceLanguage": language.value}}}
            ],
            "inputData": {"audio": [{"audioContent": base64.b64encode(audio).decode()}]},
        }
        try:
            client = await shared_client()
            response = await client.post(
                self.endpoint,
                headers={"Authorization": self._key or "", "userID": self._user_id or ""},
                json=payload,
                timeout=45.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["pipelineResponse"][0]["output"][0]["source"]
        except Exception as exc:
            raise AdapterError(self.provider, f"transcription failed: {exc}") from exc

        return Transcript(
            language=language,
            native_text=text,
            english_text="",
            confidence=None,
            duration_seconds=None,
            provider=self.provider,
        )
