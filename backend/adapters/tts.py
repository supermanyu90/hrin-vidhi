"""Text-to-speech.

The mock resolves in three steps, best first:

  1. A pre-rendered fixture at ``fixtures/audio/<language>/<sha1>.wav``.
  2. macOS ``say`` — genuinely offline, genuinely speech, and it ships with
     Hindi, Tamil, Telugu and Indian-English voices. This is what makes the
     DEMO_MODE voice note *audible* rather than a placeholder.
  3. Nothing — `audio_base64` is None and the provider says "mock-unavailable".

Step 3 used to return a silent WAV of the right duration. That was wrong twice
over: a hundred-second script is ~6 MB of base64, which exceeds a serverless
response limit, and a silent file that *looks* like a voice note is a lie the
UI cannot detect. Returning nothing lets the browser speak the script itself
with the Web Speech API, which is real speech on every platform.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import shutil
import tempfile
from pathlib import Path

from backend.adapters.base import AdapterError, TextToSpeech
from backend.adapters.text import split_on_sentences
from backend.adapters.wav import WavError, duration_seconds, join_wav
from backend.config import FIXTURES_DIR, Settings
from backend.schemas import Language, SpeechAudio

log = logging.getLogger(__name__)

AUDIO_FIXTURE_DIR = FIXTURES_DIR / "audio"

# macOS bundled voices, all offline. Marathi and Bhojpuri have no macOS voice;
# Lekha reads Devanagari, so the output is accented but intelligible. That
# compromise is visible here rather than hidden.
_MACOS_VOICES: dict[Language, str] = {
    Language.HINDI: "Lekha",
    Language.MARATHI: "Lekha",
    Language.BHOJPURI: "Lekha",
    Language.TAMIL: "Vani",
    Language.TELUGU: "Geeta",
    Language.ENGLISH: "Rishi",
}

_SAMPLE_RATE = 22050


def _script_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _estimate_duration(text: str) -> float:
    """~2.6 words/second is a comfortable pace for spoken guidance."""
    words = max(1, len(text.split()))
    return round(words / 2.6, 1)


class MockTextToSpeech(TextToSpeech):
    provider = "mock"

    async def synthesize(self, text: str, language: Language) -> SpeechAudio:
        fixture = self._find_fixture(text, language)
        if fixture is not None:
            return SpeechAudio(
                language=language,
                text=text,
                audio_base64=base64.b64encode(fixture.read_bytes()).decode(),
                mime_type="audio/wav",
                duration_seconds=_estimate_duration(text),
                provider="mock-fixture",
            )

        spoken = await self._say(text, language)
        if spoken is not None:
            return SpeechAudio(
                language=language,
                text=text,
                audio_base64=base64.b64encode(spoken).decode(),
                mime_type="audio/wav",
                duration_seconds=_estimate_duration(text),
                provider="mock-say",
            )

        # No fixture and no offline voice on this machine (any Linux host,
        # so every serverless deployment). Return the script with no audio
        # rather than a silent file: the client speaks it instead.
        log.info("No offline voice available for %s; the client will speak the script",
                 language.value)
        return SpeechAudio(
            language=language,
            text=text,
            audio_base64=None,
            mime_type="audio/wav",
            duration_seconds=_estimate_duration(text),
            provider="mock-unavailable",
        )

    def _find_fixture(self, text: str, language: Language) -> Path | None:
        candidate = AUDIO_FIXTURE_DIR / language.value / f"{_script_hash(text)}.wav"
        return candidate if candidate.is_file() else None

    async def _say(self, text: str, language: Language) -> bytes | None:
        """Drive macOS `say` off the event loop. Returns None anywhere it can't."""
        say = shutil.which("say")
        if say is None:
            return None
        voice = _MACOS_VOICES.get(language, "Rishi")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "speech.wav"
            try:
                proc = await asyncio.create_subprocess_exec(
                    say,
                    "-v",
                    voice,
                    "-o",
                    str(out),
                    "--data-format=LEI16@22050",
                    text,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except (asyncio.TimeoutError, OSError) as exc:
                log.warning("macOS say failed for %s: %s", language.value, exc)
                return None
            if proc.returncode != 0 or not out.is_file():
                log.warning(
                    "macOS say voice %r unavailable (rc=%s): %s",
                    voice,
                    proc.returncode,
                    stderr.decode(errors="replace").strip()[:200],
                )
                return None
            return out.read_bytes()


_SARVAM_CODES: dict[Language, str] = {
    Language.HINDI: "hi-IN",
    Language.MARATHI: "mr-IN",
    Language.TAMIL: "ta-IN",
    Language.TELUGU: "te-IN",
    Language.BHOJPURI: "hi-IN",
    Language.ENGLISH: "en-IN",
}


class SarvamTextToSpeech(TextToSpeech):
    """Sarvam AI Bulbul TTS. Requires SARVAM_API_KEY."""

    provider = "sarvam"
    endpoint = "https://api.sarvam.ai/text-to-speech"
    max_chars = 500
    #: bulbul:v1 and v2 are retired; v3 is current and has its own speaker
    #: list, which no longer includes the "meera" this adapter used to ask for.
    #: Both were rejected outright with a 400 until this was verified live.
    model = "bulbul:v3"
    #: A calm female Hindi voice. Valid for bulbul:v3.
    speaker = "ritu"

    def __init__(self, settings: Settings) -> None:
        if not settings.sarvam_api_key:
            raise AdapterError(self.provider, "SARVAM_API_KEY is not set", recoverable=False)
        self._key = settings.sarvam_api_key

    async def synthesize(self, text: str, language: Language) -> SpeechAudio:
        import httpx

        # Bulbul caps the length of each input, so a full rights script is sent
        # as several pieces.
        #
        # Verified against bulbul:v3 (September 2026): the API synthesises
        # EVERY input and returns them already concatenated as a single clip —
        # `["नमस्ते।"]` renders 1.23s, `["नमस्ते।", <long passage>]` renders
        # 16.59s, so nothing is being dropped. We still join whatever comes
        # back rather than taking `audios[0]`, because one clip joins to
        # itself and a future version that returns one clip per input would
        # otherwise silently truncate the borrower's voice note to its first
        # sentences — losing the spoken disclaimer §9 requires.
        pieces = split_on_sentences(text, self.max_chars)
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    self.endpoint,
                    headers={"api-subscription-key": self._key},
                    json={
                        "inputs": pieces,
                        "target_language_code": _SARVAM_CODES.get(language, "hi-IN"),
                        "speaker": self.speaker,
                        "pace": 0.9,
                        "model": self.model,
                    },
                )
                response.raise_for_status()
                audios = response.json().get("audios") or []
        except Exception as exc:
            raise AdapterError(self.provider, f"synthesis failed: {exc}") from exc

        if not audios:
            raise AdapterError(self.provider, "synthesis returned no audio")
        try:
            joined = join_wav([base64.b64decode(a) for a in audios])
        except (WavError, ValueError) as exc:
            # A join we cannot do correctly must not become garbled audio in a
            # borrower's ear. Fall back to the first clip and mark it partial.
            log.warning("Could not join %d Sarvam clips (%s); sending the first only",
                        len(audios), exc)
            return SpeechAudio(
                language=language,
                text=text,
                audio_base64=audios[0],
                mime_type="audio/wav",
                duration_seconds=_estimate_duration(text),
                provider=f"{self.provider}-partial",
            )

        return SpeechAudio(
            language=language,
            text=text,
            audio_base64=base64.b64encode(joined).decode(),
            mime_type="audio/wav",
            # Measured from the audio itself. The character-count estimate was
            # describing the whole script while only a third of it played.
            duration_seconds=duration_seconds(joined) or _estimate_duration(text),
            provider=self.provider,
        )


class GTTSTextToSpeech(TextToSpeech):
    """Google Translate TTS via the `gtts` package. Needs network, needs no key."""

    provider = "gtts"

    _LANG_CODES: dict[Language, str] = {
        Language.HINDI: "hi",
        Language.MARATHI: "mr",
        Language.TAMIL: "ta",
        Language.TELUGU: "te",
        Language.BHOJPURI: "hi",
        Language.ENGLISH: "en",
    }

    async def synthesize(self, text: str, language: Language) -> SpeechAudio:
        try:
            from gtts import gTTS  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AdapterError(self.provider, "gtts is not installed", recoverable=False) from exc

        import io

        def _render() -> bytes:
            buffer = io.BytesIO()
            gTTS(text=text, lang=self._LANG_CODES.get(language, "hi")).write_to_fp(buffer)
            return buffer.getvalue()

        try:
            audio = await asyncio.to_thread(_render)
        except Exception as exc:
            raise AdapterError(self.provider, f"synthesis failed: {exc}") from exc

        return SpeechAudio(
            language=language,
            text=text,
            audio_base64=base64.b64encode(audio).decode(),
            mime_type="audio/mpeg",
            duration_seconds=_estimate_duration(text),
            provider=self.provider,
        )
