"""The provider paths — the ones no test touched until keys were about to land.

Every other test in this suite runs against mocks, which is what makes the demo
trustworthy and also what left ~1,000 lines of adapter code unexecuted. These
cover the parts that are pure logic (chunking, WAV assembly, metadata
stripping) and stub the HTTP boundary for the rest, so a provider bug shows up
here rather than in front of a borrower.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from backend.adapters.base import AdapterError
from backend.adapters.text import split_on_sentences
from backend.adapters.tts import SarvamTextToSpeech
from backend.adapters.wav import WavError, duration_seconds, join_wav
from backend.config import Settings
from backend.privacy import strip_image_metadata
from backend.schemas import Language

# ---------------------------------------------------------------------------
# WAV assembly
# ---------------------------------------------------------------------------


def make_wav(seconds: float = 1.0, rate: int = 22050, channels: int = 1, bits: int = 16) -> bytes:
    frame = channels * (bits // 8)
    data = b"\x01" * (int(rate * seconds) * frame)
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * frame, frame, bits)
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_joined_audio_runs_for_the_sum_of_its_parts() -> None:
    joined = join_wav([make_wav(1.0), make_wav(2.0), make_wav(0.5)])
    assert duration_seconds(joined) == pytest.approx(3.5, abs=0.01)


def test_joining_one_clip_returns_it_unchanged() -> None:
    clip = make_wav(1.0)
    assert join_wav([clip]) == clip


def test_clips_in_different_formats_are_refused() -> None:
    """Garbled audio in a borrower's ear is worse than an error in a log."""
    with pytest.raises(WavError):
        join_wav([make_wav(1.0, rate=22050), make_wav(1.0, rate=8000)])


def test_joining_nothing_is_an_error() -> None:
    with pytest.raises(WavError):
        join_wav([])


def test_non_wav_bytes_are_rejected() -> None:
    with pytest.raises(WavError):
        join_wav([b"this is not audio", make_wav(1.0)])


def test_duration_of_unreadable_audio_is_unknown_not_wrong() -> None:
    assert duration_seconds(b"not a wav") is None


# ---------------------------------------------------------------------------
# Splitting text for a provider's input cap
# ---------------------------------------------------------------------------

LONG_HINDI = (
    "वे आपको सुबह आठ बजे से पहले फ़ोन नहीं कर सकते। "
    "वे आपके पड़ोसियों को नहीं बता सकते। "
    "कर्ज़ न चुका पाना अपराध नहीं है। "
) * 6


@pytest.mark.parametrize("limit", [200, 500, 1000])
def test_splitting_loses_no_text(limit: int) -> None:
    pieces = split_on_sentences(LONG_HINDI, limit)
    assert "".join(pieces).replace(" ", "") == LONG_HINDI.replace(" ", "")


@pytest.mark.parametrize("limit", [200, 500, 1000])
def test_every_piece_fits_the_providers_cap(limit: int) -> None:
    assert all(len(p) <= limit for p in split_on_sentences(LONG_HINDI, limit))


def test_pieces_break_on_sentence_ends_not_mid_word() -> None:
    """Slicing every N characters cut syllables in half and the seams were audible."""
    pieces = split_on_sentences(LONG_HINDI, 300)
    assert len(pieces) > 1
    assert all(p.rstrip()[-1] in "।॥.!?" for p in pieces[:-1])


def test_a_sentence_longer_than_the_cap_is_still_split() -> None:
    """The old translator emitted an oversized piece here, which Sarvam rejects."""
    pieces = split_on_sentences("क " * 400 + "।", 200)
    assert all(len(p) <= 200 for p in pieces)
    assert len(pieces) > 1


def test_short_text_is_left_alone() -> None:
    assert split_on_sentences("एक। दो।", 500) == ["एक। दो।"]


def test_empty_text_yields_nothing() -> None:
    assert split_on_sentences("   ", 500) == []


# ---------------------------------------------------------------------------
# Sarvam speech — the bug this file exists for
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    """Stands in for httpx, returning one WAV clip per input piece."""

    captured: dict = {}

    def __init__(self, *a, **k):
        pass

    async def post(self, url, headers=None, json=None, **kw):
        import base64

        pieces = json["inputs"]
        _FakeClient.captured = {"pieces": pieces, "json": json}
        return _FakeResponse({"audios": [base64.b64encode(make_wav(1.0)).decode() for _ in pieces]})


@pytest.fixture
def sarvam(monkeypatch):
    """Stub the pooled client, not httpx.AsyncClient.

    The adapters take a shared client now rather than opening one per call,
    so patching the constructor intercepts nothing.
    """
    import backend.adapters.tts as tts_module

    fake = _FakeClient()

    async def _shared():
        return fake

    monkeypatch.setattr(tts_module, "shared_client", _shared)
    return SarvamTextToSpeech(Settings(sarvam_api_key="test-key", demo_mode=False))


def test_the_whole_script_is_spoken_not_just_the_first_clip(sarvam) -> None:
    """Whatever clips come back are joined, rather than only the first.

    Verified live against bulbul:v3: that version synthesises every input and
    returns them already concatenated as one clip, so `audios[0]` happens to
    be the whole script today — an earlier version of this comment claimed
    otherwise and was wrong.

    The guard is still worth having, because the cost of being wrong is
    silent: a future version that returns one clip per input would truncate
    the voice note to its opening sentences and drop the spoken disclaimer,
    with nothing in the response to say so. This stubs that shape explicitly.
    """
    script = LONG_HINDI  # comfortably over the 500-character cap
    audio = asyncio.run(sarvam.synthesize(script, Language.HINDI))

    pieces = _FakeClient.captured["pieces"]
    assert len(pieces) > 1, "the test script must exceed the cap to be meaningful"
    assert duration_seconds(__import__("base64").b64decode(audio.audio_base64)) == pytest.approx(
        len(pieces) * 1.0, abs=0.05
    )


def test_the_reported_duration_matches_the_audio(sarvam) -> None:
    """It used to estimate from the full script while a third of it played."""
    audio = asyncio.run(sarvam.synthesize(LONG_HINDI, Language.HINDI))
    pieces = _FakeClient.captured["pieces"]
    assert audio.duration_seconds == pytest.approx(len(pieces) * 1.0, abs=0.05)


def test_no_audio_back_is_an_error_not_a_silent_note(sarvam, monkeypatch) -> None:
    import backend.adapters.tts as tts_module

    class _Empty(_FakeClient):
        async def post(self, *a, **k):
            return _FakeResponse({"audios": []})

    empty = _Empty()

    async def _shared():
        return empty

    monkeypatch.setattr(tts_module, "shared_client", _shared)
    with pytest.raises(AdapterError):
        asyncio.run(sarvam.synthesize("नमस्ते।", Language.HINDI))


# ---------------------------------------------------------------------------
# Image metadata — a stated privacy requirement
# ---------------------------------------------------------------------------

GPS = b"GPSLatitude 26.9124N 75.7873E SERIAL-XYZ"


def jpeg_with_exif() -> bytes:
    app1 = b"\xff\xe1" + struct.pack(">H", len(GPS) + 8) + b"Exif\x00\x00" + GPS
    sof = b"\xff\xc0" + struct.pack(">H", 11) + b"\x08\x00\x10\x00\x10\x01\x01\x11\x00"
    scan = b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00" + b"\xd2\xcf\xff\xd9"
    return (
        b"\xff\xd8\xff\xe0"
        + struct.pack(">H", 16)
        + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + app1
        + sof
        + scan
    )


def png_with_metadata() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + b"\x00\x00\x00\x00"

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", b"\x00" * 13)
        + chunk(b"eXIf", GPS)
        + chunk(b"tEXt", b"Comment\x00private")
        + chunk(b"IDAT", b"\x01\x02\x03")
        + chunk(b"IEND", b"")
    )


def test_a_photo_does_not_carry_the_borrowers_location_to_a_third_party() -> None:
    assert GPS in jpeg_with_exif()
    assert GPS not in strip_image_metadata(jpeg_with_exif())


def test_stripping_keeps_the_image_itself() -> None:
    out = strip_image_metadata(jpeg_with_exif())
    assert out.startswith(b"\xff\xd8")
    assert out.endswith(b"\xff\xd9")
    assert b"\xff\xc0" in out, "the frame header must survive or the image will not decode"


def test_png_text_and_exif_chunks_go_but_pixels_stay() -> None:
    out = strip_image_metadata(png_with_metadata())
    assert GPS not in out
    assert b"private" not in out
    assert b"IDAT" in out and b"IEND" in out


def test_a_pdf_passes_through_untouched() -> None:
    pdf = b"%PDF-1.4 loan sanction letter"
    assert strip_image_metadata(pdf) == pdf


@pytest.mark.parametrize(
    "blob", [b"", b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff\xe1\x00"]
)
def test_malformed_images_do_not_crash_the_upload(blob: bytes) -> None:
    assert isinstance(strip_image_metadata(blob), bytes)


# ---------------------------------------------------------------------------
# The transcript must be the borrower's own words, or nothing
# ---------------------------------------------------------------------------


def test_a_failed_transcription_does_not_invent_a_borrower(monkeypatch) -> None:
    """The worst bug this project has had.

    Sarvam rejected the browser's WebM with a 400, the route caught it and
    fell back to MockSpeechToText, and the interface showed what came back
    under "this is what I heard" — a detailed, convincing story about a man
    named Ramesh Kumbhar with a two-wheeler loan. Every recording produced it.
    The borrower was shown invented words as their own, and the rules engine
    then reasoned about that fictional person's debt.

    A failure must reach the borrower as a failure.
    """
    from fastapi.testclient import TestClient

    from backend.adapters.base import AdapterError
    from backend.main import app

    async def always_fails(*args, **kwargs):
        raise AdapterError("sarvam", "transcription failed (400): bad audio format")

    client = TestClient(app)
    monkeypatch.setattr(app.state.adapters.stt, "transcribe", always_fails)

    response = client.post(
        "/intake/voice",
        files={"file": ("v.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={"language": "hi"},
    )
    assert response.status_code >= 400, "a failed transcription must not read as success"
    body = response.text
    assert "Ramesh" not in body and "रमेश" not in body, (
        "the fixture leaked into a real failure — this is the fabrication bug"
    )


def test_a_silent_recording_is_reported_not_analysed(monkeypatch) -> None:
    """An empty transcript is not a borrower who said nothing worth noting."""
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.schemas import Language, Transcript

    async def heard_nothing(*args, **kwargs):
        return Transcript(
            language=Language.HINDI,
            native_text="",
            english_text="",
            confidence=None,
            duration_seconds=None,
            provider="sarvam",
        )

    client = TestClient(app)
    monkeypatch.setattr(app.state.adapters.stt, "transcribe", heard_nothing)

    response = client.post(
        "/intake/voice",
        files={"file": ("v.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
        data={"language": "hi"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    # A code as well as the words: the interface shows this in the borrower's
    # language, and a failure is the worst moment to hand them English.
    assert detail["code"] == "errNoSpeech"
    assert "did not hear" in detail["message"]


# ---------------------------------------------------------------------------
# Connection reuse
# ---------------------------------------------------------------------------


def test_the_http_client_is_shared_between_calls() -> None:
    """Every Sarvam call used to open a client and close it again, paying for
    a TCP connection and a TLS handshake each time. Measured against the live
    API: 727ms per call with a fresh client, 567ms reusing one, and a voice
    note makes two of them."""
    import asyncio

    from backend.adapters.http import aclose_shared_client, shared_client

    async def check():
        first = await shared_client()
        second = await shared_client()
        assert first is second, "a new client per call defeats the pool"
        await aclose_shared_client()

    asyncio.run(check())


def test_a_client_from_a_dead_event_loop_is_replaced() -> None:
    """Tests call asyncio.run repeatedly and each run builds a new loop.
    A client held from a previous one raises on use, so it is rebuilt."""
    import asyncio

    from backend.adapters.http import shared_client

    first = asyncio.run(shared_client())
    second = asyncio.run(shared_client())
    assert first is not second, "the client outlived its event loop"


def test_no_adapter_opens_its_own_client() -> None:
    from backend.config import BACKEND_DIR

    for name in ("stt.py", "translate.py", "tts.py"):
        source = (BACKEND_DIR / "adapters" / name).read_text(encoding="utf-8")
        assert "httpx.AsyncClient(" not in source, f"{name} bypasses the shared pool"
