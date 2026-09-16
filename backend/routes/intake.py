"""F1 — voice intake, F2 — document intake, and the session that joins them.

Both intake routes work with or without a session. Without one they are pure
functions over the uploaded bytes (which is how the tests drive them); with
one they accumulate into a borrower's conversation so the later stages have
everything they need.

Nothing uploaded here is written to disk. Audio and image bytes live only for
the duration of the request that carried them.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.adapters.base import AdapterError
from backend.analysis.rules import RuleContext, evaluate
from backend.finance import analyse_debt
from backend.privacy import redact
from backend.schemas import (
    DocumentKind,
    Language,
    ParsedDocument,
    SessionState,
    Transcript,
    VoiceIntakeResponse,
)
from backend.sessions import get_store

log = logging.getLogger(__name__)

router = APIRouter(tags=["intake"])

#: A voice note long enough to describe a loan problem, bounded so an
#: accidental video upload is refused before it reaches a speech provider.
MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024

ALLOWED_AUDIO_TYPES = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp4", "audio/m4a", "audio/aac", "audio/flac",
}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "application/pdf"}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@router.post("/session", response_model=SessionState, summary="Open a borrower session")
async def open_session(language: Language = Form(default=Language.HINDI)) -> SessionState:
    return await get_store().create(language)


@router.get("/session/{session_id}", response_model=SessionState, summary="Fetch session state")
async def read_session(session_id: str) -> SessionState:
    state = await get_store().get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return state


@router.delete("/session/{session_id}", summary="Erase a session now")
async def erase_session(session_id: str) -> dict:
    """Borrower-initiated erase. Sessions also expire on their own."""
    removed = await get_store().delete(session_id)
    return {"deleted": removed}


# ---------------------------------------------------------------------------
# F1 — voice
# ---------------------------------------------------------------------------


@router.post("/intake/voice", response_model=VoiceIntakeResponse, summary="Transcribe a voice note")
async def intake_voice(
    request: Request,
    file: UploadFile = File(..., description="Recorded voice note."),
    language: Language = Form(..., description="Language the borrower selected."),
    session_id: str | None = Form(default=None),
) -> VoiceIntakeResponse:
    """Speech to native transcript, then an English working copy.

    Both are retained: the native text is what we show and later speak back to
    the borrower, the English is what the rules engine reasons over. Language
    is declared by the UI rather than detected — see the README on why
    auto-detect is deliberately absent.
    """
    adapters = request.app.state.adapters

    if file.content_type and file.content_type.split(";")[0] not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type {file.content_type!r}.",
        )

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=422, detail="The recording is empty. Please try again.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Recording is too long ({len(audio) / 1e6:.1f} MB). "
                   f"Maximum is {MAX_AUDIO_BYTES // (1024 * 1024)} MB.",
        )

    log.info("Voice intake: %d bytes, language=%s", len(audio), language.value)

    transcript = await _transcribe(adapters, audio, language, file.content_type)
    del audio  # the bytes die with the request

    state = await _attach(session_id, lambda s: s.model_copy(
        update={"transcript": transcript, "language": transcript.language}
    ))

    return VoiceIntakeResponse(
        session_id=state.session_id if state else "",
        transcript=transcript,
        next_step=(
            "Now photograph your loan paper, or the notice they sent you, and send it here."
        ),
    )


async def _transcribe(adapters, audio: bytes, language: Language, mime: str | None) -> Transcript:
    """STT then translation, each degrading to its mock rather than failing."""
    try:
        transcript = await adapters.stt.transcribe(
            audio, language, mime_type=mime or "audio/webm"
        )
    except AdapterError as exc:
        if not exc.recoverable:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        log.warning("Speech-to-text failed (%s); falling back to the mock", exc)
        from backend.adapters.stt import MockSpeechToText

        transcript = await MockSpeechToText().transcribe(audio, language)

    # A provider that already returned English leaves nothing to translate.
    if transcript.english_text.strip():
        return transcript

    if transcript.language is Language.ENGLISH:
        return transcript.model_copy(update={"english_text": transcript.native_text})

    try:
        translated = await adapters.translate.translate(
            transcript.native_text, transcript.language, Language.ENGLISH
        )
        english = translated.text
    except AdapterError as exc:
        log.warning("Translation failed (%s); keeping the native transcript only", exc)
        english = ""

    return transcript.model_copy(update={"english_text": english})


# ---------------------------------------------------------------------------
# F2 — documents
# ---------------------------------------------------------------------------


@router.post("/intake/document", response_model=ParsedDocument, summary="Parse a loan paper or notice")
async def intake_document(
    request: Request,
    file: UploadFile = File(..., description="Photograph of the loan paper or the notice."),
    hint: str | None = Form(
        default=None, description="Optional: 'loan paper' or 'notice', from the UI."
    ),
    session_id: str | None = Form(default=None),
) -> ParsedDocument:
    """Extract typed facts from an uploaded image.

    The bytes are held in memory for this request only. Nothing about the
    image content is logged, and the filename is never echoed.
    """
    settings = request.app.state.settings
    adapters = request.app.state.adapters

    if file.content_type and file.content_type.split(";")[0] not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {file.content_type!r}. Send a photo (JPEG, PNG or PDF).",
        )

    image = await file.read()
    if not image:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(image) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large ({len(image) / 1e6:.1f} MB). Maximum is "
                   f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )

    log.info(
        "Parsing document: %d bytes, type=%s, hint=%s",
        len(image),
        file.content_type or "unknown",
        redact(hint or "none"),
    )

    try:
        parsed = await adapters.docparser.parse(
            image, mime_type=file.content_type or "image/jpeg", hint=hint
        )
    except AdapterError as exc:
        if not exc.recoverable:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        log.warning("Document parser failed (%s); falling back to the mock", exc)
        from backend.adapters.docparser import MockDocumentParser

        parsed = await MockDocumentParser().parse(image, hint=hint)

    if not settings.persist_uploads:
        del image

    await _attach(session_id, lambda s: _merge_document(s, parsed))
    return parsed


def _merge_document(state: SessionState, parsed: ParsedDocument) -> SessionState:
    """Fold a parsed document into the session and re-run the analysis.

    A borrower sends the loan paper and the notice as separate photographs, so
    each merges into whichever half of the picture it fills rather than
    replacing the other.
    """
    existing = state.parsed_document
    loan = parsed.loan_facts or (existing.loan_facts if existing else None)
    notice = parsed.notice_facts or (existing.notice_facts if existing else None)

    kind = parsed.kind
    if loan is not None and notice is not None:
        kind = DocumentKind.UNKNOWN  # the session now holds both halves

    combined = ParsedDocument(
        kind=kind,
        loan_facts=loan,
        notice_facts=notice,
        provider=parsed.provider,
        parse_confidence=parsed.parse_confidence,
        unreadable_regions=parsed.unreadable_regions,
    )

    debt = None
    notes: list[str] = []
    if loan and loan.principal and loan.quoted_rate is not None and loan.tenure_months:
        debt = analyse_debt(
            principal=loan.principal,
            quoted_rate=loan.quoted_rate,
            tenure_months=loan.tenure_months,
            rate_type=loan.rate_type,
            processing_fee=loan.processing_fee or 0.0,
            other_fees_total=sum(f.amount or 0.0 for f in loan.other_fees),
        )
    elif loan is not None:
        notes.append(
            "The loan paper did not yield a principal, rate and tenure together, so the "
            "true-cost calculation could not be run."
        )

    report = evaluate(
        RuleContext(
            loan=loan,
            notice=notice,
            debt=debt,
            transcript_english=state.transcript.english_text if state.transcript else "",
            extra_notes=notes,
        )
    )

    return state.model_copy(
        update={"parsed_document": combined, "debt_analysis": debt, "compliance_report": report}
    )


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


async def _attach(session_id: str | None, mutate) -> SessionState | None:
    """Apply `mutate` to the named session, if one was supplied and is live.

    An unknown or expired id is a 404 rather than a silent no-op: a UI that
    thinks it has a session but does not would otherwise show a borrower a
    conversation the server is not accumulating.
    """
    if not session_id:
        return None
    store = get_store()
    state = await store.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    updated = mutate(state)
    await store.update(updated)
    return updated
