"""Redaction helpers. Loan papers are sensitive; logs must not hold identifiers.

`install_redacting_logging()` attaches a filter to the root logger so *every*
log record — including ones emitted by uvicorn or a third-party SDK — passes
through `redact()` before it is formatted. This is belt-and-braces on top of
callers redacting explicitly.
"""

from __future__ import annotations

import logging
import re

# Ordered most-specific first: an Aadhaar-looking number should not be eaten by
# the generic long-digit-run rule before it can be labelled.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Aadhaar: 12 digits, often spaced 4-4-4. Never legitimately needed in a log.
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[AADHAAR-REDACTED]"),
    # PAN: AAAAA9999A
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN-REDACTED]"),
    # Indian mobile, with or without +91 / 0 prefix.
    (re.compile(r"(?:(?<=\D)|^)(?:\+?91[ -]?|0)?[6-9]\d{9}(?=\D|$)"), "[PHONE-REDACTED]"),
    # Bank / loan account numbers: 9-18 digit runs.
    (re.compile(r"\b\d{9,18}\b"), "[ACCOUNT-REDACTED]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL-REDACTED]"),
]


def redact(text: str) -> str:
    """Replace phone numbers, account numbers and other identifiers with labels."""
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    """Scrubs the formatted message and every positional arg of a log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
            else:
                record.args = tuple(
                    redact(a) if isinstance(a, str) else a for a in record.args
                )
        return True


def install_redacting_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    redactor = RedactingFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(redactor)
    # uvicorn installs its own handlers after basicConfig; cover them too.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        for handler in logging.getLogger(name).handlers:
            handler.addFilter(redactor)


# ---------------------------------------------------------------------------
# Image metadata
# ---------------------------------------------------------------------------
#
# A borrower photographs their loan paper on a phone. That photo carries EXIF:
# GPS coordinates of where it was taken — their house — plus the device serial
# and timestamp. Before this, those bytes went to Anthropic or Google exactly
# as the camera wrote them.
#
# §"Privacy by default. Treat loan papers as sensitive" is the requirement, and
# it does not stop at the log file. Stripping is deliberately dependency-free:
# the demo path installs no image library, and a privacy control that only
# works when an optional package happens to be present is not a control.

#: JPEG APPn markers (FFE0-FFEF) carry EXIF, XMP and Flashpix; COM (FFFE) is a
#: free-text comment. None affects how the image decodes.
_JPEG_STRIP = set(range(0xE0, 0xF0)) | {0xFE}
#: PNG ancillary chunks holding metadata or text. Lower-case first letter means
#: ancillary, so dropping them is safe by the format's own rules.
_PNG_STRIP = {b"eXIf", b"tEXt", b"iTXt", b"zTXt", b"tIME"}


def strip_image_metadata(data: bytes) -> bytes:
    """Remove EXIF/XMP/comment metadata from JPEG or PNG bytes.

    Returns the input unchanged for anything it does not recognise (PDF, or a
    malformed file): the caller's size and type checks still apply, and a
    parser that guesses at unknown bytes would be the more dangerous failure.
    """
    if data[:3] == b"\xff\xd8\xff":
        return _strip_jpeg(data)
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _strip_png(data)
    return data


def _strip_jpeg(data: bytes) -> bytes:
    out = bytearray(data[:2])  # SOI
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            break  # not a marker boundary; stop rewriting and copy the rest
        marker = data[i + 1]
        if marker == 0xDA:  # Start of Scan — entropy-coded data follows
            out += data[i:]
            return bytes(out)
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if length < 2 or i + 2 + length > n:
            break
        if marker not in _JPEG_STRIP:
            out += data[i : i + 2 + length]
        i += 2 + length
    else:
        return bytes(out)
    # Fell out of a malformed segment: keep what is left rather than truncating.
    out += data[i:]
    return bytes(out)


def _strip_png(data: bytes) -> bytes:
    out = bytearray(data[:8])  # signature
    i = 8
    n = len(data)
    while i + 8 <= n:
        length = int.from_bytes(data[i : i + 4], "big")
        chunk_type = data[i + 4 : i + 8]
        end = i + 12 + length  # length + type + payload + CRC
        if end > n:
            break
        if chunk_type not in _PNG_STRIP:
            out += data[i:end]
        i = end
        if chunk_type == b"IEND":
            return bytes(out)
    out += data[i:]
    return bytes(out)
