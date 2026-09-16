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
