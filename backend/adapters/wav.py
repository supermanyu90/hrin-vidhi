"""Joining RIFF/WAVE clips end to end.

A text-to-speech provider that caps its input length hands back one clip per
piece of text. Keeping only the first — which is what this code used to do —
silently truncates the borrower's voice note to its opening sentences and
drops the spoken disclaimer entirely.

Concatenating WAV bytes directly does not work: each clip carries its own
44-byte header, so a naive join plays the first clip and then reads a header
as if it were audio. The fix is small but has to be exact — parse the chunk
structure, keep one format chunk, and concatenate only the sample data under a
rewritten length.

Deliberately dependency-free and format-agnostic: it works for any PCM WAV,
not just one provider's output.
"""

from __future__ import annotations

import struct

#: RIFF chunk ids are 4 ASCII bytes; sizes are little-endian uint32.
_HEADER = 12  # "RIFF" + size + "WAVE"


class WavError(ValueError):
    """The bytes are not a WAV this joiner can read."""


def _chunks(data: bytes) -> dict[str, bytes]:
    """Map chunk id -> payload. Later chunks of the same id are ignored."""
    if len(data) < _HEADER or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE stream")

    found: dict[str, bytes] = {}
    offset = _HEADER
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4].decode("ascii", errors="replace")
        (size,) = struct.unpack_from("<I", data, offset + 4)
        start = offset + 8
        end = start + size
        if end > len(data):
            # A truncated final chunk: take what is actually there rather than
            # discarding the clip. Providers occasionally pad sloppily.
            end = len(data)
        found.setdefault(chunk_id.strip(), data[start:end])
        # Chunks are word-aligned: an odd size is followed by a pad byte.
        offset = end + (size & 1)
    return found


def join_wav(clips: list[bytes]) -> bytes:
    """Concatenate PCM WAV clips into one. Raises WavError on a bad clip.

    Every clip must share the first one's format chunk — differing sample
    rates or channel counts cannot be concatenated without resampling, and
    producing garbled audio would be worse than failing.
    """
    if not clips:
        raise WavError("no clips to join")
    if len(clips) == 1:
        return clips[0]

    parsed = [_chunks(c) for c in clips]

    fmt = parsed[0].get("fmt")
    if fmt is None:
        raise WavError("first clip has no fmt chunk")
    for index, chunk in enumerate(parsed[1:], start=1):
        if chunk.get("fmt") != fmt:
            raise WavError(f"clip {index} has a different audio format")

    payload = b"".join(chunk.get("data", b"") for chunk in parsed)
    if not payload:
        raise WavError("clips contain no sample data")

    body = (
        b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(payload)) + payload
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def duration_seconds(wav: bytes) -> float | None:
    """Actual playing time from the header, or None if it cannot be read.

    Preferred over estimating from character count, which is what the caller
    had to do before and which is wrong by a factor of three on a clip that
    was silently truncated.
    """
    try:
        chunk = _chunks(wav)
    except WavError:
        return None
    fmt, data = chunk.get("fmt"), chunk.get("data")
    if not fmt or len(fmt) < 16 or data is None:
        return None
    _, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    frame = channels * (bits // 8)
    if frame <= 0 or rate <= 0:
        return None
    return round(len(data) / frame / rate, 2)
