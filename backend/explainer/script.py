"""F4 — turning a ComplianceReport into a spoken rights explanation.

The script is **assembled, not generated**. Each finding maps to one
hand-written sentence that already exists in all six languages
(`phrasebook.json`), and the script is the sentences whose rules actually
fired, in a fixed order, de-duplicated.

Why not generate English and translate it? Because offline there is no
translator, and the mock deliberately refuses to invent one. Assembling from
pre-translated sentences means the borrower hears real sentences in their own
language whether or not a provider key is present — and it means no model can
put a legal claim in the script that the rules engine did not make.

The disclaimer is appended in every case, spoken and on screen (§9).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from backend.adapters.base import AdapterError, TextToSpeech
from backend.adapters.translate import PHRASEBOOK
from backend.schemas import (
    ComplianceReport,
    DebtAnalysis,
    Language,
    RightsExplanation,
)

log = logging.getLogger(__name__)

PHRASEBOOK_PATH = Path(__file__).parent / "phrasebook.json"

#: Finding id -> phrasebook message. A finding with no entry contributes
#: nothing to the spoken script: it still appears in the on-screen report and
#: the letter, but we will not read out a sentence we have not written in the
#: borrower's language. Prefix match, so every R-002 threat variant is covered.
RULE_TO_MESSAGE: tuple[tuple[str, str], ...] = (
    ("R-001-recovery-hours", "recovery_hours"),
    ("R-002-threat-family-contact", "third_party"),
    ("R-002-threat-public-shaming", "third_party"),
    ("R-002-threat-job-loss", "third_party"),
    ("R-002-threat-arrest-claim", "no_arrest"),
    ("R-002-threat-violence", "no_threats"),
    ("R-002-threat-obscene-language", "no_threats"),
    ("R-002-threat-property-seizure", "no_seizure"),
    ("R-003-court-notice-misrepresentation", "fake_court_notice"),
    ("R-004-undisclosed-charges", "hidden_charges"),
    ("R-005-no-key-fact-statement", "hidden_charges"),
    ("F-001-apr-understated", "true_cost"),
    ("F-003-nbfc-registration-unverified", "check_nbfc"),
)

#: Spoken order, independent of rule severity. Money first because it is the
#: borrower's own question; the harassment rights next because they are what
#: stops tonight's phone call; what to do last so it is the closing thought.
MESSAGE_ORDER: tuple[str, ...] = (
    "true_cost",
    "hidden_charges",
    "recovery_hours",
    "third_party",
    "no_threats",
    "no_arrest",
    "fake_court_notice",
    "no_seizure",
    "check_nbfc",
)


@lru_cache(maxsize=1)
def load_phrasebook(path: str = str(PHRASEBOOK_PATH)) -> dict[str, dict[str, str]]:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def phrase(key: str, language: Language) -> str:
    """One sentence, in the borrower's language, falling back to English."""
    book = load_phrasebook()
    entry = book.get(key)
    if entry is None:
        log.error("No phrasebook entry for %r", key)
        return ""
    text = entry.get(language.value)
    if not text:
        log.warning("Phrasebook %r has no %s; using English", key, language.value)
        return entry.get("en", "")
    return text


def _format_indian(amount: float) -> str:
    """Indian digit grouping, e.g. 14169 -> '14,169'. Spoken and displayed."""
    whole = f"{int(round(abs(amount)))}"
    if len(whole) <= 3:
        return whole
    head, tail = whole[:-3], whole[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def select_messages(report: ComplianceReport) -> list[str]:
    """Which sentences this report earns, de-duplicated and in spoken order."""
    fired = {v.rule_id for v in report.violations} | {f.flag_id for f in report.flags}
    keys: set[str] = set()
    for rule_id, message in RULE_TO_MESSAGE:
        if rule_id in fired:
            keys.add(message)
    return [m for m in MESSAGE_ORDER if m in keys]


def build_script(
    report: ComplianceReport,
    language: Language,
    debt: DebtAnalysis | None = None,
) -> tuple[str, list[str]]:
    """Return (script, key_points). Both are in `language`."""
    messages = select_messages(report)

    lines: list[str] = [phrase("intro", language)]
    points: list[str] = []

    if not messages:
        lines.append(phrase("nothing_found", language))
        return " ".join(line for line in lines if line), points

    for key in messages:
        if key == "true_cost":
            # The only sentence carrying numbers, and every one of them is
            # computed. If the money math did not run we simply skip it rather
            # than speak a figure we do not have.
            if debt is None:
                continue
            text = phrase(key, language).format(
                quoted=f"{debt.quoted_rate:g}",
                actual=f"{debt.all_in_apr:.0f}",
                extra=_format_indian(debt.hidden_cost_gap),
            )
        else:
            text = phrase(key, language)
        if not text:
            continue
        if key == "recovery_hours" and phrase("rights_intro", language) not in lines:
            lines.append(phrase("rights_intro", language))
        lines.append(text)
        points.append(text)

    lines.append(phrase("what_next", language))
    points.append(phrase("what_next", language))

    return " ".join(line for line in lines if line), points


def disclaimer_for(language: Language) -> str:
    """The §9 line, in the borrower's language, from the shared phrasebook."""
    entry = PHRASEBOOK["DISCLAIMER"]
    return entry.get(language, entry[Language.ENGLISH])


async def explain_rights(
    report: ComplianceReport,
    language: Language,
    debt: DebtAnalysis | None = None,
    tts: TextToSpeech | None = None,
) -> RightsExplanation:
    """Build the script and, if a TTS adapter is given, the voice note."""
    native_script, points = build_script(report, language, debt)
    english_script, _ = build_script(report, Language.ENGLISH, debt)

    disclaimer_native = disclaimer_for(language)
    disclaimer_english = disclaimer_for(Language.ENGLISH)

    # The disclaimer is spoken as well as shown — §9 says both.
    spoken = f"{native_script} {disclaimer_native}".strip()

    audio = None
    if tts is not None:
        try:
            audio = await tts.synthesize(spoken, language)
        except AdapterError as exc:
            # A missing voice note must not cost the borrower the script.
            log.warning("Text-to-speech failed (%s); returning the script only", exc)

    return RightsExplanation(
        language=language,
        script_native=native_script,
        script_english=english_script,
        disclaimer_native=disclaimer_native,
        disclaimer_english=disclaimer_english,
        voice_note=audio,
        key_points=points,
    )


def reset_phrasebook_cache() -> None:
    load_phrasebook.cache_clear()
