"""F6 — the letter skeleton and the two addressee blocks.

Kept apart from the drafter so the wording can be reviewed and replaced by a
lawyer without touching the assembly logic.

Every blank is written as a self-describing `[BRACKETED INSTRUCTION]` rather
than a bare `[NAME]`, so a borrower filling it in by hand knows exactly what
goes where — that is the F6 acceptance criterion about no unlabelled
placeholder. `drafter.enumerate_placeholders` scrapes them back out for the UI.
"""

from __future__ import annotations

from backend.schemas import Addressee

# --- the borrower's own details, which only they can supply ---------------
BORROWER_BLOCK = """\
[YOUR FULL NAME]
[YOUR FULL POSTAL ADDRESS, WITH VILLAGE OR TOWN, DISTRICT AND PIN CODE]
[YOUR MOBILE NUMBER]
[YOUR EMAIL ADDRESS, IF YOU HAVE ONE]

Date: [THE DATE ON WHICH YOU SEND THIS LETTER]"""

SIGN_OFF = """\
Yours faithfully,


[SIGN HERE]
[YOUR FULL NAME]"""

ENCLOSURES = """\
Enclosures:
1. Copy of the loan agreement or sanction letter.
2. Copy of the notice received from the recovery agency, if any.
3. [LIST ANY OTHER PAPER YOU ARE ATTACHING, SUCH AS PAYMENT RECEIPTS OR CALL RECORDS]"""


def nodal_officer(lender_name: str | None) -> Addressee:
    """Step one. Every regulated lender must have this officer and publish them."""
    lender = lender_name or "[FULL NAME OF THE LENDING COMPANY, AS PRINTED ON YOUR LOAN PAPER]"
    return Addressee(
        role="Nodal Officer / Grievance Redressal Officer",
        organisation=lender,
        address_block=(
            "The Nodal Officer / Grievance Redressal Officer\n"
            f"{lender}\n"
            "[REGISTERED OFFICE ADDRESS OF THE LENDER — take it from your loan paper, "
            "or from the lender's website where the grievance officer is listed]"
        ),
        note=(
            "A regulated lender must display its grievance officer's name and contact "
            "details at its branches and on its website. Send this letter here first: "
            "the Ombudsman will not take up a complaint that has not been put to the "
            "lender."
        ),
    )


def rbi_ombudsman() -> Addressee:
    """Step two, after 30 days with no reply or an unsatisfactory one."""
    return Addressee(
        role="Reserve Bank of India — Integrated Ombudsman (RB-IOS, 2021)",
        organisation="Reserve Bank of India",
        address_block=(
            "Online:  Complaint Management System (CMS) portal at cms.rbi.org.in\n"
            "By post: Centralised Receipt and Processing Centre,\n"
            "         Reserve Bank of India\n"
            "         [POSTAL ADDRESS OF THE CENTRALISED RECEIPT AND PROCESSING CENTRE — "
            "check the current address on rbi.org.in before posting]"
        ),
        note=(
            "Only escalate here once you have complained to the lender and either had no "
            "reply for thirty days, or had a reply you are not satisfied with. Filing "
            "costs nothing and you do not need a lawyer. RBI also runs a contact centre "
            "that will help you file and track the complaint. The Ombudsman covers "
            "entities the RBI regulates — if this lender turns out not to be registered, "
            "take the complaint to the consumer forum or the police instead."
        ),
    )


ESCALATION_NOTE = """\
If I do not receive a reply within thirty days of your receiving this letter, or if the \
reply does not resolve the matter, I intend to take this complaint to the Reserve Bank of \
India under the Reserve Bank - Integrated Ombudsman Scheme, 2021. That complaint can be \
filed online through the Complaint Management System at cms.rbi.org.in, or by post to the \
Centralised Receipt and Processing Centre. I would much prefer to settle this with you \
directly."""

CITATION_CAVEAT = """\
The references given above are plain-language summaries of the position as I understand it, \
not quotations of the rules themselves. They are given so that you can identify the \
requirement I am relying on. I am not a lawyer and this letter is not a legal notice."""

SUBJECT_TEMPLATE = (
    "Complaint under your grievance redressal mechanism — "
    "loan account [YOUR LOAN ACCOUNT NUMBER, AS PRINTED ON YOUR PAPERS]{descriptor}"
)

#: Handed to the LLM when it tightens the prose. Deliberately an editor's
#: brief, not an author's: the facts are fixed before it ever runs.
DRAFTER_SYSTEM_PROMPT = """\
You are copy-editing a complaint letter that a borrower in India will send to their lender.

Absolute rules:
- Change wording only. Do not add, remove, or alter any fact, figure, date, name,
  amount, quotation, or reference.
- Never introduce a statute name, section number, rule number, or case citation.
  If one is not already in the draft, it must not appear in your output.
- Keep every numbered heading and every [SQUARE BRACKET] placeholder exactly as it is.
- Keep the letter firm, factual and polite. No threats, no rhetoric, no apology.
- Do not add a greeting, a sign-off, or any commentary of your own.
- Return the complete edited letter and nothing else.
"""
