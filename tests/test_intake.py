"""F1 voice intake, F2 document intake, and the session that joins them."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from backend.schemas import Language
from backend.sessions import SessionStore, reset_store


@pytest.fixture(autouse=True)
def clean_store():
    """Every test gets a fresh store, so TTL and cap tests cannot leak."""
    reset_store()
    yield
    reset_store()


@pytest.fixture
def client() -> TestClient:
    from backend.main import app

    return TestClient(app)


def open_session(client: TestClient, language: str = "mr") -> str:
    response = client.post("/session", data={"language": language})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def send_voice(client: TestClient, language: str = "mr", session_id: str | None = None,
               content: bytes = b"fake-audio-bytes",
               content_type: str = "audio/webm") -> dict:
    data = {"language": language}
    if session_id:
        data["session_id"] = session_id
    response = client.post(
        "/intake/voice",
        files={"file": ("note.webm", io.BytesIO(content), content_type)},
        data=data,
    )
    assert response.status_code == 200, response.text
    return response.json()


def send_document(client: TestClient, hint: str, session_id: str | None = None) -> dict:
    data = {"hint": hint}
    if session_id:
        data["session_id"] = session_id
    response = client.post(
        "/intake/document",
        files={"file": ("doc.jpg", io.BytesIO(b"fake-jpeg-bytes"), "image/jpeg")},
        data=data,
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# F1 — /intake/voice
# ---------------------------------------------------------------------------


def test_f1_acceptance_marathi_returns_native_and_english(client: TestClient) -> None:
    """Spec F1: selecting Marathi and recording returns a Marathi transcript
    plus its English translation, and advances the flow."""
    body = send_voice(client, language="mr")
    transcript = body["transcript"]

    assert transcript["language"] == "mr"
    assert transcript["native_text"].strip()
    assert transcript["english_text"].strip()
    assert transcript["native_text"] != transcript["english_text"]
    # Devanagari must actually be present — an English string tagged "mr" would pass a
    # weaker check and show the borrower the wrong thing.
    assert any("ऀ" <= ch <= "ॿ" for ch in transcript["native_text"])
    assert body["next_step"].strip()


@pytest.mark.parametrize("language", [lang.value for lang in Language])
def test_every_selectable_language_transcribes(client: TestClient, language: str) -> None:
    body = send_voice(client, language=language)
    assert body["transcript"]["language"] == language
    assert body["transcript"]["native_text"].strip()
    assert body["transcript"]["english_text"].strip()


def test_the_five_indian_languages_the_spec_names_are_all_supported() -> None:
    for code in ("hi", "mr", "ta", "te", "bho"):
        assert code in {lang.value for lang in Language}


def test_english_intake_needs_no_translation(client: TestClient) -> None:
    body = send_voice(client, language="en")
    transcript = body["transcript"]
    assert transcript["native_text"] == transcript["english_text"]


def test_voice_rejects_an_empty_recording(client: TestClient) -> None:
    response = client.post(
        "/intake/voice",
        files={"file": ("note.webm", io.BytesIO(b""), "audio/webm")},
        data={"language": "mr"},
    )
    assert response.status_code == 422


def test_voice_rejects_a_non_audio_upload(client: TestClient) -> None:
    response = client.post(
        "/intake/voice",
        files={"file": ("photo.jpg", io.BytesIO(b"x" * 50), "image/jpeg")},
        data={"language": "mr"},
    )
    assert response.status_code == 415


def test_voice_rejects_an_oversized_recording(client: TestClient) -> None:
    from backend.routes.intake import MAX_AUDIO_BYTES

    response = client.post(
        "/intake/voice",
        files={"file": ("note.webm", io.BytesIO(b"\x00" * (MAX_AUDIO_BYTES + 1)), "audio/webm")},
        data={"language": "mr"},
    )
    assert response.status_code == 413


def test_voice_accepts_a_codec_suffixed_content_type(client: TestClient) -> None:
    """Chrome sends 'audio/webm;codecs=opus' — the suffix must not 415."""
    body = send_voice(client, content_type="audio/webm;codecs=opus")
    assert body["transcript"]["native_text"]


def test_voice_requires_a_language(client: TestClient) -> None:
    response = client.post(
        "/intake/voice", files={"file": ("n.webm", io.BytesIO(b"aa"), "audio/webm")}
    )
    assert response.status_code == 422


def test_voice_rejects_an_unknown_language(client: TestClient) -> None:
    response = client.post(
        "/intake/voice",
        files={"file": ("n.webm", io.BytesIO(b"aa"), "audio/webm")},
        data={"language": "fr"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# F2 — /intake/document
# ---------------------------------------------------------------------------


def test_uploading_a_loan_paper_returns_populated_loan_facts(client: TestClient) -> None:
    body = send_document(client, hint="loan paper")
    assert body["kind"] == "loan_paper"
    facts = body["loan_facts"]
    assert facts["rate_type"] == "flat"
    assert facts["quoted_rate"] == 12.0
    assert facts["nbfc_registration_status"] == "absent"
    assert body["notice_facts"] is None


def test_uploading_a_notice_returns_populated_notice_facts(client: TestClient) -> None:
    body = send_document(client, hint="harassment notice")
    assert body["kind"] == "notice"
    notice = body["notice_facts"]
    assert notice["sender_is_court"] is False
    assert any(c["time_24h"] == "06:30" for c in notice["contact_times_mentioned"])


def test_unreadable_regions_are_reported_rather_than_hidden(client: TestClient) -> None:
    """A parser that silently drops what it could not read would be worse than useless."""
    assert send_document(client, hint="loan paper")["unreadable_regions"]


def test_document_rejects_empty_wrong_type_and_oversized(client: TestClient) -> None:
    from backend.routes.intake import MAX_IMAGE_BYTES

    empty = client.post(
        "/intake/document", files={"file": ("x.jpg", io.BytesIO(b""), "image/jpeg")}
    )
    assert empty.status_code == 422

    wrong = client.post(
        "/intake/document", files={"file": ("clip.mp4", io.BytesIO(b"\x00" * 100), "video/mp4")}
    )
    assert wrong.status_code == 415

    big = client.post(
        "/intake/document",
        files={"file": ("big.jpg", io.BytesIO(b"\x00" * (MAX_IMAGE_BYTES + 1)), "image/jpeg")},
    )
    assert big.status_code == 413


def test_document_requires_a_file(client: TestClient) -> None:
    assert client.post("/intake/document").status_code == 422


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_session_accumulates_the_whole_conversation(client: TestClient) -> None:
    """The end-to-end flow: voice, then both documents, all in one session."""
    session_id = open_session(client, "mr")

    send_voice(client, language="mr", session_id=session_id)
    send_document(client, hint="loan paper", session_id=session_id)
    send_document(client, hint="notice", session_id=session_id)

    state = client.get(f"/session/{session_id}").json()

    assert state["transcript"]["language"] == "mr"
    # Both documents merged rather than one replacing the other.
    assert state["parsed_document"]["loan_facts"] is not None
    assert state["parsed_document"]["notice_facts"] is not None

    assert state["debt_analysis"]["effective_apr"] > 21
    report = state["compliance_report"]
    assert {"R-001-recovery-hours", "R-002-threat-family-contact"} <= {
        v["rule_id"] for v in report["violations"]
    }
    assert report["unconfirmed"] == []


def test_the_second_document_does_not_erase_the_first(client: TestClient) -> None:
    session_id = open_session(client)
    send_document(client, hint="notice", session_id=session_id)
    send_document(client, hint="loan paper", session_id=session_id)

    parsed = client.get(f"/session/{session_id}").json()["parsed_document"]
    assert parsed["loan_facts"] is not None
    assert parsed["notice_facts"] is not None


def test_analysis_reruns_as_each_document_arrives(client: TestClient) -> None:
    session_id = open_session(client)

    send_document(client, hint="notice", session_id=session_id)
    after_notice = client.get(f"/session/{session_id}").json()
    assert after_notice["debt_analysis"] is None  # no loan paper yet
    assert after_notice["compliance_report"]["violations"]

    send_document(client, hint="loan paper", session_id=session_id)
    after_loan = client.get(f"/session/{session_id}").json()
    assert after_loan["debt_analysis"] is not None
    assert len(after_loan["compliance_report"]["violations"]) > len(
        after_notice["compliance_report"]["violations"]
    )


def test_session_defaults_to_not_persisting_anything(client: TestClient) -> None:
    """The privacy default must be off unless the borrower opts in."""
    session_id = open_session(client)
    assert client.get(f"/session/{session_id}").json()["persist_opt_in"] is False


def test_an_unknown_session_is_a_404_not_a_silent_noop(client: TestClient) -> None:
    """A UI that thinks it has a session but does not must find out."""
    assert client.get("/session/does-not-exist").status_code == 404

    response = client.post(
        "/intake/voice",
        files={"file": ("n.webm", io.BytesIO(b"aa"), "audio/webm")},
        data={"language": "mr", "session_id": "does-not-exist"},
    )
    assert response.status_code == 404


def test_a_borrower_can_erase_their_session(client: TestClient) -> None:
    session_id = open_session(client)
    send_voice(client, session_id=session_id)

    assert client.delete(f"/session/{session_id}").json() == {"deleted": True}
    assert client.get(f"/session/{session_id}").status_code == 404
    assert client.delete(f"/session/{session_id}").json() == {"deleted": False}


def test_session_ids_are_unguessable(client: TestClient) -> None:
    """A session id is a bearer token for a borrower's loan documents."""
    ids = {open_session(client) for _ in range(5)}
    assert len(ids) == 5
    assert all(len(i) >= 20 for i in ids)


def test_work_without_a_session_still_returns_results(client: TestClient) -> None:
    """Stateless use must keep working — the visualizer relies on it."""
    assert send_voice(client)["transcript"]["native_text"]
    assert send_document(client, hint="loan paper")["loan_facts"]


# ---------------------------------------------------------------------------
# Session store internals
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_expired_sessions_are_dropped() -> None:
    store = SessionStore(ttl_seconds=0)
    state = await store.create(Language.HINDI)
    assert await store.get(state.session_id) is None


@pytest.mark.anyio
async def test_store_caps_how_many_sessions_it_holds() -> None:
    store = SessionStore(ttl_seconds=3600, max_sessions=3)
    created = [await store.create(Language.HINDI) for _ in range(5)]
    stats = await store.stats()
    assert stats["active_sessions"] <= 3
    # The most recent survive; the oldest were evicted.
    assert await store.get(created[-1].session_id) is not None


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# F4 / F6 over HTTP, driven through a session (the demo path)
# ---------------------------------------------------------------------------


def _full_session(client: TestClient) -> str:
    session_id = open_session(client, "mr")
    send_voice(client, language="mr", session_id=session_id)
    send_document(client, hint="loan paper", session_id=session_id)
    send_document(client, hint="notice", session_id=session_id)
    return session_id


def test_rights_explainer_over_a_session(client: TestClient) -> None:
    session_id = _full_session(client)
    response = client.post("/explain/rights", json={"session_id": session_id})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["language"] == "mr"
    assert body["script_native"].strip()
    assert body["disclaimer_native"].strip()
    assert body["key_points"]
    assert body["voice_note"]["audio_base64"]


def test_the_session_does_not_retain_the_audio_payload(client: TestClient) -> None:
    """~4 MB per voice note times the session cap would be gigabytes."""
    session_id = _full_session(client)
    client.post("/explain/rights", json={"session_id": session_id})
    stored = client.get(f"/session/{session_id}").json()["rights_explanation"]
    assert stored["script_native"].strip()
    assert stored["voice_note"] is None


def test_rights_can_skip_synthesis(client: TestClient) -> None:
    session_id = _full_session(client)
    body = client.post(
        "/explain/rights", json={"session_id": session_id, "speak": False}
    ).json()
    assert body["voice_note"] is None
    assert body["script_native"].strip()


def test_grievance_draft_over_a_session(client: TestClient) -> None:
    session_id = _full_session(client)
    response = client.post("/grievance/draft", json={"session_id": session_id})
    assert response.status_code == 200, response.text
    letter = response.json()

    assert letter["violations_cited"]
    assert letter["summary_native"].strip()
    assert letter["placeholders"]
    assert [a["role"] for a in letter["addressees"]]
    assert "cms.rbi.org.in" in letter["body_english"]


def test_the_letter_downloads_as_a_text_file(client: TestClient) -> None:
    session_id = _full_session(client)
    client.post("/grievance/draft", json={"session_id": session_id})

    response = client.get(f"/grievance/{session_id}/download")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert ".txt" in response.headers["content-disposition"]
    assert "WHERE TO SEND THIS" in response.text
    assert "Nodal Officer" in response.text
    assert "Ombudsman" in response.text
    # §9: the disclaimer travels with the file.
    assert "not legal advice" in response.text


def test_download_before_drafting_is_a_clear_error(client: TestClient) -> None:
    session_id = _full_session(client)
    response = client.get(f"/grievance/{session_id}/download")
    assert response.status_code == 409
    assert "drafted" in response.json()["detail"]


def test_output_routes_need_an_analysed_session(client: TestClient) -> None:
    empty = open_session(client)
    assert client.post("/explain/rights", json={"session_id": empty}).status_code == 409
    assert client.post("/grievance/draft", json={"session_id": empty}).status_code == 409


def test_output_routes_reject_an_unknown_session(client: TestClient) -> None:
    assert client.post("/explain/rights", json={"session_id": "nope"}).status_code == 404
    assert client.post("/grievance/draft", json={"session_id": "nope"}).status_code == 404
    assert client.get("/grievance/nope/download").status_code == 404


def test_output_routes_work_without_a_session(client: TestClient) -> None:
    """Each stage must be demoable on its own."""
    notice = send_document(client, hint="notice")["notice_facts"]
    report = client.post("/analysis/compliance", json={"notice_facts": notice}).json()

    rights = client.post(
        "/explain/rights", json={"report": report, "language": "ta", "speak": False}
    )
    assert rights.status_code == 200
    assert rights.json()["language"] == "ta"

    letter = client.post(
        "/grievance/draft",
        json={"payload": {"report": report, "notice_facts": notice, "language": "ta"}},
    )
    assert letter.status_code == 200
    assert letter.json()["violations_cited"]


def test_output_routes_need_something_to_work_from(client: TestClient) -> None:
    assert client.post("/explain/rights", json={}).status_code == 422
    assert client.post("/grievance/draft", json={}).status_code == 422


@pytest.mark.anyio
async def test_reading_stats_never_discards_a_live_session() -> None:
    """`stats` once enforced the session cap, so merely reading it deleted a
    borrower's conversation. Expiry is safe from a read; eviction is not."""
    store = SessionStore(ttl_seconds=3600, max_sessions=3)
    ids = [(await store.create(Language.HINDI)).session_id for _ in range(3)]

    for _ in range(5):
        await store.stats()

    for session_id in ids:
        assert await store.get(session_id) is not None, session_id
    assert (await store.stats())["active_sessions"] == 3


@pytest.mark.anyio
async def test_the_cap_still_holds_when_creating() -> None:
    store = SessionStore(ttl_seconds=3600, max_sessions=3)
    created = [await store.create(Language.HINDI) for _ in range(6)]
    assert (await store.stats())["active_sessions"] == 3
    # The most recent survive; the oldest were evicted to make room.
    assert await store.get(created[-1].session_id) is not None
    assert await store.get(created[0].session_id) is None
