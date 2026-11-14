"""Focused validation of the legacy robot transcript adapter contract."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from uuid import uuid4

import pytest

import robot_client
from transport import ProtocolViolationError, TransportErrorCode


def response(*, ok: bool, error: object = None, result: object = None) -> SimpleNamespace:
    return SimpleNamespace(ok=ok, error=error, result=result)


class ScriptedRobotClient:
    """Return typed response-shaped values while recording adapter requests."""

    def __init__(self, *responses: SimpleNamespace) -> None:
        self.config = SimpleNamespace(
            connect_timeout=2.5,
            read_timeout=3.0,
            request_timeout=4.0,
        )
        self.connection_generation = 7
        self.is_connected = True
        self.responses = deque(responses)
        self.calls: list[tuple[str, object, dict[str, object]]] = []

    async def request(
        self,
        command: str,
        arguments: object = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        self.calls.append((command, arguments, kwargs))
        if not self.responses:
            raise AssertionError(f"Unexpected request for {command}")
        return self.responses.popleft()


async def install_client(monkeypatch: pytest.MonkeyPatch, client: ScriptedRobotClient) -> None:
    async def get_client() -> ScriptedRobotClient:
        return client

    monkeypatch.setattr(robot_client, "get_robot_client", get_client)


@pytest.mark.parametrize(
    "error",
    [
        None,
        {"code": 3, "message": "Bad code type"},
        {"code": TransportErrorCode.INTERNAL_ERROR.value, "message": None},
        {"code": TransportErrorCode.INTERNAL_ERROR.value},
    ],
    ids=["missing-error", "non-string-code", "non-string-message", "missing-message"],
)
async def test_failed_poll_requires_a_well_formed_error_object(monkeypatch, error):
    client = ScriptedRobotClient(response(ok=False, error=error))
    await install_client(monkeypatch, client)

    with pytest.raises(ProtocolViolationError, match="Invalid transcript error response"):
        await robot_client.TranscriptPoller().next_transcript()

    assert [call[0] for call in client.calls] == ["poll_transcript"]


async def test_success_response_cannot_contain_an_error_object(monkeypatch):
    client = ScriptedRobotClient(
        response(ok=True, error={"code": "internal_error", "message": "invalid"}, result=None)
    )
    await install_client(monkeypatch, client)

    with pytest.raises(ProtocolViolationError, match="Unexpected error on successful transcript"):
        await robot_client.TranscriptPoller().next_transcript()


async def test_unsupported_poll_with_result_does_not_enter_legacy_mode(monkeypatch):
    client = ScriptedRobotClient(
        response(
            ok=False,
            error={
                "code": TransportErrorCode.UNSUPPORTED_COMMAND.value,
                "message": "poll_transcript is unavailable",
            },
            result={"id": str(uuid4()), "text": "Must not be consumed as legacy"},
        )
    )
    await install_client(monkeypatch, client)

    with pytest.raises(ProtocolViolationError) as failure:
        await robot_client.TranscriptPoller().next_transcript()

    assert failure.value.code == TransportErrorCode.MALFORMED_REQUEST
    assert [call[0] for call in client.calls] == ["poll_transcript"]


@pytest.mark.parametrize(
    "record",
    [
        "not a record",
        {},
        {"id": "not-a-uuid", "text": "Hello"},
        {"id": str(uuid4()), "text": ""},
        {"id": str(uuid4()), "text": " \n\t "},
        {"id": str(uuid4()), "text": 12},
        {"id": str(uuid4()), "text": "Hello", "extra": True},
    ],
    ids=[
        "non-dict",
        "wrong-keys",
        "bad-id",
        "blank-text",
        "whitespace-text",
        "non-string-text",
        "extra-key",
    ],
)
async def test_malformed_transcript_record_is_rejected(monkeypatch, record):
    client = ScriptedRobotClient(response(ok=True, result=record))
    await install_client(monkeypatch, client)

    with pytest.raises(ProtocolViolationError):
        await robot_client.TranscriptPoller().next_transcript()

    assert [call[0] for call in client.calls] == ["poll_transcript"]


async def test_legacy_poll_rejects_non_string_text(monkeypatch):
    client = ScriptedRobotClient(
        response(
            ok=False,
            error={
                "code": TransportErrorCode.UNSUPPORTED_COMMAND.value,
                "message": "poll_transcript is unavailable",
            },
            result=None,
        ),
        response(ok=True, result=17),
    )
    await install_client(monkeypatch, client)

    with pytest.raises(ProtocolViolationError, match="Invalid legacy transcript"):
        await robot_client.TranscriptPoller().next_transcript()

    assert [call[0] for call in client.calls] == ["poll_transcript", "check_for_response"]
    assert client.calls[1][2]["expected_connection_generation"] == 7


async def test_poll_uses_connect_budget_and_ack_uses_default_response_timeout(monkeypatch):
    transcript_id = str(uuid4())
    client = ScriptedRobotClient(
        response(ok=True, result={"id": transcript_id, "text": "Hello"}),
        response(ok=True, result=True),
    )
    await install_client(monkeypatch, client)

    assert await robot_client.TranscriptPoller().next_transcript() == "Hello"

    assert client.calls[0][0] == "poll_transcript"
    assert client.calls[0][2]["connect_budget"] == client.config.connect_timeout
    assert client.calls[1][0] == "ack_transcript"
    assert client.calls[1][2]["expected_connection_generation"] == client.connection_generation
    assert "timeout" not in client.calls[1][2]
