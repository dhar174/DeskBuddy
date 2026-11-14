"""Authenticated loopback contracts for the legacy transcript adapters."""

from __future__ import annotations

import json
import logging
import sys
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import ModuleType

import pytest

import buddy1
from transport import RobotClient, RobotServer, TransportConfig
from transport.auth import sign_response
from transport.protocol import RobotResponse
from transport.transcripts import MAX_PENDING_TRANSCRIPTS, TranscriptQueue, is_transcript_id
from windows import buddy1 as windows_buddy1
from windows import buddy2 as windows_buddy2


@pytest.fixture(
    params=[buddy1, windows_buddy1, windows_buddy2],
    ids=["main-adapter", "windows-buddy1-adapter", "windows-buddy2-adapter"],
)
def adapter_module(request: pytest.FixtureRequest) -> ModuleType:
    return request.param


@asynccontextmanager
async def _connected_client(
    server: RobotServer,
    server_config: TransportConfig,
) -> AsyncIterator[RobotClient]:
    client_config = server_config.with_overrides(port=server.port)
    async with RobotClient(client_config) as client:
        yield client


@asynccontextmanager
async def _running_client(
    adapter: ModuleType,
    server_config: TransportConfig,
) -> AsyncIterator[RobotClient]:
    server = adapter.create_robot_server(server_config)
    await server.start()
    try:
        async with _connected_client(server, server_config) as client:
            yield client
    finally:
        await server.stop()


async def _poll(client: RobotClient) -> dict[str, str] | None:
    response = await client.request("poll_transcript", {})
    assert response.ok, response.error
    assert response.result is None or isinstance(response.result, dict)
    return response.result


async def _ack(client: RobotClient, transcript_id: str) -> bool:
    response = await client.request("ack_transcript", {"id": transcript_id})
    assert response.ok, response.error
    assert isinstance(response.result, bool)
    return response.result


@pytest.mark.asyncio
async def test_authenticated_transcript_queue_round_trip(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    text = 'Could you describe the café? It says "你好" 😀.'

    async with _running_client(adapter_module, ephemeral_server_config) as client:
        await adapter_module.MyFunctions.send_response(text)
        item = await _poll(client)
        assert item is not None
        assert set(item) == {"id", "text"}
        assert item["text"] == text
        transcript_id = item["id"]
        assert str(uuid.UUID(transcript_id)) == transcript_id

        assert await _ack(client, transcript_id) is True
        assert await _poll(client) is None


@pytest.mark.asyncio
async def test_legacy_polling_keeps_latest_text_after_shadow_queue_overflow(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    async with _running_client(adapter_module, ephemeral_server_config) as client:
        for index in range(MAX_PENDING_TRANSCRIPTS + 8):
            await adapter_module.MyFunctions.send_response(f"legacy turn {index}")

        response = await client.request("check_for_response", {})

        assert response.ok, response.error
        assert response.result == f"legacy turn {MAX_PENDING_TRANSCRIPTS + 7}"
        assert adapter_module.transcript_queue.polling_mode == "legacy"
        assert adapter_module.transcript_queue.poll() is None

        for index in range(MAX_PENDING_TRANSCRIPTS + 8):
            await adapter_module.MyFunctions.send_response(f"later legacy turn {index}")
        later = await client.request("check_for_response", {})
        assert later.ok, later.error
        assert later.result == f"later legacy turn {MAX_PENDING_TRANSCRIPTS + 7}"
        assert adapter_module.transcript_queue.poll() is None


@pytest.mark.asyncio
async def test_id_poll_after_legacy_mode_returns_restart_hint(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    async with _running_client(adapter_module, ephemeral_server_config) as client:
        legacy = await client.request("check_for_response", {})
        assert legacy.ok, legacy.error

        response = await client.request("poll_transcript", {})

        assert response.ok is False
        assert response.error is not None
        assert response.error.get("code") == "invalid_arguments"
        assert "restart the robot server" in response.error.get("message", "")


@pytest.mark.asyncio
async def test_invalid_ack_request_id_does_not_select_mode_or_remove_pending_text(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    async with _running_client(adapter_module, ephemeral_server_config) as client:
        await adapter_module.MyFunctions.send_response("pending legacy text")
        queue = adapter_module.transcript_queue
        pending = queue.poll()
        assert pending is not None
        assert queue.polling_mode is None

        response = await client.request(
            "ack_transcript",
            {"id": pending["id"]},
            request_id="noncanonical-request-id",
        )

        assert response.ok is False
        assert response.error is not None
        assert response.error.get("code") == "invalid_arguments"
        assert queue.polling_mode is None
        assert queue.poll() == pending

        malformed_ack = await client.request("ack_transcript", {"id": "not-a-uuid"})
        assert malformed_ack.ok is False
        assert malformed_ack.error is not None
        assert malformed_ack.error.get("code") == "invalid_arguments"
        assert queue.poll() == pending

        legacy = await client.request("check_for_response", {})
        assert legacy.ok, legacy.error
        assert legacy.result == "pending legacy text"
        assert queue.polling_mode == "legacy"
        assert queue.poll() is None


@pytest.mark.asyncio
async def test_legacy_text_fitting_only_legacy_frame_is_still_delivered(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    max_payload_size = 1024
    config = ephemeral_server_config.with_overrides(max_payload_size=max_payload_size)
    probe = TranscriptQueue(max_payload_size=max_payload_size, secret=config.secret)
    transcript_id = str(uuid.UUID(int=1))
    text = next(
        "x" * length
        for length in range(max_payload_size)
        if probe._legacy_response_size("x" * length, config.secret) <= max_payload_size
        and probe._poll_response_size(transcript_id, "x" * length, config.secret) > max_payload_size
    )

    async with _running_client(adapter_module, config) as client:
        await adapter_module.MyFunctions.send_response(text)
        assert adapter_module.transcript_queue.poll() is None

        response = await client.request("check_for_response", {})

        assert response.ok, response.error
        assert response.result == text


@pytest.mark.asyncio
async def test_oversized_legacy_text_is_skipped_without_printing_or_logging_content(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_marker = "PRIVATE_OVERSIZED_LEGACY_SENTINEL"
    text = private_marker + ("x" * 2048)
    config = ephemeral_server_config.with_overrides(max_payload_size=1024)
    caplog.set_level(logging.WARNING, logger="transport.transcripts")

    async with _running_client(adapter_module, config) as client:
        await adapter_module.MyFunctions.send_response(text)
        response = await client.request("check_for_response", {})

        assert response.ok, response.error
        assert response.result == "No transcription"

    assert private_marker not in caplog.text
    assert private_marker not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_invalid_text_is_rejected_without_poisoning_prestart_legacy_state(
    adapter_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    queue = TranscriptQueue()
    monkeypatch.setattr(adapter_module, "transcript_queue", queue)
    monkeypatch.setattr(adapter_module, "lastTranscription", "previous valid")
    monkeypatch.setattr(adapter_module, "priorTranscription", "older")
    caplog.set_level(logging.WARNING, logger="transport.transcripts")

    await adapter_module.MyFunctions.send_response(" \t")
    await adapter_module.MyFunctions.send_response(None)
    assert adapter_module.lastTranscription == "previous valid"
    assert adapter_module.priorTranscription == "older"

    await adapter_module.MyFunctions.send_response("valid before startup")

    assert adapter_module.lastTranscription == "valid before startup"
    assert adapter_module.priorTranscription == "previous valid"
    assert queue.poll() is None
    assert queue.polling_mode is None
    assert "valid before startup" not in caplog.text


@pytest.mark.parametrize(
    "secret",
    ["transcript-boundary-test-secret", ""],
    ids=["signed", "unsigned"],
)
def test_queue_accepts_exact_signed_or_unsigned_response_limit(secret: str) -> None:
    transcript_id = str(uuid.UUID(int=1))
    text = 'A café transcript with "JSON" and 😀.'
    expected_response = RobotResponse.success(
        request_id="00000000-0000-0000-0000-000000000000",
        result={"id": transcript_id, "text": text},
        timestamp=-sys.float_info.max,
    )
    if secret:
        sign_response(secret, expected_response)
    exact_size = len(expected_response.to_json().encode("utf-8"))

    exact = TranscriptQueue(max_payload_size=exact_size, secret=secret)
    assert exact.enqueue(text)
    assert exact.poll() is not None

    one_byte_short = TranscriptQueue(max_payload_size=exact_size - 1, secret=secret)
    assert one_byte_short.enqueue(text) is False
    assert one_byte_short.poll() is None


def test_queue_fast_rejects_text_longer_than_frame_before_json_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = TranscriptQueue(max_payload_size=64, secret="fast-reject-secret")

    def sizing_must_not_run(*args: object, **kwargs: object) -> int:
        pytest.fail("oversized text reached response serialization")

    monkeypatch.setattr(TranscriptQueue, "_poll_response_size", sizing_must_not_run)

    assert queue.enqueue("x" * 65) is False


@pytest.mark.parametrize(
    ("max_payload_size", "error_type"),
    [
        (True, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
    ],
)
def test_queue_constructor_rejects_invalid_frame_limits(
    max_payload_size: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        TranscriptQueue(max_payload_size=max_payload_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_payload_size", [True, 0, -1, 1.5])
def test_invalid_reset_frame_limit_preserves_pending_head(max_payload_size: object) -> None:
    queue = TranscriptQueue(max_payload_size=65536, secret="reset-validation-secret")
    assert queue.enqueue("preserve me")
    pending = queue.poll()
    assert pending is not None

    expected_error = TypeError if isinstance(max_payload_size, (bool, float)) else ValueError
    with pytest.raises(expected_error):
        queue.reset(max_payload_size=max_payload_size, secret="replacement-secret")  # type: ignore[arg-type]

    assert queue.poll() == pending


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (str(uuid.UUID(int=1)), True),
        (str(uuid.UUID("abcdefab-cdef-abcd-efab-cdefabcdefab")).upper(), False),
        ("not-a-uuid", False),
        ("0" * 36, False),
        ("0" * 37, False),
        (None, False),
        (123, False),
    ],
)
def test_transcript_id_validation_is_canonical_and_bounded(value: object, expected: bool) -> None:
    assert is_transcript_id(value) is expected


def test_invalid_signing_configuration_preserves_pending_head() -> None:
    with pytest.raises(TypeError, match="secret must be a string"):
        TranscriptQueue(secret=None)  # type: ignore[arg-type]

    queue = TranscriptQueue(max_payload_size=65536, secret="valid-secret")
    assert queue.enqueue("pending")
    pending = queue.poll()
    assert pending is not None
    with pytest.raises(TypeError, match="secret must be a string"):
        queue.reset(max_payload_size=65536, secret=None)  # type: ignore[arg-type]

    assert queue.poll() == pending


def test_concurrent_enqueue_respects_queue_capacity() -> None:
    queue = TranscriptQueue(max_payload_size=65536, secret="queue-concurrency-secret")
    worker_count = MAX_PENDING_TRANSCRIPTS * 2
    start = threading.Barrier(worker_count + 1)
    accepted: list[str] = []
    accepted_lock = threading.Lock()

    def enqueue(index: int) -> None:
        text = f"concurrent transcript {index}"
        start.wait(timeout=5)
        if queue.enqueue(text):
            with accepted_lock:
                accepted.append(text)

    workers = [threading.Thread(target=enqueue, args=(i,)) for i in range(worker_count)]
    for worker in workers:
        worker.start()
    start.wait(timeout=5)
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()

    assert len(accepted) == MAX_PENDING_TRANSCRIPTS
    queued: list[str] = []
    while (item := queue.poll()) is not None:
        queued.append(item["text"])
        assert queue.ack(item["id"])
    assert len(queued) == MAX_PENDING_TRANSCRIPTS
    assert set(queued) == set(accepted)


def test_reset_during_size_preflight_rejects_stale_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = TranscriptQueue(max_payload_size=65536, secret="old-secret")
    preflight_started = threading.Event()
    resume_preflight = threading.Event()
    original_preflight = queue._fits_poll_response
    result: list[bool] = []

    def pause_preflight(*args: object, **kwargs: object) -> bool:
        preflight_started.set()
        assert resume_preflight.wait(timeout=5)
        return original_preflight(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(queue, "_fits_poll_response", pause_preflight)
    worker = threading.Thread(target=lambda: result.append(queue.enqueue("during reset")))
    worker.start()
    assert preflight_started.wait(timeout=5)
    queue.reset(max_payload_size=65536, secret="new-secret")
    resume_preflight.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert result == [False]
    assert queue.poll() is None


@pytest.mark.asyncio
async def test_identical_text_has_distinct_ids_and_ack_is_head_only(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    repeated_text = "Please repeat this question"

    async with _running_client(adapter_module, ephemeral_server_config) as client:
        await adapter_module.MyFunctions.send_response(repeated_text)
        await adapter_module.MyFunctions.send_response(repeated_text)
        await adapter_module.MyFunctions.send_response("A later question")

        first = await _poll(client)
        assert first is not None and first["text"] == repeated_text
        first_id = first["id"]

        assert await _ack(client, str(uuid.uuid4())) is False
        assert await _ack(client, str(uuid.UUID(int=0))) is False
        assert await _poll(client) == first
        assert await _ack(client, first_id) is True
        assert await _ack(client, first_id) is False

        second = await _poll(client)
        assert second is not None and second["text"] == repeated_text
        assert second["id"] != first_id
        assert await _ack(client, second["id"]) is True

        third = await _poll(client)
        assert third is not None and third["text"] == "A later question"
        assert await _ack(client, third["id"]) is True
        assert await _poll(client) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("poll_transcript", {"unexpected": "value"}),
        ("ack_transcript", {}),
        ("ack_transcript", {"id": "not-a-uuid"}),
        ("ack_transcript", {"id": str(uuid.uuid4()), "unexpected": "value"}),
    ],
)
async def test_transcript_commands_reject_malformed_arguments(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
    command: str,
    arguments: dict[str, str],
) -> None:
    async with _running_client(adapter_module, ephemeral_server_config) as client:
        response = await client.request(command, arguments)

        assert response.ok is False
        assert response.error is not None
        assert response.error.get("code") == "invalid_arguments"
        assert await _poll(client) is None


@pytest.mark.asyncio
async def test_transcript_commands_reject_noncanonical_request_id(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    async with _running_client(adapter_module, ephemeral_server_config) as client:
        response = await client.request("poll_transcript", {}, request_id="ordinary-invalid-id")

        assert response.ok is False
        assert response.error is not None
        assert response.error.get("code") == "invalid_arguments"
        assert await _poll(client) is None


@pytest.mark.asyncio
async def test_blank_transcripts_are_not_queued(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    async with _running_client(adapter_module, ephemeral_server_config) as client:
        for text in ("", "   ", "\t\n"):
            await adapter_module.MyFunctions.send_response(text)
            assert await _poll(client) is None


@pytest.mark.asyncio
async def test_full_queue_rejects_newest_and_warning_omits_transcript_text(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    overflow_text = "PRIVATE_OVERFLOW_TRANSCRIPT_SENTINEL"
    caplog.set_level(logging.WARNING, logger="transport.transcripts")

    async with _running_client(adapter_module, ephemeral_server_config) as client:
        for index in range(32):
            await adapter_module.MyFunctions.send_response(f"queued turn {index}")
        await adapter_module.MyFunctions.send_response(overflow_text)

        assert "PRIVATE_OVERFLOW_TRANSCRIPT_SENTINEL" not in caplog.text
        assert any(
            record.name == "transport.transcripts" and record.levelno == logging.WARNING
            for record in caplog.records
        )

        for index in range(32):
            item = await _poll(client)
            assert item is not None
            assert item["text"] == f"queued turn {index}"
            assert await _ack(client, item["id"]) is True

        assert await _poll(client) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "é" * 33_000,
        "PRIVATE_ESCAPED_TRANSCRIPT_SENTINEL" + "\x01" * 12_000,
    ],
    ids=["oversized-utf8", "oversized-json-escaped"],
)
async def test_oversized_transcripts_are_rejected_without_logging_content(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
    caplog: pytest.LogCaptureFixture,
    text: str,
) -> None:
    max_payload_size = ephemeral_server_config.max_payload_size
    if text.startswith("PRIVATE_ESCAPED_TRANSCRIPT_SENTINEL"):
        assert len(text.encode("utf-8")) < max_payload_size
        assert len(json.dumps({"text": text}, separators=(",", ":")).encode("utf-8")) > (
            max_payload_size
        )
        private_marker = "PRIVATE_ESCAPED_TRANSCRIPT_SENTINEL"
    else:
        assert len(text.encode("utf-8")) > max_payload_size
        private_marker = "é" * 16

    caplog.set_level(logging.WARNING, logger="transport.transcripts")
    async with _running_client(adapter_module, ephemeral_server_config) as client:
        await adapter_module.MyFunctions.send_response(text)

        assert private_marker not in caplog.text
        assert any(
            record.name == "transport.transcripts" and record.levelno == logging.WARNING
            for record in caplog.records
        )
        assert await _poll(client) is None


@pytest.mark.asyncio
async def test_restarting_adapter_server_discards_pending_transcripts(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    first_server = adapter_module.create_robot_server(ephemeral_server_config)
    restarted_server: RobotServer | None = None
    try:
        await first_server.start()
        async with _connected_client(first_server, ephemeral_server_config) as client:
            await adapter_module.MyFunctions.send_response("pending before recreation")
            first = await _poll(client)
            assert first is not None and first["text"] == "pending before recreation"

        await first_server.stop()

        restarted_server = adapter_module.create_robot_server(ephemeral_server_config)
        await restarted_server.start()
        async with _connected_client(restarted_server, ephemeral_server_config) as client:
            await adapter_module.MyFunctions.send_response("after server recreation")
            assert await _ack(client, first["id"]) is False
            recreated = await _poll(client)
            assert recreated is not None
            assert recreated["text"] == "after server recreation"
            assert recreated["id"] != first["id"]

        await restarted_server.stop()
        await restarted_server.start()
        async with _connected_client(restarted_server, ephemeral_server_config) as client:
            await adapter_module.MyFunctions.send_response("after same-instance restart")
            assert await _ack(client, recreated["id"]) is False
            same_instance_restart = await _poll(client)
            assert same_instance_restart is not None
            assert same_instance_restart["text"] == "after same-instance restart"
            assert same_instance_restart["id"] != recreated["id"]
            assert await _ack(client, same_instance_restart["id"]) is True
            assert await _poll(client) is None
    finally:
        await first_server.stop()
        if restarted_server is not None:
            await restarted_server.stop()
