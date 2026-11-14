"""Authenticated loopback contracts for the legacy transcript adapters."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import ModuleType

import pytest

import buddy1
from transport import RobotClient, RobotServer, TransportConfig
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
async def test_authenticated_transcript_queue_round_trip_and_legacy_client(
    adapter_module: ModuleType,
    ephemeral_server_config: TransportConfig,
) -> None:
    text = 'Could you describe the café? It says "你好" 😀.'

    async with _running_client(adapter_module, ephemeral_server_config) as client:
        await adapter_module.MyFunctions.send_response(text)

        legacy_response = await client.request("check_for_response", {})
        assert legacy_response.ok, legacy_response.error
        assert legacy_response.result == text

        item = await _poll(client)
        assert item is not None
        assert set(item) == {"id", "text"}
        assert item["text"] == text
        transcript_id = item["id"]
        assert str(uuid.UUID(transcript_id)) == transcript_id

        assert await _ack(client, transcript_id) is True
        assert await _poll(client) is None


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
