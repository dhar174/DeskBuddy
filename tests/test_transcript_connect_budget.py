"""Exercise desktop socket setup deadlines without a remote endpoint."""

import asyncio

import pytest

import robot_client
from transport import client as client_module
from transport.client import RobotClient
from transport.config import TransportConfig
from transport.protocol import TransportError, TransportErrorCode
from transport.registry import validate_reply_arguments
from transport.server import RobotServer
from transport.transcripts import TRANSCRIPT_MODE_MISMATCH_MESSAGE, TranscriptQueue


@pytest.mark.parametrize("operation", ["poll", "reply"])
async def test_desktop_connection_budget_bounds_stalled_socket_setup(
    ephemeral_server_config: TransportConfig, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    client = RobotClient(ephemeral_server_config.with_overrides(connect_timeout=0.05))
    connect_entered = asyncio.Event()
    connect_cancelled = asyncio.Event()

    async def stalled_connect() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        connect_entered.set()
        try:
            await asyncio.Future()
        finally:
            connect_cancelled.set()
        raise AssertionError("stalled connect returned")

    async def get_client() -> RobotClient:
        return client

    monkeypatch.setattr(client, "_ensure_connected", stalled_connect)
    monkeypatch.setattr(robot_client, "get_robot_client", get_client)
    request = (
        robot_client.TranscriptPoller().next_transcript()
        if operation == "poll"
        else robot_client.send_reply("Hello")
    )

    with pytest.raises(TransportError) as failure:
        await asyncio.wait_for(request, timeout=2)

    assert failure.value.code == TransportErrorCode.TIMEOUT
    assert type(failure.value) is TransportError
    assert connect_entered.is_set()
    assert connect_cancelled.is_set()
    assert client.connection_generation == 0


async def test_aged_exchange_refreshes_socket_before_other_side_effecting_command(
    ephemeral_server_config: TransportConfig,
) -> None:
    server = RobotServer(ephemeral_server_config)
    calls = []
    server.register("simple_test", lambda _args: True)
    server.register("say", lambda args: calls.append(args["text"]), validate_reply_arguments)
    await server.start()
    try:
        config = ephemeral_server_config.with_overrides(port=server.port)
        async with RobotClient(config) as client:
            assert await client.ping()
            assert client.connection_generation == 1
            assert client._last_request_started_at is not None
            client._last_request_started_at = (
                asyncio.get_running_loop().time() - config.read_timeout * 0.9
            )

            await client.say("Hello")

            assert client.connection_generation == 2
            assert calls == ["Hello"]
    finally:
        await server.stop()


async def test_pinned_ack_preserves_default_response_grace(
    ephemeral_server_config: TransportConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = TranscriptQueue()
    config = ephemeral_server_config.with_overrides(read_timeout=3.0, request_timeout=3.0)
    server = RobotServer(config, transcript_queue=queue)
    read_timeouts = []
    read_frame = client_module.read_frame

    async def record_read_timeout(reader, *, max_payload_size, timeout):
        read_timeouts.append(timeout)
        return await read_frame(reader, max_payload_size=max_payload_size, timeout=timeout)

    monkeypatch.setattr(client_module, "read_frame", record_read_timeout)
    await server.start()
    try:
        assert queue.enqueue("Hello")
        async with RobotClient(config.with_overrides(port=server.port)) as client:

            async def get_client() -> RobotClient:
                return client

            monkeypatch.setattr(robot_client, "get_robot_client", get_client)
            assert await robot_client.TranscriptPoller().next_transcript() == "Hello"
            assert client.connection_generation == 1
            assert queue.poll() is None
            assert read_timeouts == pytest.approx([4.2, 4.2])
    finally:
        await server.stop()


async def test_speech_socket_refresh_preserves_legacy_text_deduplication(
    ephemeral_server_config: TransportConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = RobotServer(ephemeral_server_config)
    spoken = []
    transcripts = iter(["Hello", "Hello", " \t\n ", "New turn"])
    server.register("check_for_response", lambda _args: next(transcripts))
    server.register("reply", lambda args: spoken.append(args["message"]), validate_reply_arguments)
    await server.start()
    try:
        config = ephemeral_server_config.with_overrides(port=server.port)
        async with RobotClient(config) as client:

            async def get_client() -> RobotClient:
                return client

            monkeypatch.setattr(robot_client, "get_robot_client", get_client)
            poller = robot_client.TranscriptPoller()
            assert await poller.next_transcript() == "Hello"
            assert client.connection_generation == 1
            assert client._last_request_started_at is not None
            client._last_request_started_at = (
                asyncio.get_running_loop().time() - config.read_timeout * 0.9
            )

            await robot_client.send_reply("Hi")

            assert client.connection_generation == 2
            assert await poller.next_transcript() is None
            assert await poller.next_transcript() is None
            assert await poller.next_transcript() == "New turn"
            assert spoken == ["Hi"]
    finally:
        await server.stop()


async def test_desktop_receives_authenticated_polling_mode_restart_hint(
    ephemeral_server_config: TransportConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = TranscriptQueue()
    server = RobotServer(ephemeral_server_config, transcript_queue=queue)
    await server.start()
    try:
        queue.select_polling_mode("legacy")
        config = ephemeral_server_config.with_overrides(port=server.port)
        async with RobotClient(config) as client:

            async def get_client() -> RobotClient:
                return client

            monkeypatch.setattr(robot_client, "get_robot_client", get_client)
            with pytest.raises(TransportError) as failure:
                await robot_client.TranscriptPoller().next_transcript()

            assert failure.value.code == TransportErrorCode.INVALID_ARGUMENTS
            assert failure.value.message == TRANSCRIPT_MODE_MISMATCH_MESSAGE
    finally:
        await server.stop()
