"""Unit tests for RobotServer and RobotClient end-to-end communication."""

import asyncio
import time

import pytest

from transport.client import RobotClient
from transport.config import TransportConfig
from transport.protocol import (
    AuthenticationError,
    ExecutionOutcomeUnknown,
    RobotRequest,
    RobotResponse,
    TransportError,
    TransportErrorCode,
)
from transport.registry import (
    CommandRegistry,
    validate_empty_arguments,
    validate_reply_arguments,
)
from transport.server import RobotServer


@pytest.mark.asyncio
async def test_server_client_roundtrip_success(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    registry.register("ping", lambda args: "pong", validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            resp = await client.request("ping", {})
            assert resp.ok is True
            assert resp.result == "pong"
            assert resp.request_id is not None
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_rejects_unauthorized_client(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    registry.register("ping", lambda args: "pong", validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        # Client has incorrect secret
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret="different-wrong-secret",
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.request("ping", {})
            assert exc_info.value.code == TransportErrorCode.UNAUTHORIZED
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_rejects_unsupported_command(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            resp = await client.request("nonexistent_command", {})
            assert resp.ok is False
            assert resp.error["code"] == TransportErrorCode.UNSUPPORTED_COMMAND
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_handler_exception_returns_internal_error(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()

    def crash_handler(args):
        raise RuntimeError("Secret internal failure detail: /path/to/private/file")

    registry.register("crash", crash_handler, validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            resp = await client.request("crash", {})
            assert resp.ok is False
            assert resp.error["code"] == TransportErrorCode.INTERNAL_ERROR
            # Must NOT leak exception details to client!
            assert "private" not in resp.error["message"]
            assert resp.error["message"] == "Internal server error during command execution"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_client_connect_timeout_when_server_not_running():
    client_cfg = TransportConfig(
        host="127.0.0.1",
        port=65432,  # Port not open
        secret="test-secret",
        connect_timeout=0.2,
        max_retries=1,
    )
    async with RobotClient(client_cfg) as client:
        with pytest.raises(TransportError):
            await client.request("ping", {})


@pytest.mark.asyncio
async def test_multiple_sequential_requests_on_single_connection(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    count = 0

    def increment(args):
        nonlocal count
        count += 1
        return count

    registry.register("inc", increment, validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            for expected in range(1, 6):
                resp = await client.request("inc", {})
                assert resp.ok is True
                assert resp.result == expected
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_rejects_unsupported_protocol_version(ephemeral_server_config: TransportConfig):
    from transport.auth import sign_request
    from transport.framing import read_frame, write_frame

    registry = CommandRegistry()
    registry.register("ping", lambda args: "pong", validate_empty_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            req = RobotRequest.create("ping", protocol_version="99.0")
            sign_request(ephemeral_server_config.secret, req)
            await write_frame(writer, req.to_json().encode("utf-8"))

            raw_resp = await read_frame(reader)
            resp = RobotResponse.from_json(raw_resp)
            assert resp.ok is False
            assert resp.error["code"] == TransportErrorCode.UNSUPPORTED_PROTOCOL
            assert "Unsupported protocol version '99.0'" in resp.error["message"]
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_rejects_malformed_json_frame(ephemeral_server_config: TransportConfig):
    from transport.framing import read_frame, write_frame

    registry = CommandRegistry()
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            malformed_json_bytes = b"{{invalid-json--not-object"
            await write_frame(writer, malformed_json_bytes)

            raw_resp = await read_frame(reader)
            resp = RobotResponse.from_json(raw_resp)
            assert resp.ok is False
            assert resp.error["code"] == TransportErrorCode.MALFORMED_REQUEST
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_rejects_invalid_schema_frame(ephemeral_server_config: TransportConfig):
    from transport.framing import read_frame, write_frame

    registry = CommandRegistry()
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            # Valid JSON but missing required fields
            invalid_schema_bytes = b'{"hello": "world"}'
            await write_frame(writer, invalid_schema_bytes)

            raw_resp = await read_frame(reader)
            resp = RobotResponse.from_json(raw_resp)
            assert resp.ok is False
            assert resp.error["code"] == TransportErrorCode.MALFORMED_REQUEST
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_rejects_invalid_command_arguments(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    registry.register("reply", lambda args: None, validate_reply_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            resp = await client.request("reply", {"wrong_argument": "hello"})
            assert resp.ok is False
            assert resp.error["code"] == TransportErrorCode.INVALID_ARGUMENTS
            assert "Unexpected arguments" in resp.error["message"]
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_handles_truncated_socket_frame_without_crashing(ephemeral_server_config: TransportConfig):
    from transport.framing import HEADER_STRUCT

    registry = CommandRegistry()
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        # Send length 100, but only 10 bytes then close connection
        writer.write(HEADER_STRUCT.pack(100) + b"1234567890")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

        # Verify server is still alive and handling next connection
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        registry.register("ping", lambda args: "pong", validate_empty_arguments)
        async with RobotClient(client_cfg) as client:
            resp = await client.request("ping", {})
            assert resp.ok is True
            assert resp.result == "pong"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_client_read_timeout_disconnects_and_prevents_stream_desync(
    ephemeral_server_config: TransportConfig,
):
    registry = CommandRegistry()
    slow_flag = True

    async def slow_handler(args):
        nonlocal slow_flag
        if slow_flag:
            slow_flag = False
            await asyncio.sleep(0.4)
            return "slow_done"
        return "fast_done"

    registry.register("test_timing", slow_handler, validate_empty_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            read_timeout=0.15,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            # 1. First request times out on read
            with pytest.raises(TransportError) as exc_info:
                await client.request("test_timing", {})
            assert exc_info.value.code == TransportErrorCode.TIMEOUT

            # Wait for server to finish slow handler and write to socket buffer
            await asyncio.sleep(0.45)

            # 2. Second request must NOT receive the first request's response
            # Connection must have been reset cleanly so Request 2 gets "fast_done"
            resp2 = await client.request("test_timing", {})
            assert resp2.ok is True
            assert resp2.result == "fast_done"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_client_reconnects_when_server_drops_idle_connection(
    ephemeral_server_config: TransportConfig,
):
    registry = CommandRegistry()
    registry.register("ping", lambda args: "pong", validate_empty_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            resp1 = await client.request("ping", {})
            assert resp1.ok is True

            # Simulate server dropping the connection from its end
            for task in list(server._active_connections):
                task.cancel()
            await asyncio.sleep(0.05)

            # Client should cleanly recover and reconnect without NameError on TruncatedFrameError
            resp2 = await client.request("ping", {})
            assert resp2.ok is True
            assert resp2.result == "pong"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_client_overall_request_timeout(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()

    async def hanging_handler(args):
        await asyncio.sleep(2.0)
        return "late"

    registry.register("hang", hanging_handler, validate_empty_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            read_timeout=5.0,
            request_timeout=0.2,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(TransportError) as exc_info:
                await client.request("hang", {})
            assert exc_info.value.code == TransportErrorCode.TIMEOUT
            assert "Overall request timed out" in str(exc_info.value)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_client_correlation_rejects_null_or_mismatched_request_id(ephemeral_server_config: TransportConfig):
    from transport.framing import read_frame, write_frame

    async def bad_responder(reader, writer):
        try:
            await read_frame(reader)
            # Reply with mismatched request_id
            bad_resp = RobotResponse.success(request_id="mismatched-id", result="bad")
            await write_frame(writer, bad_resp.to_json().encode("utf-8"))
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    custom_server = await asyncio.start_server(bad_responder, "127.0.0.1", 0)
    port = custom_server.sockets[0].getsockname()[1]
    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(TransportError) as exc_info:
                await client.request("ping", {})
            assert exc_info.value.code == TransportErrorCode.MALFORMED_REQUEST
            assert "Correlation mismatch" in str(exc_info.value)
    finally:
        custom_server.close()
        await custom_server.wait_closed()


@pytest.mark.asyncio
async def test_check_for_response_raises_on_error(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    registry.register("check_for_response", lambda args: 12345, validate_empty_arguments)  # Non-string result
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(TransportError) as exc_info:
                await client.check_for_response()
            assert exc_info.value.code == TransportErrorCode.INTERNAL_ERROR
            assert "must be a string" in str(exc_info.value)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_check_for_response_raises_on_server_failure(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        # Client with bad secret
        bad_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret="wrong-secret-12345",
            request_timeout=2.0,
        )
        async with RobotClient(bad_cfg) as client:
            with pytest.raises(TransportError) as exc_info:
                await client.check_for_response()
            assert exc_info.value.code == TransportErrorCode.UNAUTHORIZED
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_client_wait_convenience_method(ephemeral_server_config: TransportConfig):
    waited_seconds = None

    def wait_handler(args):
        nonlocal waited_seconds
        waited_seconds = args["seconds"]
        return "waited"

    from transport.registry import validate_wait_arguments
    registry = CommandRegistry()
    registry.register("wait", wait_handler, validate_wait_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=5.0,
        )
        async with RobotClient(client_cfg) as client:
            await client.wait(0.5, timeout=10.0)
            assert waited_seconds == 0.5
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_convenience_methods_support_timeout_parameter(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    from transport.registry import (
        validate_base_col_arguments,
        validate_move_arguments,
        validate_reply_arguments,
        validate_say_arguments,
    )
    registry.register("reply", lambda args: "spoken", validate_reply_arguments)
    registry.register("say", lambda args: "said", validate_say_arguments)
    registry.register("move", lambda args: "moved", validate_move_arguments)
    registry.register("base_col", lambda args: "colored", validate_base_col_arguments)
    registry.register("simple_test", lambda args: "pong", validate_empty_arguments)
    registry.register("check_for_response", lambda args: "hello", validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            assert await client.ping(timeout=3.0) is True
            assert await client.check_for_response(timeout=3.0) == "hello"
            assert await client.reply("test message", timeout=3.0) == "spoken"
            await client.say("test text", timeout=3.0)
            await client.move(0, 5.0, timeout=3.0)
            await client.base_col(1, 2, 3, timeout=3.0)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_reconnect_retry_replayed_raises_execution_outcome_unknown(
    ephemeral_server_config: TransportConfig,
):
    """Verify that if connection is lost after server processes command and retry

    is rejected as a replay, client raises ExecutionOutcomeUnknown.
    """
    registry = CommandRegistry()
    command_executed_count = 0

    async def counted_handler(args):
        nonlocal command_executed_count
        command_executed_count += 1
        return "robot_action_completed"

    registry.register("act", counted_handler, validate_empty_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    orig_send = server._send_response
    drop_first_response = True

    async def dropping_send(writer, response):
        nonlocal drop_first_response
        if drop_first_response:
            # Simulate network drop after server executes command and prepares response
            drop_first_response = False
            writer.close()
            await writer.wait_closed()
            return
        await orig_send(writer, response)

    server._send_response = dropping_send

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=3.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
                await client.request("act", {})

            # Command was executed once by server before connection drop
            assert command_executed_count == 1
            # Exception provides request_id and command name for caller investigation
            assert exc_info.value.command == "act"
            assert exc_info.value.request_id is not None
            assert "execution outcome is unknown" in str(exc_info.value)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_server_startup_port_in_use_propagates_error(
    ephemeral_server_config: TransportConfig,
):
    """Verify server startup propagates port binding conflict (fails closed)."""
    registry = CommandRegistry()
    server1 = RobotServer(ephemeral_server_config, registry)
    await server1.start()

    try:
        conflict_cfg = TransportConfig(
            host="127.0.0.1",
            port=server1.port,
            secret=ephemeral_server_config.secret,
        )
        server2 = RobotServer(conflict_cfg, registry)
        with pytest.raises(OSError):
            await server2.start()
    finally:
        await server1.stop()


@pytest.mark.asyncio
async def test_client_rejects_unsigned_server_response(ephemeral_server_config: TransportConfig):
    from transport.framing import read_frame, write_frame

    async def unsigned_responder(reader, writer):
        try:
            req_bytes = await read_frame(reader)
            req = RobotRequest.from_json(req_bytes)
            # Send response with no auth
            resp = RobotResponse.success(request_id=req.request_id, result="unsigned-secret-data")
            resp.auth = {}
            await write_frame(writer, resp.to_json().encode("utf-8"))
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    custom_server = await asyncio.start_server(unsigned_responder, "127.0.0.1", 0)
    port = custom_server.sockets[0].getsockname()[1]
    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.request("ping", {})
            assert exc_info.value.code == TransportErrorCode.UNAUTHORIZED
            assert "missing authentication" in str(exc_info.value)
    finally:
        custom_server.close()
        await custom_server.wait_closed()


@pytest.mark.asyncio
async def test_client_rejects_tampered_response_result(ephemeral_server_config: TransportConfig):
    from transport.auth import sign_response
    from transport.framing import read_frame, write_frame

    async def tampering_responder(reader, writer):
        try:
            req_bytes = await read_frame(reader)
            req = RobotRequest.from_json(req_bytes)
            # Sign response with legitimate result
            resp = RobotResponse.success(request_id=req.request_id, result={"amount": 100})
            sign_response(ephemeral_server_config.secret, resp)
            # Attacker tampers with result after signing
            resp.result = {"amount": 999999}
            await write_frame(writer, resp.to_json().encode("utf-8"))
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    custom_server = await asyncio.start_server(tampering_responder, "127.0.0.1", 0)
    port = custom_server.sockets[0].getsockname()[1]
    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.request("ping", {})
            assert exc_info.value.code == TransportErrorCode.UNAUTHORIZED
            assert "verification failed" in str(exc_info.value)
    finally:
        custom_server.close()
        await custom_server.wait_closed()


@pytest.mark.asyncio
async def test_client_rejects_tampered_response_error(ephemeral_server_config: TransportConfig):
    from transport.auth import sign_response
    from transport.framing import read_frame, write_frame

    async def tampering_responder(reader, writer):
        try:
            req_bytes = await read_frame(reader)
            req = RobotRequest.from_json(req_bytes)
            # Sign fail response
            resp = RobotResponse.fail(request_id=req.request_id, code=TransportErrorCode.INVALID_ARGUMENTS, message="original")
            sign_response(ephemeral_server_config.secret, resp)
            # Attacker tampers with error message
            resp.error["message"] = "tampered"
            await write_frame(writer, resp.to_json().encode("utf-8"))
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    custom_server = await asyncio.start_server(tampering_responder, "127.0.0.1", 0)
    port = custom_server.sockets[0].getsockname()[1]
    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.request("ping", {})
            assert exc_info.value.code == TransportErrorCode.UNAUTHORIZED
    finally:
        custom_server.close()
        await custom_server.wait_closed()


@pytest.mark.asyncio
async def test_client_rejects_response_signed_with_wrong_secret(ephemeral_server_config: TransportConfig):
    from transport.auth import sign_response
    from transport.framing import read_frame, write_frame

    async def wrong_secret_responder(reader, writer):
        try:
            req_bytes = await read_frame(reader)
            req = RobotRequest.from_json(req_bytes)
            resp = RobotResponse.success(request_id=req.request_id, result="ok")
            sign_response("adversary-different-secret-xyz", resp)
            await write_frame(writer, resp.to_json().encode("utf-8"))
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    custom_server = await asyncio.start_server(wrong_secret_responder, "127.0.0.1", 0)
    port = custom_server.sockets[0].getsockname()[1]
    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.request("ping", {})
            assert exc_info.value.code == TransportErrorCode.UNAUTHORIZED
    finally:
        custom_server.close()
        await custom_server.wait_closed()


@pytest.mark.asyncio
async def test_client_rejects_expired_server_response(ephemeral_server_config: TransportConfig):
    from transport.auth import sign_response
    from transport.framing import read_frame, write_frame

    async def expired_responder(reader, writer):
        try:
            req_bytes = await read_frame(reader)
            req = RobotRequest.from_json(req_bytes)
            # Response timestamp is 120 seconds in the past
            resp = RobotResponse.success(
                request_id=req.request_id,
                result="ok",
                timestamp=time.time() - 120.0,
            )
            sign_response(ephemeral_server_config.secret, resp)
            await write_frame(writer, resp.to_json().encode("utf-8"))
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    custom_server = await asyncio.start_server(expired_responder, "127.0.0.1", 0)
    port = custom_server.sockets[0].getsockname()[1]
    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=port,
            secret=ephemeral_server_config.secret,
            max_clock_skew=10.0,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.request("ping", {})
            assert exc_info.value.code == TransportErrorCode.EXPIRED_REQUEST
            assert "expired" in str(exc_info.value)
    finally:
        custom_server.close()
        await custom_server.wait_closed()


@pytest.mark.asyncio
async def test_normal_signed_structured_error_roundtrip(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    registry.register("fail_cmd", lambda args: None, validate_empty_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            # Command sending invalid arguments returns signed structured error
            resp = await client.request("fail_cmd", {"unexpected_arg": 123})
            assert resp.ok is False
            assert resp.error is not None
            assert resp.error["code"] == TransportErrorCode.INVALID_ARGUMENTS.value
            assert "unexpected keys" in resp.error["message"]
            assert resp.auth is not None
            assert resp.auth.get("scheme") == "hmac-sha256"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_timeout_scaling_wait_extends_beyond_base_deadline(
    ephemeral_server_config: TransportConfig,
):
    """Verify that a wait duration exceeding base timeout is NOT killed by the default timeout."""
    registry = CommandRegistry()

    async def mock_wait(args):
        sec = float(args["seconds"])
        await asyncio.sleep(sec)
        return f"waited_{sec}"

    from transport.registry import validate_wait_arguments
    registry.register("wait", mock_wait, validate_wait_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        # Client configured with very short base timeout (0.15s)
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            read_timeout=0.15,
            request_timeout=0.15,
        )
        async with RobotClient(client_cfg) as client:
            # A normal operation taking 0.3s would fail base timeout (0.15s),
            # but wait(0.3) scales the timeout to max(0.15, 0.3 + 10.0) = 10.3s
            resp = await client.request("wait", {"seconds": 0.3})
            assert resp.ok is True
            assert resp.result == "waited_0.3"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_timeout_scaling_speech_reply_extends_beyond_base_deadline(
    ephemeral_server_config: TransportConfig,
):
    """Verify that long speech/reply extends beyond base timeout."""
    registry = CommandRegistry()

    async def mock_speech_reply(args):
        await asyncio.sleep(0.3)
        return "spoken"

    from transport.registry import validate_reply_arguments
    registry.register("reply", mock_speech_reply, validate_reply_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        # Base timeout is 0.15s
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            read_timeout=0.15,
            request_timeout=0.15,
        )
        async with RobotClient(client_cfg) as client:
            # Long speech (200 characters) scales timeout to ~30s
            long_msg = "x" * 200
            resp = await client.request("reply", {"message": long_msg})
            assert resp.ok is True
            assert resp.result == "spoken"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_concurrent_duplicate_requests_stress_replay_cache(
    ephemeral_server_config: TransportConfig,
):
    """Stress test ReplayCache by flooding duplicate request IDs across simultaneous connections."""
    registry = CommandRegistry()
    execution_counter = 0

    async def counted_handler(args):
        nonlocal execution_counter
        execution_counter += 1
        await asyncio.sleep(0.02)
        return "ok"

    registry.register("ping", counted_handler, validate_empty_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    from transport.auth import sign_request
    from transport.framing import read_frame, write_frame

    shared_request_id = "concurrent-stress-req-id-999"
    fixed_request = RobotRequest.create("ping", request_id=shared_request_id, timestamp=time.time())
    sign_request(ephemeral_server_config.secret, fixed_request)
    wire_bytes = fixed_request.to_json().encode("utf-8")

    async def send_worker():
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            await write_frame(writer, wire_bytes)
            raw_resp = await read_frame(reader)
            return RobotResponse.from_json(raw_resp)
        finally:
            writer.close()
            await writer.wait_closed()

    try:
        # Launch 8 concurrent connections all sending the exact same signed request
        responses = await asyncio.gather(*(send_worker() for _ in range(8)))
        successes = [r for r in responses if r.ok]
        replays = [
            r for r in responses
            if not r.ok and r.error and r.error.get("code") == TransportErrorCode.REPLAYED_REQUEST.value
        ]

        # Exactly 1 request must succeed; remaining 7 must be detected as replays
        assert len(successes) == 1
        assert len(replays) == 7
        assert execution_counter == 1
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_robot_script_start_server_startup_failures_propagate(
    ephemeral_server_config: TransportConfig,
):
    """Verify that start_server in buddy1 and windows robot scripts propagate startup failures."""
    import buddy1
    from windows import buddy1 as win_buddy1
    from windows import buddy2 as win_buddy2

    # 1. Missing secret propagates AuthenticationError
    bad_cfg = TransportConfig(host="127.0.0.1", port=0, secret="")
    with pytest.raises(AuthenticationError):
        await buddy1.start_server(bad_cfg)
    with pytest.raises(AuthenticationError):
        await win_buddy1.start_server(bad_cfg)
    with pytest.raises(AuthenticationError):
        await win_buddy2.start_server(bad_cfg)

    # 2. Port conflict propagates OSError
    registry = CommandRegistry()
    occupying_server = RobotServer(ephemeral_server_config, registry)
    await occupying_server.start()
    try:
        conflict_cfg = TransportConfig(
            host="127.0.0.1",
            port=occupying_server.port,
            secret=ephemeral_server_config.secret,
        )
        with pytest.raises(OSError):
            await buddy1.start_server(conflict_cfg)
        with pytest.raises(OSError):
            await win_buddy1.start_server(conflict_cfg)
        with pytest.raises(OSError):
            await win_buddy2.start_server(conflict_cfg)
    finally:
        await occupying_server.stop()



