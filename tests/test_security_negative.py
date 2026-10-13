"""Security and negative tests verifying that pickle, arbitrary dispatch, and injection are prevented."""

import asyncio
import pickle

import pytest

from transport.config import TransportConfig
from transport.framing import HEADER_STRUCT, read_frame, write_frame
from transport.protocol import TransportErrorCode
from transport.registry import CommandRegistry, validate_empty_arguments
from transport.server import RobotServer


@pytest.mark.asyncio
async def test_pickle_payload_is_not_deserialized(ephemeral_server_config: TransportConfig):
    handler_invoked = False

    def dummy_handler(args):
        nonlocal handler_invoked
        handler_invoked = True
        return "ok"

    registry = CommandRegistry()
    registry.register("ping", dummy_handler, validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            # Craft an exploit-like pickle payload
            pickle_payload = pickle.dumps({"function_name": "ping", "cmd": "malicious"})
            # Send length-framed pickle bytes
            await write_frame(writer, pickle_payload)

            # Server must not unpickle; should respond with malformed_request error
            response_bytes = await read_frame(reader)
            assert b"malformed_request" in response_bytes
            assert handler_invoked is False
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_arbitrary_function_names_cannot_be_invoked(ephemeral_server_config: TransportConfig):
    dangerous_invoked = False

    class DangerousTarget:
        @staticmethod
        def shutdown_robot():
            nonlocal dangerous_invoked
            dangerous_invoked = True

    registry = CommandRegistry()
    # Only "ping" is registered
    registry.register("ping", lambda args: "pong", validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        from transport.client import RobotClient

        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            # Try to invoke arbitrary functions
            for target in ["shutdown_robot", "os.system", "eval", "exec", "__import__", "listen", "record"]:
                resp = await client.request(target, {})
                assert resp.ok is False
                assert resp.error["code"] == TransportErrorCode.UNSUPPORTED_COMMAND
            assert dangerous_invoked is False
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_dunder_attributes_cannot_be_reached_via_dispatch(ephemeral_server_config: TransportConfig):
    registry = CommandRegistry()
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        from transport.client import RobotClient

        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            for dunder in ["__class__", "__dict__", "__init__", "__globals__", "__subclasses__", "mro"]:
                resp = await client.request(dunder, {})
                assert resp.ok is False
                assert resp.error["code"] == TransportErrorCode.UNSUPPORTED_COMMAND
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_invalid_auth_never_reaches_handler(ephemeral_server_config: TransportConfig):
    handler_invoked = False

    def guarded_handler(args):
        nonlocal handler_invoked
        handler_invoked = True
        return "guarded"

    registry = CommandRegistry()
    registry.register("guarded", guarded_handler, validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        from transport.client import RobotClient
        from transport.protocol import AuthenticationError

        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret="completely-incorrect-secret",
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.request("guarded", {}, side_effects=False)
            assert exc_info.value.code == TransportErrorCode.UNAUTHORIZED
            assert handler_invoked is False
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_oversized_payload_rejected_before_dispatch(ephemeral_server_config: TransportConfig):
    handler_invoked = False

    def ping_handler(args):
        nonlocal handler_invoked
        handler_invoked = True
        return "pong"

    registry = CommandRegistry()
    registry.register("ping", ping_handler, validate_empty_arguments)

    # Server configured with 1024 bytes max
    server_cfg = TransportConfig(
        host="127.0.0.1",
        port=0,
        secret=ephemeral_server_config.secret,
        max_payload_size=1024,
    )
    server = RobotServer(server_cfg, registry)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            # Send length header declaring 50,000 bytes (far above 1024 max)
            writer.write(HEADER_STRUCT.pack(50000))
            await writer.drain()

            response_bytes = await read_frame(reader, max_payload_size=65536)
            assert b"payload_too_large" in response_bytes
            assert handler_invoked is False
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_malformed_data_never_reaches_handler(ephemeral_server_config: TransportConfig):
    handler_invoked = False

    def ping_handler(args):
        nonlocal handler_invoked
        handler_invoked = True
        return "pong"

    registry = CommandRegistry()
    registry.register("ping", ping_handler, validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            # 1. Broken JSON
            await write_frame(writer, b"{bad json: 123")
            resp1 = await read_frame(reader)
            assert b"malformed_request" in resp1
            assert handler_invoked is False

            # 2. Valid JSON but completely invalid envelope schema
            await write_frame(writer, b'{"foo": "bar"}')
            resp2 = await read_frame(reader)
            assert b"malformed_request" in resp2
            assert handler_invoked is False
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_code_injection_strings_never_evaluated(ephemeral_server_config: TransportConfig):
    def reply_handler(args):
        # Even if args contains Python code strings, it's just raw string data
        msg = args.get("message", "")
        # Confirm string was not executed
        assert "__import__" in msg
        return {"handled": True}

    from transport.registry import validate_reply_arguments
    registry = CommandRegistry()
    registry.register("reply", reply_handler, validate_reply_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        from transport.client import RobotClient

        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            injection_payload = "__import__('os').system('echo pwned')"
            resp = await client.request("reply", {"message": injection_payload})
            assert resp.ok is True
            assert resp.result == {"handled": True}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_unconfigured_secret_rejects_requests():
    from transport.client import RobotClient
    from transport.protocol import AuthenticationError

    registry = CommandRegistry()
    registry.register("ping", lambda args: "pong", validate_empty_arguments)

    # Server with empty secret must fail closed before binding
    unconfigured_cfg = TransportConfig(
        host="127.0.0.1",
        port=0,
        secret="",
    )
    with pytest.raises(AuthenticationError) as exc_server:
        RobotServer(unconfigured_cfg, registry)
    assert exc_server.value.code == TransportErrorCode.UNAUTHORIZED

    # Server with None secret must also fail closed (either at config validation or server startup)
    with pytest.raises((AuthenticationError, TypeError)):
        none_cfg = TransportConfig(
            host="127.0.0.1",
            port=0,
            secret=None,
        )
        RobotServer(none_cfg, registry)

    # Client with empty secret must fail closed before connecting
    client_cfg = TransportConfig(
        host="127.0.0.1",
        port=9999,
        secret="",
        request_timeout=1.0,
    )
    with pytest.raises(AuthenticationError) as exc_client:
        RobotClient(client_cfg)
    assert exc_client.value.code == TransportErrorCode.UNAUTHORIZED
