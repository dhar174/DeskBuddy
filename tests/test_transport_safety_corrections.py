"""Targeted regression tests for PR #28 transport safety, execution ambiguity, and concurrency corrections.

Verifies:
1. Failure before any transmission.
2. Failure after partial/possible transmission.
3. Response timeout after server-side execution may have started.
4. Successful replay rejection with an explicit unknown outcome.
5. Failed reconnection after possible transmission.
6. Legacy wrapper preserving the unknown outcome.
7. Synchronous hardware operation continuing after coroutine timeout.
8. Serialized hardware execution during an unresolved operation.
9. Long-running command followed by a queued request.
10. Correct request timestamp freshness after lock contention.
11. Client/server deadline grace behavior.
12. Invalid runtime timeout overrides.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

import robot_client as rc
from transport import (
    CommandRegistry,
    ExecutionOutcomeUnknown,
    HardwareExecutor,
    RobotClient,
    RobotServer,
    TransportConfig,
    TransportError,
    TransportErrorCode,
    reset_default_hardware_executor,
    validate_empty_arguments,
    validate_move_arguments,
    validate_reply_arguments,
    validate_say_arguments,
    validate_timeout_override,
)
from transport.framing import read_frame


@pytest.fixture(autouse=True)
def clean_hardware_executor():
    reset_default_hardware_executor()
    yield
    reset_default_hardware_executor()


# ==============================================================================
# 1. Failure before any transmission
# ==============================================================================
@pytest.mark.asyncio
async def test_failure_before_transmission_raises_ordinary_transport_error():
    """Verify failure before transmission produces ordinary TransportError, not ExecutionOutcomeUnknown."""
    # Port not open
    client_cfg = TransportConfig(
        host="127.0.0.1",
        port=65431,
        secret="test-secret-32-chars-long-auth-key!",
        connect_timeout=0.1,
        max_retries=1,
    )
    async with RobotClient(client_cfg) as client:
        with pytest.raises(TransportError) as exc_info:
            await client.request("move", {"motor": 1, "position": 5.0})

        assert not isinstance(exc_info.value, ExecutionOutcomeUnknown)
        assert exc_info.value.code in (TransportErrorCode.INTERNAL_ERROR, TransportErrorCode.TIMEOUT)


# ==============================================================================
# 2. Failure after partial/possible transmission
# ==============================================================================
@pytest.mark.asyncio
async def test_failure_after_transmission_raises_execution_outcome_unknown(
    ephemeral_server_config: TransportConfig,
):
    """Verify connection loss after transmission raises ExecutionOutcomeUnknown for side-effecting commands."""
    registry = CommandRegistry()
    executed_count = 0

    async def counting_move(args):
        nonlocal executed_count
        executed_count += 1
        return "moved"

    registry.register("move", counting_move, validate_move_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    # Hook server to drop connection after receiving the request frame
    async def dropping_stream(reader, writer):
        try:
            # Read frame, then abruptly drop socket
            await read_frame(reader)
            # Execute command on server side before drop
            await counting_move({})
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server._process_client_stream = dropping_stream

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=1.5,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
                await client.move(1, 5.0)

            assert exc_info.value.command == "move"
            assert exc_info.value.request_id is not None
            assert "execution outcome is unknown" in str(exc_info.value)
    finally:
        await server.stop()


# ==============================================================================
# 3. Response timeout after server-side execution may have started
# ==============================================================================
@pytest.mark.asyncio
async def test_response_timeout_after_transmission_raises_execution_outcome_unknown(
    ephemeral_server_config: TransportConfig,
):
    """Verify read timeout after transmission raises ExecutionOutcomeUnknown."""
    registry = CommandRegistry()

    async def slow_move(args):
        await asyncio.sleep(1.0)
        return "moved"

    registry.register("move", slow_move, validate_move_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            read_timeout=0.15,
            request_timeout=0.15,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
                # Command has side effects
                await client.move(1, 5.0)

            assert exc_info.value.command == "move"
            assert exc_info.value.request_id is not None
            assert "execution outcome is unknown" in str(exc_info.value)
    finally:
        await server.stop()


# ==============================================================================
# 4. Successful replay rejection with an explicit unknown outcome
# ==============================================================================
@pytest.mark.asyncio
async def test_signed_replay_rejection_surfaces_execution_outcome_unknown(
    ephemeral_server_config: TransportConfig,
):
    """Verify replayed_request error response raises ExecutionOutcomeUnknown via RobotClient."""
    registry = CommandRegistry()
    registry.register("move", lambda args: "moved", validate_move_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
        )
        async with RobotClient(client_cfg) as client:
            # 1. First execution succeeds with custom request_id
            custom_id = "test-replay-uuid-456"
            resp1 = await client.request(
                "move", {"motor": 1, "position": 5.0}, request_id=custom_id
            )
            assert resp1.ok is True
            assert resp1.result == "moved"

            # 2. Second transmission with exact same request ID gets rejected as replay by server.
            # Client receives signed replayed_request response and raises ExecutionOutcomeUnknown!
            with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
                await client.request(
                    "move", {"motor": 1, "position": 5.0}, request_id=custom_id
                )

            assert exc_info.value.command == "move"
            assert exc_info.value.request_id == custom_id
            assert exc_info.value.code == TransportErrorCode.REPLAYED_REQUEST
    finally:
        await server.stop()


# ==============================================================================
# 5. Failed reconnection after possible transmission
# ==============================================================================
@pytest.mark.asyncio
async def test_failed_reconnection_after_transmission_raises_execution_outcome_unknown():
    """Verify that if reconnect fails after transmission was attempted, ExecutionOutcomeUnknown is raised."""
    async def fake_server_handler(reader, writer):
        try:
            await read_frame(reader)
        finally:
            writer.close()
            await writer.wait_closed()
            # Shut down server listener so subsequent reconnect attempt fails with connection refused
            srv.close()

    srv = await asyncio.start_server(fake_server_handler, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]

    client_cfg = TransportConfig(
        host="127.0.0.1",
        port=port,
        secret="test-secret-32-chars-long-auth-key!",
        request_timeout=1.0,
        connect_timeout=0.2,
        max_retries=1,
    )
    async with RobotClient(client_cfg) as client:
        # Command with retry permitted (safe/idempotent). Attempt 0 transmits, server drops and closes.
        # Attempt 1 attempts to reconnect, but server port is closed.
        with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
            await client.request("ping", {}, side_effects=False)

        assert exc_info.value.command == "ping"
        assert exc_info.value.request_id is not None
        assert "failed to reconnect after transmission attempted" in str(exc_info.value)

    srv.close()
    await srv.wait_closed()


# ==============================================================================
# 6. Legacy wrapper preserving the unknown outcome
# ==============================================================================
@pytest.mark.asyncio
async def test_legacy_wrapper_preserves_execution_outcome_unknown(
    ephemeral_server_config: TransportConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify request_to_server propagates ExecutionOutcomeUnknown and does not convert it to None."""
    registry = CommandRegistry()

    async def dropping_reply(args):
        await asyncio.sleep(0.5)
        return "spoken"

    registry.register("reply", dropping_reply, validate_reply_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    monkeypatch.setenv("DESKBUDDY_ROBOT_HOST", "127.0.0.1")
    monkeypatch.setenv("DESKBUDDY_ROBOT_PORT", str(server.port))
    monkeypatch.setenv("DESKBUDDY_ROBOT_AUTH_SECRET", ephemeral_server_config.secret)
    monkeypatch.setenv("DESKBUDDY_ROBOT_TIMEOUT_SECONDS", "0.15")

    try:
        await rc.close_client()

        # Side-effecting reply command times out after transmission:
        # request_to_server MUST raise ExecutionOutcomeUnknown!
        with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
            await rc.request_to_server(function_name="reply", message="Test ambiguous", timeout=0.15)

        assert exc_info.value.command == "reply"
        assert exc_info.value.request_id is not None
    finally:
        await rc.close_client()
        await server.stop()


# ==============================================================================
# 7. Synchronous hardware operation continuing after coroutine timeout
# ==============================================================================
@pytest.mark.asyncio
async def test_synchronous_hardware_op_continuing_after_coroutine_timeout(
    ephemeral_server_config: TransportConfig,
):
    """Controlled fake hardware test verifying execution_outcome_unknown when synchronous hardware times out."""
    hw_executor = HardwareExecutor(thread_name_prefix="test-hw-sync")
    started_event = threading.Event()
    release_event = threading.Event()
    completed_event = threading.Event()

    def fake_blocking_hardware():
        started_event.set()
        # Block until test releases
        release_event.wait(timeout=5.0)
        completed_event.set()
        return "hardware_completed"

    registry = CommandRegistry()

    async def blocking_handler(args):
        return await hw_executor.execute(fake_blocking_hardware)

    registry.register("move", blocking_handler, validate_move_arguments)
    server = RobotServer(ephemeral_server_config, registry, hardware_executor=hw_executor)
    await server.start()

    try:
        # 1. Start command with short server execution timeout
        # Base timeout 0.15s
        server.config.request_timeout = 0.15

        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=0.15,
        )
        async with RobotClient(client_cfg) as client:
            # Send command and await result; server coroutine will time out while fake hardware is blocked
            with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
                await client.move(1, 5.0, timeout=0.3)

            assert exc_info.value.command == "move"
            # 2. Hardware started and is STILL blocked in thread pool
            assert started_event.is_set()
            assert not completed_event.is_set()
            assert hw_executor.has_unresolved_operation is True

            # 3. Attempting another conflicting command while unresolved is rejected
            with pytest.raises(ExecutionOutcomeUnknown):
                await client.move(1, 6.0, timeout=0.3)

            # 4. Release blocked hardware operation
            release_event.set()
            # Wait for thread to exit
            assert completed_event.wait(timeout=2.0)

            # Let loop process worker completion threadsafe callback
            await asyncio.sleep(0.05)
            assert hw_executor.has_unresolved_operation is False

            # 5. Subsequent promptly completing hardware operation succeeds
            def fast_hardware():
                return "fast_done"

            async def fast_handler(args):
                return await hw_executor.execute(fast_hardware)

            registry.register("move", fast_handler, validate_move_arguments)
            resp = await client.request("move", {"motor": 1, "position": 2.0}, timeout=1.0)
            assert resp.ok is True
            assert resp.result == "fast_done"

    finally:
        release_event.set()
        await hw_executor.shutdown()
        await server.stop()


# ==============================================================================
# 8. Serialized hardware execution during an unresolved operation
# ==============================================================================
@pytest.mark.asyncio
async def test_serialized_hardware_execution_prevents_concurrency():
    """Verify hardware executor strictly serializes calls and rejects conflicting ops while unresolved."""
    hw_executor = HardwareExecutor(thread_name_prefix="test-hw-ser")
    concurrent_detected = False
    in_flight = 0
    lock = threading.Lock()

    def hardware_op(duration: float):
        nonlocal concurrent_detected, in_flight
        with lock:
            in_flight += 1
            if in_flight > 1:
                concurrent_detected = True
        try:
            time.sleep(duration)
        finally:
            with lock:
                in_flight -= 1
        return "done"

    try:
        # Run 3 operations sequentially through executor
        t1 = asyncio.create_task(hw_executor.execute(hardware_op, 0.05))
        t2 = asyncio.create_task(hw_executor.execute(hardware_op, 0.05))
        t3 = asyncio.create_task(hw_executor.execute(hardware_op, 0.05))

        res = await asyncio.gather(t1, t2, t3)
        assert res == ["done", "done", "done"]
        assert not concurrent_detected
    finally:
        await hw_executor.shutdown()


# ==============================================================================
# 9. Long-running command followed by a queued request
# ==============================================================================
@pytest.mark.asyncio
async def test_queued_hardware_work_cancelled_before_pickup_never_executes():
    """Verify that queued executor work cancelled while waiting never executes on hardware."""
    hw_executor = HardwareExecutor(thread_name_prefix="test-hw-queue")
    first_block = threading.Event()
    first_release = threading.Event()
    second_executed = False

    def first_op():
        first_block.set()
        first_release.wait(timeout=2.0)
        return "first"

    def second_op():
        nonlocal second_executed
        second_executed = True
        return "second"

    try:
        task1 = asyncio.create_task(hw_executor.execute(first_op))
        # Wait until first_op starts
        await asyncio.to_thread(first_block.wait, 1.0)

        # Queue second_op
        task2 = asyncio.create_task(hw_executor.execute(second_op))
        await asyncio.sleep(0.02)  # Give time to queue behind task1

        # Cancel task2 while queued
        task2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task2

        # Release first_op
        first_release.set()
        await task1

        await asyncio.sleep(0.05)
        # Second operation must NEVER have run
        assert not second_executed
    finally:
        first_release.set()
        await hw_executor.shutdown()


# ==============================================================================
# 10. Correct request timestamp freshness after lock contention
# ==============================================================================
@pytest.mark.asyncio
async def test_request_timestamp_freshness_after_client_lock_contention(
    ephemeral_server_config: TransportConfig,
):
    """Verify that Request B waiting behind Request A is timestamped freshly after acquiring the lock."""
    registry = CommandRegistry()
    registry.register("ping", lambda args: "pong", validate_empty_arguments)

    async def slow_ping(args):
        # Hold connection for 0.4s
        await asyncio.sleep(0.4)
        return "slow_pong"

    registry.register("slow_ping", slow_ping, validate_empty_arguments)

    # Configure server with very tight clock skew: 0.3s
    tight_server_cfg = TransportConfig(
        host="127.0.0.1",
        port=0,
        secret=ephemeral_server_config.secret,
        max_clock_skew=0.3,
    )
    server = RobotServer(tight_server_cfg, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            max_clock_skew=0.3,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            # Request A runs for 0.4s (exceeding 0.3s max_clock_skew)
            task_a = asyncio.create_task(client.request("slow_ping", {}))

            # Small delay to ensure Request A holds the client lock
            await asyncio.sleep(0.05)

            # Request B queued while A is running.
            # In old buggy code, B was signed now (t=0.05). By the time A finishes (t=0.4),
            # B's timestamp was >0.35s old and rejected as expired_request.
            # In corrected code, B is signed after acquiring the lock, so timestamp is fresh!
            task_b = asyncio.create_task(client.request("ping", {}))

            resp_a, resp_b = await asyncio.gather(task_a, task_b)
            assert resp_a.ok is True
            assert resp_a.result == "slow_pong"
            assert resp_b.ok is True
            assert resp_b.result == "pong"
    finally:
        await server.stop()


# ==============================================================================
# 11. Client/server deadline grace behavior
# ==============================================================================
@pytest.mark.asyncio
async def test_client_server_deadline_grace_permits_near_deadline_success(
    ephemeral_server_config: TransportConfig,
):
    """Verify that a command completing near server deadline succeeds thanks to client response grace."""
    registry = CommandRegistry()

    # Handler executes in 0.9s (near 1.0s server execution deadline)
    async def near_deadline_handler(args):
        await asyncio.sleep(0.9)
        return "completed_just_in_time"

    registry.register("ping", near_deadline_handler, validate_empty_arguments)
    server_cfg = TransportConfig(
        host="127.0.0.1",
        port=0,
        secret=ephemeral_server_config.secret,
        request_timeout=1.0,
    )
    server = RobotServer(server_cfg, registry)
    await server.start()

    try:
        # Client configured with base timeout 1.0s and response_grace 2.0s (total client budget 1.4s)
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=1.0,
            response_grace=2.0,
        )
        async with RobotClient(client_cfg) as client:
            resp = await client.request("ping", {})
            assert resp.ok is True
            assert resp.result == "completed_just_in_time"
    finally:
        await server.stop()


# ==============================================================================
# 12. Invalid runtime timeout overrides
# ==============================================================================
def test_validate_timeout_override_rejects_invalid_values():
    """Verify validate_timeout_override rejects zero, negative, NaN, Inf, bool, and non-numeric types."""
    # Valid positive finite numbers
    assert validate_timeout_override(5.0) == 5.0
    assert validate_timeout_override(1) == 1.0
    assert validate_timeout_override(0.001) == 0.001

    # Zero and negative
    with pytest.raises(ValueError, match="positive"):
        validate_timeout_override(0)
    with pytest.raises(ValueError, match="positive"):
        validate_timeout_override(0.0)
    with pytest.raises(ValueError, match="positive"):
        validate_timeout_override(-1.0)
    with pytest.raises(ValueError, match="positive"):
        validate_timeout_override(-5)

    # NaN and infinities
    with pytest.raises(ValueError, match="finite"):
        validate_timeout_override(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        validate_timeout_override(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        validate_timeout_override(float("-inf"))

    # Booleans
    with pytest.raises(TypeError, match="numeric"):
        validate_timeout_override(True)
    with pytest.raises(TypeError, match="numeric"):
        validate_timeout_override(False)

    # Non-numeric types
    with pytest.raises(TypeError, match="numeric"):
        validate_timeout_override("5.0")
    with pytest.raises(TypeError, match="numeric"):
        validate_timeout_override([5.0])
    with pytest.raises(ValueError, match="None"):
        validate_timeout_override(None)


@pytest.mark.asyncio
async def test_client_request_rejects_invalid_timeout_before_transmission():
    """Verify RobotClient.request fails before any network transmission on invalid timeout."""
    client_cfg = TransportConfig(
        host="127.0.0.1",
        port=9999,  # Unused port
        secret="test-secret-32-chars-long-auth-key!",
    )
    client = RobotClient(client_cfg)

    # Each invalid timeout must raise immediately before touching the network
    for bad_val in (0, -1, float("nan"), float("inf"), True, False, "bad"):
        with pytest.raises((ValueError, TypeError)):
            await client.request("ping", {}, timeout=bad_val)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_legacy_request_to_server_rejects_invalid_timeout_and_does_not_leak_to_args(
    ephemeral_server_config: TransportConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify request_to_server rejects invalid timeout and never passes timeout metadata to command arguments."""
    registry = CommandRegistry()
    received_args: dict[str, Any] = {}

    def inspect_move(args):
        nonlocal received_args
        received_args = dict(args)
        return "inspected"

    registry.register("move", inspect_move, validate_move_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    monkeypatch.setenv("DESKBUDDY_ROBOT_HOST", "127.0.0.1")
    monkeypatch.setenv("DESKBUDDY_ROBOT_PORT", str(server.port))
    monkeypatch.setenv("DESKBUDDY_ROBOT_AUTH_SECRET", ephemeral_server_config.secret)

    try:
        await rc.close_client()

        # 1. Invalid timeout rejected with ValueError/TypeError
        with pytest.raises((ValueError, TypeError)):
            await rc.request_to_server(function_name="move", motor=1, position=5.0, timeout=-2.0)

        # 2. Valid timeout override executed successfully without leaking into arguments
        res = await rc.request_to_server(function_name="move", motor=1, position=5.0, timeout=3.0)
        assert res == "inspected"
        assert "timeout" not in received_args
        assert "request_timeout" not in received_args
        assert received_args == {"motor": 1, "position": 5.0}
    finally:
        await rc.close_client()
        await server.stop()


# ==============================================================================
# 13. Overall deadline expiration after transmission
# ==============================================================================
@pytest.mark.asyncio
async def test_overall_deadline_expiration_after_transmission_raises_execution_outcome_unknown(
    ephemeral_server_config: TransportConfig,
):
    """Verify overall deadline expiration after transmission raises ExecutionOutcomeUnknown."""
    registry = CommandRegistry()

    async def slow_move(args):
        await asyncio.sleep(2.0)
        return "moved"

    registry.register("move", slow_move, validate_move_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        # read_timeout is generous (2.0s), but explicit overall timeout is 0.15s
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            read_timeout=2.0,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
                await client.move(1, 5.0, timeout=0.15)

            assert exc_info.value.command == "move"
            assert exc_info.value.request_id is not None
            assert exc_info.value.code == TransportErrorCode.TIMEOUT
            assert "overall request timed out" in str(exc_info.value).lower()
    finally:
        await server.stop()


# ==============================================================================
# 14. No duplicate physical-command dispatch after an uncertain outcome
# ==============================================================================
@pytest.mark.asyncio
async def test_no_duplicate_physical_command_dispatch_after_uncertain_outcome(
    ephemeral_server_config: TransportConfig,
):
    """Verify side-effecting commands are never automatically retransmitted after uncertain delivery."""
    dispatches = 0
    registry = CommandRegistry()

    async def counting_say(args):
        nonlocal dispatches
        dispatches += 1
        await asyncio.sleep(0.3)
        return "spoken"

    registry.register("say", counting_say, validate_say_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            read_timeout=0.1,
            request_timeout=0.1,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
                await client.say("Alert message", timeout=0.1)

            assert exc_info.value.command == "say"

        # Wait to ensure no background retry was issued
        await asyncio.sleep(0.35)
        # Physical action must have been dispatched at most once
        assert dispatches == 1
    finally:
        await server.stop()


# ==============================================================================
# 15. Legacy wrapper does not automatically retry ambiguous physical action
# ==============================================================================
@pytest.mark.asyncio
async def test_legacy_wrapper_does_not_retry_ambiguous_physical_action(
    ephemeral_server_config: TransportConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify request_to_server does not automatically retry when an ambiguous outcome occurs."""
    speech_count = 0
    registry = CommandRegistry()

    async def counted_reply(args):
        nonlocal speech_count
        speech_count += 1
        await asyncio.sleep(0.5)
        return "spoken"

    registry.register("reply", counted_reply, validate_reply_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    monkeypatch.setenv("DESKBUDDY_ROBOT_HOST", "127.0.0.1")
    monkeypatch.setenv("DESKBUDDY_ROBOT_PORT", str(server.port))
    monkeypatch.setenv("DESKBUDDY_ROBOT_AUTH_SECRET", ephemeral_server_config.secret)

    try:
        await rc.close_client()
        with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
            await rc.request_to_server(function_name="reply", message="One-time action", timeout=0.15)

        assert exc_info.value.command == "reply"
        await asyncio.sleep(0.4)
        assert speech_count == 1
    finally:
        await rc.close_client()
        await server.stop()


# ==============================================================================
# 16. Legacy request_to_server accepts timeout=None without error
# ==============================================================================
@pytest.mark.asyncio
async def test_legacy_request_to_server_accepts_none_timeout(
    ephemeral_server_config: TransportConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify request_to_server handles timeout=None and request_timeout=None as default timeout."""
    registry = CommandRegistry()
    registry.register("ping", lambda args: "pong", validate_empty_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    monkeypatch.setenv("DESKBUDDY_ROBOT_HOST", "127.0.0.1")
    monkeypatch.setenv("DESKBUDDY_ROBOT_PORT", str(server.port))
    monkeypatch.setenv("DESKBUDDY_ROBOT_AUTH_SECRET", ephemeral_server_config.secret)

    try:
        await rc.close_client()
        res = await rc.request_to_server(function_name="ping", timeout=None)
        assert res == "pong"
        res2 = await rc.request_to_server(function_name="ping", request_timeout=None)
        assert res2 == "pong"
    finally:
        await rc.close_client()
        await server.stop()
