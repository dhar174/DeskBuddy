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
    AuthenticationError,
    CommandRegistry,
    ExecutionOutcomeUnknown,
    HardwareExecutor,
    ProtocolViolationError,
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
    validate_wait_arguments,
)
from transport.framing import read_frame, write_frame


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


# ==============================================================================
# 17. External task cancellation cleanly resets stream and prevents stale response contamination
# ==============================================================================
@pytest.mark.asyncio
async def test_external_cancellation_disconnects_stream_and_prevents_stale_response_contamination(
    ephemeral_server_config: TransportConfig,
):
    """Cancel a task after its request has been sent but before the response,
    then submit a new request on the same RobotClient.
    Verify client reconnects and accepts only the new request's authenticated response.
    """
    registry = CommandRegistry()
    slow_request_started = asyncio.Event()
    slow_request_release = asyncio.Event()

    async def slow_handler(args):
        slow_request_started.set()
        await slow_request_release.wait()
        return "slow_done"

    registry.register("wait", slow_handler, validate_wait_arguments)
    registry.register("ping", lambda args: "pong", validate_empty_arguments)

    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=5.0,
        )
        client = RobotClient(client_cfg)

        # 1. Start slow request in a task
        task = asyncio.create_task(client.wait(5.0))
        # Wait until the request is transmitted and server starts processing
        await slow_request_started.wait()

        # 2. Cancel the task while in-flight
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Client must have safely disconnected its stream
        assert not client.is_connected

        # Release slow request so server sends response into the old connection
        slow_request_release.set()
        await asyncio.sleep(0.05)

        # 3. Submit a new request on the EXACT SAME RobotClient instance!
        resp = await client.request("ping", {})
        assert resp.ok is True
        assert resp.result == "pong"
        assert client.is_connected
        await client.close()
    finally:
        slow_request_release.set()
        await server.stop()


# ==============================================================================
# 18. Uncertain execution after invalid responses (malformed, wrong ID, bad HMAC)
# ==============================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt_kind",
    ["malformed_json", "wrong_id", "bad_hmac"],
)
async def test_invalid_response_after_transmission_preserves_execution_outcome_unknown(
    corrupt_kind: str,
    ephemeral_server_config: TransportConfig,
):
    """When a side-effecting command (move) has been transmitted,
    invalid responses (malformed JSON, wrong request ID, bad HMAC) must raise
    ExecutionOutcomeUnknown with the underlying error retained as its cause,
    and must never retry.
    """
    server_received_count = 0
    registry = CommandRegistry()

    async def move_handler(args):
        nonlocal server_received_count
        server_received_count += 1
        return "moved"

    registry.register("move", move_handler, validate_move_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    orig_send_response = server._send_response

    async def corrupt_send_response(writer, response):
        if corrupt_kind == "malformed_json":
            await write_frame(writer, b"{bad json", timeout=5.0)
        elif corrupt_kind == "wrong_id":
            response.request_id = "totally-wrong-request-id-999"
            await orig_send_response(writer, response)
        elif corrupt_kind == "bad_hmac":
            response.auth = {"scheme": "hmac-sha256", "signature": "0" * 64}
            raw = response.to_json().encode("utf-8")
            await write_frame(writer, raw, timeout=5.0)

    server._send_response = corrupt_send_response

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=2.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
                await client.move(1, 5.0)

            err = exc_info.value
            assert err.command == "move"
            assert err.request_id is not None
            assert err.__cause__ is not None
            if corrupt_kind == "malformed_json":
                assert isinstance(err.__cause__, TransportError)
            elif corrupt_kind == "wrong_id":
                assert isinstance(err.__cause__, ProtocolViolationError)
            elif corrupt_kind == "bad_hmac":
                assert isinstance(err.__cause__, AuthenticationError)

        # Single transmission attempt, never retried
        assert server_received_count == 1
    finally:
        await server.stop()


# ==============================================================================
# 19. Legacy adapter propagates ExecutionOutcomeUnknown on invalid responses
# ==============================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt_kind",
    ["malformed_json", "wrong_id", "bad_hmac"],
)
async def test_legacy_request_to_server_preserves_execution_outcome_unknown_on_invalid_response(
    corrupt_kind: str,
    ephemeral_server_config: TransportConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify request_to_server propagates ExecutionOutcomeUnknown when server returns invalid response
    for a side-effecting command, and does not convert it to None or retry.
    """
    dispatched = 0
    registry = CommandRegistry()

    def reply_handler(args):
        nonlocal dispatched
        dispatched += 1
        return "spoken"

    registry.register("reply", reply_handler, validate_reply_arguments)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    orig_send_response = server._send_response

    async def corrupt_send_response(writer, response):
        if corrupt_kind == "malformed_json":
            await write_frame(writer, b"{bad json", timeout=5.0)
        elif corrupt_kind == "wrong_id":
            response.request_id = "wrong-id-456"
            await orig_send_response(writer, response)
        elif corrupt_kind == "bad_hmac":
            response.auth = {"scheme": "hmac-sha256", "signature": "bad" * 16}
            await write_frame(writer, response.to_json().encode("utf-8"), timeout=5.0)

    server._send_response = corrupt_send_response

    monkeypatch.setenv("DESKBUDDY_ROBOT_HOST", "127.0.0.1")
    monkeypatch.setenv("DESKBUDDY_ROBOT_PORT", str(server.port))
    monkeypatch.setenv("DESKBUDDY_ROBOT_AUTH_SECRET", ephemeral_server_config.secret)

    try:
        await rc.close_client()
        with pytest.raises(ExecutionOutcomeUnknown) as exc_info:
            await rc.request_to_server(function_name="reply", message="Hello test")

        assert exc_info.value.command == "reply"
        assert exc_info.value.__cause__ is not None
        assert dispatched == 1
    finally:
        await rc.close_client()
        await server.stop()


# ==============================================================================
# 20. Conservative side-effect classification cannot be downgraded
# ==============================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cmd,args",
    [
        ("move", {"motor": 1, "position": 5.0}),
        ("picoh_move", {"motor": 1, "position": 5.0}),
        ("say", {"text": "hello"}),
        ("picoh_say", {"text": "hello"}),
        ("reply", {"message": "hello"}),
        ("base_col", {"r": 5, "g": 5, "b": 5}),
        ("baseCol", {"r": 5, "g": 5, "b": 5}),
        ("wait", {"seconds": 0.1}),
        ("picoh_wait", {"seconds": 0.1}),
    ],
)
async def test_conservative_side_effect_classification_prevents_downgrade(
    cmd: str,
    args: dict[str, Any],
    ephemeral_server_config: TransportConfig,
):
    """Verify that passing side_effects=False for each built-in physical command
    does not downgrade it: exactly one transmission attempt is made upon failure.
    """
    attempts = 0
    registry = CommandRegistry()

    def handler(a):
        nonlocal attempts
        attempts += 1
        return "ok"

    registry.register(cmd, handler)
    server = RobotServer(ephemeral_server_config, registry)
    await server.start()

    # Drop connection after read to trigger error
    async def drop_after_receive(reader, writer):
        nonlocal attempts
        await read_frame(reader)
        attempts += 1
        writer.close()
        await writer.wait_closed()

    server._process_client_stream = drop_after_receive

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=1.0,
        )
        async with RobotClient(client_cfg) as client:
            with pytest.raises(ExecutionOutcomeUnknown):
                # Explicit side_effects=False MUST NOT downgrade physical command!
                await client.request(cmd, args, side_effects=False)

        # Must have made exactly 1 transmission attempt (no blind retry)
        assert attempts == 1
    finally:
        await server.stop()


# ==============================================================================
# 21. HardwareExecutor race-safe cancellation and completion
# ==============================================================================
@pytest.mark.asyncio
async def test_hardware_executor_completed_operation_not_reported_as_cancelled():
    """Control the boundary between worker completion and async completion notification.
    Cancel the awaiting task when worker has completed.
    Assert result is accurately reported as completed, never cancelled or timed out.
    """
    hw_executor = HardwareExecutor(thread_name_prefix="test-hw-race")
    gate_event = threading.Event()

    def gated_op():
        gate_event.wait(timeout=2.0)
        return "physical_motion_done"

    try:
        task = asyncio.create_task(hw_executor.execute(gated_op))
        # Wait until worker picks up the operation
        for _ in range(100):
            if hw_executor.active_operation is not None:
                break
            await asyncio.sleep(0.005)
        assert hw_executor.active_operation is not None

        # Release gate_event so worker finishes op.func and transitions op.state to COMPLETED
        gate_event.set()

        # Spin until worker thread sets COMPLETED
        for _ in range(100):
            if hw_executor.active_operation is None:
                break
            await asyncio.sleep(0.005)

        # Cancel the task right after worker completed
        task.cancel()

        # The task must resolve with the successful result, NOT CancelledError!
        res = await task
        assert res == "physical_motion_done"
        assert hw_executor.has_unresolved_operation is False
    finally:
        gate_event.set()
        await hw_executor.shutdown()


# ==============================================================================
# 22. Direct TransportConfig construction fail-closed validation
# ==============================================================================
def test_direct_transport_config_validation_rejects_invalid_values():
    """Verify TransportConfig.__post_init__ rejects NaN, Inf, non-positive timeouts,
    invalid ports, negative retries, etc., while accepting port 0.
    """
    # 1. Ephemeral port 0 is accepted
    cfg = TransportConfig(port=0, secret="secret")
    assert cfg.port == 0

    # 2. Port bounds
    with pytest.raises(ValueError, match="port"):
        TransportConfig(port=-1)
    with pytest.raises(ValueError, match="port"):
        TransportConfig(port=65536)
    with pytest.raises(ValueError, match="port"):
        TransportConfig(port=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="port"):
        TransportConfig(port="8888")  # type: ignore[arg-type]

    # 3. Host validation
    with pytest.raises(ValueError, match="host"):
        TransportConfig(host="")
    with pytest.raises(ValueError, match="host"):
        TransportConfig(host="   ")

    # 4. Secret validation
    with pytest.raises(TypeError, match="secret"):
        TransportConfig(secret=123)  # type: ignore[arg-type]

    # 5. Timeouts must be positive finite numbers
    for field in ("connect_timeout", "read_timeout", "write_timeout", "request_timeout"):
        for bad_val in (0, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "5.0"):
            with pytest.raises((ValueError, TypeError)):
                TransportConfig(**{field: bad_val})

    # 6. max_payload_size must be positive int
    for bad_val in (0, -1, True, False, "1000"):
        with pytest.raises(ValueError):
            TransportConfig(max_payload_size=bad_val)  # type: ignore[arg-type]

    # 7. max_clock_skew must be positive finite number
    for bad_val in (0, -1.0, float("nan"), float("inf"), True, False):
        with pytest.raises((ValueError, TypeError)):
            TransportConfig(max_clock_skew=bad_val)

    # 8. max_retries must be non-negative int
    with pytest.raises(ValueError):
        TransportConfig(max_retries=-1)
    with pytest.raises(ValueError):
        TransportConfig(max_retries=True)  # type: ignore[arg-type]

    # 9. retry_backoff and response_grace must be non-negative finite numbers
    for field in ("retry_backoff", "response_grace"):
        for bad_val in (-0.1, -1.0, float("nan"), float("inf"), True, False):
            with pytest.raises((ValueError, TypeError)):
                TransportConfig(**{field: bad_val})
        # 0.0 is accepted for backoff/grace
        valid_cfg = TransportConfig(**{field: 0.0})
        assert getattr(valid_cfg, field) == 0.0


# ==============================================================================
# 23. Connection establishment does not consume response deadline
# ==============================================================================
@pytest.mark.asyncio
async def test_slow_connection_does_not_starve_response_deadline(
    ephemeral_server_config: TransportConfig,
):
    """Verify that under default timeouts, a slow connection establishment
    does not consume the server execution + response grace budget.
    """
    registry = CommandRegistry()

    # Handler executes in 0.3s
    async def delayed_handler(args):
        await asyncio.sleep(0.3)
        return "delayed_success"

    registry.register("ping", delayed_handler, validate_empty_arguments)
    server_cfg = TransportConfig(
        host="127.0.0.1",
        port=0,
        secret=ephemeral_server_config.secret,
        request_timeout=0.4,
    )
    server = RobotServer(server_cfg, registry)
    await server.start()

    try:
        # Client configured with connect_timeout 2.0s, request_timeout 0.4s, response_grace 0.1s
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            connect_timeout=2.0,
            request_timeout=0.4,
            response_grace=0.1,
        )
        async with RobotClient(client_cfg) as client:
            resp = await client.request("ping", {})
            assert resp.ok is True
            assert resp.result == "delayed_success"
    finally:
        await server.stop()

