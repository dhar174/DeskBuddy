"""Asyncio TCP client for DeskBuddy robot transport."""

from __future__ import annotations

import abc
import asyncio
import logging
import random
from collections.abc import Mapping
from typing import Any

from transport.auth import sign_request, verify_response
from transport.config import TransportConfig, compute_command_timeout
from transport.framing import read_frame, write_frame
from transport.protocol import (
    CURRENT_PROTOCOL_VERSION,
    AuthenticationError,
    ExecutionOutcomeUnknown,
    PayloadTooLargeError,
    ProtocolViolationError,
    RobotRequest,
    RobotResponse,
    TransportError,
    TransportErrorCode,
    TruncatedFrameError,
    validate_timeout_override,
)

logger = logging.getLogger(__name__)

SAFE_IDEMPOTENT_COMMANDS: frozenset[str] = frozenset(
    {"ping", "simple_test", "check_for_response", "test_timing", "hang"}
)

KNOWN_PHYSICAL_COMMANDS: frozenset[str] = frozenset(
    {
        "move",
        "picoh_move",
        "say",
        "picoh_say",
        "reply",
        "base_col",
        "basecol",
        "wait",
        "picoh_wait",
    }
)


def is_side_effecting_command(command: str) -> bool:
    """Determine whether a command has potential side effects on physical hardware/state."""
    cmd_lower = command.lower()
    if cmd_lower in KNOWN_PHYSICAL_COMMANDS:
        return True
    return cmd_lower not in SAFE_IDEMPOTENT_COMMANDS


class RobotTransport(abc.ABC):
    """Abstract transport interface for DeskBuddy robot communications."""

    @abc.abstractmethod
    async def request(
        self,
        command: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
        side_effects: bool | None = None,
        request_id: str | None = None,
    ) -> RobotResponse:
        """Send a command request and await the typed response."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close transport resources."""

    async def __aenter__(self) -> RobotTransport:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


class RobotClient(RobotTransport):
    """Concrete authenticated, length-framed client for the DeskBuddy robot server."""

    def __init__(
        self,
        config: TransportConfig | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        secret: str | bytes | None = None,
    ) -> None:
        base_config = (
            TransportConfig.from_settings(role="client") if config is None else config
        )
        # Overrides produce a re-validated copy; the caller's config is never mutated.
        self.config = base_config.with_overrides(host=host, port=port, secret=secret)

        if not self.config.secret or not self.config.secret.strip():
            raise AuthenticationError(
                TransportErrorCode.UNAUTHORIZED,
                "Missing robot authentication secret. Set DESKBUDDY_ROBOT_AUTH_SECRET.",
            )

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def _ensure_connected(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if self._reader is not None and self._writer is not None and not self._writer.is_closing():
            return self._reader, self._writer

        await self._disconnect()

        attempt = 0
        last_exc: Exception | None = None
        max_attempts = max(1, self.config.max_retries)

        while attempt < max_attempts:
            attempt += 1
            try:
                connect_coro = asyncio.open_connection(self.config.host, self.config.port)
                reader, writer = await asyncio.wait_for(
                    connect_coro, timeout=self.config.connect_timeout
                )
                self._reader = reader
                self._writer = writer
                logger.debug(
                    "Connected to robot server at %s:%d (attempt %d/%d)",
                    self.config.host,
                    self.config.port,
                    attempt,
                    max_attempts,
                )
                return reader, writer
            except (OSError, asyncio.TimeoutError) as exc:
                last_exc = exc
                logger.debug(
                    "Connection attempt %d/%d to %s:%d failed: %s",
                    attempt,
                    max_attempts,
                    self.config.host,
                    self.config.port,
                    exc,
                )
                if attempt < max_attempts:
                    backoff = self.config.retry_backoff * (2 ** (attempt - 1))
                    jitter = random.uniform(0, 0.05)
                    await asyncio.sleep(backoff + jitter)

        raise TransportError(
            TransportErrorCode.TIMEOUT if isinstance(last_exc, asyncio.TimeoutError) else TransportErrorCode.INTERNAL_ERROR,
            f"Failed to connect to {self.config.host}:{self.config.port} after {max_attempts} attempts: {last_exc}",
        ) from last_exc

    async def _disconnect(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            try:
                writer.close()
                await asyncio.shield(writer.wait_closed())
            except Exception:
                pass

    async def close(self) -> None:
        async with self._lock:
            await self._disconnect()

    async def request(
        self,
        command: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
        side_effects: bool | None = None,
        request_id: str | None = None,
    ) -> RobotResponse:
        """Send an authenticated command and wait for a verified response.

        Never retries potentially executed physical commands.
        Retries only safe/idempotent commands or pre-transmission failures.
        """
        if timeout is not None:
            timeout = validate_timeout_override(timeout)

        # Conservative side-effect classification:
        # Known physical commands are unconditionally side-effecting.
        # Explicit side_effects=True can upgrade an otherwise safe command,
        # but side_effects=False can NEVER weaken protections on known physical commands.
        cmd_lower = command.lower()
        if cmd_lower in KNOWN_PHYSICAL_COMMANDS:
            is_side_effecting = True
        elif side_effects is not None:
            is_side_effecting = side_effects
        else:
            is_side_effecting = is_side_effecting_command(command)

        server_cmd_timeout = compute_command_timeout(
            command, arguments, self.config.request_timeout
        )
        read_cmd_timeout = compute_command_timeout(
            command, arguments, self.config.read_timeout
        )
        if timeout is not None:
            effective_timeout = timeout
            effective_read_timeout = timeout
        else:
            grace = min(self.config.response_grace, max(0.01, server_cmd_timeout * 0.4))
            read_grace = min(self.config.response_grace, max(0.01, read_cmd_timeout * 0.4))
            effective_timeout = server_cmd_timeout + grace
            effective_read_timeout = read_cmd_timeout + read_grace

        # Phase 1: Wait to acquire lock (local queueing deadline)
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        lock_timeout = (
            effective_timeout
            if timeout is not None
            else self.config.connect_timeout + self.config.request_timeout
        )
        lock_acquired = False
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=lock_timeout)
            lock_acquired = True
        except asyncio.TimeoutError as exc:
            raise TransportError(
                TransportErrorCode.TIMEOUT,
                f"Client lock acquisition timed out after {lock_timeout:.2f}s before transmission",
            ) from exc

        try:
            transmission_attempted = False
            request: RobotRequest | None = None

            # Phase 2: Ensure connection (pre-transmission)
            # Dedicated bounded budget so slow connection establishment does not
            # consume the server execution plus response-grace deadline.
            if timeout is not None:
                elapsed = loop.time() - start_time
                remaining_for_connect = timeout - elapsed
                if remaining_for_connect <= 0:
                    raise TransportError(
                        TransportErrorCode.TIMEOUT,
                        f"Request timed out after {elapsed:.2f}s before connection",
                    )
                try:
                    reader, writer = await asyncio.wait_for(
                        self._ensure_connected(), timeout=remaining_for_connect
                    )
                except asyncio.TimeoutError as exc:
                    raise TransportError(
                        TransportErrorCode.TIMEOUT,
                        f"Connection timed out after {remaining_for_connect:.2f}s before transmission",
                    ) from exc
            else:
                reader, writer = await self._ensure_connected()

            # Phase 3: Transmission and response deadline (starts after connection succeeds)
            if timeout is not None:
                elapsed = loop.time() - start_time
                remaining_for_response = timeout - elapsed
                if remaining_for_response <= 0:
                    raise TransportError(
                        TransportErrorCode.TIMEOUT,
                        f"Request timed out after {elapsed:.2f}s before transmission",
                    )
                post_connect_timeout = remaining_for_response
                effective_read_timeout = min(effective_read_timeout, post_connect_timeout)
            else:
                post_connect_timeout = effective_timeout

            async def _transmit_and_receive() -> RobotResponse:
                nonlocal transmission_attempted, request, reader, writer

                # Create and sign request IMMEDIATELY before transmission
                request = RobotRequest.create(
                    command=command,
                    arguments=arguments,
                    protocol_version=CURRENT_PROTOCOL_VERSION,
                    request_id=request_id,
                )
                sign_request(self.config.secret, request)
                request_bytes = request.to_json().encode("utf-8")

                if len(request_bytes) > self.config.max_payload_size:
                    raise PayloadTooLargeError(
                        TransportErrorCode.PAYLOAD_TOO_LARGE,
                        f"Outgoing frame size {len(request_bytes)} exceeds maximum {self.config.max_payload_size}",
                    )

                if is_side_effecting:
                    # Side-effecting command: single transmission attempt, no blind retransmission
                    transmission_attempted = True
                    try:
                        await write_frame(
                            writer,
                            request_bytes,
                            max_payload_size=self.config.max_payload_size,
                            timeout=self.config.write_timeout,
                        )
                        response_bytes = await read_frame(
                            reader,
                            max_payload_size=self.config.max_payload_size,
                            timeout=effective_read_timeout,
                        )
                    except (EOFError, ConnectionError, TruncatedFrameError) as exc:
                        logger.debug("Connection dropped after transmission attempted: %s", exc)
                        await self._disconnect()
                        raise ExecutionOutcomeUnknown(
                            request_id=request.request_id,
                            command=command,
                            message=(
                                f"Command '{command}' execution outcome is unknown: connection dropped "
                                f"after transmission was attempted: {exc}"
                            ),
                        ) from exc
                    except TransportError as exc:
                        # Any framing/transport failure after possible transmission
                        # (oversized response header, timeout, invalid frame, ...)
                        # leaves the physical outcome unknown. Never downgrade it.
                        await self._disconnect()
                        if exc.code == TransportErrorCode.TIMEOUT:
                            detail = "read timed out after request was transmitted."
                        else:
                            detail = (
                                f"response framing failed after request was transmitted "
                                f"({type(exc).__name__}: {exc.message})."
                            )
                        raise ExecutionOutcomeUnknown(
                            request_id=request.request_id,
                            command=command,
                            message=f"Command '{command}' execution outcome is unknown: {detail}",
                            code=exc.code,
                        ) from exc
                    except asyncio.TimeoutError as exc:
                        await self._disconnect()
                        raise ExecutionOutcomeUnknown(
                            request_id=request.request_id,
                            command=command,
                            message=(
                                f"Command '{command}' execution outcome is unknown: read timed out "
                                f"after request was transmitted."
                            ),
                            code=TransportErrorCode.TIMEOUT,
                        ) from exc
                    except Exception as exc:
                        await self._disconnect()
                        raise ExecutionOutcomeUnknown(
                            request_id=request.request_id,
                            command=command,
                            message=(
                                f"Command '{command}' execution outcome is unknown: unexpected error "
                                f"after transmission attempted: {exc}"
                            ),
                        ) from exc
                else:
                    # Safe/idempotent command: safe connection retry permitted
                    response_bytes = b""
                    for send_attempt in range(2):
                        if send_attempt > 0:
                            try:
                                reader, writer = await self._ensure_connected()
                            except Exception as exc:
                                if transmission_attempted:
                                    raise ExecutionOutcomeUnknown(
                                        request_id=request.request_id,
                                        command=command,
                                        message=(
                                            f"Command '{command}' execution outcome is unknown: failed to reconnect "
                                            f"after transmission attempted: {exc}"
                                        ),
                                    ) from exc
                                raise
                        try:
                            transmission_attempted = True
                            await write_frame(
                                writer,
                                request_bytes,
                                max_payload_size=self.config.max_payload_size,
                                timeout=self.config.write_timeout,
                            )
                            response_bytes = await read_frame(
                                reader,
                                max_payload_size=self.config.max_payload_size,
                                timeout=effective_read_timeout,
                            )
                            break
                        except (EOFError, ConnectionError, TruncatedFrameError) as exc:
                            logger.debug("Connection dropped during attempt %d: %s", send_attempt, exc)
                            await self._disconnect()
                            if send_attempt == 1:
                                raise TransportError(
                                    TransportErrorCode.INTERNAL_ERROR,
                                    f"Connection interrupted during request: {exc}",
                                ) from exc
                        except Exception:
                            await self._disconnect()
                            raise

                # Parse and validate response schema
                try:
                    response = RobotResponse.from_json(response_bytes)
                except Exception as exc:
                    await self._disconnect()
                    if is_side_effecting and transmission_attempted:
                        req_id = request.request_id if request else "unknown"
                        raise ExecutionOutcomeUnknown(
                            request_id=req_id,
                            command=command,
                            message=(
                                f"Command '{command}' execution outcome is unknown: server returned malformed "
                                f"response after request was transmitted: {exc}"
                            ),
                            code=TransportErrorCode.MALFORMED_REQUEST,
                        ) from exc
                    if isinstance(exc, TransportError):
                        raise
                    raise TransportError(
                        TransportErrorCode.MALFORMED_REQUEST,
                        f"Server returned malformed response: {exc}",
                    ) from exc

                # Authenticate server response. Any unexpected verification error is
                # treated as an authentication failure, never as a trusted response.
                verify_exc: Exception | None = None
                try:
                    valid, err_code, err_msg = verify_response(
                        self.config.secret,
                        response,
                        expected_request_id=request.request_id,
                        max_clock_skew=self.config.max_clock_skew,
                    )
                except Exception as exc:
                    verify_exc = exc
                    valid, err_code, err_msg = (
                        False,
                        TransportErrorCode.UNAUTHORIZED,
                        "Server response authentication could not be verified",
                    )
                if not valid:
                    await self._disconnect()
                    underlying: ProtocolViolationError | AuthenticationError
                    if err_code in (
                        TransportErrorCode.UNSUPPORTED_PROTOCOL,
                        TransportErrorCode.MALFORMED_REQUEST,
                    ):
                        underlying = ProtocolViolationError(err_code, err_msg or "Protocol violation")
                    else:
                        underlying = AuthenticationError(
                            err_code or TransportErrorCode.UNAUTHORIZED,
                            err_msg or "Failed to authenticate server response",
                        )
                    if verify_exc is not None:
                        underlying.__cause__ = verify_exc
                    if is_side_effecting and transmission_attempted:
                        req_id = request.request_id if request else "unknown"
                        raise ExecutionOutcomeUnknown(
                            request_id=req_id,
                            command=command,
                            message=(
                                f"Command '{command}' execution outcome is unknown: server response failed "
                                f"authentication/validation after request was transmitted: {err_msg}"
                            ),
                            code=err_code or TransportErrorCode.UNAUTHORIZED,
                        ) from underlying
                    raise underlying

                # Check for explicit replayed_request or execution_outcome_unknown error codes
                if not response.ok and response.error:
                    resp_err_code = response.error.get("code")
                    if resp_err_code == TransportErrorCode.REPLAYED_REQUEST.value:
                        raise ExecutionOutcomeUnknown(
                            request_id=request.request_id,
                            command=command,
                            message=(
                                f"Command '{command}' execution outcome is unknown: request was sent "
                                "but connection was lost before receiving response, and retry was rejected as duplicate."
                            ),
                            code=TransportErrorCode.REPLAYED_REQUEST,
                        )
                    elif resp_err_code == TransportErrorCode.EXECUTION_OUTCOME_UNKNOWN.value:
                        raise ExecutionOutcomeUnknown(
                            request_id=request.request_id,
                            command=command,
                            message=response.error.get(
                                "message",
                                f"Command '{command}' execution outcome is unknown on server.",
                            ),
                            code=TransportErrorCode.EXECUTION_OUTCOME_UNKNOWN,
                        )

                return response

            try:
                return await asyncio.wait_for(_transmit_and_receive(), timeout=post_connect_timeout)
            except asyncio.TimeoutError as exc:
                await self._disconnect()
                req_id = request.request_id if request else "unknown"
                if transmission_attempted and is_side_effecting:
                    raise ExecutionOutcomeUnknown(
                        request_id=req_id,
                        command=command,
                        message=(
                            f"Command '{command}' execution outcome is unknown: overall request timed out "
                            f"after {post_connect_timeout:.2f}s following transmission."
                        ),
                        code=TransportErrorCode.TIMEOUT,
                    ) from exc
                raise TransportError(
                    TransportErrorCode.TIMEOUT,
                    f"Overall request timed out after {post_connect_timeout:.2f}s",
                ) from exc
        except asyncio.CancelledError:
            req_id = request.request_id if request else "unknown"
            logger.debug(
                "Request cancelled for command '%s' (request_id=%s, side_effects=%s, transmitted=%s); disconnecting stream",
                command,
                req_id,
                is_side_effecting,
                transmission_attempted,
            )
            try:
                await asyncio.shield(self._disconnect())
            except Exception:
                pass
            raise
        finally:
            if lock_acquired:
                self._lock.release()


    # Domain convenience methods
    async def check_for_response(self, *, timeout: float | None = None) -> str:
        """Poll the robot server for speech transcription."""
        response = await self.request("check_for_response", {}, timeout=timeout, side_effects=False)
        if not response.ok:
            raise TransportError(
                response.error.get("code", TransportErrorCode.INTERNAL_ERROR) if response.error else TransportErrorCode.INTERNAL_ERROR,
                response.error.get("message", "check_for_response failed") if response.error else "check_for_response failed",
            )
        if not isinstance(response.result, str):
            raise TransportError(
                TransportErrorCode.INTERNAL_ERROR,
                f"check_for_response result must be a string, got {type(response.result).__name__}",
            )
        return response.result

    async def reply(self, message: str, *, timeout: float | None = None) -> Any:
        """Send a reply text message for Picoh to speak."""
        response = await self.request("reply", {"message": message}, timeout=timeout, side_effects=True)
        if not response.ok:
            raise TransportError(
                response.error.get("code", TransportErrorCode.INTERNAL_ERROR) if response.error else TransportErrorCode.INTERNAL_ERROR,
                response.error.get("message", "Reply command failed") if response.error else "Reply failed",
            )
        return response.result

    async def ping(self, *, timeout: float | None = None) -> bool:
        """Send simple_test connectivity check."""
        try:
            response = await self.request("simple_test", {}, timeout=timeout, side_effects=False)
            return response.ok
        except Exception:
            return False

    async def base_col(self, r: int, g: int, b: int, *, timeout: float | None = None) -> None:
        """Set Picoh base LED color (r, g, b in [0, 10])."""
        response = await self.request("base_col", {"r": r, "g": g, "b": b}, timeout=timeout, side_effects=True)
        if not response.ok:
            raise TransportError(
                response.error.get("code", TransportErrorCode.INTERNAL_ERROR) if response.error else TransportErrorCode.INTERNAL_ERROR,
                response.error.get("message", "base_col failed") if response.error else "base_col failed",
            )

    async def say(self, text: str, *, timeout: float | None = None) -> None:
        """Direct TTS command."""
        response = await self.request("say", {"text": text}, timeout=timeout, side_effects=True)
        if not response.ok:
            raise TransportError(
                response.error.get("code", TransportErrorCode.INTERNAL_ERROR) if response.error else TransportErrorCode.INTERNAL_ERROR,
                response.error.get("message", "say failed") if response.error else "say failed",
            )

    async def move(self, motor: int | str, position: float, *, timeout: float | None = None) -> None:
        """Direct servo move with safety range checking."""
        response = await self.request("move", {"motor": motor, "position": position}, timeout=timeout, side_effects=True)
        if not response.ok:
            raise TransportError(
                response.error.get("code", TransportErrorCode.INTERNAL_ERROR) if response.error else TransportErrorCode.INTERNAL_ERROR,
                response.error.get("message", "move failed") if response.error else "move failed",
            )

    async def wait(self, seconds: float, *, timeout: float | None = None) -> None:
        """Pause Picoh execution with deadline aligned to requested duration."""
        response = await self.request("wait", {"seconds": float(seconds)}, timeout=timeout, side_effects=True)
        if not response.ok:
            raise TransportError(
                response.error.get("code", TransportErrorCode.INTERNAL_ERROR) if response.error else TransportErrorCode.INTERNAL_ERROR,
                response.error.get("message", "wait failed") if response.error else "wait failed",
            )
