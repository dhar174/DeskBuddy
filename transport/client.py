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


def is_side_effecting_command(command: str) -> bool:
    """Determine whether a command has potential side effects on physical hardware/state."""
    return command.lower() not in SAFE_IDEMPOTENT_COMMANDS


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
        if config is None:
            self.config = TransportConfig.from_settings(role="client")
        else:
            self.config = config

        if host is not None:
            self.config.host = host
        if port is not None:
            self.config.port = port
        if secret is not None:
            self.config.secret = secret if isinstance(secret, str) else secret.decode("utf-8")

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
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None

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
    ) -> RobotResponse:
        """Send an authenticated command and wait for a verified response.

        Never retries potentially executed physical commands.
        Retries only safe/idempotent commands or pre-transmission failures.
        """
        if timeout is not None:
            timeout = validate_timeout_override(timeout)

        is_side_effecting = (
            side_effects if side_effects is not None else is_side_effecting_command(command)
        )

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
        lock_timeout = (
            effective_timeout
            if timeout is not None
            else self.config.connect_timeout + self.config.request_timeout
        )
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=lock_timeout)
        except asyncio.TimeoutError as exc:
            raise TransportError(
                TransportErrorCode.TIMEOUT,
                f"Client lock acquisition timed out after {lock_timeout:.2f}s before transmission",
            ) from exc

        try:
            transmission_attempted = False
            request: RobotRequest | None = None

            async def _execute_under_lock() -> RobotResponse:
                nonlocal transmission_attempted, request

                # Phase 2: Ensure connection (pre-transmission)
                try:
                    reader, writer = await self._ensure_connected()
                except Exception as exc:
                    if transmission_attempted:
                        req_id = request.request_id if request else "unknown"
                        raise ExecutionOutcomeUnknown(
                            request_id=req_id,
                            command=command,
                            message=(
                                f"Command '{command}' execution outcome is unknown: failed to reconnect "
                                f"after transmission was already attempted: {exc}"
                            ),
                        ) from exc
                    raise

                # Phase 3: Create and sign request IMMEDIATELY before transmission
                request = RobotRequest.create(
                    command=command,
                    arguments=arguments,
                    protocol_version=CURRENT_PROTOCOL_VERSION,
                )
                sign_request(self.config.secret, request)
                request_bytes = request.to_json().encode("utf-8")

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
                        await self._disconnect()
                        if exc.code == TransportErrorCode.TIMEOUT:
                            raise ExecutionOutcomeUnknown(
                                request_id=request.request_id,
                                command=command,
                                message=(
                                    f"Command '{command}' execution outcome is unknown: read timed out "
                                    f"after request was transmitted."
                                ),
                                code=TransportErrorCode.TIMEOUT,
                            ) from exc
                        raise
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
                    if isinstance(exc, TransportError):
                        raise
                    raise TransportError(
                        TransportErrorCode.MALFORMED_REQUEST,
                        f"Server returned malformed response: {exc}",
                    ) from exc

                # Authenticate server response
                valid, err_code, err_msg = verify_response(
                    self.config.secret,
                    response,
                    expected_request_id=request.request_id,
                    max_clock_skew=self.config.max_clock_skew,
                )
                if not valid:
                    await self._disconnect()
                    if err_code in (
                        TransportErrorCode.UNSUPPORTED_PROTOCOL,
                        TransportErrorCode.MALFORMED_REQUEST,
                    ):
                        raise ProtocolViolationError(err_code, err_msg or "Protocol violation")
                    raise AuthenticationError(
                        err_code or TransportErrorCode.UNAUTHORIZED,
                        err_msg or "Failed to authenticate server response",
                    )

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
                return await asyncio.wait_for(_execute_under_lock(), timeout=effective_timeout)
            except asyncio.TimeoutError as exc:
                await self._disconnect()
                req_id = request.request_id if request else "unknown"
                if transmission_attempted and is_side_effecting:
                    raise ExecutionOutcomeUnknown(
                        request_id=req_id,
                        command=command,
                        message=(
                            f"Command '{command}' execution outcome is unknown: overall request timed out "
                            f"after {effective_timeout:.2f}s following transmission."
                        ),
                        code=TransportErrorCode.TIMEOUT,
                    ) from exc
                raise TransportError(
                    TransportErrorCode.TIMEOUT,
                    f"Overall request timed out after {effective_timeout:.2f}s",
                ) from exc
        finally:
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
