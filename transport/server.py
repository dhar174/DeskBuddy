"""Asyncio TCP server with length framing, HMAC authentication, and allowlist dispatch."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from transport.auth import HMACAuthenticator, sign_response
from transport.config import TransportConfig, compute_command_timeout
from transport.framing import read_frame, write_frame
from transport.protocol import (
    CURRENT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    AuthenticationError,
    PayloadTooLargeError,
    ProtocolViolationError,
    RobotRequest,
    RobotResponse,
    TransportError,
    TransportErrorCode,
    TruncatedFrameError,
)
from transport.registry import CommandRegistry

logger = logging.getLogger(__name__)


class RobotServer:
    """Secure, authenticated, frame-based asyncio TCP server for DeskBuddy robot control."""

    def __init__(
        self,
        config: TransportConfig | None = None,
        registry: CommandRegistry | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        secret: str | bytes | None = None,
    ) -> None:
        if config is None:
            self.config = TransportConfig.from_settings(role="server")
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

        self.registry = registry or CommandRegistry()
        self.authenticator = HMACAuthenticator(
            secret=self.config.secret,
            max_clock_skew=self.config.max_clock_skew,
        )
        self._server: asyncio.Server | None = None
        self._active_connections: set[asyncio.Task] = set()
        self._is_running: bool = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def host(self) -> str:
        return self.config.host

    @property
    def port(self) -> int:
        if self._server and self._server.sockets:
            sock = self._server.sockets[0]
            return sock.getsockname()[1]
        return self.config.port

    def register(
        self,
        command_name: str,
        handler: Callable[..., Any],
        validator: Callable[..., Any] | None = None,
    ) -> None:
        """Register a command handler on the server's registry."""
        self.registry.register(command_name, handler, validator)

    async def start(self) -> None:
        """Start listening for incoming client connections."""
        if self._is_running:
            return

        self._server = await asyncio.start_server(
            self._handle_client_connection,
            host=self.config.host,
            port=self.config.port,
        )
        self._is_running = True
        logger.info(
            "RobotServer listening on %s:%d (protocol version %s)",
            self.config.host,
            self.port,
            CURRENT_PROTOCOL_VERSION,
        )

    async def stop(self) -> None:
        """Gracefully stop the server and close existing connections."""
        self._is_running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Cancel active connection tasks
        if self._active_connections:
            for task in list(self._active_connections):
                task.cancel()
            await asyncio.gather(*self._active_connections, return_exceptions=True)
            self._active_connections.clear()

        logger.info("RobotServer stopped")

    async def serve_forever(self) -> None:
        """Serve incoming requests until cancelled."""
        if not self._is_running:
            await self.start()
        if self._server:
            await self._server.serve_forever()

    async def __aenter__(self) -> RobotServer:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    def _handle_client_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.create_task(self._process_client_stream(reader, writer))
        self._active_connections.add(task)
        task.add_done_callback(self._active_connections.discard)

    async def _process_client_stream(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        client_addr = writer.get_extra_info("peername")
        logger.debug("Accepted connection from %s", client_addr)

        try:
            while self._is_running:
                # 1. Read framed request
                try:
                    payload_bytes = await read_frame(
                        reader,
                        max_payload_size=self.config.max_payload_size,
                        timeout=self.config.read_timeout,
                    )
                except EOFError:
                    logger.debug("Client %s closed connection cleanly", client_addr)
                    break
                except PayloadTooLargeError as exc:
                    logger.warning("Rejecting oversized frame from %s: %s", client_addr, exc)
                    response = RobotResponse.fail(
                        request_id=None,
                        code=TransportErrorCode.PAYLOAD_TOO_LARGE,
                        message="Declared frame length exceeds maximum allowed payload limit",
                    )
                    await self._send_response(writer, response)
                    # Connection framing is desynchronized; terminate connection
                    break
                except TruncatedFrameError as exc:
                    logger.warning("Truncated frame from %s: %s", client_addr, exc)
                    # Connection framing desynchronized or client disconnected
                    break
                except TransportError as exc:
                    if exc.code == TransportErrorCode.TIMEOUT:
                        logger.debug("Client %s read timeout reached", client_addr)
                    else:
                        logger.warning("Transport error from %s: %s", client_addr, exc)
                    break

                # 2. Process payload and produce typed response
                response = await self._process_request_payload(payload_bytes)

                # 3. Write framed response back to client
                try:
                    await self._send_response(writer, response)
                except Exception as exc:
                    logger.warning("Failed sending response to %s: %s", client_addr, exc)
                    break

        except asyncio.CancelledError:
            logger.debug("Client handler cancelled for %s", client_addr)
        except Exception as exc:
            logger.exception("Unexpected error in client loop for %s: %s", client_addr, exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug("Closed connection for %s", client_addr)

    async def _process_request_payload(self, payload_bytes: bytes) -> RobotResponse:
        """Parse, validate, authenticate, and dispatch a single request payload."""
        req_id: str | None = None

        # 1. Parse JSON and Request envelope
        try:
            request = RobotRequest.from_json(payload_bytes)
            req_id = request.request_id
        except ProtocolViolationError as exc:
            logger.warning("Protocol violation in payload: %s", exc)
            return RobotResponse.fail(
                request_id=None,
                code=exc.code,
                message=exc.message,
            )
        except Exception as exc:
            logger.warning("Malformed JSON in payload: %s", exc)
            return RobotResponse.fail(
                request_id=None,
                code=TransportErrorCode.MALFORMED_REQUEST,
                message="Malformed JSON request payload",
            )

        # 2. Validate protocol version
        if request.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            logger.warning(
                "Unsupported protocol version '%s' requested", request.protocol_version
            )
            return RobotResponse.fail(
                request_id=req_id,
                code=TransportErrorCode.UNSUPPORTED_PROTOCOL,
                message=f"Unsupported protocol version '{request.protocol_version}'",
            )

        # 3. Authenticate request
        valid, err_code, err_msg = self.authenticator.authenticate(request)
        if not valid:
            logger.warning(
                "Authentication rejected for request %s: [%s] %s",
                req_id,
                err_code,
                err_msg,
            )
            return RobotResponse.fail(
                request_id=req_id,
                code=err_code or TransportErrorCode.UNAUTHORIZED,
                message=err_msg or "Authentication failed",
            )

        # 4. Validate command existence in allowlist
        if not self.registry.is_registered(request.command):
            logger.warning("Unauthorized or unknown command '%s' requested", request.command)
            return RobotResponse.fail(
                request_id=req_id,
                code=TransportErrorCode.UNSUPPORTED_COMMAND,
                message=f"Command '{request.command}' is not supported",
            )

        # 5. Validate arguments against schema
        args_valid, arg_err = self.registry.validate_command(request.command, request.arguments)
        if not args_valid:
            logger.warning("Invalid arguments for command '%s': %s", request.command, arg_err)
            return RobotResponse.fail(
                request_id=req_id,
                code=TransportErrorCode.INVALID_ARGUMENTS,
                message=arg_err or "Invalid command arguments",
            )

        # 6. Execute registered handler safely with command-aware timeout policy
        cmd_timeout = compute_command_timeout(
            request.command, request.arguments, self.config.request_timeout
        )
        try:
            result = await asyncio.wait_for(
                self.registry.execute(request.command, request.arguments),
                timeout=cmd_timeout,
            )
            return RobotResponse.success(request_id=req_id, result=result)
        except asyncio.TimeoutError:
            logger.warning("Command '%s' timed out after %.2fs", request.command, cmd_timeout)
            return RobotResponse.fail(
                request_id=req_id,
                code=TransportErrorCode.TIMEOUT,
                message=f"Command '{request.command}' timed out after {cmd_timeout:.2f}s",
            )
        except TransportError as exc:
            logger.warning("Transport error executing command '%s': %s", request.command, exc)
            err_msg = exc.message
            if exc.code == TransportErrorCode.INTERNAL_ERROR:
                err_msg = "Internal server error during command execution"
            return RobotResponse.fail(
                request_id=req_id,
                code=exc.code,
                message=err_msg,
            )

        except Exception as exc:
            # Do NOT leak internal exception details to the network
            logger.exception("Internal exception executing command '%s': %s", request.command, exc)
            return RobotResponse.fail(
                request_id=req_id,
                code=TransportErrorCode.INTERNAL_ERROR,
                message="Internal server error during command execution",
            )

    async def _send_response(
        self, writer: asyncio.StreamWriter, response: RobotResponse
    ) -> None:
        if self.config.secret:
            sign_response(self.config.secret, response)
        raw_json = response.to_json().encode("utf-8")
        await write_frame(
            writer,
            raw_json,
            max_payload_size=self.config.max_payload_size,
            timeout=self.config.write_timeout,
        )
