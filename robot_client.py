"""Robot client module providing authenticated transport for DeskBuddy.

Replaces the legacy pickle-based RPC with a length-framed, HMAC-authenticated
JSON transport while maintaining backward compatibility for request_to_server callers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from deskbuddy.settings import load_settings
from transport.client import RobotClient, RobotTransport
from transport.config import ClientConfig, TransportConfig
from transport.protocol import (
    ExecutionOutcomeUnknown,
    ProtocolViolationError,
    RobotResponse,
    TransportError,
    TransportErrorCode,
    validate_timeout_override,
)
from transport.transcripts import ACK_TRANSCRIPT_COMMAND, POLL_TRANSCRIPT_COMMAND, is_transcript_id

logger = logging.getLogger(__name__)

# Default network configuration from canonical settings
_settings = load_settings()
HOST = _settings.robot_host
PORT = _settings.robot_port

_shared_client: RobotClient | None = None
_shared_client_loop: asyncio.AbstractEventLoop | None = None
_client_lock: asyncio.Lock | None = None


def _get_client_lock() -> asyncio.Lock:
    global _client_lock
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    return _client_lock


async def get_robot_client() -> RobotClient:
    """Return a shared singleton RobotClient instance for the running event loop."""
    global _shared_client, _shared_client_loop
    current_loop = asyncio.get_running_loop()
    async with _get_client_lock():
        if _shared_client is None or _shared_client_loop is not current_loop:
            settings = load_settings()
            _shared_client = RobotClient(TransportConfig.from_settings(settings, role="client"))
            _shared_client_loop = current_loop
        return _shared_client


async def request_to_server(**kwargs: Any) -> Any:
    """Send a command to the robot server and return the result.

    Maintains backward compatibility with legacy callers:
        kwargs = {"function_name": "check_for_response"}
        kwargs = {"function_name": "reply", "message": "..."}

    Returns the unboxed response result on success, or None on ordinary error/timeout,
    preserving expected legacy caller failure semantics.

    If a side-effecting command has an ambiguous outcome (e.g. connection lost or
    deadline exceeded after transmission), ExecutionOutcomeUnknown is raised so
    callers can distinguish physical uncertainty from successful None or ordinary failure.
    """
    call_timeout: float | None = None
    if "timeout" in kwargs and kwargs["timeout"] is not None:
        call_timeout = validate_timeout_override(kwargs["timeout"])
    elif "request_timeout" in kwargs and kwargs["request_timeout"] is not None:
        call_timeout = validate_timeout_override(kwargs["request_timeout"])

    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("timeout", "request_timeout")}

    command: str = ""
    arguments: dict[str, Any] = {}

    if "function_name" in filtered_kwargs:
        command = str(filtered_kwargs["function_name"])
        arguments = {k: v for k, v in filtered_kwargs.items() if k != "function_name"}
    elif "command" in filtered_kwargs:
        command = str(filtered_kwargs["command"])
        raw_args = filtered_kwargs.get("arguments", {})
        if isinstance(raw_args, dict):
            arguments = {
                k: v for k, v in raw_args.items() if k not in ("timeout", "request_timeout")
            }
        for k, v in filtered_kwargs.items():
            if k not in ("command", "arguments"):
                arguments[k] = v
    else:
        logger.error("request_to_server received neither 'function_name' nor 'command': %s", kwargs)
        return None

    try:
        client = await get_robot_client()
        response = await client.request(command, arguments, timeout=call_timeout)
        if response.ok:
            return response.result
        logger.warning(
            "Robot server returned error for '%s': [%s] %s",
            command,
            response.error.get("code") if response.error else "unknown",
            response.error.get("message") if response.error else "unknown error",
        )
        return None
    except ExecutionOutcomeUnknown:
        # Crucial: Propagate typed exception so callers recognize uncertain physical execution
        logger.warning("Propagating ambiguous execution outcome for '%s'", command)
        raise
    except asyncio.TimeoutError:
        logger.warning("Robot server not responding within timeout for command '%s'", command)
        return None
    except TransportError as exc:
        logger.warning("Transport error executing '%s': %s", command, exc)
        return None
    except Exception as exc:
        logger.error("Error sending request to robot server: %s", exc)
        return None


def _transcript_result(response: RobotResponse) -> Any:
    if not response.ok:
        error = response.error or {}
        raise TransportError(
            error.get("code", TransportErrorCode.INTERNAL_ERROR),
            "Robot transcript request failed",
        )
    if response.error is not None:
        raise ProtocolViolationError(
            TransportErrorCode.MALFORMED_REQUEST,
            "Unexpected error on successful transcript response",
        )
    return response.result


class TranscriptPoller:
    """Deliver only positively acknowledged utterances to the conversation loop."""

    def __init__(self) -> None:
        self._legacy_connection: tuple[RobotClient, int] | None = None
        self._last_legacy_text: str | None = None

    async def next_transcript(self) -> str | None:
        """Run one poll, with one acknowledgement if the new API delivers a record."""
        client = await get_robot_client()
        connection = (client, client.connection_generation)
        if self._legacy_connection == connection and client.is_connected:
            return await self._poll_legacy(client)

        response = await client.request(POLL_TRANSCRIPT_COMMAND)
        if (
            not response.ok
            and response.result is None
            and response.error
            and response.error["code"] == TransportErrorCode.UNSUPPORTED_COMMAND
            and isinstance(response.error["message"], str)
        ):
            self._legacy_connection = (client, client.connection_generation)
            logger.warning("Robot server lacks utterance IDs; using legacy text polling")
            return await self._poll_legacy(client)

        self._legacy_connection = None
        self._last_legacy_text = None
        record = _transcript_result(response)
        if record is None:
            return None
        if not isinstance(record, dict) or record.keys() != {"id", "text"}:
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "Invalid transcript record"
            )
        transcript_id, transcript = record["id"], record["text"]
        if (
            not is_transcript_id(transcript_id)
            or not isinstance(transcript, str)
            or not transcript.strip()
        ):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "Invalid transcript identity or text"
            )
        ack = await client.request(ACK_TRANSCRIPT_COMMAND, {"id": transcript_id})
        accepted = _transcript_result(ack)
        if not isinstance(accepted, bool):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "Invalid transcript acknowledgement"
            )
        return transcript if accepted else None

    async def _poll_legacy(self, client: RobotClient) -> str | None:
        # A new socket must explicitly report lack of support before text
        # polling can consume a turn from it.
        response = await client.request(
            "check_for_response", expected_connection_generation=client.connection_generation
        )
        transcript = _transcript_result(response)
        if transcript in ("No transcription", "", None, self._last_legacy_text):
            return None
        if not isinstance(transcript, str):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "Invalid legacy transcript"
            )
        self._last_legacy_text = transcript
        return transcript


async def send_reply(message: str) -> None:
    """Send speech, preserving typed failures for callers that need delivery status."""
    client = await get_robot_client()
    await client.reply(message)


async def close_client() -> None:
    """Close the shared client connection."""
    global _shared_client, _shared_client_loop
    async with _get_client_lock():
        if _shared_client is not None:
            await _shared_client.close()
            _shared_client = None
            _shared_client_loop = None


__all__ = [
    "HOST",
    "PORT",
    "ClientConfig",
    "ExecutionOutcomeUnknown",
    "RobotClient",
    "RobotTransport",
    "TransportConfig",
    "TransportError",
    "TranscriptPoller",
    "close_client",
    "get_robot_client",
    "request_to_server",
    "send_reply",
]
