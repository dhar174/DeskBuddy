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
from transport.config import ClientConfig, TransportConfig, compute_command_timeout
from transport.protocol import ExecutionOutcomeUnknown, TransportError

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

    Maintains backward compatibility with legacy calls from start_bot.py:
        kwargs = {"function_name": "check_for_response"}
        kwargs = {"function_name": "reply", "message": "..."}

    Returns the unboxed response result on success, or None on error/timeout,
    preserving expected legacy caller failure semantics.
    """
    command: str = ""
    arguments: dict[str, Any] = {}

    call_timeout: float | None = None
    if "timeout" in kwargs:
        call_timeout = float(kwargs["timeout"])
    elif "request_timeout" in kwargs:
        call_timeout = float(kwargs["request_timeout"])

    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("timeout", "request_timeout")}

    if "function_name" in filtered_kwargs:
        command = str(filtered_kwargs["function_name"])
        arguments = {k: v for k, v in filtered_kwargs.items() if k != "function_name"}
    elif "command" in filtered_kwargs:
        command = str(filtered_kwargs["command"])
        arguments = dict(filtered_kwargs.get("arguments", {}) or {})
        for k, v in filtered_kwargs.items():
            if k not in ("command", "arguments"):
                arguments[k] = v
    else:
        logger.error("request_to_server received neither 'function_name' nor 'command': %s", kwargs)
        return None

    try:
        client = await get_robot_client()
        if call_timeout is None:
            call_timeout = compute_command_timeout(command, arguments, client.config.request_timeout)
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
    except ExecutionOutcomeUnknown as exc:
        logger.warning("Ambiguous execution outcome for '%s': %s", command, exc)
        return None
    except asyncio.TimeoutError:
        logger.warning("Robot server not responding within timeout for command '%s'", command)
        return None
    except TransportError as exc:
        logger.warning("Transport error executing '%s': %s", command, exc)
        return None
    except Exception as exc:
        logger.error("Error sending request to robot server: %s", exc)
        return None


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
    "close_client",
    "get_robot_client",
    "request_to_server",
]
