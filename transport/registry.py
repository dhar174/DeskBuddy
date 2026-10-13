"""Strict command allowlist, argument validators, and dispatch registry for DeskBuddy."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from transport.protocol import TransportError, TransportErrorCode

ValidatorFunc = Callable[[dict[str, Any]], tuple[bool, str | None]]
HandlerFunc = Callable[[dict[str, Any]], Awaitable[Any] | Any]


def validate_empty_arguments(arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate that command accepts no arguments."""
    if arguments:
        extra = sorted(arguments.keys())
        return False, f"Command accepts no arguments, received unexpected keys: {extra}"
    return True, None


def validate_reply_arguments(arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate arguments for 'reply' command."""
    allowed_keys = {"message", "text"}
    unexpected = sorted(set(arguments.keys()) - allowed_keys)
    if unexpected:
        return False, f"Unexpected arguments for reply: {unexpected}"

    message_val = arguments.get("message")
    if message_val is None:
        message_val = arguments.get("text")

    if message_val is None:
        return False, "Missing required argument 'message' or 'text'"

    if not isinstance(message_val, str):
        return False, f"Argument 'message' must be a string, got {type(message_val).__name__}"

    if len(message_val) > 10000:
        return False, f"Message length exceeds 10,000 characters limit ({len(message_val)})"

    return True, None


def validate_base_col_arguments(arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate arguments for 'base_col' command (LED color)."""
    expected_keys = {"r", "g", "b"}
    if set(arguments.keys()) != expected_keys:
        return (
            False,
            f"Expected exactly arguments {sorted(expected_keys)}, got {sorted(arguments.keys())}",
        )

    for channel in ("r", "g", "b"):
        val = arguments[channel]
        if not isinstance(val, int) or isinstance(val, bool):
            return (
                False,
                f"Channel '{channel}' must be an integer, got {type(val).__name__}",
            )
        if not (0 <= val <= 10):
            return False, f"Channel '{channel}' must be in range [0, 10], got {val}"

    return True, None


def validate_say_arguments(arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate arguments for direct TTS 'say' command."""
    if set(arguments.keys()) != {"text"}:
        return False, f"Expected argument 'text', got {sorted(arguments.keys())}"

    text = arguments["text"]
    if not isinstance(text, str):
        return False, f"Argument 'text' must be a string, got {type(text).__name__}"

    if len(text) > 10000:
        return False, f"Text length exceeds 10,000 characters limit ({len(text)})"

    return True, None


VALID_MOTOR_NAMES: frozenset[str] = frozenset(
    {"BOTTOMLIP", "EYETILT", "EYETURN", "HEADNOD", "HEADTURN", "LIDBLINK", "TOPLIP"}
)


def validate_move_arguments(arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate arguments for 'move' command with servo safety limits [0.0, 10.0]."""
    if set(arguments.keys()) != {"motor", "position"}:
        return (
            False,
            f"Expected arguments 'motor' and 'position', got {sorted(arguments.keys())}",
        )

    motor = arguments["motor"]
    if not isinstance(motor, (int, str)) or isinstance(motor, bool):
        return False, f"Argument 'motor' must be an integer or string, got {type(motor).__name__}"

    if isinstance(motor, int):
        if not (0 <= motor <= 6):
            return False, f"Motor index {motor} must be in range [0, 6]"
    elif isinstance(motor, str):
        if motor.strip().upper() not in VALID_MOTOR_NAMES:
            try:
                m_int = int(motor)
                if not (0 <= m_int <= 6):
                    return False, f"Motor index {m_int} must be in range [0, 6]"
            except ValueError:
                return (
                    False,
                    f"Unknown motor '{motor}'. Allowed motor names: {sorted(VALID_MOTOR_NAMES)} or index [0, 6]",
                )

    pos = arguments["position"]
    if not isinstance(pos, (int, float)) or isinstance(pos, bool):
        return False, f"Argument 'position' must be a numeric value, got {type(pos).__name__}"

    if not (0.0 <= float(pos) <= 10.0):
        return (
            False,
            f"Position {pos} violates physical safety range [0.0, 10.0]",
        )

    return True, None


def validate_wait_arguments(arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate arguments for 'wait' command."""
    if set(arguments.keys()) != {"seconds"}:
        return False, f"Expected argument 'seconds', got {sorted(arguments.keys())}"

    sec = arguments["seconds"]
    if not isinstance(sec, (int, float)) or isinstance(sec, bool):
        return False, f"Argument 'seconds' must be a numeric value, got {type(sec).__name__}"

    if not (0.0 <= float(sec) <= 60.0):
        return False, f"Wait duration {sec} must be in range [0.0, 60.0] seconds"

    return True, None


class CommandRegistry:
    """Explicit, allowlist-only command registry preventing arbitrary dynamic dispatch."""

    def __init__(self) -> None:
        self._handlers: dict[str, HandlerFunc] = {}
        self._validators: dict[str, ValidatorFunc] = {}

    def register(
        self,
        command_name: str,
        handler: HandlerFunc,
        validator: ValidatorFunc | None = None,
    ) -> None:
        """Register a command handler with an optional argument validator."""
        if not command_name or not isinstance(command_name, str):
            raise ValueError("Command name must be a non-empty string")
        if not callable(handler):
            raise TypeError("Handler must be callable")
        if validator is not None and not callable(validator):
            raise TypeError("Validator must be callable")

        self._handlers[command_name] = handler
        self._validators[command_name] = validator or validate_empty_arguments

    def unregister(self, command_name: str) -> None:
        self._handlers.pop(command_name, None)
        self._validators.pop(command_name, None)

    def is_registered(self, command_name: str) -> bool:
        return command_name in self._handlers

    @property
    def registered_commands(self) -> frozenset[str]:
        return frozenset(self._handlers.keys())

    def validate_command(
        self, command_name: str, arguments: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate that the command is supported and its arguments match expectations."""
        if command_name not in self._handlers:
            return False, f"Command '{command_name}' is not supported"

        validator = self._validators[command_name]
        return validator(arguments)

    async def execute(self, command_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered command handler after validating arguments.

        Raises TransportError if the command is unsupported, arguments are invalid,
        or execution fails.
        """
        if command_name not in self._handlers:
            raise TransportError(
                TransportErrorCode.UNSUPPORTED_COMMAND,
                f"Command '{command_name}' is not in the allowed command registry",
            )

        validator = self._validators[command_name]
        valid, err_msg = validator(arguments)
        if not valid:
            raise TransportError(
                TransportErrorCode.INVALID_ARGUMENTS,
                err_msg or "Invalid command arguments",
            )

        handler = self._handlers[command_name]
        try:
            if inspect.iscoroutinefunction(handler):
                return await handler(arguments)
            res = handler(arguments)
            if inspect.isawaitable(res):
                return await res
            return res
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError(
                TransportErrorCode.INTERNAL_ERROR,
                f"Handler execution for '{command_name}' failed",
            ) from exc


def create_default_registry() -> CommandRegistry:
    """Create a registry pre-configured with standard DeskBuddy command validators.

    Handlers must be attached before serving requests.
    """
    registry = CommandRegistry()
    # Placeholder registrations for schema validation tests
    return registry
