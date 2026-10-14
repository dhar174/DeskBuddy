"""Unit tests for command allowlist registry and argument validators."""

import pytest

from transport.protocol import TransportError, TransportErrorCode
from transport.registry import (
    CommandRegistry,
    validate_base_col_arguments,
    validate_empty_arguments,
    validate_move_arguments,
    validate_reply_arguments,
    validate_say_arguments,
    validate_wait_arguments,
)


def test_validate_empty_arguments():
    valid, err = validate_empty_arguments({})
    assert valid is True

    valid, err = validate_empty_arguments({"unexpected": 123})
    assert valid is False
    assert "unexpected keys" in err


def test_validate_reply_arguments():
    # Valid with message
    valid, err = validate_reply_arguments({"message": "Hello Cooper"})
    assert valid is True

    # Valid with text alias
    valid, err = validate_reply_arguments({"text": "Hello Cooper"})
    assert valid is True

    # Missing argument
    valid, err = validate_reply_arguments({})
    assert valid is False
    assert "Missing required argument" in err

    # Non-string argument
    valid, err = validate_reply_arguments({"message": 12345})
    assert valid is False
    assert "must be a string" in err

    # Unexpected argument
    valid, err = validate_reply_arguments({"message": "hi", "extra": "bad"})
    assert valid is False
    assert "Unexpected arguments" in err

    # Exceeds max length
    valid, err = validate_reply_arguments({"message": "A" * 10001})
    assert valid is False
    assert "exceeds 10,000 characters" in err


def test_validate_base_col_arguments():
    valid, err = validate_base_col_arguments({"r": 0, "g": 5, "b": 10})
    assert valid is True

    # Out of range (< 0)
    valid, err = validate_base_col_arguments({"r": -1, "g": 5, "b": 10})
    assert valid is False
    assert "must be in range [0, 10]" in err

    # Out of range (> 10)
    valid, err = validate_base_col_arguments({"r": 0, "g": 11, "b": 10})
    assert valid is False
    assert "must be in range [0, 10]" in err

    # Non-integer (float)
    valid, err = validate_base_col_arguments({"r": 5.5, "g": 5, "b": 5})
    assert valid is False
    assert "must be an integer" in err

    # Non-integer (bool)
    valid, err = validate_base_col_arguments({"r": True, "g": 5, "b": 5})
    assert valid is False
    assert "must be an integer" in err

    # Missing channel
    valid, err = validate_base_col_arguments({"r": 5, "g": 5})
    assert valid is False
    assert "Expected exactly arguments" in err


def test_validate_move_arguments():
    valid, err = validate_move_arguments({"motor": 1, "position": 5.0})
    assert valid is True

    valid, err = validate_move_arguments({"motor": "HEADNOD", "position": 7})
    assert valid is True

    valid, err = validate_move_arguments({"motor": "BOTTOMLIP", "position": 5.0})
    assert valid is True

    valid, err = validate_move_arguments({"motor": "TOPLIP", "position": 5.0})
    assert valid is True

    # Valid motor boundaries: 0 and 6
    valid, err = validate_move_arguments({"motor": 0, "position": 0.0})
    assert valid is True
    valid, err = validate_move_arguments({"motor": 6, "position": 10.0})
    assert valid is True

    # Out of range (< 0.0)
    valid, err = validate_move_arguments({"motor": 1, "position": -0.5})
    assert valid is False
    assert "violates physical safety range [0.0, 10.0]" in err

    # Out of range (> 10.0)
    valid, err = validate_move_arguments({"motor": 1, "position": 10.1})
    assert valid is False
    assert "violates physical safety range [0.0, 10.0]" in err

    # Motor index out of range (< 0)
    valid, err = validate_move_arguments({"motor": -1, "position": 5.0})
    assert valid is False
    assert "Motor index -1 must be in range [0, 6]" in err

    # Motor index out of range (>= 7)
    valid, err = validate_move_arguments({"motor": 7, "position": 5.0})
    assert valid is False
    assert "Motor index 7 must be in range [0, 6]" in err

    valid, err = validate_move_arguments({"motor": 15, "position": 5.0})
    assert valid is False
    assert "Motor index 15 must be in range [0, 6]" in err

    # Unknown motor name
    valid, err = validate_move_arguments({"motor": "UNKNOWN_MOTOR", "position": 5.0})
    assert valid is False
    assert "Unknown motor 'UNKNOWN_MOTOR'" in err

    # Numeric string motors (including whitespace and signed)
    valid, err = validate_move_arguments({"motor": "2", "position": 5.0})
    assert valid is True

    valid, err = validate_move_arguments({"motor": " 1 ", "position": 5.0})
    assert valid is True

    valid, err = validate_move_arguments({"motor": "+1", "position": 5.0})
    assert valid is True

    valid, err = validate_move_arguments({"motor": "7", "position": 5.0})
    assert valid is False
    assert "Motor index 7 must be in range [0, 6]" in err

    # Unexpected argument
    valid, err = validate_move_arguments({"motor": 1, "position": 5.0, "speed": 100})
    assert valid is False
    assert "Expected arguments 'motor' and 'position'" in err


def test_validate_say_arguments():
    valid, err = validate_say_arguments({"text": "Hello"})
    assert valid is True

    valid, err = validate_say_arguments({"text": 123})
    assert valid is False
    assert "must be a string" in err

    valid, err = validate_say_arguments({"text": "X" * 10001})
    assert valid is False
    assert "exceeds 10,000 characters" in err


def test_validate_wait_arguments():
    valid, err = validate_wait_arguments({"seconds": 1.5})
    assert valid is True

    valid, err = validate_wait_arguments({"seconds": -1.0})
    assert valid is False
    assert "must be in range [0.0, 60.0]" in err

    valid, err = validate_wait_arguments({"seconds": 65.0})
    assert valid is False
    assert "must be in range [0.0, 60.0]" in err


@pytest.mark.asyncio
async def test_command_registry_dispatch_flow():
    registry = CommandRegistry()

    received_text = None

    async def mock_speak(args):
        nonlocal received_text
        received_text = args["message"]
        return {"spoken": True}

    registry.register("speak", mock_speak, validate_reply_arguments)

    # 1. Valid execution
    res = await registry.execute("speak", {"message": "Picoh speaks!"})
    assert res == {"spoken": True}
    assert received_text == "Picoh speaks!"

    # 2. Unsupported command
    with pytest.raises(TransportError) as exc_info:
        await registry.execute("delete_database", {})
    assert exc_info.value.code == TransportErrorCode.UNSUPPORTED_COMMAND

    # 3. Invalid arguments
    with pytest.raises(TransportError) as exc_info:
        await registry.execute("speak", {"wrong_arg": 123})
    assert exc_info.value.code == TransportErrorCode.INVALID_ARGUMENTS


@pytest.mark.asyncio
async def test_registry_wraps_internal_exceptions():
    registry = CommandRegistry()

    def faulty_handler(args):
        raise ZeroDivisionError("division by zero in motor math")

    registry.register("faulty", faulty_handler, validate_empty_arguments)

    with pytest.raises(TransportError) as exc_info:
        await registry.execute("faulty", {})

    assert exc_info.value.code == TransportErrorCode.INTERNAL_ERROR
    assert "Handler execution for 'faulty' failed" in str(exc_info.value)
