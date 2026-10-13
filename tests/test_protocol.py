"""Unit tests for RobotRequest and RobotResponse protocol schemas and error codes."""

import time

import pytest

from transport.protocol import (
    CURRENT_PROTOCOL_VERSION,
    ProtocolViolationError,
    RobotRequest,
    RobotResponse,
    TransportErrorCode,
)


def test_robot_request_creation_and_serialization():
    req = RobotRequest.create("ping", {"key": "value"})
    assert req.command == "ping"
    assert req.arguments == {"key": "value"}
    assert req.protocol_version == CURRENT_PROTOCOL_VERSION
    assert req.request_id is not None
    assert isinstance(req.timestamp, float)

    json_str = req.to_json()
    assert '"command":"ping"' in json_str

    deserialized = RobotRequest.from_json(json_str)
    assert deserialized.request_id == req.request_id
    assert deserialized.command == "ping"
    assert deserialized.arguments == {"key": "value"}
    assert deserialized.protocol_version == CURRENT_PROTOCOL_VERSION


def test_robot_request_malformed_json_rejection():
    with pytest.raises(ProtocolViolationError) as exc_info:
        RobotRequest.from_json("{not valid json")
    assert exc_info.value.code == TransportErrorCode.MALFORMED_REQUEST


def test_robot_request_missing_required_fields():
    # Missing command
    with pytest.raises(ProtocolViolationError) as exc_info:
        RobotRequest.from_dict({
            "request_id": "123",
            "protocol_version": "1.0",
            "timestamp": time.time(),
        })
    assert exc_info.value.code == TransportErrorCode.MALFORMED_REQUEST

    # Non-dict payload
    with pytest.raises(ProtocolViolationError):
        RobotRequest.from_dict(["not", "a", "dict"])


def test_robot_response_creation_and_serialization():
    # Success response
    resp = RobotResponse.success("req-123", result={"status": "ok"})
    assert resp.ok is True
    assert resp.request_id == "req-123"
    assert resp.result == {"status": "ok"}
    assert resp.error is None

    json_str = resp.to_json()
    deserialized = RobotResponse.from_json(json_str)
    assert deserialized.ok is True
    assert deserialized.result == {"status": "ok"}

    # Failure response
    err_resp = RobotResponse.fail("req-456", TransportErrorCode.UNAUTHORIZED, "Bad auth")
    assert err_resp.ok is False
    assert err_resp.request_id == "req-456"
    assert err_resp.error == {"code": "unauthorized", "message": "Bad auth"}

    err_json = err_resp.to_json()
    deserialized_err = RobotResponse.from_json(err_json)
    assert deserialized_err.ok is False
    assert deserialized_err.error["code"] == "unauthorized"


def test_robot_response_malformed_rejection():
    with pytest.raises(ProtocolViolationError):
        RobotResponse.from_json("invalid json string")

    with pytest.raises(ProtocolViolationError):
        RobotResponse.from_dict({"result": 123})  # missing ok and protocol_version

    # ok: False without error block must be rejected
    with pytest.raises(ProtocolViolationError):
        RobotResponse.from_dict({"ok": False, "protocol_version": "1.0"})


def test_robot_request_edge_cases():
    # Boolean timestamp must be rejected
    with pytest.raises(ProtocolViolationError):
        RobotRequest.from_dict({
            "request_id": "123",
            "protocol_version": "1.0",
            "timestamp": True,
            "command": "ping",
        })

    # Blank command must be rejected
    with pytest.raises(ProtocolViolationError):
        RobotRequest.from_dict({
            "request_id": "123",
            "protocol_version": "1.0",
            "timestamp": time.time(),
            "command": "   ",
        })


def test_transport_error_custom_code():
    from transport.protocol import TransportError
    # Custom string code must not raise ValueError
    err = TransportError("custom_vendor_error", "Something vendor-specific failed")
    assert err.code == "custom_vendor_error"
    assert "custom_vendor_error" in str(err)


def test_robot_request_non_finite_timestamps():
    for bad_ts in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ProtocolViolationError) as exc_info:
            RobotRequest.from_dict({
                "request_id": "123",
                "protocol_version": "1.0",
                "timestamp": bad_ts,
                "command": "ping",
            })
        assert "finite" in exc_info.value.message
