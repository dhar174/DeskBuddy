"""Protocol definitions and typed message schemas for DeskBuddy robot transport."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

CURRENT_PROTOCOL_VERSION = "1.0"
SUPPORTED_PROTOCOL_VERSIONS = frozenset({"1.0"})


class TransportErrorCode(str, Enum):
    """Machine-readable transport error codes."""

    MALFORMED_REQUEST = "malformed_request"
    UNSUPPORTED_PROTOCOL = "unsupported_protocol"
    UNAUTHORIZED = "unauthorized"
    EXPIRED_REQUEST = "expired_request"
    REPLAYED_REQUEST = "replayed_request"
    UNSUPPORTED_COMMAND = "unsupported_command"
    INVALID_ARGUMENTS = "invalid_arguments"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"
    EXECUTION_OUTCOME_UNKNOWN = "execution_outcome_unknown"


class TransportError(Exception):
    """Base exception for robot transport failures."""

    def __init__(self, code: TransportErrorCode | str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code: TransportErrorCode | str
        if isinstance(code, TransportErrorCode):
            self.code = code
        else:
            try:
                self.code = TransportErrorCode(code)
            except ValueError:
                self.code = code
        self.message = message


class FramingError(TransportError):
    """Raised when framing read or write operations encounter an error."""


class TruncatedFrameError(FramingError):
    """Raised when a frame payload is shorter than declared in its header."""


class PayloadTooLargeError(FramingError):
    """Raised when an incoming frame header declares a length beyond the allowed limit."""


class AuthenticationError(TransportError):
    """Raised on authentication failure or signature mismatch."""


class ProtocolViolationError(TransportError):
    """Raised when payload content violates expected protocol structure."""


@dataclass(slots=True)
class RobotRequest:
    """Strongly typed robot command request."""

    request_id: str
    command: str
    arguments: dict[str, Any] = field(default_factory=dict)
    auth: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = CURRENT_PROTOCOL_VERSION
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        command: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
        protocol_version: str = CURRENT_PROTOCOL_VERSION,
        timestamp: float | None = None,
        auth: Mapping[str, Any] | None = None,
    ) -> RobotRequest:
        return cls(
            request_id=request_id or str(uuid.uuid4()),
            command=command,
            arguments=dict(arguments or {}),
            auth=dict(auth or {}),
            protocol_version=protocol_version,
            timestamp=time.time() if timestamp is None else float(timestamp),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "protocol_version": self.protocol_version,
            "timestamp": self.timestamp,
            "command": self.command,
            "arguments": self.arguments,
            "auth": self.auth,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Any) -> RobotRequest:
        if not isinstance(data, dict):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "Request payload must be a JSON object"
            )

        for key in ("request_id", "protocol_version", "timestamp", "command"):
            if key not in data:
                raise ProtocolViolationError(
                    TransportErrorCode.MALFORMED_REQUEST, f"Missing required field: {key}"
                )

        req_id = data["request_id"]
        if not isinstance(req_id, str) or not req_id.strip():
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "request_id must be a non-empty string"
            )

        proto_ver = data["protocol_version"]
        if not isinstance(proto_ver, str) or not proto_ver.strip():
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "protocol_version must be a non-empty string"
            )

        ts = data["timestamp"]
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "timestamp must be a finite numeric timestamp"
            )

        cmd = data["command"]
        if not isinstance(cmd, str) or not cmd.strip():
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "command must be a non-empty string"
            )

        raw_args = data.get("arguments", {})
        if not isinstance(raw_args, dict):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "arguments must be a JSON object"
            )

        raw_auth = data.get("auth", {})
        if not isinstance(raw_auth, dict):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "auth must be a JSON object"
            )

        return cls(
            request_id=req_id,
            protocol_version=proto_ver,
            timestamp=float(ts),
            command=cmd,
            arguments=raw_args,
            auth=raw_auth,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> RobotRequest:
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, f"Invalid JSON payload: {exc}"
            ) from exc
        return cls.from_dict(parsed)


def validate_timeout_override(timeout: Any) -> float:
    """Validate an explicit runtime timeout override.

    Must be a positive finite numeric value. Booleans and non-numeric types are rejected.
    """
    if timeout is None:
        raise ValueError("Timeout override cannot be None")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError(f"Timeout override must be a numeric value, got {type(timeout).__name__}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"Timeout override must be a positive finite number, got {timeout}")
    return float(timeout)


class ExecutionOutcomeUnknown(TransportError):
    """Raised when request delivery/execution state cannot be confirmed after transport loss."""

    def __init__(
        self,
        request_id: str,
        command: str,
        message: str = (
            "Execution outcome unknown: request was sent but outcome could not be confirmed."
        ),
        code: TransportErrorCode | str = TransportErrorCode.EXECUTION_OUTCOME_UNKNOWN,
    ) -> None:
        super().__init__(code, message)
        self.request_id = request_id
        self.command = command


@dataclass(slots=True)
class RobotResponse:
    """Strongly typed robot command response."""

    request_id: str | None
    ok: bool
    result: Any = None
    error: dict[str, str] | None = None
    protocol_version: str = CURRENT_PROTOCOL_VERSION
    timestamp: float = field(default_factory=time.time)
    auth: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        request_id: str | None,
        result: Any = None,
        protocol_version: str = CURRENT_PROTOCOL_VERSION,
        *,
        timestamp: float | None = None,
        auth: Mapping[str, Any] | None = None,
    ) -> RobotResponse:
        return cls(
            request_id=request_id,
            ok=True,
            result=result,
            error=None,
            protocol_version=protocol_version,
            timestamp=time.time() if timestamp is None else float(timestamp),
            auth=dict(auth or {}),
        )

    @classmethod
    def fail(
        cls,
        request_id: str | None,
        code: TransportErrorCode | str,
        message: str,
        protocol_version: str = CURRENT_PROTOCOL_VERSION,
        *,
        timestamp: float | None = None,
        auth: Mapping[str, Any] | None = None,
    ) -> RobotResponse:
        err_code_str = code.value if isinstance(code, TransportErrorCode) else str(code)
        return cls(
            request_id=request_id,
            ok=False,
            result=None,
            error={"code": err_code_str, "message": message},
            protocol_version=protocol_version,
            timestamp=time.time() if timestamp is None else float(timestamp),
            auth=dict(auth or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "protocol_version": self.protocol_version,
            "timestamp": self.timestamp,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "auth": self.auth,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Any) -> RobotResponse:
        if not isinstance(data, dict):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "Response payload must be a JSON object"
            )

        if "ok" not in data or "protocol_version" not in data:
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "Missing required fields in response"
            )

        req_id = data.get("request_id")
        if req_id is not None and not isinstance(req_id, str):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "request_id must be null or string"
            )

        proto_ver = data["protocol_version"]
        if not isinstance(proto_ver, str):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "protocol_version must be a string"
            )

        ts = data.get("timestamp")
        if ts is not None and (
            isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts)
        ):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "timestamp must be a finite numeric timestamp"
            )
        response_ts = time.time() if ts is None else float(ts)

        ok_val = data["ok"]
        if not isinstance(ok_val, bool):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "ok must be a boolean"
            )

        err_val = data.get("error")
        if not ok_val and err_val is None:
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST,
                "error object must be provided when ok is false",
            )
        if err_val is not None:
            if not isinstance(err_val, dict) or "code" not in err_val or "message" not in err_val:
                raise ProtocolViolationError(
                    TransportErrorCode.MALFORMED_REQUEST,
                    "error must be a dict with code and message",
                )

        raw_auth = data.get("auth", {})
        if not isinstance(raw_auth, dict):
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, "auth must be a JSON object"
            )

        return cls(
            request_id=req_id,
            ok=ok_val,
            result=data.get("result"),
            error=err_val,
            protocol_version=proto_ver,
            timestamp=response_ts,
            auth=raw_auth,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> RobotResponse:
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise ProtocolViolationError(
                TransportErrorCode.MALFORMED_REQUEST, f"Invalid JSON payload: {exc}"
            ) from exc
        return cls.from_dict(parsed)
