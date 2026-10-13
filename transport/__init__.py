"""DeskBuddy Authenticated Robot Transport Package.

Provides secure, typed, frame-based JSON messaging with HMAC-SHA256 authentication
and strict command allowlisting.
"""

from __future__ import annotations

from transport.auth import (
    AUTH_SCHEME,
    REQUEST_DOMAIN_PREFIX,
    RESPONSE_DOMAIN_PREFIX,
    HMACAuthenticator,
    ReplayCache,
    compute_request_signature,
    compute_response_signature,
    compute_signature,
    sign_request,
    sign_response,
    verify_response,
)
from transport.client import RobotClient, RobotTransport
from transport.config import (
    ClientConfig,
    ServerConfig,
    TransportConfig,
    compute_command_timeout,
)
from transport.framing import (
    DEFAULT_MAX_PAYLOAD_SIZE,
    read_frame,
    write_frame,
)
from transport.protocol import (
    CURRENT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    AuthenticationError,
    ExecutionOutcomeUnknown,
    FramingError,
    PayloadTooLargeError,
    ProtocolViolationError,
    RobotRequest,
    RobotResponse,
    TransportError,
    TransportErrorCode,
    TruncatedFrameError,
)
from transport.registry import (
    CommandRegistry,
    create_default_registry,
    validate_base_col_arguments,
    validate_empty_arguments,
    validate_move_arguments,
    validate_reply_arguments,
    validate_say_arguments,
    validate_wait_arguments,
)
from transport.server import RobotServer

__all__ = [
    "AUTH_SCHEME",
    "CURRENT_PROTOCOL_VERSION",
    "DEFAULT_MAX_PAYLOAD_SIZE",
    "REQUEST_DOMAIN_PREFIX",
    "RESPONSE_DOMAIN_PREFIX",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "AuthenticationError",
    "ClientConfig",
    "CommandRegistry",
    "ExecutionOutcomeUnknown",
    "FramingError",
    "HMACAuthenticator",
    "PayloadTooLargeError",
    "ProtocolViolationError",
    "ReplayCache",
    "RobotClient",
    "RobotRequest",
    "RobotResponse",
    "RobotServer",
    "RobotTransport",
    "ServerConfig",
    "TransportConfig",
    "TransportError",
    "TransportErrorCode",
    "TruncatedFrameError",
    "compute_command_timeout",
    "compute_request_signature",
    "compute_response_signature",
    "compute_signature",
    "create_default_registry",
    "read_frame",
    "sign_request",
    "sign_response",
    "validate_base_col_arguments",
    "validate_empty_arguments",
    "validate_move_arguments",
    "validate_reply_arguments",
    "validate_say_arguments",
    "validate_wait_arguments",
    "verify_response",
    "write_frame",
]
