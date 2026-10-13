"""DeskBuddy Authenticated Robot Transport Package.

Provides secure, typed, frame-based JSON messaging with HMAC-SHA256 authentication
and strict command allowlisting.

Note on Packaging Architecture:
Per PR #29 and Issue #5 baseline, the published deskbuddy package wheel/sdist
deliberately ships only the minimal installable core (src/deskbuddy), while robot
hardware transport and control currently run from repository checkouts.
Relocation of the transport package into `src/deskbuddy/transport/` and exposure
via the distribution package is tracked under Issue #7.
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
from transport.client import (
    KNOWN_PHYSICAL_COMMANDS,
    SAFE_IDEMPOTENT_COMMANDS,
    RobotClient,
    RobotTransport,
    is_side_effecting_command,
)
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
from transport.hardware import (
    HardwareExecutor,
    HardwareOperation,
    OperationState,
    get_default_hardware_executor,
    reset_default_hardware_executor,
    run_hardware_op,
    run_hardware_sync,
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
    validate_timeout_override,
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
    "HardwareExecutor",
    "HardwareOperation",
    "KNOWN_PHYSICAL_COMMANDS",
    "OperationState",
    "PayloadTooLargeError",
    "ProtocolViolationError",
    "ReplayCache",
    "RobotClient",
    "RobotRequest",
    "RobotResponse",
    "RobotServer",
    "RobotTransport",
    "SAFE_IDEMPOTENT_COMMANDS",
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
    "get_default_hardware_executor",
    "is_side_effecting_command",
    "read_frame",
    "reset_default_hardware_executor",
    "run_hardware_op",
    "run_hardware_sync",
    "sign_request",
    "sign_response",
    "validate_base_col_arguments",
    "validate_empty_arguments",
    "validate_move_arguments",
    "validate_reply_arguments",
    "validate_say_arguments",
    "validate_timeout_override",
    "validate_wait_arguments",
    "verify_response",
    "write_frame",
]
