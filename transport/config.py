"""Configuration objects for DeskBuddy robot transport and networking boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from deskbuddy.settings import Settings, load_settings
from transport.protocol import AuthenticationError, TransportErrorCode

DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8888
DEFAULT_CONNECT_TIMEOUT: Final[float] = 5.0
DEFAULT_TIMEOUT: Final[float] = 5.0
DEFAULT_MAX_PAYLOAD_SIZE: Final[int] = 65536  # 64 KiB
DEFAULT_MAX_CLOCK_SKEW: Final[float] = 30.0  # seconds
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_BACKOFF: Final[float] = 0.2  # seconds
DEFAULT_RESPONSE_GRACE: Final[float] = 2.0  # seconds


def compute_command_timeout(
    command: str,
    arguments: Mapping[str, Any] | None,
    base_timeout: float,
    margin: float = 10.0,
) -> float:
    """Derive dynamic timeout based on command semantics and arguments.

    - normal command: base robot_timeout_seconds
    - wait/picoh_wait: max(base_timeout, requested seconds + margin)
    - reply/say/picoh_say: max(base_timeout, speech duration margin + margin)
    """
    args = arguments or {}
    if command in ("wait", "picoh_wait"):
        seconds = args.get("seconds")
        if seconds is not None:
            try:
                sec_val = float(seconds)
                return max(base_timeout, sec_val + margin)
            except (ValueError, TypeError):
                pass
    elif command in ("reply", "say", "picoh_say"):
        text = args.get("message")
        if text is None:
            text = args.get("text")
        if text is not None and isinstance(text, str):
            speech_margin = len(text) / 10.0
            return max(base_timeout, speech_margin + margin)
    return base_timeout


@dataclass(slots=True)
class TransportConfig:
    """Explicit network, timeout, and authentication configuration for robot transport.

    Constructed from validated DeskBuddy Settings or direct arguments.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    secret: str = ""
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    read_timeout: float = DEFAULT_TIMEOUT
    write_timeout: float = DEFAULT_CONNECT_TIMEOUT
    request_timeout: float = DEFAULT_TIMEOUT
    max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE
    max_clock_skew: float = DEFAULT_MAX_CLOCK_SKEW
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff: float = DEFAULT_RETRY_BACKOFF
    response_grace: float = DEFAULT_RESPONSE_GRACE

    def __post_init__(self) -> None:
        """Fail-closed validation for security-sensitive network and protocol parameters."""
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError(f"host must be a non-empty string, got {self.host!r}")

        if isinstance(self.port, bool) or not isinstance(self.port, int) or not (0 <= self.port <= 65535):
            raise ValueError(f"port must be an integer in range [0, 65535], got {self.port!r}")

        if not isinstance(self.secret, str):
            raise TypeError(f"secret must be a string, got {type(self.secret).__name__}")

        for name, val in (
            ("connect_timeout", self.connect_timeout),
            ("read_timeout", self.read_timeout),
            ("write_timeout", self.write_timeout),
            ("request_timeout", self.request_timeout),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a numeric value, got {type(val).__name__}")
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be a positive finite number, got {val!r}")

        if isinstance(self.max_payload_size, bool) or not isinstance(self.max_payload_size, int) or self.max_payload_size <= 0:
            raise ValueError(f"max_payload_size must be a positive integer, got {self.max_payload_size!r}")

        if isinstance(self.max_clock_skew, bool) or not isinstance(self.max_clock_skew, (int, float)) or not math.isfinite(self.max_clock_skew) or self.max_clock_skew <= 0:
            raise ValueError(f"max_clock_skew must be a positive finite number, got {self.max_clock_skew!r}")

        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError(f"max_retries must be a non-negative integer, got {self.max_retries!r}")

        if isinstance(self.retry_backoff, bool) or not isinstance(self.retry_backoff, (int, float)) or not math.isfinite(self.retry_backoff) or self.retry_backoff < 0:
            raise ValueError(f"retry_backoff must be a non-negative finite number, got {self.retry_backoff!r}")

        if isinstance(self.response_grace, bool) or not isinstance(self.response_grace, (int, float)) or not math.isfinite(self.response_grace) or self.response_grace < 0:
            raise ValueError(f"response_grace must be a non-negative finite number, got {self.response_grace!r}")

        # Normalize numeric float fields to float
        self.connect_timeout = float(self.connect_timeout)
        self.read_timeout = float(self.read_timeout)
        self.write_timeout = float(self.write_timeout)
        self.request_timeout = float(self.request_timeout)
        self.max_clock_skew = float(self.max_clock_skew)
        self.retry_backoff = float(self.retry_backoff)
        self.response_grace = float(self.response_grace)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        role: Literal["client", "server"] = "client",
    ) -> TransportConfig:
        """Construct transport configuration from validated canonical Settings.

        Fails before network startup if the authentication secret is missing or blank.
        """
        if role not in ("client", "server"):
            raise ValueError(f"Invalid role '{role}': must be 'client' or 'server'")

        if settings is None:
            settings = load_settings()

        if settings.robot_auth_secret is None:
            raise AuthenticationError(
                TransportErrorCode.UNAUTHORIZED,
                "Missing robot authentication secret. Set DESKBUDDY_ROBOT_AUTH_SECRET.",
            )

        secret = settings.robot_auth_secret.get_secret_value()
        if not secret.strip():
            raise AuthenticationError(
                TransportErrorCode.UNAUTHORIZED,
                "Missing robot authentication secret. Set DESKBUDDY_ROBOT_AUTH_SECRET.",
            )

        target_host = settings.robot_host if role == "client" else settings.robot_bind_address

        return cls(
            host=target_host,
            port=settings.robot_port,
            secret=secret,
            connect_timeout=settings.robot_connect_timeout_seconds,
            read_timeout=settings.robot_timeout_seconds,
            write_timeout=settings.robot_timeout_seconds,
            request_timeout=settings.robot_timeout_seconds,
        )

    @classmethod
    def for_client(cls, settings: Settings | None = None) -> TransportConfig:
        """Construct client transport configuration from Settings."""
        return cls.from_settings(settings, role="client")

    @classmethod
    def for_server(cls, settings: Settings | None = None) -> TransportConfig:
        """Construct server transport configuration from Settings."""
        return cls.from_settings(settings, role="server")


# Convenience aliases for server/client contexts
ServerConfig = TransportConfig
ClientConfig = TransportConfig
