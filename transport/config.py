"""Configuration objects for DeskBuddy robot transport and networking boundaries."""

from __future__ import annotations

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
