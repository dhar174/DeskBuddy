"""HMAC authentication and replay protection for DeskBuddy robot transport."""

from __future__ import annotations

import collections
import hashlib
import hmac
import json
import time
from typing import Any, Final

from transport.protocol import (
    SUPPORTED_PROTOCOL_VERSIONS,
    AuthenticationError,
    RobotRequest,
    RobotResponse,
    TransportErrorCode,
)

AUTH_SCHEME: Final[str] = "hmac-sha256"
DEFAULT_MAX_CLOCK_SKEW_SECONDS: Final[float] = 30.0
DEFAULT_REPLAY_CACHE_MAX_ENTRIES: Final[int] = 10000
REQUEST_DOMAIN_PREFIX: Final[str] = "deskbuddy-request-v1"
RESPONSE_DOMAIN_PREFIX: Final[str] = "deskbuddy-response-v1"


def canonical_json_string(value: Any) -> str:
    """Produce a deterministic, sorted, canonical JSON string."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_arguments_string(arguments: dict) -> str:
    """Produce a deterministic, sorted, canonical JSON string for arguments."""
    return canonical_json_string(arguments)


def compute_request_signature(secret: str | bytes, request: RobotRequest) -> str:
    """Compute HMAC-SHA256 signature for a RobotRequest with domain separation."""
    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    else:
        secret_bytes = bytes(secret)

    if not secret_bytes:
        raise AuthenticationError(
            TransportErrorCode.UNAUTHORIZED, "Transport secret cannot be empty"
        )

    canonical_args = canonical_arguments_string(request.arguments)
    message_material = (
        f"{REQUEST_DOMAIN_PREFIX}|"
        f"{request.protocol_version}|"
        f"{request.request_id}|"
        f"{request.timestamp:.6f}|"
        f"{request.command}|"
        f"{canonical_args}"
    ).encode("utf-8")

    return hmac.new(secret_bytes, message_material, hashlib.sha256).hexdigest()


# Backward-compatible alias
compute_signature = compute_request_signature


def sign_request(secret: str | bytes, request: RobotRequest) -> RobotRequest:
    """Sign a RobotRequest with HMAC-SHA256 in-place and return it."""
    sig = compute_request_signature(secret, request)
    request.auth = {
        "scheme": AUTH_SCHEME,
        "signature": sig,
    }
    return request


def compute_response_signature(secret: str | bytes, response: RobotResponse) -> str:
    """Compute HMAC-SHA256 signature for a RobotResponse with domain separation."""
    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    else:
        secret_bytes = bytes(secret)

    if not secret_bytes:
        raise AuthenticationError(
            TransportErrorCode.UNAUTHORIZED, "Transport secret cannot be empty"
        )

    req_id_str = response.request_id or ""
    canonical_result = canonical_json_string(response.result)
    canonical_error = canonical_json_string(response.error)
    ok_str = "1" if response.ok else "0"

    message_material = (
        f"{RESPONSE_DOMAIN_PREFIX}|"
        f"{response.protocol_version}|"
        f"{req_id_str}|"
        f"{response.timestamp:.6f}|"
        f"{ok_str}|"
        f"{canonical_result}|"
        f"{canonical_error}"
    ).encode("utf-8")

    return hmac.new(secret_bytes, message_material, hashlib.sha256).hexdigest()


def sign_response(secret: str | bytes, response: RobotResponse) -> RobotResponse:
    """Sign a RobotResponse with HMAC-SHA256 in-place and return it."""
    sig = compute_response_signature(secret, response)
    response.auth = {
        "scheme": AUTH_SCHEME,
        "signature": sig,
    }
    return response


def verify_response(
    secret: str | bytes,
    response: RobotResponse,
    *,
    expected_request_id: str | None = None,
    max_clock_skew: float = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
    current_time: float | None = None,
) -> tuple[bool, TransportErrorCode | None, str | None]:
    """Verify HMAC signature, request ID correlation, and freshness of a RobotResponse.

    Returns (is_valid, error_code, error_message).
    """
    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    else:
        secret_bytes = bytes(secret)

    if not secret_bytes:
        return False, TransportErrorCode.UNAUTHORIZED, "Transport secret cannot be empty"

    if response.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
        return (
            False,
            TransportErrorCode.UNSUPPORTED_PROTOCOL,
            f"Unsupported protocol version '{response.protocol_version}'",
        )

    if expected_request_id is not None:
        if not response.request_id or response.request_id != expected_request_id:
            return (
                False,
                TransportErrorCode.MALFORMED_REQUEST,
                f"Correlation mismatch in response: expected '{expected_request_id}', got '{response.request_id}'",
            )

    now = time.time() if current_time is None else float(current_time)
    skew = abs(now - response.timestamp)
    if skew > max_clock_skew:
        return (
            False,
            TransportErrorCode.EXPIRED_REQUEST,
            f"Response timestamp expired (skew: {skew:.2f}s > allowed: {max_clock_skew:.2f}s)",
        )

    if not isinstance(response.auth, dict) or not response.auth:
        return False, TransportErrorCode.UNAUTHORIZED, "Response is missing authentication"

    scheme = response.auth.get("scheme")
    if scheme != AUTH_SCHEME:
        return (
            False,
            TransportErrorCode.UNAUTHORIZED,
            f"Unsupported authentication scheme '{scheme}'",
        )

    sig = response.auth.get("signature")
    if not isinstance(sig, str) or not sig:
        return False, TransportErrorCode.UNAUTHORIZED, "Missing or invalid response signature"

    expected_sig = compute_response_signature(secret_bytes, response)
    if not hmac.compare_digest(sig, expected_sig):
        return (
            False,
            TransportErrorCode.UNAUTHORIZED,
            "Response authentication signature verification failed",
        )

    return True, None, None


class ReplayCache:
    """Bounded, TTL-aware cache of seen request IDs to prevent replay attacks."""

    def __init__(self, max_entries: int = DEFAULT_REPLAY_CACHE_MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        # Mapping request_id -> arrival timestamp (insertion ordered in Python 3.7+)
        self._seen: collections.OrderedDict[str, float] = collections.OrderedDict()

    def prune(self, current_time: float, max_age: float) -> None:
        cutoff = current_time - max_age
        while self._seen:
            oldest_key, oldest_ts = next(iter(self._seen.items()))
            if oldest_ts < cutoff:
                self._seen.pop(oldest_key)
            else:
                break

        while len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)

    def is_seen(self, request_id: str, current_time: float, max_clock_skew: float) -> bool:
        """Check if request_id has already been recorded in the active validity window."""
        self.prune(current_time, max_clock_skew * 2.0)
        return request_id in self._seen

    def record(self, request_id: str, current_time: float, max_clock_skew: float) -> None:
        """Record an authentic request_id into the cache."""
        self.prune(current_time, max_clock_skew * 2.0)
        self._seen[request_id] = current_time
        if len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)

    def check_and_record(
        self, request_id: str, request_timestamp: float, current_time: float, max_clock_skew: float
    ) -> bool:
        """Check if request_id has already been seen, and record if fresh."""
        if self.is_seen(request_id, current_time, max_clock_skew):
            return False
        self.record(request_id, current_time, max_clock_skew)
        return True

    def clear(self) -> None:
        self._seen.clear()

    def __len__(self) -> int:
        return len(self._seen)


class HMACAuthenticator:
    """Authenticates incoming RobotRequest objects and rejects expired/replayed requests."""

    def __init__(
        self,
        secret: str | bytes,
        *,
        max_clock_skew: float = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
        replay_cache: ReplayCache | None = None,
    ) -> None:
        if isinstance(secret, str):
            self._secret_bytes = secret.encode("utf-8")
        else:
            self._secret_bytes = bytes(secret)
        self.max_clock_skew = float(max_clock_skew)
        self.replay_cache = replay_cache if replay_cache is not None else ReplayCache()

    @property
    def has_secret(self) -> bool:
        return bool(self._secret_bytes)

    def authenticate(
        self,
        request: RobotRequest,
        *,
        current_time: float | None = None,
    ) -> tuple[bool, TransportErrorCode | None, str | None]:
        """Verify the authentication and validity of an incoming request.

        Returns (is_valid, error_code, error_message).
        """
        now = time.time() if current_time is None else float(current_time)

        if not self._secret_bytes:
            return False, TransportErrorCode.INTERNAL_ERROR, "Server secret is not configured"

        # Check auth block presence
        if not isinstance(request.auth, dict) or not request.auth:
            return False, TransportErrorCode.UNAUTHORIZED, "Request is missing authentication"

        scheme = request.auth.get("scheme")
        if scheme != AUTH_SCHEME:
            return (
                False,
                TransportErrorCode.UNAUTHORIZED,
                f"Unsupported authentication scheme '{scheme}'",
            )

        signature = request.auth.get("signature")
        if not isinstance(signature, str) or not signature:
            return False, TransportErrorCode.UNAUTHORIZED, "Missing or invalid signature"

        # Check clock skew / expiry
        skew = abs(now - request.timestamp)
        if skew > self.max_clock_skew:
            return (
                False,
                TransportErrorCode.EXPIRED_REQUEST,
                f"Request timestamp expired (skew: {skew:.2f}s > allowed: {self.max_clock_skew:.2f}s)",
            )

        # Constant-time signature verification
        expected_sig = compute_signature(self._secret_bytes, request)
        if not hmac.compare_digest(signature, expected_sig):
            return (
                False,
                TransportErrorCode.UNAUTHORIZED,
                "Authentication signature verification failed",
            )

        # Check replay protection after confirming authentic signature
        if self.replay_cache.is_seen(request.request_id, now, self.max_clock_skew):
            return (
                False,
                TransportErrorCode.REPLAYED_REQUEST,
                f"Request ID '{request.request_id}' has already been processed (replay detected)",
            )

        # Record in replay cache ONLY after signature verification passes!
        # This prevents unauthenticated attackers from poisoning or evicting the replay cache.
        self.replay_cache.record(request.request_id, now, self.max_clock_skew)

        return True, None, None
