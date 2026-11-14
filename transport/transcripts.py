"""Thread-safe, bounded transcript delivery for desktop consumers."""

from __future__ import annotations

import logging
import sys
import threading
import uuid
from collections import deque
from typing import Any, Literal, TypeGuard

from transport.auth import AUTH_SCHEME, SIGNATURE_HEX_LENGTH
from transport.protocol import RobotResponse, TransportError, TransportErrorCode
from transport.registry import CommandRegistry, validate_empty_arguments

MAX_PENDING_TRANSCRIPTS = 32
POLL_TRANSCRIPT_COMMAND = "poll_transcript"
ACK_TRANSCRIPT_COMMAND = "ack_transcript"

_logger = logging.getLogger(__name__)
_SIZE_CHECK_REQUEST_ID = "00000000-0000-0000-0000-000000000000"
TranscriptPollingMode = Literal["id", "legacy"]


def is_transcript_id(value: object) -> TypeGuard[str]:
    """Return whether value is a canonical, lowercase UUID string."""
    if not isinstance(value, str) or len(value) != 36:
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return str(parsed) == value


def is_valid_transcript_text(value: object) -> TypeGuard[str]:
    """Return whether a finalized transcript is nonblank text."""
    return isinstance(value, str) and bool(value.strip())


def validate_ack_transcript_arguments(arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate the exact acknowledgement envelope without echoing user data."""
    if set(arguments) != {"id"}:
        return False, "Expected exactly argument 'id'"
    if not is_transcript_id(arguments["id"]):
        return False, "Argument 'id' must be a canonical UUID"
    return True, None


class TranscriptQueue:
    """A process-local FIFO whose head stays pending until explicitly acknowledged."""

    def __init__(
        self,
        *,
        max_payload_size: int | None = None,
        secret: str = "",
    ) -> None:
        if max_payload_size is not None:
            if isinstance(max_payload_size, bool) or not isinstance(max_payload_size, int):
                raise TypeError("max_payload_size must be an integer")
            if max_payload_size <= 0:
                raise ValueError("max_payload_size must be positive")
        if not isinstance(secret, str):
            raise TypeError("secret must be a string")
        self._max_payload_size = max_payload_size if max_payload_size is not None else 0
        self._secret = secret
        self._pending: deque[tuple[str, str]] = deque()
        self._lock = threading.Lock()
        self._generation = 0
        self._polling_mode: TranscriptPollingMode | None = None

    def reset(self, *, max_payload_size: int, secret: str) -> None:
        """Clear pending work and publish the server's current wire configuration."""
        if isinstance(max_payload_size, bool) or not isinstance(max_payload_size, int):
            raise TypeError("max_payload_size must be an integer")
        if max_payload_size <= 0:
            raise ValueError("max_payload_size must be positive")
        if not isinstance(secret, str):
            raise TypeError("secret must be a string")
        with self._lock:
            self._pending.clear()
            self._max_payload_size = max_payload_size
            self._secret = secret
            self._generation += 1
            self._polling_mode = None

    @property
    def polling_mode(self) -> TranscriptPollingMode | None:
        """Return the polling mode selected by the first real transcript poll."""
        with self._lock:
            return self._polling_mode

    def select_polling_mode(self, mode: TranscriptPollingMode) -> TranscriptPollingMode:
        """Select a polling mode once per server run and reject mixed-mode clients."""
        if mode not in ("id", "legacy"):
            raise ValueError("mode must be 'id' or 'legacy'")
        with self._lock:
            if self._polling_mode is None:
                self._polling_mode = mode
                if mode == "legacy":
                    # Keep legacy text delivery independent of this bounded shadow queue.
                    self._pending.clear()
            elif self._polling_mode != mode:
                raise TransportError(
                    TransportErrorCode.INVALID_ARGUMENTS,
                    "Transcript polling mode changed; restart the robot server before switching clients.",
                )
            selected_mode = self._polling_mode
        assert selected_mode is not None
        return selected_mode

    def enqueue(self, text: object) -> bool:
        """Append one transcript if it fits the signed response envelope and queue cap."""
        if not isinstance(text, str):
            _logger.warning("Transcript queue rejected a non-text value.")
            return False
        if not text.strip():
            _logger.warning("Transcript queue rejected an empty transcript.")
            return False

        transcript_id = str(uuid.uuid4())
        with self._lock:
            if self._polling_mode == "legacy":
                return False
            if self._max_payload_size == 0:
                _logger.warning(
                    "Transcript queue is not configured; newest transcript was rejected."
                )
                return False
            if len(self._pending) >= MAX_PENDING_TRANSCRIPTS:
                _logger.warning("Transcript queue is full; newest transcript was rejected.")
                return False
            max_payload_size = self._max_payload_size
            secret = self._secret
            generation = self._generation

        # JSON serialization scales with transcript size. Keep it outside the
        # lock and reject if reset changed the wire configuration.
        if not self._fits_poll_response(
            transcript_id,
            text,
            max_payload_size=max_payload_size,
            secret=secret,
        ):
            _logger.warning("Transcript exceeds the configured response frame limit.")
            return False

        with self._lock:
            if generation != self._generation:
                _logger.warning(
                    "Transcript queue configuration changed; newest transcript was rejected."
                )
                return False
            if self._polling_mode == "legacy":
                return False
            if len(self._pending) >= MAX_PENDING_TRANSCRIPTS:
                _logger.warning("Transcript queue is full; newest transcript was rejected.")
                return False
            self._pending.append((transcript_id, text))
            return True

    def poll(self) -> dict[str, str] | None:
        """Return a copy of the oldest pending item without removing it."""
        with self._lock:
            if not self._pending:
                return None
            transcript_id, text = self._pending[0]
            return {"id": transcript_id, "text": text}

    def ack(self, transcript_id: object) -> bool:
        """Remove the head only when its canonical ID matches the acknowledgement."""
        if not is_transcript_id(transcript_id):
            return False
        with self._lock:
            if not self._pending or self._pending[0][0] != transcript_id:
                return False
            self._pending.popleft()
            return True

    def fits_legacy_response(self, text: object) -> bool:
        """Check whether legacy text polling can serialize this text safely."""
        if not is_valid_transcript_text(text):
            return False
        with self._lock:
            if self._max_payload_size == 0:
                return False
            max_payload_size = self._max_payload_size
            secret = self._secret
            generation = self._generation
        if len(text) > max_payload_size:
            return False
        response_size = self._legacy_response_size(text, secret)
        with self._lock:
            return generation == self._generation and response_size <= max_payload_size

    @staticmethod
    def _response_size(response: RobotResponse, secret: str) -> int:
        if secret:
            # A real HMAC always has a fixed 64-character hex digest. Use a
            # placeholder to measure wire size without signing transcript data.
            response.auth = {
                "scheme": AUTH_SCHEME,
                "signature": "0" * SIGNATURE_HEX_LENGTH,
            }
        return len(response.to_json().encode("utf-8"))

    @classmethod
    def _poll_response_size(cls, transcript_id: str, text: str, secret: str) -> int:
        """Measure a complete signed ID response using conservative placeholders."""
        response = RobotResponse.success(
            request_id=_SIZE_CHECK_REQUEST_ID,
            result={"id": transcript_id, "text": text},
            timestamp=-sys.float_info.max,
        )
        return cls._response_size(response, secret)

    @classmethod
    def _legacy_response_size(cls, text: str, secret: str) -> int:
        response = RobotResponse.success(
            request_id=_SIZE_CHECK_REQUEST_ID,
            result=text,
            timestamp=-sys.float_info.max,
        )
        return cls._response_size(response, secret)

    @classmethod
    def _fits_poll_response(
        cls,
        transcript_id: str,
        text: str,
        *,
        max_payload_size: int,
        secret: str,
    ) -> bool:
        if len(text) > max_payload_size:
            return False
        return cls._poll_response_size(transcript_id, text, secret) <= max_payload_size


def register_transcript_commands(registry: CommandRegistry, queue: TranscriptQueue) -> None:
    """Register the two bounded, explicit transcript delivery operations."""

    async def _handle_poll_transcript(arguments: dict[str, Any]) -> dict[str, str] | None:
        queue.select_polling_mode("id")
        return queue.poll()

    async def _handle_ack_transcript(arguments: dict[str, Any]) -> bool:
        return queue.ack(arguments["id"])

    registry.register(
        POLL_TRANSCRIPT_COMMAND,
        _handle_poll_transcript,
        validate_empty_arguments,
    )
    registry.register(
        ACK_TRANSCRIPT_COMMAND,
        _handle_ack_transcript,
        validate_ack_transcript_arguments,
    )
