"""Length-prefixed binary framing for DeskBuddy transport connections."""

from __future__ import annotations

import asyncio
import struct
from typing import Final

from transport.protocol import (
    PayloadTooLargeError,
    TransportError,
    TransportErrorCode,
    TruncatedFrameError,
)

HEADER_STRUCT: Final = struct.Struct(">I")
HEADER_SIZE: Final[int] = HEADER_STRUCT.size
DEFAULT_MAX_PAYLOAD_SIZE: Final[int] = 65536  # 64 KiB default


async def read_frame(
    reader: asyncio.StreamReader,
    *,
    max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
    timeout: float | None = None,
) -> bytes:
    """Read a single length-prefixed frame from an asyncio StreamReader.

    Rejects frames larger than max_payload_size before allocating memory for the body.
    Raises EOFError if connection was closed cleanly with no bytes read.
    Raises TruncatedFrameError if connection closes prematurely.
    Raises PayloadTooLargeError if declared length exceeds limit.
    Raises TransportError(TIMEOUT) if timeout expires.
    """

    async def _read() -> bytes:
        try:
            length_bytes = await reader.readexactly(HEADER_SIZE)
        except asyncio.IncompleteReadError as exc:
            if len(exc.partial) == 0:
                raise EOFError("Connection closed before reading frame header") from exc
            raise TruncatedFrameError(
                TransportErrorCode.MALFORMED_REQUEST,
                f"Connection closed during frame header read (read {len(exc.partial)} of {HEADER_SIZE} bytes)",
            ) from exc

        (payload_length,) = HEADER_STRUCT.unpack(length_bytes)

        if payload_length > max_payload_size:
            raise PayloadTooLargeError(
                TransportErrorCode.PAYLOAD_TOO_LARGE,
                f"Frame size {payload_length} bytes exceeds maximum allowed {max_payload_size} bytes",
            )

        if payload_length == 0:
            return b""

        try:
            payload = await reader.readexactly(payload_length)
        except asyncio.IncompleteReadError as exc:
            raise TruncatedFrameError(
                TransportErrorCode.MALFORMED_REQUEST,
                f"Connection closed during payload read (read {len(exc.partial)} of {payload_length} bytes)",
            ) from exc

        return payload

    if timeout is not None and timeout > 0:
        try:
            return await asyncio.wait_for(_read(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TransportError(
                TransportErrorCode.TIMEOUT, f"Timed out reading frame after {timeout:.2f}s"
            ) from exc
    return await _read()


async def write_frame(
    writer: asyncio.StreamWriter,
    payload: bytes,
    *,
    max_payload_size: int = DEFAULT_MAX_PAYLOAD_SIZE,
    timeout: float | None = None,
) -> None:
    """Write a single length-prefixed frame to an asyncio StreamWriter.

    Raises PayloadTooLargeError if payload exceeds max_payload_size.
    Raises TransportError(TIMEOUT) if write/drain exceeds timeout.
    """
    if len(payload) > max_payload_size:
        raise PayloadTooLargeError(
            TransportErrorCode.PAYLOAD_TOO_LARGE,
            f"Outgoing frame size {len(payload)} bytes exceeds maximum {max_payload_size} bytes",
        )

    header = HEADER_STRUCT.pack(len(payload))
    writer.write(header + payload)

    async def _drain() -> None:
        await writer.drain()

    if timeout is not None and timeout > 0:
        try:
            await asyncio.wait_for(_drain(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TransportError(
                TransportErrorCode.TIMEOUT, f"Timed out draining write buffer after {timeout:.2f}s"
            ) from exc
    else:
        await _drain()
