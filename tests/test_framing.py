"""Unit tests for binary length-prefixed framing, fragmentation, and size enforcement."""

import asyncio

import pytest

from transport.framing import (
    HEADER_STRUCT,
    read_frame,
)
from transport.protocol import (
    PayloadTooLargeError,
    TransportError,
    TransportErrorCode,
    TruncatedFrameError,
)


@pytest.mark.asyncio
async def test_frame_roundtrip_below_1024_bytes():
    payload = b"Hello, Picoh! This is a short payload under 1024 bytes."
    reader = asyncio.StreamReader()
    # Write framed data into reader buffer
    reader.feed_data(HEADER_STRUCT.pack(len(payload)) + payload)
    reader.feed_eof()

    received = await read_frame(reader)
    assert received == payload


@pytest.mark.asyncio
async def test_frame_roundtrip_above_1024_bytes():
    # 8192 bytes payload
    payload = b"X" * 8192
    reader = asyncio.StreamReader()
    reader.feed_data(HEADER_STRUCT.pack(len(payload)) + payload)
    reader.feed_eof()

    received = await read_frame(reader, max_payload_size=16384)
    assert received == payload
    assert len(received) == 8192


@pytest.mark.asyncio
async def test_oversized_frame_rejected_before_buffer_allocation():
    # Declared length is 100,000 bytes, limit is 1,000 bytes
    declared_size = 100000
    reader = asyncio.StreamReader()
    reader.feed_data(HEADER_STRUCT.pack(declared_size))

    with pytest.raises(PayloadTooLargeError) as exc_info:
        await read_frame(reader, max_payload_size=1000)

    assert exc_info.value.code == TransportErrorCode.PAYLOAD_TOO_LARGE
    assert "exceeds maximum allowed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_truncated_frame_header():
    reader = asyncio.StreamReader()
    reader.feed_data(b"\x00\x00")  # Only 2 bytes instead of 4
    reader.feed_eof()

    with pytest.raises(TruncatedFrameError):
        await read_frame(reader)


@pytest.mark.asyncio
async def test_truncated_frame_payload():
    reader = asyncio.StreamReader()
    # Declare 20 bytes, feed only 10
    reader.feed_data(HEADER_STRUCT.pack(20) + b"1234567890")
    reader.feed_eof()

    with pytest.raises(TruncatedFrameError):
        await read_frame(reader)


@pytest.mark.asyncio
async def test_clean_eof_raises_eof_error():
    reader = asyncio.StreamReader()
    reader.feed_eof()

    with pytest.raises(EOFError):
        await read_frame(reader)


@pytest.mark.asyncio
async def test_fragmented_frame_delivery():
    payload = b"Fragmented message arriving in tiny 3-byte TCP packets across multiple read cycles."
    full_wire_bytes = HEADER_STRUCT.pack(len(payload)) + payload

    reader = asyncio.StreamReader()

    async def feeder():
        # Feed data in 3-byte chunks with slight delays
        chunk_size = 3
        for i in range(0, len(full_wire_bytes), chunk_size):
            reader.feed_data(full_wire_bytes[i : i + chunk_size])
            await asyncio.sleep(0.001)
        reader.feed_eof()

    feeder_task = asyncio.create_task(feeder())
    received = await read_frame(reader)
    await feeder_task

    assert received == payload


@pytest.mark.asyncio
async def test_multiple_sequential_frames_over_single_connection():
    payloads = [
        b"First message",
        b"Second longer message" * 50,
        b"Third message",
    ]

    reader = asyncio.StreamReader()
    for p in payloads:
        reader.feed_data(HEADER_STRUCT.pack(len(p)) + p)
    reader.feed_eof()

    for expected in payloads:
        actual = await read_frame(reader)
        assert actual == expected


@pytest.mark.asyncio
async def test_read_frame_timeout():
    reader = asyncio.StreamReader()
    # No data fed into reader; should time out
    with pytest.raises(TransportError) as exc_info:
        await read_frame(reader, timeout=0.05)
    assert exc_info.value.code == TransportErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_frame_exact_boundary_at_max_payload_size():
    # Exactly max_payload_size bytes must be accepted and read
    limit = 2048
    payload = b"Z" * limit
    reader = asyncio.StreamReader()
    reader.feed_data(HEADER_STRUCT.pack(limit) + payload)
    reader.feed_eof()

    received = await read_frame(reader, max_payload_size=limit)
    assert received == payload
    assert len(received) == limit


@pytest.mark.asyncio
async def test_frame_exact_boundary_exceeding_by_one_byte():
    # Exactly limit + 1 bytes must be rejected before reading payload
    limit = 2048
    reader = asyncio.StreamReader()
    reader.feed_data(HEADER_STRUCT.pack(limit + 1) + b"Z" * (limit + 1))
    reader.feed_eof()

    with pytest.raises(PayloadTooLargeError) as exc_info:
        await read_frame(reader, max_payload_size=limit)
    assert exc_info.value.code == TransportErrorCode.PAYLOAD_TOO_LARGE
    assert f"Frame size {limit + 1} bytes exceeds maximum allowed {limit} bytes" in str(exc_info.value)


@pytest.mark.asyncio
async def test_frame_exact_max_payload_fragmented():
    # Exactly default max payload (65,536 bytes) fragmented across chunks
    limit = 65536
    payload = b"M" * limit
    wire = HEADER_STRUCT.pack(limit) + payload

    reader = asyncio.StreamReader()

    async def feeder():
        chunk_size = 4096
        for i in range(0, len(wire), chunk_size):
            reader.feed_data(wire[i : i + chunk_size])
            await asyncio.sleep(0.001)
        reader.feed_eof()

    feeder_task = asyncio.create_task(feeder())
    received = await read_frame(reader, max_payload_size=limit)
    await feeder_task

    assert received == payload
    assert len(received) == limit
