"""Pytest configuration, import setup, and shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# Enable headless/mock mode for test suite execution by default
os.environ.setdefault("DESKBUDDY_MOCK_HARDWARE", "1")

from transport.config import TransportConfig  # noqa: E402

TEST_SECRET = "test-deskbuddy-secret-32-chars-long-abc!"


@pytest.fixture
def test_secret() -> str:
    return TEST_SECRET


@pytest.fixture
def ephemeral_server_config(test_secret: str) -> TransportConfig:
    return TransportConfig(
        host="127.0.0.1",
        port=0,  # OS will assign ephemeral port
        secret=test_secret,
        connect_timeout=2.0,
        read_timeout=2.0,
        write_timeout=2.0,
        request_timeout=3.0,
        max_payload_size=65536,
        max_clock_skew=10.0,
        max_retries=2,
        retry_backoff=0.05,
    )
