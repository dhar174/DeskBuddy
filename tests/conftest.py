"""Pytest configuration, import setup, and shared fixtures."""

from __future__ import annotations

import os
import socket
import sys
from ipaddress import ip_address
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from transport.config import TransportConfig

TEST_SECRET = "test-deskbuddy-secret-32-chars-long-abc!"
OPTIONAL_MARKERS = ("hardware", "model_weights", "external_service")
OPTIONAL_TESTS = Path(__file__).resolve().parent / "optional"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-optional",
        choices=OPTIONAL_MARKERS,
        help="Run only the specified optional integration category.",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    # Avoid importing opt-in modules (and their optional dependencies) in Gate 1.
    if not config.getoption("--run-optional") and collection_path.is_relative_to(OPTIONAL_TESTS):
        return True
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    category = config.getoption("--run-optional")
    allowed = {category} if category else set()
    selected, deselected = [], []
    for item in items:
        markers = {marker.name for marker in item.iter_markers()}.intersection(OPTIONAL_MARKERS)
        (selected if markers == allowed else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _is_loopback(host: str | bytes) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if host.lower() in {"localhost", "localhost."}:
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return address.is_loopback


def pytest_configure(config: pytest.Config) -> None:
    category = config.getoption("--run-optional")
    env = pytest.MonkeyPatch()
    config.add_cleanup(env.undo)
    if category in (None, "model_weights"):
        env.setenv("HF_HUB_OFFLINE", "1")
        env.setenv("HF_DATASETS_OFFLINE", "1")
        env.setenv("TRANSFORMERS_OFFLINE", "1")
    if category is not None:
        return

    test_profile = TemporaryDirectory(prefix="deskbuddy-tests-")
    config.add_cleanup(test_profile.cleanup)
    for name in tuple(os.environ):
        if name.startswith(("DESKBUDDY_", "OPENAI_")) or name in {
            "API_KEY",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        }:
            env.delenv(name)
    for name in (
        "HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "WIN_PD_OVERRIDE_LOCAL_APPDATA",
        "HF_HOME",
    ):
        env.setenv(name, test_profile.name)
    env.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    env.setenv("DESKBUDDY_MOCK_HARDWARE", "1")
    env.setenv("UV_OFFLINE", "1")
    # The legacy adapter tries the real driver before falling back to MockPicoh.
    env.setitem(sys.modules, "picoh", None)

    active = True

    def guard_network(event: str, args: tuple) -> None:
        if not active or not event.startswith("socket."):
            return
        if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"}:
            sock, address = args
            if sock.family not in (socket.AF_INET, socket.AF_INET6) or address is None:
                return
            host = address[0]
        elif event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
            host = args[0]
            if host is None:
                return
        elif event == "socket.getnameinfo":
            host = args[0][0]
        else:
            return
        if not _is_loopback(host):
            raise AssertionError(f"Gate 1 requires loopback networking ({event})")

    def release_network_guard() -> None:
        nonlocal active
        active = False

    sys.addaudithook(guard_network)
    config.add_cleanup(release_network_guard)


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
