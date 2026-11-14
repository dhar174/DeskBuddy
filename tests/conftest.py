"""Pytest configuration, import setup, and shared fixtures."""

from __future__ import annotations

import os
import socket
import sys
from ipaddress import ip_address
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tests.support.legacy_runtime import HEAVY_MODULES
from transport.config import TransportConfig

TEST_SECRET = "test-deskbuddy-secret-32-chars-long-abc!"
OPTIONAL_MARKERS = ("hardware", "model_weights", "external_service")
OPTIONAL_TESTS = Path(__file__).resolve().parent / "optional"
# The robot adapter imports tkinter but does not open a GUI for loopback tests.
BLOCKED_RUNTIME_IMPORTS = HEAVY_MODULES - {"tkinter"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-optional",
        choices=OPTIONAL_MARKERS,
        help="Run only the specified optional integration category.",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    category = config.getoption("--run-optional")
    # Optional runs select only their isolated templates. This also means a
    # command that names the repository root cannot include ordinary contract
    # modules without the Gate 1 boundary.
    if category is not None:
        path = Path(collection_path)
        return not (path.is_relative_to(OPTIONAL_TESTS) or OPTIONAL_TESTS.is_relative_to(path))
    # Avoid importing opt-in modules (and their optional dependencies) in Gate 1.
    if collection_path.is_relative_to(OPTIONAL_TESTS):
        return True
    return None


def _validate_explicit_collection_args(
    args: list[str], invocation_dir: Path, category: str | None
) -> None:
    for argument in args:
        # Pytest accepts file::test selectors; only the filesystem portion is
        # relevant to the isolation boundary.
        path_text = argument.split("::", maxsplit=1)[0]
        if not path_text or path_text.startswith("-"):
            continue
        path = (invocation_dir / path_text).resolve()
        is_optional = path.is_relative_to(OPTIONAL_TESTS)
        if category is not None:
            if not is_optional and not OPTIONAL_TESTS.is_relative_to(path):
                raise pytest.UsageError(
                    "--run-optional only accepts paths under tests/optional "
                    "or their directory ancestors"
                )
        elif is_optional:
            raise pytest.UsageError(
                "tests/optional paths require an explicit --run-optional category"
            )


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
    _validate_explicit_collection_args(config.args, config.invocation_params.dir, category)
    env = pytest.MonkeyPatch()
    config.add_cleanup(env.undo)
    if category in (None, "model_weights"):
        env.setenv("HF_HUB_OFFLINE", "1")
        env.setenv("HF_DATASETS_OFFLINE", "1")
        env.setenv("TRANSFORMERS_OFFLINE", "1")
    if category in {"hardware", "model_weights"}:
        # These categories deliberately exercise user-provisioned devices,
        # weights, runtimes, credentials, and configuration in tests/optional.
        return
    test_profile = TemporaryDirectory(prefix="deskbuddy-tests-")
    config.add_cleanup(test_profile.cleanup)
    for name in tuple(os.environ):
        explicit_optional_robot_config = category == "external_service" and name in {
            "DESKBUDDY_TEST_ROBOT_HOST",
            "DESKBUDDY_TEST_ROBOT_PORT",
            "DESKBUDDY_TEST_ROBOT_AUTH_SECRET",
        }
        if (
            name.startswith(("DESKBUDDY_", "OPENAI_")) and not explicit_optional_robot_config
        ) or name in {
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
    if category is None:
        env.setenv("DESKBUDDY_MOCK_HARDWARE", "1")
    env.setenv("UV_OFFLINE", "1")
    # Block optional runtimes in Gate 1 and the read-only external smoke test.
    # Hardware/model tests are explicitly selected templates where the requested
    # runtime is the subject of the test.
    if category in (None, "external_service"):
        preloaded = {
            module.partition(".")[0]
            for module, loaded in sys.modules.copy().items()
            if loaded is not None and module.partition(".")[0] in BLOCKED_RUNTIME_IMPORTS
        }
        if preloaded:
            raise pytest.UsageError(
                f"Gate 1 runtime imported before isolation: {', '.join(sorted(preloaded))}"
            )
        for module in BLOCKED_RUNTIME_IMPORTS:
            env.setitem(sys.modules, module, None)

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
