"""Assertions for the default CPU-only pytest process boundary."""

from __future__ import annotations

import os
import socket
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from deskbuddy.settings import load_settings
from tests.conftest import (
    OPTIONAL_TESTS,
    _validate_explicit_collection_args,
    pytest_ignore_collect,
)
from tests.support.legacy_runtime import HEAVY_MODULES
from tests.support.optional_robot import validated_loopback_host, validated_secret


def test_default_pytest_process_uses_offline_temporary_mock_profile() -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert os.environ["UV_OFFLINE"] == "1"
    assert os.environ["DESKBUDDY_MOCK_HARDWARE"] == "1"
    assert sys.modules.get("picoh") is None

    profile = Path(os.environ["HOME"]).resolve()
    for name in (
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "WIN_PD_OVERRIDE_LOCAL_APPDATA",
    ):
        assert Path(os.environ[name]).resolve().is_relative_to(profile)
    settings = load_settings()
    assert settings.data_dir.is_relative_to(profile)
    assert settings.cache_dir.is_relative_to(profile)
    assert not any(
        name.startswith(("DESKBUDDY_", "OPENAI_"))
        or name in {"API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}
        for name in os.environ
        if name not in {"DESKBUDDY_MOCK_HARDWARE"}
    )


def test_default_pytest_socket_guard_blocks_non_loopback_before_connect() -> None:
    with pytest.raises(AssertionError, match="Gate 1 requires loopback networking"):
        socket.getaddrinfo("example.com", 443)


def test_default_pytest_blocks_optional_runtime_imports() -> None:
    for module in sorted(HEAVY_MODULES - {"tkinter"}):
        with pytest.raises(ModuleNotFoundError) as error:
            import_module(module)
        assert error.value.name == module


def test_optional_mode_only_collects_optional_tree() -> None:
    config = SimpleNamespace(getoption=lambda _name: "external_service")
    root = OPTIONAL_TESTS.parents[1]
    optional_file = OPTIONAL_TESTS / "test_external_robot_service.py"
    ordinary_test = root / "tests" / "test_settings.py"

    assert pytest_ignore_collect(root, config) is False
    assert pytest_ignore_collect(OPTIONAL_TESTS, config) is False
    assert pytest_ignore_collect(optional_file, config) is False
    assert pytest_ignore_collect(ordinary_test, config) is True


def test_optional_mode_rejects_explicit_ordinary_paths_before_collection() -> None:
    root = OPTIONAL_TESTS.parents[1]
    with pytest.raises(pytest.UsageError, match="only accepts paths under tests/optional"):
        _validate_explicit_collection_args(
            ["tests/test_settings.py::test_example"], root, "external_service"
        )


def test_default_mode_rejects_explicit_optional_paths_before_collection() -> None:
    root = OPTIONAL_TESTS.parents[1]
    with pytest.raises(pytest.UsageError, match="require an explicit --run-optional"):
        _validate_explicit_collection_args(
            ["tests/optional/test_external_robot_service.py::test_example"], root, None
        )


@pytest.mark.parametrize(
    "argument",
    [".", "tests", "tests/optional", "tests/optional/test_external_robot_service.py"],
)
def test_optional_mode_allows_optional_paths_and_ancestors(argument) -> None:
    root = OPTIONAL_TESTS.parents[1]
    _validate_explicit_collection_args([argument], root, "external_service")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("localhost", "127.0.0.1"),
        ("LOCALHOST.", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "::1"),
    ],
)
def test_optional_robot_host_validation_is_loopback_only(configured, expected) -> None:
    assert validated_loopback_host(configured) == expected


@pytest.mark.parametrize("configured", ["example.com", "192.0.2.1"])
def test_optional_robot_host_validation_rejects_non_loopback(configured) -> None:
    with pytest.raises(pytest.fail.Exception, match="loopback"):
        validated_loopback_host(configured)


def test_optional_robot_secret_rejects_whitespace_and_preserves_valid_value() -> None:
    with pytest.raises(pytest.fail.Exception, match="non-whitespace"):
        validated_secret(" \t\r\n")
    assert validated_secret("  secret with spaces  ") == "  secret with spaces  "
