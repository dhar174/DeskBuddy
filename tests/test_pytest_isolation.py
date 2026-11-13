"""Assertions for the default CPU-only pytest process boundary."""

from __future__ import annotations

import os
import socket
import sys
from importlib import import_module
from pathlib import Path

import pytest

from deskbuddy.settings import load_settings
from tests.support.legacy_runtime import HEAVY_MODULES


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
