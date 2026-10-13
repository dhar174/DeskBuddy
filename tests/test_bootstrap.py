from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import deskbuddy.cli as cli_module

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _config_text(tmp_path: Path) -> str:
    return "\n".join(
        [
            "[deskbuddy]",
            'model_provider = "disabled"',
            f"data_dir = {json.dumps(str(tmp_path / 'data'))}",
            f"cache_dir = {json.dumps(str(tmp_path / 'cache'))}",
            f"offload_dir = {json.dumps(str(tmp_path / 'offload'))}",
            "",
        ]
    )


def _isolated_env() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("DESKBUDDY_") or name == "OPENAI_API_KEY":
            environment.pop(name)
    current_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(SOURCE_ROOT) + (
        os.pathsep + current_path if current_path else ""
    )
    return environment


@pytest.mark.parametrize("entrypoint", ["module", "start_script"])
def test_headless_bootstrap_never_imports_models_or_hardware(
    tmp_path: Path, entrypoint: str
) -> None:
    config = tmp_path / "deskbuddy.toml"
    config.write_text(_config_text(tmp_path), encoding="utf-8")
    blocked_modules = [
        "torch",
        "transformers",
        "whisper",
        "cv2",
        "picoh",
        "openai",
        "sounddevice",
        "pyaudio",
        "deepspeed",
        "torchaudio",
        "torchvision",
        "requests",
        "httpx",
    ]
    script_path = REPOSITORY_ROOT / "start_bot.py"
    if entrypoint == "module":
        launch = "from deskbuddy.cli import main; raise SystemExit(main(['--headless', '--config', sys.argv[1]]))"
    else:
        launch = (
            "import runpy; sys.argv = [sys.argv[1], '--headless', '--config', sys.argv[2]]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        )
    script = f"""
import importlib.abc
import socket
import sys

blocked = set({blocked_modules!r})
class HeavyImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('unexpected heavyweight import: ' + fullname)
        return None
sys.meta_path.insert(0, HeavyImportBlocker())
def forbidden_network(*args, **kwargs):
    raise AssertionError('headless bootstrap attempted network access')
class NetworkBlockedSocket(socket.socket):
    def connect(self, *args, **kwargs):
        forbidden_network(*args, **kwargs)
    def connect_ex(self, *args, **kwargs):
        forbidden_network(*args, **kwargs)
    def sendto(self, *args, **kwargs):
        forbidden_network(*args, **kwargs)
socket.socket = NetworkBlockedSocket
socket.create_connection = forbidden_network
socket.getaddrinfo = forbidden_network
{launch}
"""
    arguments = [sys.executable, "-c", script]
    arguments.extend(
        [str(script_path), str(config)] if entrypoint == "start_script" else [str(config)]
    )

    result = subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        env=_isolated_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["status"] == "configuration valid"
    assert payload["mode"] == "headless"
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "offload").exists()


def test_cli_treats_missing_optional_distributions_as_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "deskbuddy.toml"
    config.write_text(_config_text(tmp_path), encoding="utf-8")

    def missing_distribution(distribution: str) -> str:
        raise importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(cli_module, "version", missing_distribution)
    for name in tuple(os.environ):
        if name.startswith("DESKBUDDY_") or name == "OPENAI_API_KEY":
            monkeypatch.delenv(name, raising=False)

    exit_status = cli_module.main(["--headless", "--config", str(config)])

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_status == 0
    assert captured.err == ""
    assert result["model_provider"] == "disabled"
    assert result["optional_distributions"] == {
        "cloud": None,
        "local": None,
        "speech": None,
        "vision": None,
        "picoh": None,
    }


def test_cli_returns_two_for_invalid_configuration_and_does_not_echo_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "invalid.toml"
    config.write_text(
        '[deskbuddy]\nmodel_provider = "openai"\nunknown = "do-not-leak-this"\n',
        encoding="utf-8",
    )
    for name in tuple(os.environ):
        if name.startswith("DESKBUDDY_") or name == "OPENAI_API_KEY":
            monkeypatch.delenv(name, raising=False)

    exit_status = cli_module.main(["--headless", "--config", str(config)])

    captured = capsys.readouterr()
    assert exit_status == 2
    assert captured.out == ""
    assert "Invalid DeskBuddy configuration" in captured.err
    assert "do-not-leak-this" not in captured.err
