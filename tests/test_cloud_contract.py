from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import deskbuddy.cli as cli_module
import deskbuddy.cloud as cloud_module
from deskbuddy.settings import Settings, SettingsError

HEAVY_IMPORTS = {
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "deepspeed",
    "whisper",
    "cv2",
    "picoh",
}


class _HeavyImportBlocker(MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".", maxsplit=1)[0] in HEAVY_IMPORTS:
            raise AssertionError(f"cloud diagnostic imported local/hardware module {fullname}")
        return None


def _openai_settings(secret: str = "test-secret") -> Settings:
    return Settings(
        model_provider="openai",
        openai_model="test-chat-model",
        openai_api_key=secret,
    )


def _fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    create: Callable[..., Any],
) -> SimpleNamespace:
    class FakeOpenAIError(Exception):
        pass

    class FakeChatCompletion:
        @staticmethod
        def create(**kwargs: Any) -> Any:
            return create(**kwargs)

    sdk = SimpleNamespace(
        ChatCompletion=FakeChatCompletion,
        error=SimpleNamespace(OpenAIError=FakeOpenAIError),
        api_key="preexisting-sdk-global",
    )

    def import_sdk(name: str) -> SimpleNamespace:
        assert name == "openai"
        return sdk

    monkeypatch.setattr(cloud_module, "import_module", import_sdk)
    return sdk


def test_cloud_request_passes_credentials_per_call_without_local_imports_or_sdk_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request: dict[str, Any] = {}

    def create(**kwargs: Any) -> dict[str, Any]:
        request.update(kwargs)
        return {"choices": [{"message": {"content": "fake reply"}}]}

    sdk = _fake_sdk(monkeypatch, create)
    monkeypatch.setattr(sys, "meta_path", [_HeavyImportBlocker(), *sys.meta_path])
    loaded_before = {
        name.split(".", maxsplit=1)[0]
        for name in sys.modules
        if name.split(".", maxsplit=1)[0] in HEAVY_IMPORTS
    }

    reply = cloud_module.request_reply(_openai_settings(), "hello from a test")

    assert reply == "fake reply"
    assert request == {
        "model": "test-chat-model",
        "messages": [{"role": "user", "content": "hello from a test"}],
        "api_key": "test-secret",
        "request_timeout": 30,
    }
    assert sdk.api_key == "preexisting-sdk-global"
    loaded_after = {
        name.split(".", maxsplit=1)[0]
        for name in sys.modules
        if name.split(".", maxsplit=1)[0] in HEAVY_IMPORTS
    }
    assert loaded_after == loaded_before


@pytest.mark.parametrize(
    ("settings", "prompt", "message"),
    [
        (Settings(), "hello", "model_provider=openai"),
        (_openai_settings(), "  \n", "prompt must not be blank"),
    ],
)
def test_provider_and_prompt_guards_run_before_sdk_import(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    prompt: str,
    message: str,
) -> None:
    imports: list[str] = []

    def forbidden_import(name: str) -> None:
        imports.append(name)
        raise AssertionError("SDK must not be imported before input validation")

    monkeypatch.setattr(cloud_module, "import_module", forbidden_import)

    with pytest.raises(SettingsError, match=message):
        cloud_module.request_reply(settings, prompt)

    assert imports == []


def test_missing_openai_extra_has_a_direct_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_sdk(name: str) -> None:
        assert name == "openai"
        raise ModuleNotFoundError("No module named 'openai'", name="openai")

    monkeypatch.setattr(cloud_module, "import_module", missing_sdk)

    with pytest.raises(cloud_module.CloudRequestError, match=r"uv sync --locked --extra cloud"):
        cloud_module.request_reply(_openai_settings(), "hello")


def test_openai_sdk_errors_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "do-not-print-this-api-secret"

    def fail_request(**kwargs: Any) -> None:
        raise FakeOpenAIError(f"rejected key {kwargs['api_key']}")

    sdk = _fake_sdk(monkeypatch, fail_request)
    FakeOpenAIError = sdk.error.OpenAIError

    with pytest.raises(cloud_module.CloudRequestError) as raised:
        cloud_module.request_reply(_openai_settings(secret), "hello")

    assert "FakeOpenAIError" in str(raised.value)
    assert secret not in str(raised.value)
    assert "rejected key" not in str(raised.value)


def test_cli_prompt_branch_uses_selected_provider_and_explicit_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in tuple(__import__("os").environ):
        if name.startswith("DESKBUDDY_") or name == "OPENAI_API_KEY":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "cli-test-secret")
    config = tmp_path / "cloud.toml"
    config.write_text(
        '[deskbuddy]\nmodel_provider = "openai"\nopenai_model = "configured-model"\n',
        encoding="utf-8",
    )
    received: dict[str, Any] = {}

    def fake_request(settings: Settings, prompt: str) -> str:
        received["provider"] = settings.model_provider
        received["model"] = settings.openai_model
        received["key"] = settings.openai_api_key.get_secret_value()
        received["prompt"] = prompt
        return "offline fake response"

    monkeypatch.setattr(cloud_module, "request_reply", fake_request)

    exit_status = cli_module.main(["--config", str(config), "--prompt", "diagnostic prompt"])

    captured = capsys.readouterr()
    assert exit_status == 0
    assert captured.err == ""
    assert captured.out == "offline fake response\n"
    assert received == {
        "provider": "openai",
        "model": "configured-model",
        "key": "cli-test-secret",
        "prompt": "diagnostic prompt",
    }


def test_cli_rejects_offline_headless_mode_with_prompt_without_calling_cloud(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_request(settings: Settings, prompt: str) -> str:
        pytest.fail("mutually exclusive CLI options must be rejected before cloud dispatch")

    monkeypatch.setattr(cloud_module, "request_reply", forbidden_request)

    with pytest.raises(SystemExit) as raised:
        cli_module.main(["--headless", "--prompt", "must not be sent"])

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "not allowed with argument" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("error_type", ["authentication", "timeout", "connection", "missing_sdk"])
def test_cli_cloud_failures_are_distinct_from_invalid_configuration(
    error_type: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_module, "load_settings", lambda **kwargs: _openai_settings())

    def fail_request(settings: Settings, prompt: str) -> str:
        raise cloud_module.CloudRequestError(f"{error_type} failure (redacted)")

    monkeypatch.setattr(cloud_module, "request_reply", fail_request)

    status = cli_module.main(["--prompt", "hello"])

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert f"DeskBuddy cloud diagnostic failed: {error_type}" in captured.err
    assert "Invalid DeskBuddy configuration" not in captured.err
    assert "test-secret" not in captured.err


def test_cli_invalid_settings_fail_before_cloud_dispatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def invalid_settings(**kwargs: Any) -> Settings:
        raise SettingsError("model_provider=openai requires OPENAI_API_KEY")

    def forbidden_request(settings: Settings, prompt: str) -> str:
        pytest.fail("invalid settings must be rejected before SDK execution")

    monkeypatch.setattr(cli_module, "load_settings", invalid_settings)
    monkeypatch.setattr(cloud_module, "request_reply", forbidden_request)

    assert cli_module.main(["--prompt", "hello"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid DeskBuddy configuration" in captured.err
    assert "OPENAI_API_KEY" in captured.err


def test_cli_blank_prompt_is_an_input_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_module, "load_settings", lambda **kwargs: _openai_settings())

    assert cli_module.main(["--prompt", "   "]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid cloud diagnostic input" in captured.err
    assert "prompt must not be blank" in captured.err
    assert "Invalid DeskBuddy configuration" not in captured.err
