from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

import deskbuddy.settings as settings_module
from deskbuddy.settings import Settings, SettingsError, load_settings


@pytest.fixture
def app_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "user-data"
    cache_root = tmp_path / "user-cache"
    config_root = tmp_path / "user-config"
    monkeypatch.setattr(settings_module, "user_data_path", lambda *args, **kwargs: data_root)
    monkeypatch.setattr(settings_module, "user_cache_path", lambda *args, **kwargs: cache_root)
    monkeypatch.setattr(settings_module, "user_config_path", lambda *args, **kwargs: config_root)
    return data_root, cache_root, config_root


def test_defaults_are_safe_immutable_and_do_not_create_directories(
    app_roots: tuple[Path, Path, Path],
) -> None:
    data_root, cache_root, _ = app_roots

    settings = Settings()

    assert settings.model_provider == "disabled"
    assert settings.device == "auto"
    assert settings.robot_host == "127.0.0.1"
    assert settings.robot_bind_address == "127.0.0.1"
    assert settings.robot_port == 8888
    assert settings.openai_api_key is None
    assert settings.checkpoint_dir is None
    assert settings.tokenizer_dir is None
    assert settings.data_dir == data_root.resolve()
    assert settings.cache_dir == cache_root.resolve()
    assert settings.offload_dir == (cache_root / "offload").resolve()
    assert not data_root.exists()
    assert not cache_root.exists()
    with pytest.raises((ValidationError, TypeError)):
        settings.robot_port = 9000  # type: ignore[misc]


def test_relative_paths_resolve_from_user_data_not_current_working_directory(
    app_roots: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root, _, _ = app_roots
    checkout_cwd = tmp_path / "checkout"
    checkout_cwd.mkdir()
    monkeypatch.chdir(checkout_cwd)

    settings = Settings(
        data_dir=Path("deskbuddy"), cache_dir=Path("cache"), offload_dir=Path("offload")
    )

    assert settings.data_dir == (data_root / "deskbuddy").resolve()
    assert settings.cache_dir == (settings.data_dir / "cache").resolve()
    assert settings.offload_dir == (settings.data_dir / "offload").resolve()
    assert not (checkout_cwd / "deskbuddy").exists()


def test_data_path_stays_inside_application_data_directory(
    app_roots: tuple[Path, Path, Path],
) -> None:
    settings = Settings()

    assert settings.data_path("history.json") == settings.data_dir / "history.json"
    with pytest.raises(SettingsError, match="inside data_dir"):
        settings.data_path("../outside.json")


def test_directories_are_created_only_after_explicit_prepare(
    app_roots: tuple[Path, Path, Path],
) -> None:
    data_root, cache_root, _ = app_roots
    settings = Settings()

    assert not data_root.exists()
    assert not cache_root.exists()
    settings.prepare_directories()
    assert data_root.is_dir()
    assert cache_root.is_dir()
    assert (cache_root / "offload").is_dir()


def test_source_precedence_is_overrides_then_environment_then_dotenv_then_toml(
    tmp_path: Path, app_roots: tuple[Path, Path, Path]
) -> None:
    config_file = tmp_path / "deskbuddy.toml"
    config_file.write_text("[deskbuddy]\nrobot_port = 6100\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("DESKBUDDY_ROBOT_PORT=6200\n", encoding="utf-8")

    from_toml = load_settings(config_file=config_file, environ={})
    from_dotenv = load_settings(config_file=config_file, env_file=env_file, environ={})
    from_environment = load_settings(
        config_file=config_file,
        env_file=env_file,
        environ={"DESKBUDDY_ROBOT_PORT": "6300"},
    )
    from_override = load_settings(
        config_file=config_file,
        env_file=env_file,
        environ={"DESKBUDDY_ROBOT_PORT": "6300"},
        overrides={"robot_port": 6400},
    )

    assert from_toml.robot_port == 6100
    assert from_dotenv.robot_port == 6200
    assert from_environment.robot_port == 6300
    assert from_override.robot_port == 6400


def test_environment_variable_names_are_provider_specific_and_secrets_are_redacted(
    app_roots: tuple[Path, Path, Path],
) -> None:
    settings = load_settings(environ={"OPENAI_API_KEY": "super-secret"})

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "super-secret"
    assert "super-secret" not in repr(settings)
    assert "super-secret" not in settings.model_dump_json()
    assert "**********" in settings.model_dump_json()
    with pytest.raises(SettingsError, match="OPENAI_API_KEY") as error:
        load_settings(environ={"DESKBUDDY_MODEL_PROVIDER": "openai"})
    assert "super-secret" not in str(error.value)


@pytest.mark.parametrize("contents", ["", "robot_port=3000\n", "[deskbuddy]\nport = 3000\n"])
def test_explicit_config_requires_a_valid_deskbuddy_table(
    tmp_path: Path, contents: str, app_roots: tuple[Path, Path, Path]
) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text(contents, encoding="utf-8")

    with pytest.raises(SettingsError):
        load_settings(config_file=config_file, environ={})


def test_missing_or_malformed_explicit_config_fails_with_actionable_error(
    tmp_path: Path, app_roots: tuple[Path, Path, Path]
) -> None:
    missing = tmp_path / "missing.toml"
    malformed = tmp_path / "malformed.toml"
    malformed.write_text("[deskbuddy\nrobot_port = 42", encoding="utf-8")

    with pytest.raises(SettingsError, match="Cannot read configuration"):
        load_settings(config_file=missing, environ={})
    with pytest.raises(SettingsError, match="Cannot read configuration"):
        load_settings(config_file=malformed, environ={})


def test_missing_explicit_dotenv_file_fails(
    tmp_path: Path, app_roots: tuple[Path, Path, Path]
) -> None:
    with pytest.raises(SettingsError, match="Environment file does not exist"):
        load_settings(env_file=tmp_path / "missing.env", environ={})


def test_unknown_config_fields_fail_closed(
    tmp_path: Path, app_roots: tuple[Path, Path, Path]
) -> None:
    config_file = tmp_path / "unknown.toml"
    config_file.write_text("[deskbuddy]\nfuture_typo = true\n", encoding="utf-8")

    with pytest.raises(SettingsError, match="Extra inputs are not permitted"):
        load_settings(config_file=config_file, environ={})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("robot_port", "not-a-port", "robot_port"),
        ("robot_port", 0, "robot_port"),
        ("robot_port", 65536, "robot_port"),
        ("robot_timeout_seconds", 0, "robot_timeout_seconds"),
        ("robot_connect_timeout_seconds", -1, "robot_connect_timeout_seconds"),
        ("model_provider", "other", "model_provider"),
        ("model_provider", "openai", "OPENAI_API_KEY"),
        ("device", "tpu", "device"),
        ("gpu_memory_limit", "a lot", "gpu_memory_limit"),
        ("cpu_memory_limit", "0GiB", "cpu_memory_limit"),
        ("local_model_endpoint", "file:///tmp/model", "local_model_endpoint"),
        ("robot_host", " ", "robot_host"),
    ],
)
def test_invalid_values_fail_with_the_field_or_requirement_named(
    field: str, value: object, message: str, app_roots: tuple[Path, Path, Path]
) -> None:
    with pytest.raises(SettingsError, match=message):
        load_settings(overrides={field: value}, environ={})


@pytest.mark.parametrize("endpoint", ["http://localhost:65536", "http://localhost:not-a-port"])
def test_invalid_local_endpoint_ports_fail_early(
    endpoint: str, app_roots: tuple[Path, Path, Path]
) -> None:
    with pytest.raises(SettingsError, match="URL port"):
        load_settings(overrides={"local_model_endpoint": endpoint}, environ={})


def test_resource_limit_mapping_is_rejected_as_invalid_configuration(
    app_roots: tuple[Path, Path, Path],
) -> None:
    with pytest.raises(SettingsError, match="gpu_memory_limit"):
        load_settings(overrides={"gpu_memory_limit": {0: "4096MiB"}}, environ={})


@pytest.mark.parametrize("camera_source", ["0", 0, "http://127.0.0.1:8080/stream"])
def test_camera_indices_and_stream_urls_are_valid_optional_sources(
    camera_source: str | int, app_roots: tuple[Path, Path, Path]
) -> None:
    settings = load_settings(overrides={"camera_source": camera_source}, environ={})

    if camera_source == "0":
        assert settings.camera_source == 0
    else:
        assert settings.camera_source == camera_source


def test_camera_image_path_resolves_from_data_dir_and_must_exist(
    tmp_path: Path, app_roots: tuple[Path, Path, Path]
) -> None:
    data_dir = tmp_path / "app-data"
    data_dir.mkdir()
    image = data_dir / "frame.png"
    image.write_bytes(b"image fixture; no decoder is invoked")

    settings = load_settings(
        overrides={"data_dir": data_dir, "camera_source": "frame.png"}, environ={}
    )

    assert settings.camera_source == str(image.resolve())
    with pytest.raises(SettingsError, match="camera_source"):
        load_settings(overrides={"data_dir": data_dir, "camera_source": "missing.png"}, environ={})


@pytest.mark.parametrize("field", ["openai_api_key", "robot_auth_secret"])
@pytest.mark.parametrize("empty_value", ["", "   ", SecretStr(" ")])
def test_empty_optional_credentials_are_normalized_to_unconfigured(
    field: str,
    empty_value: str | SecretStr,
    app_roots: tuple[Path, Path, Path],
) -> None:
    settings = load_settings(overrides={field: empty_value}, environ={})

    assert getattr(settings, field) is None


def test_blank_secret_cannot_satisfy_selected_cloud_provider(
    app_roots: tuple[Path, Path, Path],
) -> None:
    with pytest.raises(SettingsError, match="requires OPENAI_API_KEY"):
        load_settings(
            overrides={
                "model_provider": "openai",
                "openai_api_key": SecretStr("   "),
            },
            environ={},
        )


@pytest.mark.parametrize("field", ["checkpoint_dir", "tokenizer_dir"])
def test_explicit_model_directories_must_exist(
    field: str, tmp_path: Path, app_roots: tuple[Path, Path, Path]
) -> None:
    absent = tmp_path / "not-installed"

    with pytest.raises(SettingsError, match=field):
        load_settings(overrides={field: absent}, environ={})
    with pytest.raises(SettingsError, match=f"DESKBUDDY_{field.upper()}"):
        Settings().require_path(field)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["checkpoint_dir", "tokenizer_dir"])
def test_empty_optional_model_paths_stay_unconfigured_until_required(
    field: str, app_roots: tuple[Path, Path, Path]
) -> None:
    settings = load_settings(overrides={field: ""}, environ={})

    assert getattr(settings, field) is None
    with pytest.raises(SettingsError, match=f"DESKBUDDY_{field.upper()}"):
        settings.require_path(field)  # type: ignore[arg-type]


def test_checkpoint_and_tokenizer_paths_accept_existing_directories(
    tmp_path: Path, app_roots: tuple[Path, Path, Path]
) -> None:
    checkpoint = tmp_path / "checkpoint"
    tokenizer = tmp_path / "tokenizer"
    checkpoint.mkdir()
    tokenizer.mkdir()

    settings = Settings(checkpoint_dir=checkpoint, tokenizer_dir=tokenizer)

    assert settings.require_path("checkpoint_dir") == checkpoint.resolve()
    assert settings.require_path("tokenizer_dir") == tokenizer.resolve()


def test_non_directory_storage_path_fails(
    tmp_path: Path, app_roots: tuple[Path, Path, Path]
) -> None:
    file_path = tmp_path / "file"
    file_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SettingsError, match="data_dir must be a directory"):
        load_settings(overrides={"data_dir": file_path}, environ={})


def test_optional_capabilities_remain_unconfigured_by_default(
    app_roots: tuple[Path, Path, Path],
) -> None:
    settings = Settings()

    assert settings.model_provider == "disabled"
    assert settings.openai_api_key is None
    assert settings.camera_source is None
    assert settings.checkpoint_dir is None
    assert settings.tokenizer_dir is None


def test_resource_limits_and_device_resolution_are_lazy() -> None:
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return False

    assert Settings().resource_limits() is None
    assert Settings(gpu_memory_limit="8GiB", cpu_memory_limit="4096MiB").resource_limits() == {
        0: "8GiB",
        "cpu": "4096MiB",
    }
    assert Settings().torch_device(FakeTorch) == "cpu"
    with pytest.raises(SettingsError, match="available CUDA device"):
        Settings(device="cuda").torch_device(FakeTorch)
