"""Validated, immutable configuration with explicit, side-effect-free loading."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from dotenv import dotenv_values
from platformdirs import user_cache_path, user_config_path, user_data_path
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)


class SettingsError(ValueError):
    """Actionable configuration error safe to display without secret values."""


def _data_dir() -> Path:
    return user_data_path("DeskBuddy", appauthor=False)


def _cache_dir() -> Path:
    return user_cache_path("DeskBuddy", appauthor=False)


class Settings(BaseModel):
    """No I/O beyond checking explicitly supplied paths; no global instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    model_provider: Literal["disabled", "local", "openai"] = "disabled"
    local_model_endpoint: str = "http://127.0.0.1:11434"
    openai_model: str = "gpt-3.5-turbo"
    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    medium_model_id: str = "PygmalionAI/pygmalion-350m"
    fast_model_id: str = "microsoft/DialoGPT-large"
    small_model_id: str = "facebook/blenderbot_small-90M"
    blenderbot_model_id: str = "facebook/blenderbot-400M-distill"
    vision_model_id: str = "microsoft/git-large-vqav2"
    topic_model_id: str = "knkarthick/TOPIC-DIALOGSUM"
    summary_model_id: str = "philschmid/bart-large-cnn-samsum"
    short_summary_model_id: str = "knkarthick/meeting-summary-samsum"
    title_model_id: str = "fabiochiu/t5-small-medium-title-generation"
    spacy_model_id: str = "en_core_web_trf"
    question_classifier_model_id: str = "shahrukhx01/question-vs-statement-classifier"
    nli_model_id: str = "valhalla/distilbart-mnli-12-3"
    command_model_id: str = "gokuls/bert-tiny-Massive-intent-KD-BERT"
    question_generator_model_id: str = "voidful/context-only-question-generator"
    embedding_model_id: str = "sentence-transformers/all-mpnet-base-v2"
    completion_model_id: str = "text-curie-001"
    sd_classifier_model_id: str = "curie"
    qa_model_id: str = "distilbert-base-cased-distilled-squad"
    checkpoint_dir: Path | None = None
    tokenizer_dir: Path | None = None
    data_dir: Path = Field(default_factory=_data_dir)
    cache_dir: Path = Field(default_factory=_cache_dir)
    offload_dir: Path | None = None
    robot_host: str = "127.0.0.1"
    robot_bind_address: str = "127.0.0.1"
    robot_port: int = Field(default=8888, ge=1, le=65535)
    robot_auth_secret: SecretStr | None = Field(default=None, repr=False)
    robot_connect_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    robot_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    camera_source: str | int | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    gpu_memory_limit: str | None = None
    cpu_memory_limit: str | None = None

    @field_validator("robot_port", mode="before")
    @classmethod
    def port_is_integer(cls, value: Any) -> Any:
        if isinstance(value, bool) or isinstance(value, float):
            raise ValueError("must be an integer from 1 to 65535")
        return value

    @field_validator("checkpoint_dir", "tokenizer_dir", "offload_dir", mode="before")
    @classmethod
    def blank_optional_path(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("data_dir", "cache_dir", mode="before")
    @classmethod
    def storage_path_is_not_blank(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError(
                "omit this setting for automatic paths; an explicit path must not be blank"
            )
        return value

    @field_validator(
        "robot_host",
        "robot_bind_address",
        "openai_model",
        "medium_model_id",
        "fast_model_id",
        "small_model_id",
        "blenderbot_model_id",
        "vision_model_id",
        "topic_model_id",
        "summary_model_id",
        "short_summary_model_id",
        "title_model_id",
        "spacy_model_id",
        "question_classifier_model_id",
        "nli_model_id",
        "command_model_id",
        "question_generator_model_id",
        "embedding_model_id",
        "completion_model_id",
        "sd_classifier_model_id",
        "qa_model_id",
    )
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("openai_api_key", "robot_auth_secret", mode="before")
    @classmethod
    def empty_secret(cls, value: Any) -> Any:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("local_model_endpoint")
    @classmethod
    def endpoint_is_http(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("must be an http:// or https:// URL with a host")
        try:
            parsed.port
        except ValueError:
            raise ValueError("URL port must be an integer from 1 to 65535") from None
        return value

    @field_validator("camera_source", mode="before")
    @classmethod
    def camera_is_optional(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("use a nonnegative camera index, URL, or image path")
        if isinstance(value, str) and value.isdecimal():
            return int(value)
        if isinstance(value, int) and value < 0:
            raise ValueError("camera index must be nonnegative")
        if isinstance(value, str) and "://" in value:
            try:
                parsed = urlsplit(value)
                port = parsed.port
                if (
                    not re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*", parsed.scheme)
                    or not parsed.hostname
                    or any(character.isspace() for character in value)
                    or "\\" in parsed.netloc
                    or parsed.netloc.endswith(":")
                    or (port is not None and not 1 <= port <= 65535)
                ):
                    raise ValueError
            except ValueError:
                raise ValueError(
                    "camera URL requires a scheme, hostname and valid port (1 to 65535)"
                ) from None
        return value

    @field_validator("gpu_memory_limit", "cpu_memory_limit", mode="before")
    @classmethod
    def memory_is_positive(cls, value: Any) -> str | None:
        if value is None or (isinstance(value, str) and value in {"", "auto"}):
            return None
        if not isinstance(value, str) or not re.fullmatch(r"[1-9]\d*(?:MiB|GiB)", value):
            raise ValueError("use auto or a positive integer with MiB/GiB, e.g. 4096MiB")
        return value

    @model_validator(mode="after")
    def resolve_and_validate(self) -> Self:
        # Relative data_dir starts at the OS user-data directory, never the checkout.
        data = self.data_dir.expanduser()
        if not data.is_absolute():
            data = _data_dir() / data
        object.__setattr__(self, "data_dir", data.resolve())
        for name in ("cache_dir", "offload_dir", "checkpoint_dir", "tokenizer_dir"):
            value = getattr(self, name)
            if name == "offload_dir" and value is None:
                value = self.cache_dir / "offload"
            if value is None:
                continue
            value = value.expanduser()
            if not value.is_absolute():
                value = self.data_dir / value
            value = value.resolve()
            if name in {"checkpoint_dir", "tokenizer_dir"} and not value.is_dir():
                raise ValueError(f"{name} must name an existing directory: {value}")
            if value.exists() and not value.is_dir():
                raise ValueError(f"{name} must be a directory: {value}")
            object.__setattr__(self, name, value)
        if self.data_dir.exists() and not self.data_dir.is_dir():
            raise ValueError("data_dir must be a directory")
        if isinstance(self.camera_source, str) and "://" not in self.camera_source:
            image_path = Path(self.camera_source).expanduser()
            if not image_path.is_absolute():
                image_path = self.data_dir / image_path
            if not image_path.is_file():
                raise ValueError("camera_source image path must name an existing file")
            object.__setattr__(self, "camera_source", str(image_path.resolve()))
        if self.model_provider == "openai" and self.openai_api_key is None:
            raise ValueError("model_provider=openai requires OPENAI_API_KEY")
        return self

    def require_path(self, field_name: Literal["checkpoint_dir", "tokenizer_dir"]) -> Path:
        value = getattr(self, field_name)
        if value is None or not value.is_dir():
            raise SettingsError(f"Set DESKBUDDY_{field_name.upper()} to an existing directory")
        return value

    def data_path(self, filename: str) -> Path:
        path = (self.data_dir / filename).resolve()
        if not path.is_relative_to(self.data_dir):
            raise SettingsError("Data filename must stay inside data_dir")
        return path

    def prepare_directories(self) -> None:
        """Explicit runtime step; construction and bootstrap do not create directories."""
        for path in (self.data_dir, self.cache_dir, self.offload_dir):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

    def resource_limits(self) -> dict[int | str, str] | None:
        """Merge explicit budgets with Accelerate's auto values at the caller."""
        limits: dict[int | str, str] = {}
        if self.gpu_memory_limit is not None:
            limits[0] = self.gpu_memory_limit
        if self.cpu_memory_limit is not None:
            limits["cpu"] = self.cpu_memory_limit
        return limits or None

    def torch_device(self, torch_module: Any) -> str:
        """Resolve hardware only when legacy runtime explicitly requests it."""
        if self.device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch_module.cuda.is_available():
            raise SettingsError("DESKBUDDY_DEVICE=cuda requires an available CUDA device")
        if self.device == "mps" and not torch_module.backends.mps.is_available():
            raise SettingsError("DESKBUDDY_DEVICE=mps requires an available Apple MPS device")
        return self.device


def load_settings(
    *,
    config_file: str | Path | None = None,
    env_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Explicit overrides > process env > explicit dotenv > TOML > safe defaults.

    No implicit checkout .env search, environment mutation, or network/hardware calls.
    Relative setting paths resolve against data_dir (including paths from TOML).
    """
    environment = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    selected_config = config_file or environment.get("DESKBUDDY_CONFIG_FILE")
    config_path = (
        Path(selected_config).expanduser()
        if selected_config
        else user_config_path("DeskBuddy", appauthor=False) / "deskbuddy.toml"
    )
    try:
        if selected_config or config_path.exists():
            document = tomllib.loads(config_path.read_text(encoding="utf-8"))
            if set(document) != {"deskbuddy"} or not isinstance(document["deskbuddy"], dict):
                raise SettingsError("Config file must contain exactly one [deskbuddy] table")
            values.update(document["deskbuddy"])
        selected_env = env_file or environment.get("DESKBUDDY_ENV_FILE")
        dotenv = (
            dotenv_values(Path(selected_env).expanduser(), interpolate=False)
            if selected_env
            else {}
        )
        if selected_env and not Path(selected_env).expanduser().is_file():
            raise SettingsError(f"Environment file does not exist: {selected_env}")
        for source in (dotenv, environment):
            for field in Settings.model_fields:
                key = (
                    "OPENAI_API_KEY" if field == "openai_api_key" else f"DESKBUDDY_{field.upper()}"
                )
                if key in source and source[key] is not None:
                    values[field] = source[key]
        values.update(overrides or {})
        return Settings.model_validate(values)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(map(str, item['loc'])) or 'settings'}: {item['msg']}"
            for item in error.errors(include_input=False, include_url=False)
        )
        raise SettingsError(details) from None
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise SettingsError(f"Cannot read configuration: {error}") from None
