"""One-shot API diagnostic, isolated from the legacy hybrid conversation runtime."""

from __future__ import annotations

from importlib import import_module

from deskbuddy.settings import Settings, SettingsError


class CloudRequestError(RuntimeError):
    """Redacted cloud capability/request failure, distinct from invalid settings."""


def request_reply(settings: Settings, prompt: str) -> str:
    """Make one explicitly requested OpenAI call, without importing local models.

    This diagnostic has no conversation history, classifiers or robot actions.
    Credentials are passed per request rather than stored in mutable SDK globals.
    """
    if settings.model_provider != "openai" or settings.openai_api_key is None:
        raise SettingsError("API diagnostic requires model_provider=openai and OPENAI_API_KEY")
    if not prompt.strip():
        raise SettingsError("API diagnostic prompt must not be blank")
    try:
        sdk = import_module("openai")
    except ModuleNotFoundError as error:
        if error.name != "openai":
            raise
        raise CloudRequestError(
            "Install the OpenAI extra: uv sync --locked --extra cloud"
        ) from None
    try:
        response = sdk.ChatCompletion.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=settings.openai_api_key.get_secret_value(),
            request_timeout=30,
        )
    except sdk.error.OpenAIError as error:
        raise CloudRequestError(
            f"OpenAI request failed ({type(error).__name__}); check credentials, model and network"
        ) from None
    return str(response["choices"][0]["message"]["content"])
