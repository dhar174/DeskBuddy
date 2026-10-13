"""Offline configuration validation. Never imports the legacy application."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from deskbuddy.settings import SettingsError, load_settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate DeskBuddy configuration without starting hardware or models"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="offline bootstrap (the default)")
    parser.add_argument("--config", type=Path, help="TOML with a [deskbuddy] table")
    parser.add_argument("--env-file", type=Path, help="explicit dotenv file")
    mode.add_argument(
        "--prompt", help="explicitly send one OpenAI API diagnostic request (may incur API charges)"
    )
    arguments = parser.parse_args(argv)
    try:
        settings = load_settings(config_file=arguments.config, env_file=arguments.env_file)
    except SettingsError as error:
        print(f"Invalid DeskBuddy configuration: {error}", file=sys.stderr)
        return 2
    if arguments.prompt is not None:
        from deskbuddy.cloud import CloudRequestError, request_reply

        try:
            reply = request_reply(settings, arguments.prompt)
        except SettingsError as error:
            print(f"Invalid cloud diagnostic input: {error}", file=sys.stderr)
            return 2
        except CloudRequestError as error:
            print(f"DeskBuddy cloud diagnostic failed: {error}", file=sys.stderr)
            return 1
        print(reply)
        return 0
    installed: dict[str, str | None] = {}
    for capability, distribution in {
        "cloud": "openai",
        "local": "transformers",
        "speech": "openai-whisper",
        "vision": "opencv-python-headless",
        "picoh": "picoh",
    }.items():
        try:
            installed[capability] = version(distribution)
        except PackageNotFoundError:
            installed[capability] = None
    print(
        json.dumps(
            {
                "status": "configuration valid",
                "mode": "headless",
                "model_provider": settings.model_provider,
                "data_dir": str(settings.data_dir),
                "cache_dir": str(settings.cache_dir),
                "optional_distributions": installed,
            },
            indent=2,
        )
    )
    return 0
