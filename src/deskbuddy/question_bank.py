"""Persistence for the legacy UTF-8 JSON question bank."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Literal, NoReturn, TypeGuard

_bank_lock = Lock()


class QuestionBankDataError(ValueError):
    """An existing bank could not be decoded or has the wrong shape."""

    def __init__(self, path: Path, reason: Literal["malformed-file", "invalid-shape"]) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Question bank {path}: {reason}")


def _is_question_bank(value: object) -> TypeGuard[dict[str, str]]:
    return isinstance(value, dict) and all(
        isinstance(question, str) and isinstance(answer, str) and bool(answer.strip())
        for question, answer in value.items()
    )


def _reject_json_constant(constant: str) -> NoReturn:
    """Reject JavaScript constants that Python's JSON parser otherwise accepts."""
    raise ValueError(f"Invalid JSON constant: {constant}")


def _snapshot_question_bank(entries: object) -> dict[str, str]:
    """Copy caller data before validating or serializing it."""
    if not isinstance(entries, Mapping):
        raise ValueError("Question bank entries must map question strings to answer strings")
    try:
        snapshot = dict(entries)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "Question bank entries must map question strings to answer strings"
        ) from error
    if not _is_question_bank(snapshot):
        raise ValueError(
            "Question bank entries must map question strings to nonblank answer strings"
        )
    return snapshot


def _read_bank(path: Path) -> dict[str, str]:
    try:
        source = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return {}
    with source:
        try:
            entries = json.load(source, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise QuestionBankDataError(path, "malformed-file") from None
    if not _is_question_bank(entries):
        raise QuestionBankDataError(path, "invalid-shape")
    return entries


def load_question_bank(path: Path) -> dict[str, str]:
    """Read a complete bank, returning an empty bank if the file is missing."""
    with _bank_lock:
        return _read_bank(path)


def save_question_bank(path: Path, entries: dict[str, str]) -> None:
    """Replace the whole bank atomically within one process."""
    snapshot = _snapshot_question_bank(entries)
    with _bank_lock:
        # Validate the existing file before replacing it. The same lock covers
        # validation and publication so concurrent calls cannot race this check.
        _read_bank(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                errors="backslashreplace",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                # Keep ordinary Unicode readable while escaping isolated
                # surrogates as JSON-compatible backslash sequences.
                json.dump(snapshot, temporary, ensure_ascii=False)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                # Keep the original write or publication error if cleanup fails.
                with suppress(OSError):
                    temporary_path.unlink()
