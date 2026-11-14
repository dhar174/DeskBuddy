from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread, current_thread
from typing import Any, cast

import pytest

from deskbuddy import question_bank


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        (b'{"unfinished":', "malformed-file"),
        (b"\xff", "malformed-file"),
        (b"[]", "invalid-shape"),
        (b'{"question": 42}', "invalid-shape"),
    ],
)
def test_load_rejects_malformed_or_wrong_shaped_data(
    tmp_path: Path, contents: bytes, reason: str
) -> None:
    path = tmp_path / "question_file.json"
    path.write_bytes(contents)

    with pytest.raises(question_bank.QuestionBankDataError) as caught:
        question_bank.load_question_bank(path)

    assert caught.value.path == path
    assert caught.value.reason == reason


def test_missing_bank_is_empty_without_creating_its_directory(tmp_path: Path) -> None:
    path = tmp_path / "not-created" / "question_file.json"

    assert question_bank.load_question_bank(path) == {}
    assert not path.parent.exists()
    assert not path.exists()


def test_save_round_trips_utf8_and_replaces_the_complete_mapping(tmp_path: Path) -> None:
    path = tmp_path / "question_file.json"
    first = {"¿Qué tal?": "你好，世界 🌍", "astronaut": "👩‍🚀"}
    replacement = {"new question": "new answer"}

    question_bank.save_question_bank(path, first)

    assert json.loads(path.read_text(encoding="utf-8")) == first
    assert question_bank.load_question_bank(path) == first

    question_bank.save_question_bank(path, replacement)

    assert question_bank.load_question_bank(path) == replacement


@pytest.mark.parametrize(
    "contents",
    [b"{broken", b"\xfe\xff", b'{"question": null}'],
)
def test_save_refuses_to_replace_corrupt_bank_and_preserves_its_bytes(
    tmp_path: Path, contents: bytes
) -> None:
    path = tmp_path / "question_file.json"
    path.write_bytes(contents)

    with pytest.raises(question_bank.QuestionBankDataError):
        question_bank.save_question_bank(path, {"replacement": "answer"})

    assert path.read_bytes() == contents
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "invalid_entries",
    [
        {"question": 42},
        {3: "answer"},
        None,
    ],
)
def test_invalid_proposed_mapping_is_rejected_before_disk_access(
    tmp_path: Path, invalid_entries: object
) -> None:
    path = tmp_path / "uncreated" / "question_file.json"

    with pytest.raises(ValueError) as caught:
        question_bank.save_question_bank(path, cast(dict[str, str], invalid_entries))

    assert not isinstance(caught.value, question_bank.QuestionBankDataError)
    assert not path.parent.exists()


def test_filesystem_errors_remain_distinct_from_question_bank_data_errors(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    blocker = tmp_path / "parent-is-a-file"
    blocker.write_bytes(b"preserve me")

    with pytest.raises(OSError) as load_error:
        question_bank.load_question_bank(directory)
    with pytest.raises(OSError) as save_error:
        question_bank.save_question_bank(blocker / "question_file.json", {"q": "a"})

    assert not isinstance(load_error.value, question_bank.QuestionBankDataError)
    assert not isinstance(save_error.value, question_bank.QuestionBankDataError)
    assert blocker.read_bytes() == b"preserve me"


@pytest.mark.parametrize("failure_point", ["write", "sync", "replace"])
def test_prepublication_failure_preserves_old_bytes_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    path = tmp_path / "question_file.json"
    old_bytes = b'{ "old": "snapshot" }\n'
    path.write_bytes(old_bytes)

    def fail(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError(f"injected {failure_point} failure")

    if failure_point == "write":

        def fail_after_partial_write(value: Any, stream: Any, *args: Any, **kwargs: Any) -> None:
            del value, args, kwargs
            stream.write('{"partial":')
            raise OSError(f"injected {failure_point} failure")

        monkeypatch.setattr(question_bank.json, "dump", fail_after_partial_write)
    elif failure_point == "sync":
        monkeypatch.setattr(question_bank.os, "fsync", fail)
    else:
        monkeypatch.setattr(question_bank.os, "replace", fail)

    with pytest.raises(OSError, match=f"injected {failure_point} failure"):
        question_bank.save_question_bank(path, {"new": "snapshot"})

    assert path.read_bytes() == old_bytes
    assert {item.name for item in tmp_path.iterdir()} == {path.name}


def _join_worker(thread: Thread) -> None:
    thread.join(timeout=5)
    assert not thread.is_alive(), "coordinated persistence worker did not finish"


def test_reader_overlaps_publication_and_observes_a_complete_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "question_file.json"
    old = {"version": "old", "complete": "snapshot"}
    new = {"version": "new", "complete": "snapshot"}
    question_bank.save_question_bank(path, old)

    real_replace = question_bank.os.replace
    real_lock = question_bank._bank_lock
    replace_entered = Event()
    allow_replace = Event()
    reader_started = Event()
    reader_blocked_on_writer = Event()
    read_results: list[dict[str, str]] = []
    worker_errors: list[Exception] = []

    class ObservedBankLock:
        def __enter__(self) -> ObservedBankLock:
            if current_thread().name == "question-bank-reader":
                if not real_lock.acquire(blocking=False):
                    reader_blocked_on_writer.set()
                    real_lock.acquire()
                return self
            real_lock.acquire()
            return self

        def __exit__(self, *args: Any) -> None:
            del args
            real_lock.release()

    def gated_replace(source: Path, destination: Path) -> None:
        replace_entered.set()
        if not allow_replace.wait(timeout=5):
            raise TimeoutError("test did not release the staged replacement")
        real_replace(source, destination)

    def write_new_snapshot() -> None:
        try:
            question_bank.save_question_bank(path, new)
        except Exception as error:
            worker_errors.append(error)

    def read_snapshot() -> None:
        reader_started.set()
        try:
            read_results.append(question_bank.load_question_bank(path))
        except Exception as error:
            worker_errors.append(error)

    monkeypatch.setattr(question_bank.os, "replace", gated_replace)
    monkeypatch.setattr(question_bank, "_bank_lock", ObservedBankLock())
    writer = Thread(target=write_new_snapshot, daemon=True)
    reader = Thread(target=read_snapshot, name="question-bank-reader", daemon=True)
    writer.start()
    try:
        assert replace_entered.wait(timeout=5), "writer never reached atomic publication"
        reader.start()
        assert reader_started.wait(timeout=5), "reader did not start during publication"
        assert reader_blocked_on_writer.wait(timeout=5), "reader did not contend with active save"
        assert json.loads(path.read_text(encoding="utf-8")) == old
    finally:
        allow_replace.set()
        _join_worker(writer)
        if reader.ident is not None:
            _join_worker(reader)

    assert not worker_errors
    assert read_results == [new]
    assert question_bank.load_question_bank(path) == new


def test_failed_save_cannot_invalidate_a_concurrent_successful_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "question_file.json"
    old = {"version": "old"}
    broken = {"version": "broken"}
    successful = {"version": "successful", "complete": "mapping"}
    question_bank.save_question_bank(path, old)

    real_dump = question_bank.json.dump
    failing_dump_entered = Event()
    allow_failure = Event()
    successful_call_started = Event()
    failure_errors: list[Exception] = []
    success_errors: list[Exception] = []

    def coordinated_dump(value: object, stream: Any, *args: Any, **kwargs: Any) -> None:
        if value == broken:
            failing_dump_entered.set()
            if not allow_failure.wait(timeout=5):
                raise TimeoutError("test did not release the failing writer")
            raise OSError("injected concurrent write failure")
        real_dump(value, stream, *args, **kwargs)

    def save(entries: dict[str, str], errors: list[Exception]) -> None:
        try:
            question_bank.save_question_bank(path, entries)
        except Exception as error:
            errors.append(error)

    def save_successful() -> None:
        successful_call_started.set()
        save(successful, success_errors)

    monkeypatch.setattr(question_bank.json, "dump", coordinated_dump)
    failing_writer = Thread(target=save, args=(broken, failure_errors), daemon=True)
    successful_writer = Thread(target=save_successful, daemon=True)
    failing_writer.start()
    try:
        assert failing_dump_entered.wait(timeout=5), "failing writer did not reach serialization"
        successful_writer.start()
        assert successful_call_started.wait(timeout=5), "successful save did not overlap failure"
    finally:
        allow_failure.set()
        _join_worker(failing_writer)
        if successful_writer.ident is not None:
            _join_worker(successful_writer)

    assert len(failure_errors) == 1 and isinstance(failure_errors[0], OSError)
    assert not success_errors
    assert question_bank.load_question_bank(path) == successful
    assert {item.name for item in tmp_path.iterdir()} == {path.name}
