"""Production-path responsiveness and cancellation contracts for the legacy desktop loop."""

from __future__ import annotations

import asyncio
import copy
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from deskbuddy.legacy_worker import LegacyWorker
from deskbuddy.settings import Settings
from tests.support.fakes import CallRecorder, FakeNLIClassifier, FakeOpenAISDK, nli_routes
from tests.support.legacy_runtime import load_botcortex_runtime
from tests.test_conversation_robot_integration import conversation_server


class ThreadGate:
    """Block a synchronous fake with thread events and report entry to asyncio."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.thread_ids: list[int] = []

    def call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.thread_ids.append(threading.get_ident())
        self.loop.call_soon_threadsafe(self.entered.set)
        if not self.release.wait(5):
            raise AssertionError("the test did not release the blocked dependency")
        return function(*args, **kwargs)

    def wrap(self, function: Callable[..., Any]) -> Callable[..., Any]:
        def gated(*args: Any, **kwargs: Any) -> Any:
            return self.call(function, *args, **kwargs)

        return gated


def _make_runtime(
    *,
    local_only_mode: bool = True,
    is_question: bool = False,
    settings: Settings | None = None,
    qdict_data: dict[str, str] | None = None,
    embedding_scores: list[float] | None = None,
    openai_sdk: FakeOpenAISDK | None = None,
) -> tuple[dict[str, Any], CallRecorder, LegacyWorker]:
    recorder = CallRecorder()
    worker = LegacyWorker("desktop-contract")
    cortex, _ = load_botcortex_runtime(
        settings=settings,
        recorder=recorder,
        local_only_mode=local_only_mode,
        is_question=is_question,
        nli_classifier=FakeNLIClassifier(recorder, routes=nli_routes()),
        qdict_data=qdict_data,
        embedding_scores=embedding_scores,
        openai_sdk=openai_sdk,
        legacy_worker=worker,
    )
    cortex["start_workers"]()
    return cortex, recorder, worker


async def _assert_loop_progresses_while_blocked(
    task: asyncio.Task[Any], gate: ThreadGate, worker: LegacyWorker
) -> None:
    # These timeouts are deadlock guards; responsiveness is proven by event order.
    await asyncio.wait_for(gate.entered.wait(), timeout=5)
    assert not task.done()
    assert worker.busy
    assert gate.thread_ids
    assert all(thread_id != gate.loop_thread for thread_id in gate.thread_ids)

    heartbeat = asyncio.Event()
    gate.loop.call_soon(heartbeat.set)
    await asyncio.wait_for(heartbeat.wait(), timeout=5)
    assert not task.done()


def _current_job(worker: LegacyWorker) -> Any:
    job = worker._active
    assert job is not None
    return job


async def _wait_for_job_completion(job: Any) -> None:
    # Unlike the awaiter's ready event, the concurrent future completes only when
    # the synchronous callable has actually returned from its worker thread.
    await asyncio.wait_for(asyncio.wrap_future(job.result), timeout=5)


def _turn_state(cortex: dict[str, Any]) -> dict[str, Any]:
    names = (
        "history",
        "history_nonlocal",
        "named_entities",
        "after_first",
        "after_second",
        "asked_follow_up",
        "unfound_answer",
        "count_index",
    )
    return {name: copy.deepcopy(cortex[name]) for name in names}


async def _classification_call(cortex: dict[str, Any]) -> Any:
    history = cortex["history_nonlocal"] if not cortex["local_only_mode"] else cortex["history"]
    return await cortex["classify_input"]("Let's chat about games.", history)


@pytest.mark.parametrize("blocked_step", ["tokenize", "generate", "decode"])
async def test_local_generation_is_responsive_cancel_safe_and_reusable(
    monkeypatch: pytest.MonkeyPatch, blocked_step: str
) -> None:
    cortex, recorder, worker = _make_runtime()
    gate = ThreadGate()
    target, method = {
        "tokenize": (type(cortex["tokenizer"]), "__call__"),
        "generate": (cortex["model"], "generate"),
        "decode": (cortex["tokenizer"], "decode"),
    }[blocked_step]
    original = getattr(target, method)
    if isinstance(target, dict):
        monkeypatch.setitem(target, method, gate.wrap(original))
    else:
        monkeypatch.setattr(target, method, gate.wrap(original))
    before = _turn_state(cortex)
    first_turn = asyncio.create_task(cortex["talk_history"]("First blocked turn"))
    job = None

    try:
        await _assert_loop_progresses_while_blocked(first_turn, gate, worker)
        job = _current_job(worker)
        first_turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_turn

        assert _turn_state(cortex) == before
        assert recorder.count("save_history_summary") == 0
        assert recorder.count("save_history_verbatim") == 0

        gate.release.set()
        await _wait_for_job_completion(job)
        assert _turn_state(cortex) == before
        assert not worker.busy

        reply = await cortex["talk_history"]("Second turn after release")
        assert reply == "LOCAL_MODEL_REPLY"
        assert cortex["history"] == before["history"] + ["Second turn after release"]
        assert cortex["count_index"] == before["count_index"]
    finally:
        gate.release.set()
        cortex["shutdown_workers"]()


@pytest.mark.parametrize(
    "stage",
    [
        "cloud-classification",
        "cloud-reply",
        "nli-classification",
        "clarification",
        "memory-search",
        "summary",
    ],
)
async def test_blocking_production_stages_leave_the_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    local_only = stage not in {"cloud-classification", "cloud-reply"}
    qdict = {"Known question?": "Known answer", "Other question?": "Other answer"}
    cortex, _recorder, worker = _make_runtime(
        local_only_mode=local_only,
        qdict_data=qdict if stage == "memory-search" else None,
        embedding_scores=[0.96, 0.2],
    )
    gate = ThreadGate()

    if stage == "cloud-classification":
        target, method = cortex["openai"].Completion, "create"
        operation = _classification_call(cortex)
    elif stage == "cloud-reply":
        target, method = cortex["openai"].ChatCompletion, "create"
        operation = cortex["talk_medium_API_with_history"]("Hello from cloud mode")
    elif stage == "nli-classification":
        target, method = type(cortex["nli_classifier"]), "__call__"
        operation = _classification_call(cortex)
    elif stage == "clarification":
        target, method = cortex["qgen_tokenizer"], "encode"
        operation = cortex["get_follow_up_question"]("I am not sure yet", cortex["history"])
    elif stage == "memory-search":
        target = cortex
        method = "_get_embeddings_sync"
        operation = cortex["query_memory"]("Known question?")
    else:
        target, method = cortex["longsummarizer"], "generate"
        operation = cortex["generate_summary"]("A short summary input")

    if isinstance(target, dict):
        original = target[method]
        monkeypatch.setitem(target, method, gate.wrap(original))
    else:
        original = getattr(target, method)
        monkeypatch.setattr(target, method, gate.wrap(original))
    task = asyncio.create_task(operation)

    try:
        await _assert_loop_progresses_while_blocked(task, gate, worker)
        gate.release.set()
        result = await asyncio.wait_for(task, timeout=5)
        if stage in {"cloud-classification", "nli-classification"}:
            assert result == ("Let's chat about games.", False)
        elif stage == "cloud-reply":
            assert result == "OPENAI_CHAT_FAKE"
        elif stage == "clarification":
            assert result == "Could you clarify what you mean?"
        elif stage == "memory-search":
            assert result == ("Known answer", True)
        else:
            assert result == "SUMMARY:5"
        assert not worker.busy
    finally:
        gate.release.set()
        cortex["shutdown_workers"]()


async def test_cancelled_cloud_turn_is_not_committed_or_spoken_and_loopback_recovers(
    monkeypatch: pytest.MonkeyPatch, ephemeral_server_config: Any
) -> None:
    gate = ThreadGate()
    first_worker = LegacyWorker("desktop-cancelled-cloud")
    captured: list[dict[str, Any]] = []
    real_loader = load_botcortex_runtime

    def capture_runtime(**kwargs: Any) -> tuple[dict[str, Any], CallRecorder]:
        cortex, recorder = real_loader(**kwargs)
        captured.append(cortex)
        return cortex, recorder

    monkeypatch.setattr(
        "tests.test_conversation_robot_integration.load_botcortex_runtime",
        capture_runtime,
    )
    blocked_sdk = FakeOpenAISDK(chat_reply=lambda _kwargs: gate.call(lambda: "LATE_CLOUD_REPLY"))

    async with conversation_server(
        monkeypatch, ephemeral_server_config, legacy_server=False
    ) as flow:
        first_task = asyncio.create_task(
            flow.run(
                ["First cloud turn"],
                local_only=False,
                is_question=False,
                openai_sdk=blocked_sdk,
                legacy_worker=first_worker,
            )
        )
        first_cortex: dict[str, Any] | None = None
        job = None

        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=5)
            first_cortex = captured[0]
            job = _current_job(first_worker)
            before = _turn_state(first_cortex)

            heartbeat = asyncio.Event()
            gate.loop.call_soon(heartbeat.set)
            await asyncio.wait_for(heartbeat.wait(), timeout=5)
            assert not first_task.done()
            assert first_worker.busy
            assert all(thread_id != gate.loop_thread for thread_id in gate.thread_ids)

            first_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_task
            assert flow.received_replies == []
            assert flow.spoken == []
            assert _turn_state(first_cortex) == before

            gate.release.set()
            await _wait_for_job_completion(job)
            assert _turn_state(first_cortex) == before
            assert flow.received_replies == []
            assert flow.spoken == []

            second_worker = LegacyWorker("desktop-successful-cloud")
            await flow.run(
                ["Second cloud turn"],
                local_only=False,
                is_question=False,
                openai_sdk=FakeOpenAISDK(chat_reply="Fresh reply after cancellation"),
                legacy_worker=second_worker,
            )
            assert flow.received_replies == ["Fresh reply after cancellation"]
            assert flow.spoken == ["Fresh reply after cancellation"]
        finally:
            gate.release.set()
            if first_cortex is not None:
                first_cortex["shutdown_workers"]()


def _checkpoint_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
) -> tuple[dict[str, Any], CallRecorder, LegacyWorker]:
    cortex, recorder, worker = _make_runtime(settings=settings)
    cortex["after_first"] = True
    cortex["after_second"] = False
    cortex["count_index"] = 10

    async def fixed_context(_text: str) -> list[Any]:
        return ["checkpoint summary", ["Alex"], ["topic"]]

    monkeypatch.setitem(cortex, "get_list_of_list_of_strings", fixed_context)
    return cortex, recorder, worker


async def test_cancelled_history_export_does_not_publish_or_commit_turn_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path, cache_dir=tmp_path / "cache")
    summary_path = tmp_path / "history_summary.txt"
    verbatim_path = tmp_path / "history_verbatim.txt"
    summary_path.write_text("existing-summary|", encoding="utf-8")
    verbatim_path.write_text("existing-history", encoding="utf-8")
    cortex, _recorder, worker = _checkpoint_runtime(monkeypatch, settings=settings)
    gate = ThreadGate()
    original = cortex["_save_history_export_sync"]
    monkeypatch.setitem(cortex, "_save_history_export_sync", gate.wrap(original))
    before = _turn_state(cortex)
    checkpoint_consumed = {**before, "count_index": 0}
    turn = asyncio.create_task(cortex["talk_history"]("Checkpoint turn"))
    job = None

    try:
        await _assert_loop_progresses_while_blocked(turn, gate, worker)
        job = _current_job(worker)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        assert _turn_state(cortex) == checkpoint_consumed
        assert summary_path.read_text(encoding="utf-8") == "existing-summary|"
        assert verbatim_path.read_text(encoding="utf-8") == "existing-history"

        gate.release.set()
        await _wait_for_job_completion(job)
        assert summary_path.read_text(encoding="utf-8") == "existing-summary|"
        assert verbatim_path.read_text(encoding="utf-8") == "existing-history"
        assert _turn_state(cortex) == checkpoint_consumed
        assert not worker.busy
    finally:
        gate.release.set()
        cortex["shutdown_workers"]()


async def test_successful_history_checkpoint_publishes_summary_then_committed_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path, cache_dir=tmp_path / "cache")
    cortex, _recorder, worker = _checkpoint_runtime(monkeypatch, settings=settings)
    write_order: list[str] = []
    original_replace = cortex["os"].replace

    def record_export_replace(source: Any, destination: Any) -> None:
        destination_path = Path(destination)
        if destination_path.name in {"history_summary.txt", "history_verbatim.txt"}:
            write_order.append(destination_path.name)
        original_replace(source, destination)

    monkeypatch.setattr(cortex["os"], "replace", record_export_replace)

    try:
        reply = await cortex["talk_history"]("Published checkpoint turn")
        assert isinstance(reply, str) and reply
        assert write_order == ["history_summary.txt", "history_verbatim.txt"]
        assert (tmp_path / "history_summary.txt").read_text(
            encoding="utf-8"
        ) == "checkpoint summary"
        assert (tmp_path / "history_verbatim.txt").read_text(encoding="utf-8") == "\n".join(
            cortex["history"]
        )
        assert "Published checkpoint turn" in cortex["history"][-1]
        assert cortex["count_index"] == 0
    finally:
        cortex["shutdown_workers"]()
