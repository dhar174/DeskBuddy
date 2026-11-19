"""Offline contracts for the retained robot audio adapters."""

from __future__ import annotations

import asyncio
import threading
import time
import wave
from array import array
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import buddy1
from deskbuddy import legacy_audio
from deskbuddy.legacy_worker import LegacyWorker
from transport import RobotClient, TransportConfig
from transport.transcripts import TranscriptQueue
from windows import buddy1 as windows_buddy1
from windows import buddy2 as windows_buddy2

ADAPTERS: tuple[tuple[ModuleType, int], ...] = (
    (buddy1, 30),
    (windows_buddy1, 30),
    (windows_buddy2, 80),
)


class FakeKeyboard:
    def __init__(self, pressed: bool) -> None:
        self.pressed = pressed

    def is_pressed(self, key: str) -> bool:
        assert key == "space"
        return self.pressed


class FakeStream:
    def __init__(self, backend: FakeAudioBackend) -> None:
        self.backend = backend
        self.read_count = 0
        self.stop_count = 0
        self.close_count = 0

    def read(self, frame_count: int) -> bytes:
        self.backend.read_threads.append(threading.get_ident())
        self.read_count += 1
        value = 1000 if self.read_count == 1 else 0
        pcm = array("h", [value]) * frame_count
        if (
            self.backend.cancel_stop is not None
            and self.backend.cancel_after_read == self.read_count
        ):
            self.backend.cancel_stop.set()
        return pcm.tobytes()

    def stop_stream(self) -> None:
        self.stop_count += 1

    def close(self) -> None:
        self.close_count += 1


class FakeAudioBackend:
    def __init__(
        self,
        *,
        cancel_stop: threading.Event | None = None,
        cancel_after_read: int | None = None,
    ) -> None:
        self.cancel_stop = cancel_stop
        self.cancel_after_read = cancel_after_read
        self.constructor_threads: list[int] = []
        self.open_threads: list[int] = []
        self.read_threads: list[int] = []
        self.open_arguments: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.terminate_count = 0

    def as_module(self) -> SimpleNamespace:
        backend = self

        class FakePyAudio:
            def __init__(self) -> None:
                backend.constructor_threads.append(threading.get_ident())

            def open(self, **kwargs: Any) -> FakeStream:
                backend.open_threads.append(threading.get_ident())
                backend.open_arguments.append(kwargs)
                stream = FakeStream(backend)
                backend.streams.append(stream)
                return stream

            def get_sample_size(self, audio_format: int) -> int:
                return 2

            def terminate(self) -> None:
                backend.terminate_count += 1

        return SimpleNamespace(paInt16=8, PyAudio=FakePyAudio)


class FakeWhisper:
    def __init__(
        self,
        texts: tuple[str, ...] = ("fresh transcript",),
        *,
        releases: tuple[threading.Event | None, ...] = (),
    ) -> None:
        self.texts = texts
        self.releases = releases
        self.started = [threading.Event() for _ in range(max(2, len(texts)))]
        self.paths: list[Path] = []
        self.threads: list[int] = []
        self.wav_parameters: list[tuple[int, int, int]] = []

    def transcribe(self, path: str) -> dict[str, str]:
        index = len(self.paths)
        wav_path = Path(path)
        assert wav_path.is_file()
        with wave.open(str(wav_path), "rb") as recording:
            self.wav_parameters.append(
                (
                    recording.getnchannels(),
                    recording.getframerate(),
                    recording.getsampwidth(),
                )
            )
            assert recording.getnframes() > 0
        self.paths.append(wav_path)
        self.threads.append(threading.get_ident())
        self.started[min(index, len(self.started) - 1)].set()
        if index < len(self.releases) and self.releases[index] is not None:
            if not self.releases[index].wait(5):
                raise TimeoutError("controlled fake Whisper call was not released")
        return {"text": self.texts[min(index, len(self.texts) - 1)]}


def _configure_adapter(
    adapter: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    backend: FakeAudioBackend,
    model: FakeWhisper,
    *,
    pressed: bool = True,
) -> TranscriptQueue:
    previous_worker = getattr(adapter, "_shared_audio_worker", None)
    if previous_worker is not None:
        previous_worker.shutdown()
        assert previous_worker.can_replace
    monkeypatch.setattr(adapter, "_shared_audio_worker", None)
    monkeypatch.setattr(adapter, "_shared_audio_users", 0)
    monkeypatch.setattr(adapter, "_shared_audio_worker_owned", False)
    monkeypatch.setattr(adapter, "pyaudio", backend.as_module())
    monkeypatch.setattr(adapter, "model", model, raising=False)
    monkeypatch.setattr(adapter, "keyboard", FakeKeyboard(pressed))
    monkeypatch.setattr(adapter, "FORMAT", 8)
    monkeypatch.setattr(adapter, "RATE", 44_100)
    monkeypatch.setattr(adapter, "CHUNK_SIZE", 1024)
    monkeypatch.setattr(adapter, "THRESHOLD", 500)
    monkeypatch.setattr(adapter, "listening", True)
    monkeypatch.setattr(adapter, "recording", False, raising=False)
    monkeypatch.setattr(adapter, "recorded", False, raising=False)
    monkeypatch.setattr(adapter, "result", None, raising=False)
    monkeypatch.setattr(adapter, "lastTranscription", "")
    monkeypatch.setattr(adapter, "priorTranscription", "")
    if hasattr(adapter, "can_listen"):
        monkeypatch.setattr(adapter, "can_listen", True)
    queue = TranscriptQueue(max_payload_size=65_536, secret="audio-test-secret")
    monkeypatch.setattr(adapter, "transcript_queue", queue)

    async def no_op(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(adapter.MyFunctions, "baseCol", no_op)
    monkeypatch.setattr(adapter.MyFunctions, "picoh_wait", no_op)
    return queue


def _shutdown_external_worker(adapter: ModuleType, worker: LegacyWorker) -> None:
    adapter._shutdown_runtime_audio_worker(worker)


async def _wait_for_thread_event(event: threading.Event, timeout: float = 2.0) -> None:
    reached = await asyncio.wait_for(asyncio.to_thread(event.wait, timeout), timeout + 1)
    assert reached


@pytest.mark.parametrize(("adapter", "silence_chunks"), ADAPTERS)
@pytest.mark.asyncio
async def test_listener_runs_full_audio_and_whisper_turn_off_loop(
    adapter: ModuleType,
    silence_chunks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeAudioBackend()
    model = FakeWhisper(("hello Cooper",))
    queue = _configure_adapter(adapter, monkeypatch, backend, model)
    stop = asyncio.Event()
    capture_stop = threading.Event()
    worker = LegacyWorker(f"test-{adapter.__name__}-audio")
    loop_thread = threading.get_ident()
    original_send_response = adapter.MyFunctions.send_response

    async def publish_and_stop(text: str) -> None:
        await original_send_response(text)
        stop.set()

    monkeypatch.setattr(adapter.MyFunctions, "send_response", publish_and_stop)
    listener = asyncio.create_task(adapter.MyFunctions.listen(stop, worker, capture_stop))
    try:
        await _wait_for_thread_event(model.started[0])
        loop_heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(loop_heartbeat.set)
        await asyncio.wait_for(loop_heartbeat.wait(), 0.25)
        await asyncio.wait_for(listener, 2)

        assert adapter.lastTranscription == "hello Cooper"
        queued = queue.poll()
        assert queued is not None and queued["text"] == "hello Cooper"
        assert len(model.paths) == 1
        assert not model.paths[0].exists()
        assert model.wav_parameters == [(1, 44_100, 2)]
        assert len(backend.streams) == 1
        stream = backend.streams[0]
        assert stream.read_count == silence_chunks + 2
        assert stream.stop_count == stream.close_count == 1
        assert backend.terminate_count == 1
        assert backend.open_arguments == [
            {
                "format": 8,
                "channels": 1,
                "rate": 44_100,
                "input": True,
                "output": True,
                "frames_per_buffer": 1024,
            }
        ]
        worker_thread = backend.constructor_threads[0]
        assert worker_thread != loop_thread
        assert backend.open_threads == [worker_thread]
        assert backend.read_threads == [worker_thread] * stream.read_count
        assert model.threads == [worker_thread]
    finally:
        stop.set()
        capture_stop.set()
        if not listener.done():
            listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)
        _shutdown_external_worker(adapter, worker)


@pytest.mark.parametrize(("adapter", "silence_chunks"), ADAPTERS)
@pytest.mark.asyncio
async def test_listener_idle_yields_and_stops_promptly(
    adapter: ModuleType,
    silence_chunks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del silence_chunks
    backend = FakeAudioBackend()
    model = FakeWhisper()
    _configure_adapter(adapter, monkeypatch, backend, model, pressed=False)
    stop = asyncio.Event()
    worker = LegacyWorker(f"test-{adapter.__name__}-idle")
    listener = asyncio.create_task(adapter.MyFunctions.listen(stop, worker))
    try:
        await asyncio.sleep(0.02)
        started = time.perf_counter()
        stop.set()
        await asyncio.wait_for(listener, 0.3)
        assert time.perf_counter() - started < 0.25
        assert backend.constructor_threads == []
        assert model.paths == []
    finally:
        stop.set()
        if not listener.done():
            listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)
        _shutdown_external_worker(adapter, worker)


@pytest.mark.parametrize(("adapter", "silence_chunks"), ADAPTERS)
@pytest.mark.asyncio
async def test_capture_stop_between_reads_and_before_whisper_cleans_stream(
    adapter: ModuleType,
    silence_chunks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del silence_chunks
    for stop_stage in ("between_reads", "before_whisper"):
        capture_stop = threading.Event()
        backend = FakeAudioBackend(
            cancel_stop=capture_stop,
            cancel_after_read=1 if stop_stage == "between_reads" else None,
        )
        model = FakeWhisper()
        _configure_adapter(adapter, monkeypatch, backend, model)
        stop = asyncio.Event()
        worker = LegacyWorker(f"test-{adapter.__name__}-{stop_stage}")
        temporary_paths: list[Path] = []
        original_write_wav = legacy_audio.write_wav

        if stop_stage == "before_whisper":

            def stop_after_wav(*args: Any, **kwargs: Any) -> None:
                original_write_wav(*args, **kwargs)
                temporary_paths.append(Path(args[0]))
                capture_stop.set()

            monkeypatch.setattr(legacy_audio, "write_wav", stop_after_wav)
        listener = asyncio.create_task(adapter.MyFunctions.listen(stop, worker, capture_stop))
        try:
            await asyncio.wait_for(listener, 2)
            assert backend.terminate_count == 1
            assert len(backend.streams) == 1
            assert backend.streams[0].stop_count == 1
            assert backend.streams[0].close_count == 1
            assert model.paths == []
            assert adapter.lastTranscription == ""
            if stop_stage == "between_reads":
                assert backend.streams[0].read_count == 1
            else:
                assert len(temporary_paths) == 1
                assert not temporary_paths[0].exists()
        finally:
            stop.set()
            capture_stop.set()
            if not listener.done():
                listener.cancel()
                await asyncio.gather(listener, return_exceptions=True)
            _shutdown_external_worker(adapter, worker)
            monkeypatch.setattr(legacy_audio, "write_wav", original_write_wav)


@pytest.mark.parametrize(("adapter", "silence_chunks"), ADAPTERS)
@pytest.mark.asyncio
async def test_completed_transcript_is_not_published_after_runtime_stop(
    adapter: ModuleType,
    silence_chunks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del silence_chunks
    backend = FakeAudioBackend()
    model = FakeWhisper(("too late",))
    queue = _configure_adapter(adapter, monkeypatch, backend, model)
    stop = asyncio.Event()
    worker = LegacyWorker(f"test-{adapter.__name__}-publish-guard")

    async def stop_before_publish(delay: float) -> None:
        stop.set()

    monkeypatch.setattr(adapter.MyFunctions, "picoh_wait", stop_before_publish)
    listener = asyncio.create_task(adapter.MyFunctions.listen(stop, worker))
    try:
        await asyncio.wait_for(listener, 2)
        assert len(model.paths) == 1
        assert not model.paths[0].exists()
        assert adapter.lastTranscription == ""
        assert queue.poll() is None
    finally:
        stop.set()
        if not listener.done():
            listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)
        _shutdown_external_worker(adapter, worker)


@pytest.mark.parametrize(("adapter", "silence_chunks"), ADAPTERS)
@pytest.mark.asyncio
async def test_task_cancellation_at_post_capture_gate_clears_loop_state(
    adapter: ModuleType,
    silence_chunks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del silence_chunks
    backend = FakeAudioBackend()
    model = FakeWhisper(("must not publish",))
    queue = _configure_adapter(adapter, monkeypatch, backend, model)
    at_post_capture_gate = asyncio.Event()
    release_post_capture_gate = asyncio.Event()
    worker = LegacyWorker(f"test-{adapter.__name__}-cancel-at-publish")

    async def hold_before_publish(delay: float) -> None:
        at_post_capture_gate.set()
        await release_post_capture_gate.wait()

    monkeypatch.setattr(adapter.MyFunctions, "picoh_wait", hold_before_publish)
    listener = asyncio.create_task(adapter.MyFunctions.listen(asyncio.Event(), worker))
    try:
        await asyncio.wait_for(at_post_capture_gate.wait(), 2)
        assert adapter.recorded is True
        assert adapter.result == {"text": "must not publish"}
        listener.cancel()
        completion = await asyncio.gather(listener, return_exceptions=True)
        assert isinstance(completion[0], asyncio.CancelledError)
        assert adapter.recording is False
        assert adapter.recorded is False
        assert adapter.result is None
        assert adapter.lastTranscription == ""
        assert queue.poll() is None
    finally:
        release_post_capture_gate.set()
        if not listener.done():
            listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)
        _shutdown_external_worker(adapter, worker)


@pytest.mark.parametrize(("adapter", "silence_chunks"), ADAPTERS)
@pytest.mark.asyncio
async def test_late_whisper_result_is_abandoned_then_next_turn_succeeds(
    adapter: ModuleType,
    silence_chunks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del silence_chunks
    release_late = threading.Event()
    backend = FakeAudioBackend()
    model = FakeWhisper(("late result", "next turn"), releases=(release_late, None))
    queue = _configure_adapter(adapter, monkeypatch, backend, model)
    worker = LegacyWorker(f"test-{adapter.__name__}-late")
    first_stop = asyncio.Event()
    first_capture_stop = threading.Event()
    first_listener = asyncio.create_task(
        adapter.MyFunctions.listen(first_stop, worker, first_capture_stop)
    )
    await _wait_for_thread_event(model.started[0])
    first_stop.set()
    await asyncio.wait_for(first_listener, 0.5)
    assert worker.busy
    assert worker.abandoned
    assert adapter.lastTranscription == ""
    assert queue.poll() is None

    second_stop = asyncio.Event()
    second_capture_stop = threading.Event()
    monkeypatch.setattr(adapter, "listening", True)
    original_send_response = adapter.MyFunctions.send_response

    async def publish_and_stop(text: str) -> None:
        await original_send_response(text)
        second_stop.set()

    monkeypatch.setattr(adapter.MyFunctions, "send_response", publish_and_stop)
    second_listener = asyncio.create_task(
        adapter.MyFunctions.listen(second_stop, worker, second_capture_stop)
    )
    try:
        await asyncio.sleep(0.03)
        assert len(backend.constructor_threads) == 1
        release_late.set()
        await _wait_for_thread_event(model.started[1])
        await asyncio.wait_for(second_listener, 2)
        assert adapter.lastTranscription == "next turn"
        queued = queue.poll()
        assert queued is not None and queued["text"] == "next turn"
        assert len(model.paths) == 2
        assert all(not path.exists() for path in model.paths)
        assert len(backend.streams) == 2
    finally:
        release_late.set()
        first_stop.set()
        second_stop.set()
        first_capture_stop.set()
        second_capture_stop.set()
        if not second_listener.done():
            second_listener.cancel()
            await asyncio.gather(second_listener, return_exceptions=True)
        _shutdown_external_worker(adapter, worker)


@pytest.mark.parametrize(("adapter", "silence_chunks"), ADAPTERS)
@pytest.mark.asyncio
async def test_adapter_main_owns_authenticated_server_and_stops_cleanly(
    adapter: ModuleType,
    silence_chunks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del silence_chunks
    monkeypatch.setattr(adapter, "keyboard", FakeKeyboard(False))
    if adapter is buddy1:
        monkeypatch.setattr(adapter, "init_hardware", lambda: None)
    if adapter is windows_buddy2:

        async def no_hardware_operation(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(adapter, "run_hardware_op", no_hardware_operation)
        monkeypatch.setattr(adapter, "firstLoop", False, raising=False)

    stop = asyncio.Event()

    async def wait_until_stopped(*args: Any, **kwargs: Any) -> None:
        await stop.wait()

    for motion_task in ("blinkLids", "randomLook", "randomNod", "randomTurn"):
        monkeypatch.setattr(adapter.MyFunctions, motion_task, wait_until_stopped)

    config = TransportConfig(
        host="127.0.0.1",
        port=0,
        secret="authenticated-audio-loopback-secret",
    )
    server = adapter.create_robot_server(config)
    server_started = asyncio.Event()
    original_start = server.start

    async def start_and_signal() -> None:
        await original_start()
        server_started.set()

    monkeypatch.setattr(server, "start", start_and_signal)
    main_task = asyncio.create_task(adapter.main(config, server, stop))
    try:
        await asyncio.wait_for(server_started.wait(), 2)
        client_config = config.with_overrides(port=server.port)
        async with RobotClient(client_config) as client:
            response = await client.request("ping", {})
        assert response.ok and response.result == "Hello"
        stop.set()
        await asyncio.wait_for(main_task, 2)
        assert not server._is_running
        assert server._server is None
        assert adapter._shared_audio_worker is None
    finally:
        stop.set()
        if not main_task.done():
            await asyncio.wait_for(main_task, 2)
        await server.stop()


@pytest.mark.parametrize(("adapter", "silence_chunks"), ADAPTERS)
def test_sigint_handler_requests_runtime_shutdown(
    adapter: ModuleType,
    silence_chunks: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del silence_chunks
    stop = asyncio.Event()
    monkeypatch.setattr(adapter, "_runtime_stop_event", stop)
    adapter.signal_handler(2, None)
    assert stop.is_set()
