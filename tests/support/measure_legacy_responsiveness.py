"""CPU-only, repeatable responsiveness measurements for legacy DeskBuddy paths.

Run this file with a target checkout and an absolute JSON output path. The runner
loads production functions from the target checkout, injects the repository's
deterministic fakes, and isolates long-lived robot listeners in watchdog-bounded
child processes.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import wave
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

BASELINE_COMMIT = "fb5f2c9db5e36de2cbf5884fec7f539a30dd5681"
ROBOT_VARIANTS = {
    "buddy1": "buddy1",
    "windows.buddy1": "windows.buddy1",
    "windows.buddy2": "windows.buddy2",
}
BLOCKED_OPTIONAL_MODULES = (
    "picoh",
    "pyaudio",
    "whisper",
    "keyboard",
    "sounddevice",
    "cv2",
    "torch",
    "transformers",
    "PIL",
    "spacy",
    "sentence_transformers",
)
TEST_SECRET = "issue-37-benchmark-loopback-secret-32-chars"
RUNTIME_ROOTS = (
    "BotCortex.py",
    "helpers.py",
    "start_bot.py",
    "robot_client.py",
    "buddy1.py",
    "windows/",
    "src/",
    "transport/",
)
METRIC_NAMES = (
    "operation_ms",
    "turn_ms",
    "shutdown_ms",
    "ping_rtt_ms",
    "poll_rtt_ms",
    "ack_rtt_ms",
)


class QuietStream:
    """Discard verbose legacy prints without buffering benchmark output."""

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


def _run_async_quiet(coroutine: Awaitable[Any]) -> Any:
    """Suppress legacy prints around a single synchronous asyncio.run boundary."""
    with contextlib.redirect_stdout(QuietStream()):
        return asyncio.run(coroutine)


class LoopProbe:
    """5 ms fixed-deadline loop heartbeat with explicit missed-period counts."""

    def __init__(self, interval_ms: float) -> None:
        self.interval_s = interval_ms / 1000.0
        self.origin = 0.0
        self.samples: list[dict[str, float | int]] = []
        self.missed_probes = 0

    async def run(self, stop: asyncio.Event) -> None:
        self.origin = time.perf_counter()
        interval = self.interval_s
        due = self.origin + interval
        while not stop.is_set():
            await asyncio.sleep(max(0.0, due - time.perf_counter()))
            if stop.is_set():
                break
            observed = time.perf_counter()
            missed = max(0, int((observed - due) // interval))
            self.missed_probes += missed
            self.samples.append(
                {
                    "scheduled_ms": (due - self.origin) * 1000.0,
                    "observed_ms": (observed - self.origin) * 1000.0,
                    "lag_ms": max(0.0, observed - due) * 1000.0,
                    "missed_before_sample": missed,
                }
            )
            due += (missed + 1) * interval


class AuthenticatedLoopbackProber:
    """Keep authenticated ping/poll requests in flight beside a measured workload."""

    def __init__(self, client: Any, *, interval_ms: float = 20.0, timeout_s: float = 1.0) -> None:
        self.client = client
        self.interval_s = interval_ms / 1000.0
        self.timeout_s = timeout_s
        self.origin = 0.0
        self.stop_event = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.samples: list[dict[str, Any]] = []
        self.missed_probes = 0
        self.timeouts = 0
        self.errors: list[str] = []
        self.connection_warmup_ms = 0.0

    async def start(self) -> None:
        started = time.perf_counter()
        response = await self.client.request("ping", {})
        self.connection_warmup_ms = (time.perf_counter() - started) * 1000.0
        if not response.ok:
            raise RuntimeError(f"authenticated loopback warmup ping failed: {response.error}")
        self.origin = time.perf_counter()
        self.task = asyncio.create_task(self._run())
        await asyncio.sleep(0)

    async def _request(self, command: str, arguments: dict[str, Any] | None = None) -> Any:
        return await asyncio.wait_for(
            self.client.request(command, arguments or {}),
            timeout=self.timeout_s,
        )

    async def _run(self) -> None:
        due = self.origin
        while not self.stop_event.is_set():
            await asyncio.sleep(max(0.0, due - time.perf_counter()))
            if self.stop_event.is_set():
                break
            request_started = time.perf_counter()
            missed = max(0, int((request_started - due) // self.interval_s))
            self.missed_probes += missed
            sample: dict[str, Any] = {
                "scheduled_ms": (due - self.origin) * 1000.0,
                "request_started_ms": (request_started - self.origin) * 1000.0,
                "missed_before_probe": missed,
                "status": "ok",
            }
            try:
                ping_started = time.perf_counter()
                ping = await self._request("ping")
                ping_responded = time.perf_counter()
                if not ping.ok:
                    raise RuntimeError(f"ping failed: {ping.error}")
                poll_started = time.perf_counter()
                poll = await self._request("poll_transcript")
                poll_responded = time.perf_counter()
                if not poll.ok:
                    raise RuntimeError(f"poll failed: {poll.error}")
                ack_ms = None
                if isinstance(poll.result, dict) and isinstance(poll.result.get("id"), str):
                    ack_started = time.perf_counter()
                    ack = await self._request("ack_transcript", {"id": poll.result["id"]})
                    ack_responded = time.perf_counter()
                    if not ack.ok or ack.result is not True:
                        raise RuntimeError(f"ack failed: {ack.error}")
                    ack_ms = (ack_responded - ack_started) * 1000.0
                    completed = ack_responded
                else:
                    completed = poll_responded
                sample.update(
                    {
                        "response_ms": (completed - self.origin) * 1000.0,
                        "due_to_response_ms": max(0.0, completed - due) * 1000.0,
                        "ping_rtt_ms": (ping_responded - ping_started) * 1000.0,
                        "poll_rtt_ms": (poll_responded - poll_started) * 1000.0,
                        "ack_rtt_ms": ack_ms,
                    }
                )
            except TimeoutError:
                self.timeouts += 1
                sample.update(
                    {
                        "status": "timeout",
                        "response_ms": (time.perf_counter() - self.origin) * 1000.0,
                        "due_to_response_ms": max(0.0, time.perf_counter() - due) * 1000.0,
                    }
                )
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
                sample.update(
                    {
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                        "response_ms": (time.perf_counter() - self.origin) * 1000.0,
                        "due_to_response_ms": max(0.0, time.perf_counter() - due) * 1000.0,
                    }
                )
            self.samples.append(sample)
            due += (missed + 1) * self.interval_s

    async def stop(self) -> float:
        self.stop_event.set()
        if self.task is not None:
            await self.task
        started = time.perf_counter()
        await self.client.close()
        return (time.perf_counter() - started) * 1000.0

    def report(self) -> dict[str, Any]:
        def values(key: str) -> list[float]:
            return [float(item[key]) for item in self.samples if item.get(key) is not None]

        due_response = values("due_to_response_ms")
        return {
            "interval_ms": self.interval_s * 1000.0,
            "timeout_seconds": self.timeout_s,
            "sample_count": len(self.samples),
            "missed_probe_count": self.missed_probes,
            "timeout_count": self.timeouts,
            "error_count": len(self.errors),
            "errors": self.errors,
            "due_to_response_ms": _summarize(due_response),
            "ping_rtt_ms": _summarize(values("ping_rtt_ms")),
            "poll_rtt_ms": _summarize(values("poll_rtt_ms")),
            "ack_rtt_ms": _summarize(values("ack_rtt_ms")),
            "samples": self.samples,
            "connection_warmup_ms": self.connection_warmup_ms,
        }


async def _open_authenticated_probe(
    *,
    target: Path,
    variant: str,
    interval_ms: float = 20.0,
) -> tuple[Any, Any, AuthenticatedLoopbackProber, float]:
    module = importlib.import_module(ROBOT_VARIANTS[variant])
    from transport import RobotClient, TransportConfig

    config = TransportConfig(
        host="127.0.0.1",
        port=0,
        secret=TEST_SECRET,
        connect_timeout=2.0,
        read_timeout=2.0,
        write_timeout=2.0,
        request_timeout=3.0,
        max_retries=0,
    )
    server = module.create_robot_server(config)
    started = time.perf_counter()
    await server.start()
    startup_ms = (time.perf_counter() - started) * 1000.0
    client_config = config.with_overrides(port=server.port)
    client = RobotClient(client_config)
    prober = AuthenticatedLoopbackProber(client, interval_ms=interval_ms)
    await prober.start()
    return module, server, prober, startup_ms


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1)
    return ordered[rank]


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else None,
    }


def _metric_summary(samples: Mapping[str, list[float]]) -> dict[str, Any]:
    return {name: _summarize(samples.get(name, [])) for name in METRIC_NAMES}


async def _maybe_call(function: Callable[..., Any] | None) -> Any:
    if function is None:
        return None
    value = function()
    if inspect.isawaitable(value):
        return await value
    return value


async def _collect_samples(
    *,
    operation: Callable[[], Awaitable[Mapping[str, Any] | None]],
    warmups: int,
    repetitions: int,
    heartbeat_ms: float,
    settle_ms: float,
    shutdown: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    extra: dict[str, list[float]] = {}
    probe = LoopProbe(heartbeat_ms)
    stop_probe = asyncio.Event()
    probe_task = asyncio.create_task(probe.run(stop_probe))
    await asyncio.sleep(0)
    try:
        total = warmups + repetitions
        for index in range(total):
            started = time.perf_counter()
            outcome = await operation()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if index >= warmups:
                result = dict(outcome or {})
                operation_ms = float(result.get("operation_ms", elapsed_ms))
                samples["operation_ms"].append(operation_ms)
                turn_ms = result.get("turn_ms", operation_ms)
                if turn_ms is not None:
                    samples["turn_ms"].append(float(turn_ms))
                for key in ("ping_rtt_ms", "poll_rtt_ms", "ack_rtt_ms"):
                    if result.get(key) is not None:
                        samples[key].append(float(result[key]))
                for key, value in result.items():
                    if key.endswith("_ms") and key not in METRIC_NAMES and value is not None:
                        extra.setdefault(key, []).append(float(value))
                if result.get("shutdown_ms") is not None:
                    samples["shutdown_ms"].append(float(result["shutdown_ms"]))
            await asyncio.sleep(settle_ms / 1000.0)

        shutdown_ms = None
        if shutdown is not None:
            shutdown_started = time.perf_counter()
            await _maybe_call(shutdown)
            shutdown_ms = (time.perf_counter() - shutdown_started) * 1000.0
            samples["shutdown_ms"].append(shutdown_ms)
            await asyncio.sleep(settle_ms / 1000.0)
    finally:
        stop_probe.set()
        await probe_task

    lag_values = [float(sample["lag_ms"]) for sample in probe.samples]
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "timings_ms": _metric_summary(samples),
        "additional_timings_ms": {key: _summarize(values) for key, values in extra.items()},
        "loop_lag_ms": _summarize(lag_values),
        "lag_count_ge_20ms": sum(value >= 20.0 for value in lag_values),
        "lag_count_gt_50ms": sum(value > 50.0 for value in lag_values),
        "missed_probe_count": probe.missed_probes,
        "heartbeat_samples": probe.samples,
        "shutdown_ms_single_observation": shutdown_ms,
    }


def _configure_offline_environment(profile: Path) -> None:
    """Keep settings, model caches, credentials, and devices isolated to fakes."""
    for name in tuple(os.environ):
        if name.startswith(("DESKBUDDY_", "OPENAI_")) or name in {
            "API_KEY",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        }:
            os.environ.pop(name, None)
    data_dir = profile / "deskbuddy-data"
    cache_dir = profile / "deskbuddy-cache"
    hf_home = profile / "hf-home"
    for path in (data_dir, cache_dir, hf_home):
        path.mkdir(parents=True, exist_ok=True)
    config_file = profile / "deskbuddy.toml"
    config_file.write_text("[deskbuddy]\n", encoding="utf-8")
    os.environ["DESKBUDDY_CONFIG_FILE"] = str(config_file)
    os.environ["DESKBUDDY_DATA_DIR"] = str(data_dir)
    os.environ["DESKBUDDY_CACHE_DIR"] = str(cache_dir)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["DESKBUDDY_MOCK_HARDWARE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["UV_OFFLINE"] = "1"


def _install_target_imports(target: Path) -> None:
    """Make Python imports resolve to the requested checkout, including tests.*."""
    for name in BLOCKED_OPTIONAL_MODULES:
        sys.modules.pop(name, None)
        sys.modules[name] = None
    for path in (target, target / "src"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    for name in (
        "tests.support.legacy_runtime",
        "tests.support.fakes",
        "tests.support",
        "tests",
    ):
        sys.modules.pop(name, None)
    tests_package = types.ModuleType("tests")
    tests_package.__path__ = [str(target / "tests")]
    tests_package.__package__ = "tests"
    sys.modules["tests"] = tests_package
    support_package = types.ModuleType("tests.support")
    support_package.__path__ = [str(target / "tests" / "support")]
    support_package.__package__ = "tests.support"
    sys.modules["tests.support"] = support_package


def _git_text(target: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(target), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _runtime_metadata(target: Path, parameters: Mapping[str, Any]) -> dict[str, Any]:
    head = _git_text(target, "rev-parse", "HEAD")
    branch = _git_text(target, "branch", "--show-current")
    status = _git_text(target, "status", "--short", "--untracked-files=all") or ""
    dirty_paths = [line[3:].replace("\\", "/") for line in status.splitlines() if len(line) >= 4]
    runtime_dirty = [
        path
        for path in dirty_paths
        if any(path == root.rstrip("/") or path.startswith(root) for root in RUNTIME_ROOTS)
    ]
    measurement_dirty = [
        path
        for path in dirty_paths
        if path.startswith("tests/support/")
        and not path.endswith("measure_legacy_responsiveness.py")
    ]
    lock_path = target / "uv.lock"
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else None
    uv_error = None
    try:
        uv = subprocess.run(["uv", "--version"], check=False, capture_output=True, text=True)
    except OSError as error:
        uv = None
        uv_error = str(error)
    return {
        "target_checkout": str(target),
        "target_head": head,
        "target_branch": branch,
        "baseline_reference": BASELINE_COMMIT,
        "runtime_sources_dirty": bool(runtime_dirty),
        "runtime_source_dirty_paths": runtime_dirty,
        "measurement_support_dirty_paths": measurement_dirty,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_full": sys.version,
            "executable": sys.executable,
        },
        "uv": {
            "version": uv.stdout.strip() if uv is not None and uv.returncode == 0 else None,
            "error": (
                uv.stderr.strip()
                if uv is not None and uv.returncode != 0
                else uv_error
                if uv is None
                else None
            ),
        },
        "uv_lock_sha256": lock_hash,
        "parameters": dict(parameters),
    }


def _install_record_delay(
    recorder: Any,
    delay_ms: float,
    stall_ms: float,
) -> dict[str, Any]:
    counter = {
        "fake_sync_calls": 0,
        "fake_sync_delay_ms": delay_ms,
        "targeted_fake_stall_ms": stall_ms,
        "targeted_fake_stall_armed": False,
        "targeted_fake_stall_fired": False,
        "targeted_fake_stall_fired_this_operation": False,
        "targeted_fake_stall_injections": 0,
    }
    original = recorder.record

    def record_with_delay(*args: Any, **kwargs: Any) -> Any:
        counter["fake_sync_calls"] += 1
        if (
            counter["targeted_fake_stall_armed"]
            and not counter["targeted_fake_stall_fired_this_operation"]
        ):
            time.sleep(stall_ms / 1000.0)
            counter["targeted_fake_stall_fired"] = True
            counter["targeted_fake_stall_fired_this_operation"] = True
            counter["targeted_fake_stall_injections"] += 1
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        return original(*args, **kwargs)

    recorder.record = record_with_delay
    return counter


def _reset_bot_state(namespace: dict[str, Any], *, local_only: bool) -> None:
    namespace["local_only_mode"] = local_only
    namespace["history"] = [
        " Human: Hi! My name is Human. I like cats and video games!",
        "Cooper: Hello, I am Cooper. How can I help?",
    ]
    namespace["history_nonlocal"] = [
        {"role": "system", "content": "Cooper's Persona: friendly kid robot."},
        {"role": "user", "content": "Hello Cooper!"},
        {"role": "assistant", "content": "Hi! Ready to chat!"},
    ]
    namespace["unfound_answer"] = False
    namespace["asked_follow_up"] = False
    namespace["after_first"] = False
    namespace["after_second"] = False
    namespace["count_index"] = 0
    namespace["named_entities"] = []
    namespace["important_words"] = []


def _make_legacy_worker() -> Any | None:
    try:
        worker_module = importlib.import_module("deskbuddy.legacy_worker")
    except ImportError:
        return None
    return worker_module.LegacyWorker("issue37-cpu-benchmark", use_default_executor=False)


def _make_settings(Settings: Any, data_dir: Path, *, camera_source: str | None = None) -> Any:
    return Settings(
        data_dir=data_dir,
        cache_dir=data_dir / "cache",
        model_provider="disabled",
        openai_model="gpt-3.5-turbo-test",
        camera_source=camera_source,
        device="cpu",
    )


class _BenchmarkTensor:
    shape = (1,)

    def to(self, _device: Any) -> "_BenchmarkTensor":
        return self

    def unsqueeze(self, _axis: int) -> "_BenchmarkTensor":
        return self


class _BenchmarkImage:
    @staticmethod
    def fromarray(_image: Any) -> "_BenchmarkImage":
        return _BenchmarkImage()

    def convert(self, _format: str) -> "_BenchmarkImage":
        return self


class _BenchmarkProcessor:
    tokenizer = types.SimpleNamespace(cls_token_id=1)

    def __init__(self, delay_ms: float) -> None:
        self.delay_ms = delay_ms

    def __call__(self, *, images: Any = None, text: str | None = None, **_kwargs: Any) -> Any:
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        if images is not None:
            return types.SimpleNamespace(pixel_values=_BenchmarkTensor())
        if text is None:
            raise AssertionError("VQA text processor requires a question")
        return types.SimpleNamespace(input_ids=[2])

    def batch_decode(self, _outputs: Any, **_kwargs: Any) -> list[str]:
        return ["What is shown?a cat"]


class _BenchmarkVisionModel:
    def __init__(
        self,
        delay_ms: float,
        stall_ms: float,
        state: dict[str, int | bool],
        stall_calls: set[int],
    ) -> None:
        self.delay_ms = delay_ms
        self.stall_ms = stall_ms
        self.state = state
        self.stall_calls = stall_calls

    def to(self, _device: Any) -> "_BenchmarkVisionModel":
        return self

    def generate(self, **_kwargs: Any) -> list[int]:
        self.state["generate_calls"] = int(self.state["generate_calls"]) + 1
        if self.state["generate_calls"] in self.stall_calls:
            time.sleep(self.stall_ms / 1000.0)
            self.state["stall_fired"] = True
            self.state["stall_injections"] = int(self.state["stall_injections"]) + 1
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        return [1]


def _install_vqa_fakes(
    namespace: dict[str, Any],
    settings: Any,
    delay_ms: float,
    stall_ms: float,
    warmups: int,
    repetitions: int,
) -> dict[str, int | bool]:
    state: dict[str, int | bool] = {
        "generate_calls": 0,
        "stall_fired": False,
        "stall_injections": 0,
    }
    stall_calls = set(range(warmups + 1, warmups + repetitions + 1))

    class BenchmarkCV:
        __version__ = "cpu-fake"

        @staticmethod
        def imread(_path: str) -> object:
            return object()

        @staticmethod
        def VideoCapture(_source: Any) -> Any:
            return types.SimpleNamespace(
                read=lambda: (True, object()),
                release=lambda: None,
            )

    class BenchmarkTorch:
        version = types.SimpleNamespace(cuda=None)
        cuda = types.SimpleNamespace(
            is_available=lambda: False,
            amp=types.SimpleNamespace(autocast=contextlib.nullcontext),
        )

        @staticmethod
        def device(name: str) -> str:
            return name

        @staticmethod
        def tensor(_values: Any) -> _BenchmarkTensor:
            return _BenchmarkTensor()

    class ProcessorClass:
        @staticmethod
        def from_pretrained(_model_id: str) -> _BenchmarkProcessor:
            return _BenchmarkProcessor(delay_ms)

    class ModelClass:
        @staticmethod
        def from_pretrained(_model_id: str) -> _BenchmarkVisionModel:
            return _BenchmarkVisionModel(delay_ms, stall_ms, state, stall_calls)

    namespace.update(
        {
            "cv": BenchmarkCV(),
            "Image": _BenchmarkImage,
            "torch": BenchmarkTorch,
            "GitProcessor": ProcessorClass,
            "GitForCausalLM": ModelClass,
            "load_settings": lambda: settings,
            "visionQA_processor": None,
            "visionQA_model": None,
            "_vqa_lock": threading.Lock(),
        }
    )
    return state


def _load_bot_case(
    *,
    target: Path,
    data_dir: Path,
    local_only: bool,
    use_disk_question_bank: bool,
    fake_delay_ms: float,
    camera_source: str | None = None,
    stall_ms: float = 0.0,
) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    from deskbuddy.settings import Settings

    loader = importlib.import_module("tests.support.legacy_runtime")
    worker = _make_legacy_worker()
    settings = _make_settings(Settings, data_dir, camera_source=camera_source)
    kwargs: dict[str, Any] = {
        "settings": settings,
        "local_only_mode": local_only,
        "use_disk_question_bank": use_disk_question_bank,
    }
    if (
        worker is not None
        and "legacy_worker" in inspect.signature(loader.load_botcortex_runtime).parameters
    ):
        kwargs["legacy_worker"] = worker
    namespace, recorder = loader.load_botcortex_runtime(**kwargs)
    counters = _install_record_delay(recorder, fake_delay_ms, stall_ms)
    namespace["_benchmark_support"] = loader
    namespace["_benchmark_settings"] = settings
    namespace["_benchmark_worker"] = worker
    return namespace, recorder, loader, counters


def _ensure_production_vqa(
    namespace: dict[str, Any],
    *,
    target: Path,
    loader: Any,
    fake_delay_ms: float,
    stall_ms: float,
    warmups: int,
    repetitions: int,
) -> tuple[bool, dict[str, int | bool]]:
    vqa_state = _install_vqa_fakes(
        namespace,
        namespace["_benchmark_settings"],
        fake_delay_ms,
        stall_ms,
        warmups,
        repetitions,
    )
    target_file = (target / "BotCortex.py").resolve()
    functions_to_restore: set[str] = set()
    for function_name in ("answer_visual_question", "_answer_visual_question_sync"):
        function = namespace.get(function_name)
        code = getattr(function, "__code__", None)
        if not code or Path(code.co_filename).resolve() != target_file:
            functions_to_restore.add(function_name)
    if functions_to_restore:
        namespace.update(
            loader.extract_module_functions(
                target_file,
                functions_to_restore,
                namespace,
            )
        )
    production_vqa = True
    for function_name in ("answer_visual_question", "_answer_visual_question_sync"):
        function = namespace.get(function_name)
        code = getattr(function, "__code__", None)
        production_vqa = production_vqa and bool(
            code and Path(code.co_filename).resolve() == target_file
        )
    return production_vqa, vqa_state


BOT_CASES: tuple[tuple[str, str, bool, bool], ...] = (
    ("desktop.local_fast", "talk_fast", True, False),
    ("desktop.local_small", "talk_small", True, False),
    ("desktop.local_medium", "talk_medium_with_history", True, False),
    ("desktop.local_large_history", "talk_history", True, False),
    ("desktop.cloud_medium", "talk_medium_API_with_history", False, False),
    ("desktop.classifier", "classify_input", True, False),
    ("desktop.clarification", "get_follow_up_question", True, False),
    ("desktop.embedding_search", "search_question_in_bank", True, False),
    ("desktop.summary_and_topics", "get_list_of_list_of_strings", True, False),
    ("desktop.question_bank_persistence", "question_bank", True, True),
    ("desktop.vqa", "answer_visual_question", True, False),
)


async def _bot_operation(namespace: dict[str, Any], case_name: str) -> Any:
    if case_name == "talk_fast":
        return await namespace[case_name]("Tell me a short story about a friendly robot.")
    if case_name == "talk_small":
        return await namespace[case_name]("Tell me a short story about a friendly robot.")
    if case_name == "talk_medium_with_history":
        return await namespace[case_name]("Tell me a short story about a friendly robot.")
    if case_name == "talk_history":
        return await namespace[case_name]("Tell me a short story about a friendly robot.")
    if case_name == "talk_medium_API_with_history":
        return await namespace[case_name]("Tell me a short story about a friendly robot.")
    if case_name == "classify_input":
        return await namespace[case_name]("Can you tell me about robots?", namespace["history"])
    if case_name == "get_follow_up_question":
        return await namespace[case_name]("I went there recently", namespace["history"])
    if case_name == "search_question_in_bank":
        return await namespace[case_name](
            {"What is your favorite color?": "Blue", "What is your favorite food?": "Pasta"},
            "favorite color?",
        )
    if case_name == "get_list_of_list_of_strings":
        return await namespace[case_name]("Human: I like space and robots.")
    if case_name == "answer_visual_question":
        return await namespace[case_name]("What is shown in the picture?")
    raise ValueError(f"unknown BotCortex case {case_name!r}")


async def _run_bot_case(
    *,
    target: Path,
    name: str,
    callable_name: str,
    local_only: bool,
    use_disk_question_bank: bool,
    data_dir: Path,
    warmups: int,
    repetitions: int,
    heartbeat_ms: float,
    settle_ms: float,
    fake_delay_ms: float,
    stall_ms: float,
) -> dict[str, Any]:
    if callable_name == "answer_visual_question":
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "benchmark-image.png").write_bytes(b"CPU fake image fixture")
    namespace, recorder, loader, counters = _load_bot_case(
        target=target,
        data_dir=data_dir,
        local_only=local_only,
        use_disk_question_bank=use_disk_question_bank,
        fake_delay_ms=fake_delay_ms,
        camera_source=(
            str(data_dir / "benchmark-image.png")
            if callable_name == "answer_visual_question"
            else None
        ),
        stall_ms=stall_ms
        if callable_name not in {"question_bank", "answer_visual_question"}
        else 0.0,
    )
    production_vqa = False
    vqa_state: dict[str, int | bool] = {}
    if callable_name == "answer_visual_question":
        production_vqa, vqa_state = _ensure_production_vqa(
            namespace,
            target=target,
            loader=loader,
            fake_delay_ms=fake_delay_ms,
            stall_ms=stall_ms,
            warmups=warmups,
            repetitions=repetitions,
        )

    probe_module, probe_server, prober, probe_server_start_ms = await _open_authenticated_probe(
        target=target,
        variant="buddy1",
    )
    _ = probe_module
    storage_state: dict[str, Any] = {
        "armed": False,
        "fired": False,
        "fired_this_operation": False,
        "injections": 0,
    }
    original_save_question_bank = None
    if callable_name == "question_bank":
        question_bank_module = namespace["question_bank"]
        original_save_question_bank = question_bank_module.save_question_bank

        def save_question_bank_with_stall(*args: Any, **kwargs: Any) -> Any:
            if storage_state["armed"] and not storage_state["fired_this_operation"]:
                time.sleep(stall_ms / 1000.0)
                storage_state["fired"] = True
                storage_state["fired_this_operation"] = True
                storage_state["injections"] += 1
            return original_save_question_bank(*args, **kwargs)

        question_bank_module.save_question_bank = save_question_bank_with_stall

    iteration = 0

    async def operation() -> Mapping[str, Any]:
        nonlocal iteration
        measured_operation = iteration >= warmups
        if callable_name == "question_bank":
            storage_state["armed"] = measured_operation
            storage_state["fired_this_operation"] = False
        else:
            counters["targeted_fake_stall_armed"] = measured_operation
            counters["targeted_fake_stall_fired_this_operation"] = False
        iteration += 1
        _reset_bot_state(namespace, local_only=local_only)
        started = time.perf_counter()
        if callable_name == "question_bank":
            await namespace["load_qdict_from_json"]()
            await namespace["save_qdict_to_json"](
                {"What is your favorite color?": "Blue", "What is your favorite food?": "Pasta"}
            )
            await namespace["load_qdict_from_json"]()
        else:
            await _bot_operation(namespace, callable_name)
        elapsed = (time.perf_counter() - started) * 1000.0
        return {"operation_ms": elapsed, "turn_ms": elapsed}

    start_workers = namespace.get("start_workers")
    worker_start_ms = 0.0
    if start_workers is not None:
        worker_started = time.perf_counter()
        start_workers()
        worker_start_ms = (time.perf_counter() - worker_started) * 1000.0

    workers_shutdown = False

    async def stop_workers() -> None:
        nonlocal workers_shutdown
        if not workers_shutdown:
            await _maybe_call(namespace.get("shutdown_workers"))
            workers_shutdown = True

    try:
        result = await _collect_samples(
            operation=operation,
            warmups=warmups,
            repetitions=repetitions,
            heartbeat_ms=heartbeat_ms,
            settle_ms=settle_ms,
            shutdown=stop_workers,
        )
    finally:
        await stop_workers()
        if original_save_question_bank is not None:
            namespace["question_bank"].save_question_bank = original_save_question_bank
        probe_client_shutdown_ms = await prober.stop()
        server_shutdown_started = time.perf_counter()
        await probe_server.stop()
        server_shutdown_ms = (time.perf_counter() - server_shutdown_started) * 1000.0
    probe_report = prober.report()
    for key in ("ping_rtt_ms", "poll_rtt_ms", "ack_rtt_ms"):
        result["timings_ms"][key] = probe_report[key]
    result.update(
        {
            "name": name,
            "status": "ok",
            "path": f"BotCortex.{callable_name}"
            if callable_name != "question_bank"
            else "BotCortex.load_qdict_from_json/save_qdict_to_json",
            "runtime_mode": "local" if local_only else "cloud fake",
            "fake_sync_calls": counters["fake_sync_calls"],
            "fake_sync_delay_ms_per_record": fake_delay_ms,
            "targeted_fake_stall_ms": stall_ms,
            "targeted_fake_stall_fired": bool(
                vqa_state.get("stall_fired", False)
                if callable_name == "answer_visual_question"
                else storage_state["fired"]
                if callable_name == "question_bank"
                else counters["targeted_fake_stall_fired"]
            ),
            "targeted_fake_stall_expected_injections": repetitions if stall_ms > 0 else 0,
            "targeted_fake_stall_injections": int(
                vqa_state.get("stall_injections", 0)
                if callable_name == "answer_visual_question"
                else storage_state["injections"]
                if callable_name == "question_bank"
                else counters["targeted_fake_stall_injections"]
            ),
            "warmup_stall_policy": "no targeted 250 ms stalls during warmups",
            "worker_startup_ms_single_observation": worker_start_ms,
            "authenticated_probe": probe_report,
            "probe_server_startup_ms_single_observation": probe_server_start_ms,
            "probe_client_shutdown_ms_single_observation": probe_client_shutdown_ms,
            "probe_server_shutdown_ms_single_observation": server_shutdown_ms,
            "uses_production_vqa_function": production_vqa
            if callable_name == "answer_visual_question"
            else None,
        }
    )
    return result


class _OneTurnPoller:
    def __init__(self, reply_sent: asyncio.Event) -> None:
        self.reply_sent = reply_sent
        self.delivered = False

    async def next_transcript(self) -> str | None:
        if not self.delivered:
            self.delivered = True
            return "Tell me a short story about a friendly robot."
        await self.reply_sent.wait()
        raise asyncio.CancelledError


def _fake_robot_client(reply_sent: asyncio.Event) -> Any:
    class FakeTransportError(Exception):
        code = "transport_error"
        message = "fake transport error"

    class FakeSettingsError(Exception):
        pass

    class FakeExecutionOutcomeUnknown(Exception):
        command = "reply"
        request_id = "benchmark"

    poller = _OneTurnPoller(reply_sent)

    class FakeRobotClient:
        TransportError = FakeTransportError
        SettingsError = FakeSettingsError
        ExecutionOutcomeUnknown = FakeExecutionOutcomeUnknown
        TRANSCRIPT_MODE_MISMATCH_MESSAGE = "polling mode mismatch"

        def TranscriptPoller(self) -> Any:
            return poller

        async def send_reply(self, _message: str) -> None:
            await asyncio.sleep(0.002)
            reply_sent.set()

        async def close_client(self) -> None:
            return None

    return FakeRobotClient()


async def _run_desktop_turn_case(
    *,
    target: Path,
    local_only: bool,
    data_dir: Path,
    warmups: int,
    repetitions: int,
    heartbeat_ms: float,
    settle_ms: float,
    fake_delay_ms: float,
    stall_ms: float,
) -> dict[str, Any]:
    runtimes = [
        _load_bot_case(
            target=target,
            data_dir=data_dir / f"turn-{index:02d}",
            local_only=local_only,
            use_disk_question_bank=False,
            fake_delay_ms=fake_delay_ms,
            stall_ms=stall_ms,
        )
        for index in range(warmups + repetitions)
    ]
    _, probe_server, prober, probe_server_start_ms = await _open_authenticated_probe(
        target=target,
        variant="buddy1",
    )
    iteration = 0

    async def operation() -> Mapping[str, Any]:
        nonlocal iteration
        namespace, _recorder, loader, counters = runtimes[iteration]
        measured_operation = iteration >= warmups
        counters["targeted_fake_stall_armed"] = measured_operation
        counters["targeted_fake_stall_fired_this_operation"] = False
        iteration += 1
        _reset_bot_state(namespace, local_only=local_only)
        start_workers = namespace.get("start_workers")
        worker_start_ms = 0.0
        if start_workers is not None:
            worker_started = time.perf_counter()
            start_workers()
            worker_start_ms = (time.perf_counter() - worker_started) * 1000.0
        reply_sent = asyncio.Event()
        fake_rc = _fake_robot_client(reply_sent)
        parameters = inspect.signature(loader.load_start_bot_main).parameters
        kwargs: dict[str, Any] = {
            "local_only": local_only,
            "button_text": "Medium",
        }
        if "poller" in parameters:
            kwargs["poller"] = fake_rc.TranscriptPoller()
        if "max_turns" in parameters:
            kwargs["max_turns"] = 1
        main = loader.load_start_bot_main(namespace, fake_rc, **kwargs)
        started = time.perf_counter()
        task = asyncio.create_task(main())
        try:
            await asyncio.wait_for(reply_sent.wait(), timeout=15.0)
            turn_ms = (time.perf_counter() - started) * 1000.0
        finally:
            shutdown_started = time.perf_counter()
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await _maybe_call(namespace.get("shutdown_workers"))
        shutdown_ms = (time.perf_counter() - shutdown_started) * 1000.0
        return {
            "operation_ms": turn_ms,
            "turn_ms": turn_ms,
            "shutdown_ms": shutdown_ms,
            "worker_startup_ms": worker_start_ms,
        }

    try:
        result = await _collect_samples(
            operation=operation,
            warmups=warmups,
            repetitions=repetitions,
            heartbeat_ms=heartbeat_ms,
            settle_ms=settle_ms,
            shutdown=None,
        )
    finally:
        probe_client_shutdown_ms = await prober.stop()
        server_shutdown_started = time.perf_counter()
        await probe_server.stop()
        server_shutdown_ms = (time.perf_counter() - server_shutdown_started) * 1000.0
    probe_report = prober.report()
    for key in ("ping_rtt_ms", "poll_rtt_ms", "ack_rtt_ms"):
        result["timings_ms"][key] = probe_report[key]
    result.update(
        {
            "name": f"desktop.main_{'local' if local_only else 'cloud'}_one_turn",
            "status": "ok",
            "path": "start_bot.main -> TranscriptPoller -> dispatch_reply -> send_reply(fake)",
            "transcript_source": "deterministic fake poller; authenticated loopback is measured separately",
            "fake_sync_calls": sum(context[3]["fake_sync_calls"] for context in runtimes),
            "fake_sync_delay_ms_per_record": fake_delay_ms,
            "targeted_fake_stall_ms": stall_ms,
            "targeted_fake_stall_fired": any(
                context[3]["targeted_fake_stall_fired"] for context in runtimes
            ),
            "targeted_fake_stall_expected_injections": repetitions if stall_ms > 0 else 0,
            "targeted_fake_stall_injections": sum(
                context[3]["targeted_fake_stall_injections"] for context in runtimes
            ),
            "warmup_stall_policy": "no targeted 250 ms stalls during warmups",
            "worker_startup_ms": result["additional_timings_ms"].get("worker_startup_ms"),
            "authenticated_probe": probe_report,
            "probe_server_startup_ms_single_observation": probe_server_start_ms,
            "probe_client_shutdown_ms_single_observation": probe_client_shutdown_ms,
            "probe_server_shutdown_ms_single_observation": server_shutdown_ms,
        }
    )
    return result


def _direct_get_input_path_await(target: Path, variant: str) -> bool:
    if variant != "windows.buddy2":
        return False
    source = (target / "windows" / "buddy2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "get_input_path"
        for node in ast.walk(tree)
    )


class _AwaitablePath(str):
    """Benchmark-only bridge for the baseline sync-return/await mismatch."""

    def __await__(self):
        async def return_path() -> str:
            return str(self)

        return return_path().__await__()


class _FakeAudioStream:
    def __init__(self, chunk_size: int, delay_ms: float, counts: dict[str, int]) -> None:
        self.chunk_size = chunk_size
        self.delay_ms = delay_ms
        self.counts = counts
        self.read_index = 0
        self.speech = struct.pack("<" + "h" * chunk_size, *([1200] * chunk_size))
        self.silence = bytes(chunk_size * 2)

    def read(self, _chunk_size: int) -> bytes:
        self.counts["audio_reads"] += 1
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        self.read_index += 1
        return self.speech if self.read_index == 1 else self.silence

    def stop_stream(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeAudioBackend:
    paInt16 = 8

    def __init__(self, chunk_size: int, delay_ms: float, counts: dict[str, int]) -> None:
        self.chunk_size = chunk_size
        self.delay_ms = delay_ms
        self.counts = counts

    def PyAudio(self) -> Any:
        self.counts["audio_backend_constructions"] += 1
        backend = self

        class Audio:
            def open(self, **_kwargs: Any) -> _FakeAudioStream:
                return _FakeAudioStream(backend.chunk_size, backend.delay_ms, backend.counts)

            @staticmethod
            def get_sample_size(_format: Any) -> int:
                return 2

            def terminate(self) -> None:
                backend.counts["audio_backend_terminations"] += 1

        return Audio()


class _FakeKeyboard:
    def __init__(self, *, capture: bool) -> None:
        self.capture = capture
        self.calls = 0

    def is_pressed(self, _key: str) -> bool:
        self.calls += 1
        return self.capture and self.calls <= 2


class _FakePicoh:
    HEADNOD = 0
    HEADTURN = 1
    EYETURN = 2
    LIDBLINK = 3
    TOPLIP = 4
    BOTTOMLIP = 5
    EYETILT = 6

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_input_path(self) -> str:
        return str(self.path)

    def reset(self) -> None:
        return None

    def wait(self, _seconds: float) -> None:
        return None

    def setEyeShape(self, _shape: Any) -> None:
        return None

    def setEyeBrightness(self, _value: Any) -> None:
        return None

    def say(self, _text: str, untilDone: bool = True) -> None:
        _ = untilDone

    def move(self, _motor: Any, _position: Any) -> None:
        return None

    def setBaseColour(self, _red: Any, _green: Any, _blue: Any) -> None:
        return None


class _FakeWhisper:
    def __init__(
        self,
        counts: dict[str, int],
        delay_ms: float,
        stall_ms: float,
        stall_calls: set[int],
    ) -> None:
        self.counts = counts
        self.delay_ms = delay_ms
        self.stall_ms = stall_ms
        self.stall_calls = stall_calls
        self.stall_fired = False
        self.stall_injections = 0

    def transcribe(self, _path: str) -> dict[str, str]:
        self.counts["whisper_calls"] += 1
        if self.counts["whisper_calls"] in self.stall_calls:
            time.sleep(self.stall_ms / 1000.0)
            self.stall_fired = True
            self.stall_injections += 1
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        return {"text": "benchmark spoken turn"}


async def _listener_kwargs(
    listener: Callable[..., Any],
) -> tuple[dict[str, Any], asyncio.Event | None]:
    parameters = inspect.signature(listener).parameters
    stop_event = asyncio.Event() if "stop_event" in parameters else None
    kwargs: dict[str, Any] = {}
    if stop_event is not None:
        kwargs["stop_event"] = stop_event
    if "capture_stop" in parameters:
        kwargs["capture_stop"] = threading.Event()
    return kwargs, stop_event


def _reset_robot_state(module: Any, keyboard: _FakeKeyboard, *, capture: bool) -> None:
    keyboard.capture = capture
    keyboard.calls = 0
    for name, value in (
        ("can_listen", True),
        ("listening", True),
        ("recording", False),
        ("recorded", False),
        ("result", None),
        ("lastTranscription", ""),
        ("priorTranscription", ""),
    ):
        # Some legacy modules only assign these globals from init_hardware();
        # seed them directly so the CPU-only adapter follows its post-startup state.
        setattr(module, name, value)


def _idle_coroutines(module: Any) -> list[tuple[str, Callable[[], Awaitable[Any]]]]:
    names = ("blinkLids", "randomLook", "randomNod", "randomTurn")
    return [
        (name, coroutine)
        for name in names
        if callable(coroutine := getattr(module.MyFunctions, name, None))
    ]


async def _run_robot_listener_case(
    *,
    target: Path,
    variant: str,
    mode: str,
    profile: Path,
    warmups: int,
    repetitions: int,
    heartbeat_ms: float,
    settle_ms: float,
    idle_window_ms: float,
    audio_read_delay_ms: float,
    whisper_delay_ms: float,
    stall_ms: float,
) -> dict[str, Any]:
    _configure_offline_environment(profile)
    _install_target_imports(target)
    module = importlib.import_module(ROBOT_VARIANTS[variant])
    module.wave = wave
    listener = module.MyFunctions.listen
    target_path = profile / f"{variant.replace('.', '-')}-benchmark.wav"
    counts: dict[str, int] = {
        "audio_backend_constructions": 0,
        "audio_backend_terminations": 0,
        "audio_reads": 0,
        "whisper_calls": 0,
        "transcript_enqueues": 0,
    }
    source_bug = _direct_get_input_path_await(target, variant)
    picoh = _FakePicoh(target_path)
    module.picoh = picoh
    module.pyaudio = _FakeAudioBackend(1024, audio_read_delay_ms, counts)
    whisper_stall_calls = set(range(warmups + 1, warmups + repetitions + 1))
    whisper_fake = _FakeWhisper(
        counts,
        whisper_delay_ms,
        stall_ms if mode == "capture" else 0.0,
        whisper_stall_calls if mode == "capture" else set(),
    )
    module.model = whisper_fake
    keyboard = _FakeKeyboard(capture=mode == "capture")
    module.keyboard = keyboard
    base_send_response = module.MyFunctions.send_response
    compatibility = {
        "applied": bool(source_bug and mode == "capture"),
        "name": "awaitable-str-return shim",
        "reason": "baseline windows.buddy2 awaits a synchronous Picoh path result; shim preserves the same returned path while allowing the remaining capture/Whisper path to run",
    }
    strict_case: dict[str, Any] = {"status": "not_applicable"}
    if source_bug:

        async def strict_probe() -> dict[str, str]:
            try:
                await picoh.get_input_path()
            except TypeError as error:
                return {"status": "expected_type_error", "error": str(error)}
            return {"status": "await_succeeded"}

        strict_case = await strict_probe()
        if mode == "capture":
            picoh.get_input_path = lambda: _AwaitablePath(str(target_path))  # type: ignore[method-assign]

    original_callable = listener
    transcript_event = asyncio.Event()
    idle_coroutines = _idle_coroutines(module) if mode == "idle" else []

    async def send_response(text: str) -> Any:
        result = await base_send_response(text)
        counts["transcript_enqueues"] += 1
        transcript_event.set()
        return result

    async def operation() -> Mapping[str, Any]:
        transcript_event.clear()
        _reset_robot_state(module, keyboard, capture=mode == "capture")
        module.MyFunctions.send_response = send_response
        kwargs, stop_event = await _listener_kwargs(original_callable)
        started = time.perf_counter()
        task = asyncio.create_task(original_callable(**kwargs))
        idle_tasks = [asyncio.create_task(coroutine()) for _, coroutine in idle_coroutines]
        if mode == "capture":
            try:
                await asyncio.wait_for(transcript_event.wait(), timeout=10.0)
            except TimeoutError as error:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                module.MyFunctions.send_response = base_send_response
                raise RuntimeError(
                    f"{variant} listener did not publish fake transcript; "
                    f"calls={counts}, keyboard_checks={keyboard.calls}, "
                    f"recording={getattr(module, 'recording', None)}, "
                    f"recorded={getattr(module, 'recorded', None)}"
                ) from error
            turn_ms = (time.perf_counter() - started) * 1000.0
        else:
            await asyncio.sleep(idle_window_ms / 1000.0)
            turn_ms = None
        shutdown_started = time.perf_counter()
        if stop_event is not None:
            stop_event.set()
        elif not task.done():
            task.cancel()
        for idle_task in idle_tasks:
            idle_task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(task, *idle_tasks, return_exceptions=True),
                timeout=3.0,
            )
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            pass
        shutdown_ms = (time.perf_counter() - shutdown_started) * 1000.0
        module.MyFunctions.send_response = base_send_response
        return {
            "operation_ms": turn_ms if turn_ms is not None else idle_window_ms,
            "turn_ms": turn_ms,
            "shutdown_ms": shutdown_ms,
        }

    probe_module, probe_server, prober, probe_server_start_ms = await _open_authenticated_probe(
        target=target,
        variant=variant,
    )
    _ = probe_module
    try:
        result = await _collect_samples(
            operation=operation,
            warmups=warmups,
            repetitions=repetitions,
            heartbeat_ms=heartbeat_ms,
            settle_ms=settle_ms,
            shutdown=None,
        )
    finally:
        module.MyFunctions.send_response = base_send_response
        probe_client_shutdown_ms = await prober.stop()
        server_shutdown_started = time.perf_counter()
        await probe_server.stop()
        server_shutdown_ms = (time.perf_counter() - server_shutdown_started) * 1000.0
    probe_report = prober.report()
    for key in ("ping_rtt_ms", "poll_rtt_ms", "ack_rtt_ms"):
        result["timings_ms"][key] = probe_report[key]
    result.update(
        {
            "name": f"robot.{variant}.{mode}",
            "status": "ok",
            "path": f"{variant}.MyFunctions.listen -> capture -> Whisper -> send_response",
            "idle_coroutines_started": [name for name, _ in idle_coroutines],
            "fake_audio_read_delay_ms": audio_read_delay_ms,
            "fake_whisper_delay_ms": whisper_delay_ms,
            "targeted_fake_stall_ms": stall_ms if mode == "capture" else 0.0,
            "targeted_fake_stall_fired": whisper_fake.stall_fired,
            "targeted_fake_stall_expected_injections": repetitions if mode == "capture" else 0,
            "targeted_fake_stall_injections": whisper_fake.stall_injections,
            "warmup_stall_policy": "no targeted 250 ms stalls during warmups",
            "calls": counts,
            "strict_get_input_path_case": strict_case,
            "compatibility_shim": compatibility,
            "authenticated_probe": probe_report,
            "probe_server_startup_ms_single_observation": probe_server_start_ms,
            "probe_client_shutdown_ms_single_observation": probe_client_shutdown_ms,
            "probe_server_shutdown_ms_single_observation": server_shutdown_ms,
        }
    )
    return result


async def _run_robot_loopback_case(
    *,
    target: Path,
    variant: str,
    profile: Path,
    warmups: int,
    repetitions: int,
    heartbeat_ms: float,
    settle_ms: float,
) -> dict[str, Any]:
    _configure_offline_environment(profile)
    _install_target_imports(target)
    module = importlib.import_module(ROBOT_VARIANTS[variant])
    from transport import RobotClient, TransportConfig

    settings = TransportConfig(
        host="127.0.0.1",
        port=0,
        secret=TEST_SECRET,
        connect_timeout=2.0,
        read_timeout=2.0,
        write_timeout=2.0,
        request_timeout=3.0,
        max_retries=0,
    )
    server = module.create_robot_server(settings)
    started = time.perf_counter()
    await server.start()
    startup_ms = (time.perf_counter() - started) * 1000.0
    client_config = settings.with_overrides(port=server.port)
    client = RobotClient(client_config)
    connection_started = time.perf_counter()
    connection_warmup = await client.request("ping", {})
    client_connection_warmup_ms = (time.perf_counter() - connection_started) * 1000.0
    if not connection_warmup.ok:
        raise RuntimeError(
            f"authenticated loopback client warmup failed: {connection_warmup.error}"
        )

    async def operation() -> Mapping[str, Any]:
        ping_started = time.perf_counter()
        ping = await client.request("ping", {})
        ping_ms = (time.perf_counter() - ping_started) * 1000.0
        if not ping.ok:
            raise RuntimeError(f"authenticated ping failed: {ping.error}")
        await module.MyFunctions.send_response("loopback benchmark transcript")
        poll_started = time.perf_counter()
        poll = await client.request("poll_transcript", {})
        poll_ms = (time.perf_counter() - poll_started) * 1000.0
        if not poll.ok or not isinstance(poll.result, dict):
            raise RuntimeError(f"authenticated transcript poll failed: {poll.error}")
        ack_started = time.perf_counter()
        ack = await client.request("ack_transcript", {"id": poll.result["id"]})
        ack_ms = (time.perf_counter() - ack_started) * 1000.0
        if not ack.ok or ack.result is not True:
            raise RuntimeError(f"authenticated transcript ack failed: {ack.error}")
        return {
            "ping_rtt_ms": ping_ms,
            "poll_rtt_ms": poll_ms,
            "ack_rtt_ms": ack_ms,
            "operation_ms": ping_ms + poll_ms + ack_ms,
            "turn_ms": ping_ms + poll_ms + ack_ms,
        }

    result = await _collect_samples(
        operation=operation,
        warmups=warmups,
        repetitions=repetitions,
        heartbeat_ms=heartbeat_ms,
        settle_ms=settle_ms,
    )
    await client.close()
    stopped = time.perf_counter()
    await server.stop()
    shutdown_ms = (time.perf_counter() - stopped) * 1000.0
    result["timings_ms"]["shutdown_ms"] = _summarize([shutdown_ms])
    result["shutdown_ms_single_observation"] = shutdown_ms
    result.update(
        {
            "name": f"robot.{variant}.authenticated_loopback_ping_poll_ack",
            "status": "ok",
            "path": "production RobotServer + RobotClient, authenticated 127.0.0.1 only",
            "server_startup_ms_single_observation": startup_ms,
            "client_connection_warmup_ms_single_observation": client_connection_warmup_ms,
            "server_shutdown_ms_single_observation": shutdown_ms,
            "network_scope": "loopback only",
        }
    )
    return result


def _robot_child(args: argparse.Namespace) -> int:
    target = Path(args.target_checkout).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="deskbuddy-issue37-robot-") as profile_text:
        profile = Path(profile_text)
        _configure_offline_environment(profile)
        captured_stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_stdout):
                if args.robot_child_mode == "loopback":
                    result = asyncio.run(
                        _run_robot_loopback_case(
                            target=target,
                            variant=args.robot_variant,
                            profile=profile,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                            heartbeat_ms=args.heartbeat_ms,
                            settle_ms=args.settle_ms,
                        )
                    )
                else:
                    result = asyncio.run(
                        _run_robot_listener_case(
                            target=target,
                            variant=args.robot_variant,
                            mode=args.robot_child_mode,
                            profile=profile,
                            warmups=args.warmups,
                            repetitions=args.repetitions,
                            heartbeat_ms=args.heartbeat_ms,
                            settle_ms=args.settle_ms,
                            idle_window_ms=args.idle_window_ms,
                            audio_read_delay_ms=args.audio_read_delay_ms,
                            whisper_delay_ms=args.whisper_delay_ms,
                            stall_ms=args.stall_ms,
                        )
                    )
        except BaseException as error:
            result = {
                "name": f"robot.{args.robot_variant}.{args.robot_child_mode}",
                "status": "failed",
                "comparison_limitation": None,
                "error": f"{type(error).__name__}: {error}",
                "system_exit_code": error.code if isinstance(error, SystemExit) else None,
            }
        captured = captured_stdout.getvalue()
        if captured and result.get("status") != "ok":
            result["captured_stdout_tail"] = captured[-1000:]
    sys.__stdout__.write(json.dumps(result, sort_keys=True) + "\n")
    sys.__stdout__.flush()
    return 0 if result.get("status") in {"ok", "baseline_comparison_limited"} else 1


def _run_robot_child(
    *,
    script: Path,
    target: Path,
    variant: str,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--target-checkout",
        str(target),
        "--warmups",
        str(args.warmups),
        "--repetitions",
        str(args.repetitions),
        "--heartbeat-ms",
        str(args.heartbeat_ms),
        "--settle-ms",
        str(args.settle_ms),
        "--idle-window-ms",
        str(args.idle_window_ms),
        "--audio-read-delay-ms",
        str(args.audio_read_delay_ms),
        "--whisper-delay-ms",
        str(args.whisper_delay_ms),
        "--stall-ms",
        str(args.stall_ms),
        "--robot-child-mode",
        mode,
        "--robot-variant",
        variant,
    ]
    try:
        watchdog_seconds = args.watchdog_seconds
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=watchdog_seconds,
        )
    except subprocess.TimeoutExpired:
        baseline = _git_text(target, "rev-parse", "HEAD") == BASELINE_COMMIT
        limited = baseline and mode == "idle"
        return {
            "name": f"robot.{variant}.{mode}",
            "status": "baseline_comparison_limited" if limited else "watchdog_timeout",
            "watchdog_seconds": watchdog_seconds,
            "error": "child process exceeded watchdog; isolated listener was terminated",
            "idle_coroutines_exercised": (
                ["blinkLids", "randomLook", "randomNod", "randomTurn"] if mode == "idle" else []
            ),
            "comparison_limitation": (
                "legacy disabled idle coroutines did not return control to the event loop before the watchdog; listener metrics were excluded"
                if limited and mode == "idle"
                else "legacy listener exceeded the watchdog and was excluded from numeric comparison"
                if limited
                else None
            ),
        }
    if completed.returncode != 0:
        baseline = _git_text(target, "rev-parse", "HEAD") == BASELINE_COMMIT
        limited = baseline and mode == "idle"
        return {
            "name": f"robot.{variant}.{mode}",
            "status": "baseline_comparison_limited" if limited else "child_failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-4000:],
            "comparison_limitation": (
                "legacy listener failed this CPU-fake case; retain as a baseline limitation"
                if limited
                else None
            ),
        }
    output_lines = completed.stdout.splitlines()
    result = None
    json_line = None
    parse_error = None
    for line in reversed(output_lines):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError as error:
            parse_error = error
            continue
        if isinstance(candidate, dict):
            result = candidate
            json_line = line
            break
    if result is None:
        return {
            "name": f"robot.{variant}.{mode}",
            "status": "invalid_child_output",
            "error": str(parse_error) if parse_error else "child produced no JSON object",
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
    result["child_stdout_non_json_lines"] = sum(line != json_line for line in output_lines)
    result["child_python_version"] = platform.python_version()
    result["watchdog_seconds"] = watchdog_seconds
    return result


async def _run_stall_probe(
    *,
    stall_ms: float,
    warmups: int,
    repetitions: int,
    heartbeat_ms: float,
    settle_ms: float,
) -> dict[str, Any]:
    iteration = 0

    async def block_loop() -> Mapping[str, Any]:
        nonlocal iteration
        inject_stall = iteration >= warmups
        iteration += 1
        started = time.perf_counter()
        if inject_stall:
            time.sleep(stall_ms / 1000.0)
        elapsed = (time.perf_counter() - started) * 1000.0
        return {"operation_ms": elapsed, "turn_ms": elapsed}

    result = await _collect_samples(
        operation=block_loop,
        warmups=warmups,
        repetitions=repetitions,
        heartbeat_ms=heartbeat_ms,
        settle_ms=settle_ms,
    )
    result.update(
        {
            "name": "calibration.synthetic_event_loop_stall",
            "status": "ok",
            "path": "synthetic time.sleep on the event-loop thread; calibrates heartbeat only",
            "stall_ms": stall_ms,
            "targeted_fake_stall_expected_injections": repetitions,
            "targeted_fake_stall_injections": repetitions,
            "warmup_stall_policy": "no targeted 250 ms stalls during warmups",
        }
    )
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-checkout", required=True, help="Absolute target Git checkout")
    parser.add_argument("--json-output", help="Absolute output JSON path for the full run")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--heartbeat-ms", type=float, default=5.0)
    parser.add_argument("--stall-ms", type=float, default=250.0)
    parser.add_argument("--settle-ms", type=float, default=10.0)
    parser.add_argument("--fake-sync-delay-ms", type=float, default=10.0)
    parser.add_argument("--idle-window-ms", type=float, default=250.0)
    parser.add_argument("--audio-read-delay-ms", type=float, default=1.0)
    parser.add_argument("--whisper-delay-ms", type=float, default=10.0)
    parser.add_argument("--watchdog-seconds", type=float, default=60.0)
    parser.add_argument("--robot-child-mode", choices=("capture", "idle", "loopback"))
    parser.add_argument("--robot-variant", choices=tuple(ROBOT_VARIANTS))
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be >= 0 and repetitions must be >= 1")
    if args.heartbeat_ms <= 0 or args.stall_ms < 0 or args.watchdog_seconds <= 0:
        parser.error("heartbeat and watchdog must be positive; stall must be non-negative")
    target_text = Path(args.target_checkout)
    if not target_text.is_absolute():
        parser.error("--target-checkout must be an absolute path")
    if args.robot_child_mode is not None:
        if args.robot_variant is None:
            parser.error("robot child mode requires --robot-variant")
    else:
        if args.json_output is None:
            parser.error("--json-output is required for a full benchmark run")
        output_text = Path(args.json_output)
        if not output_text.is_absolute():
            parser.error("--json-output must be an absolute path")
    return args


def _run_full(args: argparse.Namespace) -> int:
    target = Path(args.target_checkout).resolve(strict=True)
    output = Path(args.json_output).resolve(strict=False)
    if not (target / ".git").exists() and not (target / ".git").is_file():
        raise ValueError(f"target checkout does not contain Git metadata: {target}")
    required = (
        target / "BotCortex.py",
        target / "start_bot.py",
        target / "tests" / "support" / "legacy_runtime.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"target checkout is missing required paths: {missing}")

    parameters = {
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "heartbeat_ms": args.heartbeat_ms,
        "stall_ms": args.stall_ms,
        "settle_ms": args.settle_ms,
        "fake_sync_delay_ms": args.fake_sync_delay_ms,
        "idle_window_ms": args.idle_window_ms,
        "audio_read_delay_ms": args.audio_read_delay_ms,
        "whisper_delay_ms": args.whisper_delay_ms,
        "watchdog_seconds": args.watchdog_seconds,
        "stall_injection_policy": "one production-facing fake stall per measured repetition where a blocking seam exists; direct RPC-only loopback has no artificial stall",
        "warmup_stall_policy": "no targeted 250 ms stalls during warmups",
        "loopback_host": "127.0.0.1",
        "models": "deterministic CPU fakes only",
        "audio": "synthetic in-memory PCM only",
        "devices": "none",
        "external_services": "none",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deskbuddy-issue37-") as temporary:
        profile = Path(temporary)
        _configure_offline_environment(profile)
        _install_target_imports(target)
        data_dir = profile / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        failures: list[str] = []

        try:
            for name, callable_name, local_only, use_disk in BOT_CASES:
                result = _run_async_quiet(
                    _run_bot_case(
                        target=target,
                        name=name,
                        callable_name=callable_name,
                        local_only=local_only,
                        use_disk_question_bank=use_disk,
                        data_dir=data_dir / name.replace(".", "-"),
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                        heartbeat_ms=args.heartbeat_ms,
                        settle_ms=args.settle_ms,
                        fake_delay_ms=args.fake_sync_delay_ms,
                        stall_ms=args.stall_ms,
                    )
                )
                results.append(result)
                if result.get("status") != "ok":
                    failures.append(str(result.get("name")))

            for local_only in (True, False):
                result = _run_async_quiet(
                    _run_desktop_turn_case(
                        target=target,
                        local_only=local_only,
                        data_dir=data_dir / f"desktop-main-{local_only}",
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                        heartbeat_ms=args.heartbeat_ms,
                        settle_ms=args.settle_ms,
                        fake_delay_ms=args.fake_sync_delay_ms,
                        stall_ms=args.stall_ms,
                    )
                )
                results.append(result)
                if result.get("status") != "ok":
                    failures.append(str(result.get("name")))

            stall = _run_async_quiet(
                _run_stall_probe(
                    stall_ms=args.stall_ms,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    heartbeat_ms=args.heartbeat_ms,
                    settle_ms=args.settle_ms,
                )
            )
            results.append(stall)

            script = Path(__file__).resolve()
            for variant in ROBOT_VARIANTS:
                for mode in ("capture", "idle", "loopback"):
                    result = _run_robot_child(
                        script=script,
                        target=target,
                        variant=variant,
                        mode=mode,
                        args=args,
                    )
                    results.append(result)
                    if result.get("status") not in {"ok", "baseline_comparison_limited"}:
                        failures.append(str(result.get("name")))
        except Exception as error:
            failures.append(f"benchmark exception: {type(error).__name__}: {error}")

        report = {
            "schema_version": 1,
            "status": "failed" if failures else "ok",
            "metadata": _runtime_metadata(target, parameters),
            "scenarios": results,
            "failures": failures,
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        readback = json.loads(output.read_text(encoding="utf-8"))
        if readback != report:
            raise RuntimeError("benchmark JSON readback differs from written report")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(output),
                    "target_head": report["metadata"]["target_head"],
                    "runtime_sources_dirty": report["metadata"]["runtime_sources_dirty"],
                    "scenario_count": len(results),
                    "failures": failures,
                },
                sort_keys=True,
            )
        )
        return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.robot_child_mode is not None:
        return _robot_child(args)
    return _run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
