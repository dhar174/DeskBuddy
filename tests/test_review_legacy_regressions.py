from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOT_CORTEX_PATH = REPOSITORY_ROOT / "BotCortex.py"


def _vqa_runtime(
    *,
    camera_source: str | int = "fixture.png",
    processor_loader: Callable[[str], Any] | None = None,
    model_loader: Callable[[str], Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tree = ast.parse(BOT_CORTEX_PATH.read_text(encoding="utf-8"), filename=str(BOT_CORTEX_PATH))
    names = {"_answer_visual_question_sync", "answer_visual_question"}
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }
    assert functions.keys() == names
    module = ast.Module(body=list(functions.values()), type_ignores=[])
    ast.fix_missing_locations(module)

    class FakeTensor:
        shape = (1,)

        def to(self, _device: Any) -> "FakeTensor":
            return self

        def unsqueeze(self, _axis: int) -> "FakeTensor":
            return self

    class FakeImage:
        @staticmethod
        def fromarray(_image: Any) -> "FakeImage":
            return FakeImage()

        def convert(self, _format: str) -> "FakeImage":
            return self

    class FakeProcessor:
        tokenizer = SimpleNamespace(cls_token_id=1)

        def __call__(self, *, images: Any = None, text: str | None = None, **_kwargs: Any):
            if images is not None:
                return SimpleNamespace(pixel_values=FakeTensor())
            assert text is not None
            return SimpleNamespace(input_ids=[2])

        def batch_decode(self, _outputs: Any, **_kwargs: Any) -> list[str]:
            return ["What is shown?a cat"]

    class FakeModel:
        def to(self, _device: Any) -> "FakeModel":
            return self

        def generate(self, **_kwargs: Any) -> list[int]:
            return [1]

    class FakeProcessorClass:
        @staticmethod
        def from_pretrained(model_id: str) -> Any:
            return processor_loader(model_id) if processor_loader else FakeProcessor()

    class FakeModelClass:
        @staticmethod
        def from_pretrained(model_id: str) -> Any:
            return model_loader(model_id) if model_loader else FakeModel()

    class FakeCamera:
        def read(self) -> tuple[bool, object]:
            return True, object()

        def release(self) -> None:
            return None

    class FakeCV:
        __version__ = "fake"

        @staticmethod
        def imread(_path: str) -> object:
            return object()

        @staticmethod
        def VideoCapture(_source: Any) -> FakeCamera:
            return FakeCamera()

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"unexpected OpenCV access: {name}")

    class FakeTorch:
        @staticmethod
        def device(name: str) -> str:
            return name

        @staticmethod
        def tensor(_values: Any) -> FakeTensor:
            return FakeTensor()

    settings = SimpleNamespace(
        camera_source=camera_source,
        vision_model_id="test/vision-model",
        torch_device=lambda _torch: "cpu",
    )
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "asyncio": asyncio,
        "threading": threading,
        "time": SimpleNamespace(sleep=lambda _seconds: None),
        "cv": FakeCV(),
        "Image": FakeImage,
        "torch": FakeTorch,
        "GitProcessor": FakeProcessorClass,
        "GitForCausalLM": FakeModelClass,
        "load_settings": lambda: settings,
        "visionQA_processor": None,
        "visionQA_model": None,
        "_vqa_lock": threading.Lock(),
    }
    exec(compile(module, str(BOT_CORTEX_PATH), "exec"), namespace)
    return namespace, settings.__dict__


def test_headless_vqa_stream_does_not_call_opencv_highgui() -> None:
    namespace, _settings = _vqa_runtime(camera_source=0)
    result = asyncio.run(namespace["answer_visual_question"]("What is shown?"))

    assert result == "I think it is a cat"


def test_empty_and_single_entry_question_banks_return_without_embedding() -> None:
    tree = ast.parse(BOT_CORTEX_PATH.read_text(encoding="utf-8"), filename=str(BOT_CORTEX_PATH))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.AsyncFunctionDef) and item.name == "search_question_in_bank"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "get_embeddings": lambda _value: (_ for _ in ()).throw(
            AssertionError("embeddings should not be needed for fewer than two entries")
        ),
    }
    exec(compile(module, str(BOT_CORTEX_PATH), "exec"), namespace)

    async def exercise() -> None:
        assert await namespace["search_question_in_bank"]({}, "question") == ("", False)
        assert await namespace["search_question_in_bank"]({"one": "answer"}, "question") == (
            "",
            False,
        )

    asyncio.run(exercise())


def test_two_entry_question_bank_preserves_high_confidence_match() -> None:
    tree = ast.parse(BOT_CORTEX_PATH.read_text(encoding="utf-8"), filename=str(BOT_CORTEX_PATH))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.AsyncFunctionDef) and item.name == "search_question_in_bank"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)

    class Vector:
        def __init__(self, values: list[float | int]):
            self.values = values

        def detach(self) -> "Vector":
            return self

        def transpose(self, *_args: Any) -> "Vector":
            return self

        def squeeze(self) -> "Vector":
            return self

        def size(self, _dimension: int) -> int:
            return len(self.values)

        def __getitem__(self, index: int | slice) -> Any:
            if isinstance(index, slice):
                return Vector(self.values[index])
            return Scalar(self.values[index])

        def item(self) -> float:
            return float(self.values[0])

    class Scalar:
        def __init__(self, value: float | int):
            self.value = value

        def item(self) -> float:
            return float(self.value)

        def __index__(self) -> int:
            return int(self.value)

    class FakeFunctional:
        @staticmethod
        def normalize(value: Vector, **_kwargs: Any) -> Vector:
            return value

    class FakeTorch:
        @staticmethod
        def mm(_left: Vector, _right: Vector) -> Vector:
            return Vector([0.95, 0.5])

        @staticmethod
        def sort(value: Vector, **_kwargs: Any) -> tuple[Vector, Vector]:
            pairs = sorted(enumerate(value.values), key=lambda pair: pair[1], reverse=True)
            return Vector([score for _index, score in pairs]), Vector(
                [index for index, _score in pairs]
            )

    embedding_calls = 0

    async def embeddings(_value: Any) -> Vector:
        nonlocal embedding_calls
        embedding_calls += 1
        return Vector([1.0, 0.0])

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "get_embeddings": embeddings,
        "F": FakeFunctional,
        "torch": FakeTorch,
    }
    exec(compile(module, str(BOT_CORTEX_PATH), "exec"), namespace)

    result = asyncio.run(
        namespace["search_question_in_bank"](
            {"first question": "first answer", "second question": "second answer"},
            "question",
        )
    )

    assert result == ("first answer", True)
    assert embedding_calls == 2


def test_vqa_model_loading_runs_off_the_event_loop() -> None:
    entered_loader = threading.Event()
    release_loader = threading.Event()
    event_loop_progress = threading.Event()
    observed_progress_during_load: list[bool] = []

    def load_processor(_model_id: str) -> Any:
        entered_loader.set()
        release_loader.wait()
        observed_progress_during_load.append(event_loop_progress.is_set())
        return Processor()

    class Processor:
        tokenizer = SimpleNamespace(cls_token_id=1)

        def __call__(self, *, images: Any = None, text: str | None = None, **_kwargs: Any):
            if images is not None:
                return SimpleNamespace(
                    pixel_values=SimpleNamespace(to=lambda _device: SimpleNamespace(shape=(1,)))
                )
            return SimpleNamespace(input_ids=[2])

        def batch_decode(self, _outputs: Any, **_kwargs: Any) -> list[str]:
            return ["What is shown?a cat"]

    namespace, _settings = _vqa_runtime(processor_loader=load_processor)
    watchdog = threading.Timer(2, release_loader.set)
    watchdog.daemon = True
    watchdog.start()

    async def mark_progress() -> None:
        await asyncio.to_thread(entered_loader.wait, 1)
        event_loop_progress.set()

    async def exercise() -> str:
        progress_task = asyncio.create_task(mark_progress())
        answer_task = asyncio.create_task(namespace["answer_visual_question"]("What is shown?"))
        started = await asyncio.to_thread(entered_loader.wait, 1)
        assert started, "VQA model loader did not start"
        await asyncio.wait_for(progress_task, timeout=1)
        release_loader.set()
        return await answer_task

    try:
        result = asyncio.run(exercise())
    finally:
        release_loader.set()
        watchdog.cancel()

    assert observed_progress_during_load == [True]
    assert result == "I think it is a cat"


def test_failed_vqa_initialization_leaves_cache_empty_and_retries() -> None:
    counts = {"processor": 0, "model": 0}

    class Processor:
        tokenizer = SimpleNamespace(cls_token_id=1)

        def __call__(self, *, images: Any = None, text: str | None = None, **_kwargs: Any):
            if images is not None:
                return SimpleNamespace(
                    pixel_values=SimpleNamespace(to=lambda _device: SimpleNamespace(shape=(1,)))
                )
            return SimpleNamespace(input_ids=[2])

        def batch_decode(self, _outputs: Any, **_kwargs: Any) -> list[str]:
            return ["What is shown?a cat"]

    class Model:
        def to(self, _device: Any) -> "Model":
            return self

        def generate(self, **_kwargs: Any) -> list[int]:
            return [1]

    def load_processor(_model_id: str) -> Processor:
        counts["processor"] += 1
        return Processor()

    def load_model(_model_id: str) -> Model:
        counts["model"] += 1
        if counts["model"] == 1:
            raise RuntimeError("temporary model load failure")
        return Model()

    namespace, _settings = _vqa_runtime(
        processor_loader=load_processor,
        model_loader=load_model,
    )

    async def exercise() -> tuple[str, str]:
        failed = await namespace["answer_visual_question"]("What is shown?")
        assert namespace["visionQA_processor"] is None
        assert namespace["visionQA_model"] is None
        succeeded = await namespace["answer_visual_question"]("What is shown?")
        return failed, succeeded

    failed, succeeded = asyncio.run(exercise())

    assert "temporary model load failure" in failed
    assert succeeded == "I think it is a cat"
    assert counts == {"processor": 2, "model": 2}
    assert namespace["visionQA_processor"] is not None
    assert namespace["visionQA_model"] is not None


def test_concurrent_cancelled_first_vqa_requests_publish_one_complete_cache() -> None:
    entered_loader = threading.Event()
    release_loader = threading.Event()
    second_lock_attempt = threading.Event()
    counts = {"processor": 0, "model": 0}
    counts_lock = threading.Lock()

    class ProbeLock:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.attempts = 0

        def __enter__(self) -> "ProbeLock":
            with counts_lock:
                self.attempts += 1
                if self.attempts == 2:
                    second_lock_attempt.set()
            self.lock.acquire()
            return self

        def __exit__(self, *_exc: Any) -> None:
            self.lock.release()

    class Processor:
        tokenizer = SimpleNamespace(cls_token_id=1)

        def __call__(self, *, images: Any = None, text: str | None = None, **_kwargs: Any):
            if images is not None:
                return SimpleNamespace(
                    pixel_values=SimpleNamespace(to=lambda _device: SimpleNamespace(shape=(1,)))
                )
            return SimpleNamespace(input_ids=[2])

        def batch_decode(self, _outputs: Any, **_kwargs: Any) -> list[str]:
            return ["What is shown?a cat"]

    class Model:
        def to(self, _device: Any) -> "Model":
            return self

        def generate(self, **_kwargs: Any) -> list[int]:
            return [1]

    def load_processor(_model_id: str) -> Processor:
        with counts_lock:
            counts["processor"] += 1
        entered_loader.set()
        release_loader.wait()
        return Processor()

    def load_model(_model_id: str) -> Model:
        with counts_lock:
            counts["model"] += 1
        return Model()

    namespace, _settings = _vqa_runtime(
        processor_loader=load_processor,
        model_loader=load_model,
    )
    namespace["_vqa_lock"] = ProbeLock()
    watchdog = threading.Timer(2, release_loader.set)
    watchdog.daemon = True
    watchdog.start()

    async def exercise() -> tuple[str, bool]:
        first = asyncio.create_task(namespace["answer_visual_question"]("What is shown?"))
        started = await asyncio.to_thread(entered_loader.wait, 1)
        assert started, "first VQA request did not reach model loading"
        second = asyncio.create_task(namespace["answer_visual_question"]("What is shown?"))
        waiting = await asyncio.to_thread(second_lock_attempt.wait, 1)
        assert waiting, "second VQA request did not wait for the shared lock"
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("the first coroutine should be cancellable")
        assert namespace["visionQA_processor"] is None
        assert namespace["visionQA_model"] is None
        release_loader.set()
        return await second, True

    try:
        result, _completed = asyncio.run(exercise())
    finally:
        release_loader.set()
        watchdog.cancel()

    assert result == "I think it is a cat"
    assert counts == {"processor": 1, "model": 1}
    assert namespace["visionQA_processor"] is not None
    assert namespace["visionQA_model"] is not None
