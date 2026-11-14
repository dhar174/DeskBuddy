from __future__ import annotations

import ast
import asyncio
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deskbuddy import question_bank
from deskbuddy.settings import Settings, SettingsError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOT_CORTEX_PATH = REPOSITORY_ROOT / "BotCortex.py"


def _extract_functions(path: Path, names: set[str], globals_: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }
    assert functions.keys() == names, f"Missing expected functions: {names - functions.keys()}"
    module = ast.Module(body=list(functions.values()), type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"__builtins__": __builtins__, **globals_}
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _settings_at(data_dir: Path, cache_dir: Path, offload_dir: Path | None = None) -> Settings:
    return Settings(data_dir=data_dir, cache_dir=cache_dir, offload_dir=offload_dir)


def _question_bank_adapters(
    settings: Settings,
    *,
    settings_loader: Any | None = None,
) -> dict[str, Any]:
    return _extract_functions(
        BOT_CORTEX_PATH,
        {
            "load_qdict_from_json",
            "save_qdict_to_json",
            "_clear_question_bank_warning",
            "_warn_question_bank_unavailable",
        },
        {
            "asyncio": asyncio,
            "logging": logging,
            "__name__": "BotCortex",
            "question_bank": question_bank,
            "load_settings": settings_loader or (lambda: settings),
            "_question_bank_warning_lock": threading.Lock(),
            "_question_bank_warning_reasons": {},
        },
    )


def test_question_bank_missing_file_is_empty_and_has_no_cwd_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    data_dir = tmp_path / "user-data"
    settings = _settings_at(data_dir, tmp_path / "user-cache")
    load = _question_bank_adapters(settings)["load_qdict_from_json"]

    assert asyncio.run(load()) == {}
    assert not data_dir.exists()
    assert not (checkout / "question_file.json").exists()
    assert not settings.cache_dir.exists()


def test_question_bank_existing_file_is_read_from_user_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    data_dir = tmp_path / "user-data"
    data_dir.mkdir()
    expected = {"hello": "Hi there"}
    (data_dir / "question_file.json").write_text(json.dumps(expected), encoding="utf-8")
    settings = _settings_at(data_dir, tmp_path / "user-cache")
    load = _question_bank_adapters(settings)["load_qdict_from_json"]

    assert asyncio.run(load()) == expected
    assert not (checkout / "question_file.json").exists()


def test_question_bank_save_targets_user_data_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    sentinel = checkout / "question_file.json"
    sentinel.write_text("production data sentinel", encoding="utf-8")
    data_dir = tmp_path / "user-data"
    settings = _settings_at(data_dir, tmp_path / "user-cache")
    save = _question_bank_adapters(settings)["save_qdict_to_json"]
    qdict = {"where are you?": "right here"}

    asyncio.run(save(qdict))

    assert json.loads((data_dir / "question_file.json").read_text(encoding="utf-8")) == qdict
    assert sentinel.read_text(encoding="utf-8") == "production data sentinel"
    assert not settings.cache_dir.exists()
    assert settings.offload_dir is not None
    assert not settings.offload_dir.exists()


@pytest.mark.parametrize("adapter", ["load", "save"])
def test_question_bank_disk_adapters_yield_to_event_loop(
    adapter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_at(
        tmp_path / "user-data", tmp_path / "user-cache", tmp_path / "model-offload"
    )
    adapters = _question_bank_adapters(settings)
    entered = threading.Event()
    release = threading.Event()

    def blocking_bank_operation(*_args: Any) -> dict[str, str] | None:
        entered.set()
        if not release.wait(timeout=1.5):
            raise AssertionError("question-bank I/O blocked the event loop")
        return {} if adapter == "load" else None

    if adapter == "load":
        monkeypatch.setattr(question_bank, "load_question_bank", blocking_bank_operation)
    else:
        monkeypatch.setattr(question_bank, "save_question_bank", blocking_bank_operation)

    async def exercise() -> None:
        loop = asyncio.get_running_loop()

        def release_from_another_thread() -> None:
            if entered.wait(timeout=2):
                loop.call_soon_threadsafe(release.set)

        release_thread = threading.Thread(target=release_from_another_thread, daemon=True)
        release_thread.start()
        try:
            if adapter == "load":
                result = await asyncio.wait_for(adapters["load_qdict_from_json"](), timeout=2)
                assert result == {}
            else:
                await asyncio.wait_for(
                    adapters["save_qdict_to_json"]({"hello": "there"}), timeout=2
                )
        finally:
            release.set()
            release_thread.join(timeout=1)
        assert entered.is_set()

    asyncio.run(exercise())


def test_question_bank_settings_error_still_propagates(tmp_path: Path) -> None:
    settings = _settings_at(tmp_path / "data", tmp_path / "cache")

    def broken_settings_loader() -> Settings:
        raise SettingsError("invalid local settings")

    load = _question_bank_adapters(settings, settings_loader=broken_settings_loader)[
        "load_qdict_from_json"
    ]
    with pytest.raises(SettingsError, match="invalid local settings"):
        asyncio.run(load())


def test_question_bank_warning_latch_resets_after_successful_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings_at(tmp_path / "data", tmp_path / "cache")
    adapters = _question_bank_adapters(settings)
    warn = adapters["_warn_question_bank_unavailable"]
    bank = settings.data_path("question_file.json")

    with caplog.at_level(logging.WARNING, logger="BotCortex"):
        warn(bank, "malformed-file")
        warn(bank, "malformed-file")
        warn(bank, "PermissionError")
        assert asyncio.run(adapters["load_qdict_from_json"]()) == {}
        warn(bank, "PermissionError")

    warnings = [record for record in caplog.records if record.name == "BotCortex"]
    assert [record.getMessage().split("(")[-1].split(")")[0] for record in warnings] == [
        "malformed-file",
        "PermissionError",
        "PermissionError",
    ]


@pytest.mark.parametrize("failure", ["malformed", "replace"])
def test_question_bank_save_adapter_preserves_existing_file_on_error(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "user-data"
    data_dir.mkdir()
    bank = data_dir / "question_file.json"
    original = b'{"existing question":"existing answer"}'
    if failure == "malformed":
        original = b'{"damaged question":'
    bank.write_bytes(original)
    settings = _settings_at(data_dir, tmp_path / "user-cache")
    save = _question_bank_adapters(settings)["save_qdict_to_json"]

    if failure == "replace":
        real_os = question_bank.os

        def failed_replace(*_args: Any) -> None:
            raise OSError("replacement failed")

        monkeypatch.setattr(
            question_bank,
            "os",
            SimpleNamespace(fsync=real_os.fsync, replace=failed_replace),
        )
        expected_error = OSError
    else:
        expected_error = question_bank.QuestionBankDataError

    with pytest.raises(expected_error):
        asyncio.run(save({"new question": "new answer"}))

    assert bank.read_bytes() == original
    assert list(data_dir.glob(".question_file.json.*.tmp")) == []


@pytest.mark.parametrize(
    ("function_name", "filename", "initial", "new_content", "expected"),
    [
        ("save_history_verbatim", "history_verbatim.txt", "old", "new", "new"),
        ("save_history_summary", "history_summary.txt", "old", "new", "oldnew"),
    ],
)
def test_history_persistence_uses_user_data_not_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    filename: str,
    initial: str,
    new_content: str,
    expected: str,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    data_dir = tmp_path / "user-data"
    data_dir.mkdir()
    (data_dir / filename).write_text(initial, encoding="utf-8")
    settings = _settings_at(data_dir, tmp_path / "user-cache")
    save = _extract_functions(
        BOT_CORTEX_PATH,
        {function_name},
        {"load_settings": lambda: settings},
    )[function_name]

    asyncio.run(save(new_content))

    assert (data_dir / filename).read_text(encoding="utf-8") == expected
    assert not (checkout / filename).exists()


def test_missing_camera_returns_message_without_touching_vision_dependencies() -> None:
    class ForbiddenRuntime:
        def __getattr__(self, name: str) -> Any:
            pytest.fail(f"camera/vision runtime accessed without a source: {name}")

    forbidden = ForbiddenRuntime()
    answer = _extract_functions(
        BOT_CORTEX_PATH,
        {"answer_visual_question"},
        {
            "load_settings": lambda: SimpleNamespace(camera_source=None),
            "cv": forbidden,
            "torch": forbidden,
            "GitProcessor": forbidden,
            "GitForCausalLM": forbidden,
        },
    )["answer_visual_question"]

    result = asyncio.run(answer("What is in the picture?"))

    assert result == "No camera or image source configured. Set DESKBUDDY_CAMERA_SOURCE."


def _vqa_pretrained_classes(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr != "from_pretrained" or not isinstance(child.func.value, ast.Name):
            continue
        if child.func.value.id in {"GitProcessor", "GitForCausalLM"}:
            found.add(child.func.value.id)
    return found


def test_vqa_weights_are_constructed_only_inside_on_demand_function() -> None:
    tree = ast.parse(BOT_CORTEX_PATH.read_text(encoding="utf-8"), filename=str(BOT_CORTEX_PATH))
    vqa_functions = {
        node.name: _vqa_pretrained_classes(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert vqa_functions.get("_answer_visual_question_sync") == {
        "GitProcessor",
        "GitForCausalLM",
    }
    assert not any(
        classes for name, classes in vqa_functions.items() if name != "_answer_visual_question_sync"
    )
    assert all(
        not _vqa_pretrained_classes(node)
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _memory_test_function(
    settings: Settings,
    torch_module: Any,
    available: Any,
    gpu_available: Any = 8 * 1024**3,
):
    class FakePsutil:
        @staticmethod
        def virtual_memory() -> SimpleNamespace:
            return SimpleNamespace(available=available)

    functions = _extract_functions(
        BOT_CORTEX_PATH,
        {"_max_memory_with_settings", "_memory_limit_bytes"},
        {
            "SettingsError": SettingsError,
            "get_max_memory": lambda: {0: gpu_available, "cpu": "40GiB"},
            "load_settings": lambda: settings,
            "psutil": FakePsutil,
            "torch": torch_module,
        },
    )
    return functions["_max_memory_with_settings"]


@pytest.mark.parametrize(
    ("total_gib", "free_gib", "expected_gib"),
    [
        (4, 4, 3.2),
        (8, 8, 6.4),
        (24, 24, 19.2),
        (8, 2, 0.4),
        (8, 1, 0),
    ],
)
def test_legacy_gpu_budget_reserves_twenty_percent_of_vram(
    tmp_path: Path, total_gib: int, free_gib: int, expected_gib: float
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(device: int) -> SimpleNamespace:
            assert device == 0
            return SimpleNamespace(total_memory=total_gib * 1024**3)

    class FakeTorch:
        cuda = FakeCuda

    settings = Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        device="auto",
    )
    budget = _memory_test_function(
        settings,
        FakeTorch,
        40 * 1024**3,
        gpu_available=free_gib * 1024**3,
    )

    assert budget() == {0: int(expected_gib * 1024**3), "cpu": "40GiB"}


def test_explicit_gpu_budget_within_available_headroom_is_preserved(tmp_path: Path) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(device: int) -> SimpleNamespace:
            assert device == 0
            return SimpleNamespace(total_memory=8 * 1024**3)

    class FakeTorch:
        cuda = FakeCuda

    settings = Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        device="cuda",
        gpu_memory_limit="4GiB",
    )
    budget = _memory_test_function(settings, FakeTorch, 40 * 1024**3)

    assert budget() == {0: "4GiB", "cpu": "40GiB"}


def test_explicit_gpu_budget_above_free_memory_after_reserve_fails(tmp_path: Path) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(device: int) -> SimpleNamespace:
            assert device == 0
            return SimpleNamespace(total_memory=8 * 1024**3)

    class FakeTorch:
        cuda = FakeCuda

    settings = Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        device="cuda",
        gpu_memory_limit="7GiB",
    )
    budget = _memory_test_function(settings, FakeTorch, 40 * 1024**3)

    with pytest.raises(SettingsError, match="reserving 20%"):
        budget()


def test_explicit_cpu_budget_does_not_touch_cuda_or_accelerate(tmp_path: Path) -> None:
    class ForbiddenTorch:
        def __getattr__(self, name: str) -> Any:
            pytest.fail(f"explicit CPU mode accessed torch.{name}")

    settings = Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        device="cpu",
        gpu_memory_limit="8GiB",
        cpu_memory_limit="16GiB",
    )

    class FakePsutil:
        @staticmethod
        def virtual_memory() -> SimpleNamespace:
            return SimpleNamespace(available=40 * 1024**3)

    budget = _extract_functions(
        BOT_CORTEX_PATH,
        {"_max_memory_with_settings", "_memory_limit_bytes"},
        {
            "SettingsError": SettingsError,
            "get_max_memory": lambda: pytest.fail("CPU mode queried Accelerate"),
            "load_settings": lambda: settings,
            "psutil": FakePsutil,
            "torch": ForbiddenTorch(),
        },
    )["_max_memory_with_settings"]

    assert budget() == {"cpu": "16GiB"}
