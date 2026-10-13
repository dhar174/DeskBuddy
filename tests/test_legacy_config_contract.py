from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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


def _settings_at(data_dir: Path, cache_dir: Path) -> Settings:
    return Settings(data_dir=data_dir, cache_dir=cache_dir)


def test_question_bank_missing_file_is_empty_and_has_no_cwd_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    data_dir = tmp_path / "user-data"
    settings = _settings_at(data_dir, tmp_path / "user-cache")
    load = _extract_functions(
        BOT_CORTEX_PATH,
        {"load_qdict_from_json"},
        {"json": json, "load_settings": lambda: settings},
    )["load_qdict_from_json"]

    assert asyncio.run(load()) == {}
    assert not data_dir.exists()
    assert not (checkout / "question_file.json").exists()


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
    load = _extract_functions(
        BOT_CORTEX_PATH,
        {"load_qdict_from_json"},
        {"json": json, "load_settings": lambda: settings},
    )["load_qdict_from_json"]

    assert asyncio.run(load()) == expected
    assert not (checkout / "question_file.json").exists()


def test_question_bank_save_targets_user_data_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    data_dir = tmp_path / "user-data"
    settings = _settings_at(data_dir, tmp_path / "user-cache")
    save = _extract_functions(
        BOT_CORTEX_PATH,
        {"save_qdict_to_json"},
        {"json": json, "load_settings": lambda: settings},
    )["save_qdict_to_json"]
    qdict = {"where are you?": "right here"}

    asyncio.run(save(qdict))

    assert json.loads((data_dir / "question_file.json").read_text(encoding="utf-8")) == qdict
    assert not (checkout / "question_file.json").exists()


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
