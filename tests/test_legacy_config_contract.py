from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deskbuddy.settings import Settings

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

    assert vqa_functions.get("answer_visual_question") == {"GitProcessor", "GitForCausalLM"}
    assert not any(
        classes for name, classes in vqa_functions.items() if name != "answer_visual_question"
    )
    assert all(
        not _vqa_pretrained_classes(node)
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


@pytest.mark.parametrize(
    ("device", "cuda_available", "gpu_limit", "cpu_limit", "expected"),
    [
        ("cpu", False, "8GiB", "16GiB", {"cpu": "16GiB"}),
        ("auto", True, "4GiB", "32GiB", {0: "4GiB", "cpu": "32GiB"}),
        ("cpu", False, None, None, {"cpu": "40GiB"}),
    ],
)
def test_legacy_memory_map_respects_settings_without_importing_torch(
    tmp_path: Path,
    device: str,
    cuda_available: bool,
    gpu_limit: str | None,
    cpu_limit: str | None,
    expected: dict[int | str, str],
) -> None:
    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return cuda_available

    settings = Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        device=device,
        gpu_memory_limit=gpu_limit,
        cpu_memory_limit=cpu_limit,
    )
    max_memory = _extract_functions(
        BOT_CORTEX_PATH,
        {"_max_memory_with_settings"},
        {
            "get_max_memory": lambda: {0: "7GiB", "cpu": "40GiB"},
            "load_settings": lambda: settings,
            "torch": FakeTorch,
        },
    )["_max_memory_with_settings"]

    assert max_memory() == expected
