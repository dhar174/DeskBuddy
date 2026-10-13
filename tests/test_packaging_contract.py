from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from dotenv import dotenv_values

from deskbuddy.settings import Settings, load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_CAPABILITIES = {"local", "speech", "vision", "picoh"}
UV_EXECUTABLE = shutil.which("uv")
HEAVY_DISTRIBUTIONS = {
    "torch",
    "transformers",
    "openai-whisper",
    "opencv-python",
    "opencv-python-headless",
    "picoh",
    "deepspeed",
}


def _distribution_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    assert match, f"Could not read dependency name from {requirement!r}"
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def test_core_install_excludes_optional_gpu_hardware_and_legacy_packages() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    core = project["dependencies"]
    optional = project["optional-dependencies"]

    assert OPTIONAL_CAPABILITIES <= optional.keys()
    core_names = {_distribution_name(requirement) for requirement in core}
    assert not (core_names & HEAVY_DISTRIBUTIONS)
    assert "deepspeed" not in core_names


def test_declared_dependency_sources_are_portable_and_publicly_resolvable() -> None:
    document = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = document["project"]
    declared = list(project.get("dependencies", []))
    declared.extend(
        requirement
        for group in project.get("optional-dependencies", {}).values()
        for requirement in group
    )
    declared.extend(
        requirement
        for group in document.get("dependency-groups", {}).values()
        for requirement in group
    )

    for requirement in declared:
        normalized = requirement.replace("\\", "/").lower()
        assert "file://" not in normalized, requirement
        assert "git+ssh://" not in normalized, requirement
        assert not re.search(r"(?:^|\s)-e(?:\s|$)", normalized), requirement
        assert "/home/darf3/" not in normalized, requirement
        assert "c:/users/darf3/" not in normalized, requirement

    for source in _walk_strings(document.get("tool", {}).get("uv", {}).get("sources", {})):
        normalized = source.replace("\\", "/").lower()
        assert "file://" not in normalized, source
        assert "git+ssh://" not in normalized, source
        assert "/home/darf3/" not in normalized, source
        assert "c:/users/darf3/" not in normalized, source


def test_uv_lock_uses_only_pypi_sources_and_matches_the_core_dependency_set() -> None:
    project_document = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"))
    roots = [package for package in lock["package"] if package["name"] == "deskbuddy"]

    assert lock["version"] == 1
    assert len(roots) == 1
    root = roots[0]
    assert root["source"] == {"editable": "."}
    expected_core = {
        _distribution_name(requirement)
        for requirement in project_document["project"]["dependencies"]
    }
    locked_core = {dependency["name"] for dependency in root["dependencies"]}
    assert locked_core == expected_core

    for package in lock["package"]:
        if package["name"] == "deskbuddy":
            continue
        assert package["source"] == {"registry": "https://pypi.org/simple"}, package["name"]
        assert package.get("sdist") or package.get("wheels"), package["name"]

    assert project_document.get("tool", {}).get("uv", {}).get("default-groups", []) == []


def _walk_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk_strings(child)]
    return []


def _run_uv(*arguments: str) -> subprocess.CompletedProcess[str]:
    assert UV_EXECUTABLE is not None
    result = subprocess.run(
        [UV_EXECUTABLE, *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"uv {' '.join(arguments)} exited {result.returncode}\n"
        f"stdout:\n{result.stdout[:4000]}\nstderr:\n{result.stderr[:4000]}"
    )
    return result


def _noncomment_requirements(contents: str) -> list[str]:
    return [
        line.strip()
        for line in contents.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.skipif(UV_EXECUTABLE is None, reason="uv is not installed")
def test_uv_lock_is_current_without_network_access() -> None:
    _run_uv("lock", "--check", "--offline")


@pytest.mark.skipif(UV_EXECUTABLE is None, reason="uv is not installed")
def test_requirements_freeze_matches_locked_core_export_without_installing() -> None:
    exported = _run_uv(
        "export",
        "--locked",
        "--offline",
        "--no-dev",
        "--no-emit-project",
        "--format",
        "requirements-txt",
        "--quiet",
    )
    committed = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert _noncomment_requirements(exported.stdout) == _noncomment_requirements(committed)


def test_production_python_has_no_developer_specific_absolute_paths() -> None:
    runtime_files = [
        *((REPOSITORY_ROOT / "src").rglob("*.py")),
        REPOSITORY_ROOT / "start_bot.py",
        REPOSITORY_ROOT / "BotCortex.py",
        REPOSITORY_ROOT / "buddy1.py",
        REPOSITORY_ROOT / "robot_client.py",
        REPOSITORY_ROOT / "train_h_memory.py",
        *((REPOSITORY_ROOT / "windows").rglob("*.py")),
    ]

    for path in runtime_files:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            normalized = node.value.replace("\\", "/").lower()
            assert "/home/darf3/" not in normalized, f"{path}:{node.lineno}: {node.value}"
            assert "c:/users/darf3/" not in normalized, f"{path}:{node.lineno}: {node.value}"


def test_environment_example_matches_the_settings_schema_and_has_no_credentials(
    tmp_path: Path,
) -> None:
    example_path = REPOSITORY_ROOT / ".env.example"
    example_text = example_path.read_text(encoding="utf-8")
    example = dotenv_values(example_path, interpolate=False)
    required = {
        "DESKBUDDY_MODEL_PROVIDER",
        "DESKBUDDY_ROBOT_HOST",
        "DESKBUDDY_ROBOT_BIND_ADDRESS",
        "DESKBUDDY_ROBOT_PORT",
        "DESKBUDDY_ROBOT_AUTH_SECRET",
        "DESKBUDDY_ROBOT_CONNECT_TIMEOUT_SECONDS",
        "DESKBUDDY_ROBOT_TIMEOUT_SECONDS",
        "DESKBUDDY_CAMERA_SOURCE",
        "DESKBUDDY_DEVICE",
        "DESKBUDDY_GPU_MEMORY_LIMIT",
        "DESKBUDDY_CPU_MEMORY_LIMIT",
        "OPENAI_API_KEY",
    }

    assert required <= example.keys()
    assert "API_KEY=" not in example_text.replace("OPENAI_API_KEY=", "")
    assert example["OPENAI_API_KEY"] in {None, ""}
    assert example["DESKBUDDY_ROBOT_AUTH_SECRET"] in {None, ""}
    normalized = example_text.replace("\\", "/").lower()
    assert "/home/darf3/" not in normalized
    assert "c:/users/darf3/" not in normalized

    config_file = tmp_path / "empty.toml"
    config_file.write_text("[deskbuddy]\n", encoding="utf-8")
    settings = load_settings(config_file=config_file, env_file=example_path, environ={})
    assert settings.model_provider == "disabled"
    assert settings.openai_api_key is None


def test_toml_example_parses_against_the_typed_settings_schema() -> None:
    example = tomllib.loads(
        (REPOSITORY_ROOT / "deskbuddy.example.toml").read_text(encoding="utf-8")
    )

    assert set(example) == {"deskbuddy"}
    settings = Settings.model_validate(example["deskbuddy"])
    assert settings.model_provider == "disabled"
