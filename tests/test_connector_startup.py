from __future__ import annotations

import ast
import asyncio
import sys
import types
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ENTRYPOINTS = ("buddy1.py", "windows/buddy1.py", "windows/buddy2.py")


def _connector_startup_nodes(
    tree: ast.Module,
) -> tuple[list[ast.stmt], set[str], set[str]]:
    import_nodes: list[ast.stmt] = []
    aiohttp_names: set[str] = set()
    connector_names: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "aiohttp":
                    import_nodes.append(node)
                    aiohttp_names.add(imported.asname or "aiohttp")
        elif isinstance(node, ast.ImportFrom) and node.module == "aiohttp":
            import_nodes.append(node)
            for imported in node.names:
                if imported.name == "TCPConnector":
                    connector_names.add(imported.asname or imported.name)

    connector_nodes: list[ast.stmt] = []
    for node in tree.body:
        if node in import_nodes:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            function = child.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "TCPConnector"
                and isinstance(function.value, ast.Name)
                and function.value.id in aiohttp_names
            ) or (isinstance(function, ast.Name) and function.id in connector_names):
                connector_nodes.append(node)
                break

    return import_nodes + connector_nodes, aiohttp_names, connector_names


@pytest.mark.parametrize("relative_path", LEGACY_ENTRYPOINTS)
def test_legacy_startup_does_not_eagerly_construct_aiohttp_connector(
    relative_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute only aiohttp imports and connector setup, excluding hardware imports."""
    path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    startup_nodes, aiohttp_names, connector_names = _connector_startup_nodes(tree)

    constructions: list[object] = []

    def trap_tcp_connector(*args: object, **kwargs: object) -> object:
        constructions.append((args, kwargs))
        asyncio.get_running_loop()
        return object()

    fake_aiohttp = types.ModuleType("aiohttp")
    fake_aiohttp.TCPConnector = trap_tcp_connector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    isolated_startup = ast.Module(body=startup_nodes, type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(isolated_startup, str(path), "exec"), namespace)

    assert not constructions, f"{relative_path} constructed TCPConnector at import time"
    assert not aiohttp_names, f"{relative_path} retains an unused module-level aiohttp import"
    assert not connector_names, f"{relative_path} retains an unused TCPConnector import"
