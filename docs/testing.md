# DeskBuddy testing

## Gate 1 developer checks

Use CPython 3.11 or 3.12 and uv 0.9.26 or newer. The checkout defaults to 3.12,
and CI uses uv 0.9.26. Run these commands from the repository root:

```console
uv sync --locked --group dev
uv lock --check --offline
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync mypy
uv run --no-sync ruff format --check src transport tests
```

To test the other interpreter, set `UV_PYTHON` to `3.11` or `3.12` for the
whole command sequence (for example, `export UV_PYTHON=3.11` in Bash or
`$env:UV_PYTHON = '3.11'` in PowerShell).
See [setup](setup.md) for installation and optional capabilities.

The `CI` workflow runs on pull requests and pushes to `main`. Its check names are
`CPU Gate 1 (Python 3.11)` and `CPU Gate 1 (Python 3.12)`. Both use GitHub-hosted
Ubuntu CPU runners. The install can fetch locked core and dev packages. Pytest
runs offline with no repository secrets, GPU libraries, pretrained model weights,
physical devices, or live services.

Ruff lint and formatting cover Python files in `src/`, `transport/`, and `tests/`.
Mypy discovers production packages from the repository root, currently `src/` and
`transport/`. Its configuration excludes tests, build outputs, Windows vendor
copies, and flat legacy scripts. Tests may still exercise the legacy entry points
with fakes. Those tests provide behavioral evidence within their asserted scope.

## Offline contract tests

Add `test_*.py` modules under `tests/`. Use `tmp_path` for files, deterministic
fakes at model, API, camera, and audio boundaries, and event synchronization for
async tests. Assert the observable result or error for each branch. Reuse
`ephemeral_server_config` for authenticated robot tests on `127.0.0.1` with an
OS-assigned port and a dummy secret. Exercise the Issue #4 length-framed JSON
contract, including authenticated responses and safe timeout handling. These
loopback tests run in the ordinary suite.

Before test modules load, pytest isolates DeskBuddy's settings and known
credential variables, sets Hugging Face offline controls, and masks the installed
Picoh driver so legacy modules use their mock fallback. The Python socket guard
permits loopback and blocks ordinary non-loopback socket calls and lookups in
pytest. It applies to the pytest process. Use offline modes for subprocesses and
fake native I/O boundaries in the tests that exercise them.

The interactive root `test.py` was removed because it called async application
functions without awaiting them. Use the pytest suite for automated verification.

## Optional integration tests

Place tests that need real prerequisites in `tests/optional/` and mark each test
with exactly one of the following markers. The default suite ignores that
directory and deselects any test with an optional marker elsewhere.

| Marker | Required prerequisites | Selective run |
| --- | --- | --- |
| `hardware` | The actual robot, camera, microphone, or speakers needed by the test | `uv run --no-sync pytest --run-optional=hardware` |
| `model_weights` | Locally provisioned weights and a compatible inference runtime | `uv run --no-sync pytest --run-optional=model_weights` |
| `external_service` | A configured live endpoint and any credentials the test requires | `uv run --no-sync pytest --run-optional=external_service` |

Install the relevant optional dependencies alongside `--group dev` in a separate
environment. These commands select only the requested category. The repository
currently has no tests in these categories, so selection reports no tests (exit
code 5). A future test must check prerequisites in a fixture or in the test and
call `pytest.skip()` with an explicit reason, or raise a configuration error,
before initializing any device, model, or service client. Keep module imports safe
for collection. Model runs retain Hugging Face offline controls and must use
already provisioned assets. Live service and hardware runs use the operator's
configured environment.

Qualify real classifier accuracy and GPU memory, event-loop and lookup latency
under Gate 2. Qualify physical movement, real speech latency, dialogue safety,
and live failover under Gate 3 on a configured system. See the
[verification gates](../.agent/rules/deskbuddy-verification-gates.md) for the targets.
