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
uv run --no-sync ruff format --check src transport tests robot_client.py
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

Ruff lint and formatting cover Python files in `src/`, `transport/`, `tests/`,
and the desktop transport adapter `robot_client.py`. Mypy checks `src/`,
`transport/`, and that adapter. Its configuration excludes tests, build outputs,
Windows vendor copies, and other flat legacy scripts. Tests exercise legacy entry
points with fakes and provide behavioral evidence within their asserted scope.

## Offline contract tests

Add `test_*.py` modules under `tests/`. Use `tmp_path` for files, deterministic
fakes at model, API, camera, and audio boundaries, and event synchronization for
async tests. Assert the observable result or error for each branch. Reuse
`ephemeral_server_config` for authenticated robot tests on `127.0.0.1` with an
OS-assigned port and a dummy secret. Exercise the Issue #4 length-framed JSON
contract, including authenticated responses and safe timeout handling. These
loopback tests run in the ordinary suite.

Before test modules load, pytest isolates DeskBuddy's settings and known
credential variables, sets Hugging Face offline controls, and blocks imports of
optional model, cloud, and device runtimes. The legacy robot adapter uses its
mock Picoh fallback; tests that need an SDK or driver can supply an explicit fake.
The Python socket guard permits loopback and blocks ordinary non-loopback
socket calls and lookups in the pytest process. Plugin startup precedes these
guards; use a clean core/dev install. Subprocesses and native I/O need their
own offline or fake boundaries.

`tests/test_pytest_isolation.py` checks the default process profile and proves a
non-loopback lookup is stopped by the socket audit hook. `tmp_path` tests in
`tests/test_legacy_config_contract.py` verify that memory files stay under an
explicit temporary data directory and leave a checkout sentinel unchanged.

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
| `external_service` | A configured local robot service bound to loopback | `uv run --no-sync pytest --run-optional=external_service` |

Install the relevant optional dependencies alongside `--group dev` in a separate
environment. These commands select only the requested category. The external
service template in `tests/optional/test_external_robot_service.py` requires
explicit `DESKBUDDY_TEST_ROBOT_HOST`, `DESKBUDDY_TEST_ROBOT_PORT`, and
`DESKBUDDY_TEST_ROBOT_AUTH_SECRET` values. It accepts only a loopback host and
performs the read-only connectivity check (`simple_test`). Without configuration it skips before
creating a client. Import and collection make no connection. Supply the settings
only for an explicit run:

```powershell
$env:DESKBUDDY_TEST_ROBOT_HOST = '127.0.0.1'
$env:DESKBUDDY_TEST_ROBOT_PORT = '8888'
$env:DESKBUDDY_TEST_ROBOT_AUTH_SECRET = '<local test service secret>'
uv run --no-sync pytest --run-optional=external_service
```

When `--run-optional` is supplied, pytest collects only from
`tests/optional/`; the session rejects explicit paths outside that tree before
collection, including mixed optional and ordinary test paths. Explicitly naming
an optional test without `--run-optional` is also rejected; default root
discovery still skips the optional directory. For the external service category,
the temporary profile,
credential scrubbing, runtime import block, offline flags, and loopback socket
guard remain active. Only the three explicit `DESKBUDDY_TEST_ROBOT_*` variables
are preserved for the external service category. The smoke test accepts
`localhost` by mapping it directly to `127.0.0.1`, or accepts a loopback IP
literal; it does not resolve arbitrary hostnames. A whitespace-only secret fails
configuration validation, while a nonblank secret is passed unchanged.

The `hardware` and `model_weights` categories preserve the operator's original
environment and optional runtime imports so those tests can use provisioned
devices, weights, settings, and credentials. Model weight tests still receive
Hugging Face offline flags.

The hardware and model-weight categories remain templates for future tests.
Their tests must check prerequisites and skip before initializing a device,
model, or service client. Model runs retain Hugging Face offline controls and
must use already provisioned assets.

Qualify real classifier accuracy and GPU memory, event-loop and lookup latency
under Gate 2. Qualify physical movement, real speech latency, dialogue safety,
and live failover under Gate 3 on a configured system. See the
[verification gates](../.agent/rules/deskbuddy-verification-gates.md) for the targets.
