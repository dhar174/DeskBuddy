# Packaging and configuration baseline

## Python and dependency decisions

The installable baseline supports CPython 3.11 and 3.12. Python 3.11 supplies
stdlib `tomllib`; 3.12 is the checkout default. We do not infer 3.14 compatibility
from REPORT_2026_UPDATE.md. Old Torch 1.13 and SciPy 1.7 wheels do not cover the
chosen range, and the old Torch/torchvision freeze is internally inconsistent.

The optional local compatibility set uses Torch 2.4.1 (Python 3.11/3.12 wheels),
Transformers 4.36.2 (GPT-J/GIT APIs and Python 3.12 tokenizer wheels) and Accelerate
0.25.x. NumPy stays below 2 for the retained Torch generation. No new model or
conversation logic is introduced. These pins establish a resolvable optional
dependency set; GPU inference, custom Accelerate fork equivalence and the old
unversioned local DeepSpeed checkout are not certified by core tests.

The existing OpenAI calls require SDK 0.x. `cloud` therefore deliberately uses
0.28.1 until the cloud integration issue migrates the API. This package and SDK
can import without loading local model weights; the legacy GUI's cloud branch
still loads local classifiers and embeddings. No new provider is claimed.
For a weights-free API-only diagnostic use the packaged `deskbuddy --prompt`
path after installing the cloud extra and selecting OpenAI in settings. That
explicit command sends one user message to the configured model using the existing
SDK, with a 30-second request timeout. It never imports BotCortex, local classifiers
or model runtimes. It is separate from the preserved legacy conversation flow;
headless validation without `--prompt` remains entirely offline.
Configuration/input errors return exit code 2. Missing cloud capabilities or
OpenAI authentication, network, request and malformed-response failures return
exit code 1 with a cloud diagnostic message; provider error text, malformed
response contents and credential values are redacted.

Official compatibility references: [PyTorch releases](https://github.com/pytorch/pytorch/blob/main/RELEASE.md),
[compatible Torch releases](https://pytorch.org/get-started/previous-versions/),
[Whisper metadata](https://github.com/openai/whisper/blob/main/pyproject.toml),
[Picoh distribution](https://pypi.org/project/picoh/),
[OpenAI SDK migration](https://github.com/openai/openai-python/discussions/1495).

## Lock workflow and wheel installation

`uv sync --locked --no-editable` installs core and refuses a stale lock.
Source-file cache keys ensure subsequent non-editable syncs rebuild the package
when its Python sources change.
The supported uv minimum is 0.9.26. `uv sync` is also a one-command developer
bootstrap but installs the project editable; an editable install is not required.
Developer tools are explicitly enabled by `--group dev`, including pytest-asyncio
for the transport suite's coroutine tests. pytest-asyncio 1.4 requires pytest 8.4
or newer; the dev group retains pytest below 9 and the lock selects compatible
versions. Async tests use automatic discovery and function-scoped fixture loops.
uv resolves optional extras into one universal lock without installing them during
core synchronization.

To intentionally change dependencies, edit pyproject.toml, run `uv lock`, review
the lock diff and run `uv lock --check`. Regenerate the compatibility export with:

```console
uv export --locked --no-dev --no-emit-project --format requirements-txt --output-file requirements.txt
```

That export contains core third-party dependencies only. It does not install the
DeskBuddy package; use the documented uv sync or install a wheel. Build with
`uv build`, then install `dist/deskbuddy-0.1.0-py3-none-any.whl` in an independent
environment to test the ordinary package boundary. No root legacy script is
bundled into the wheel.

## Configuration API and path resolution

```python
from deskbuddy import load_settings

settings = load_settings(overrides={"robot_host": "127.0.0.1"})
# Immutable settings; no shared mutable singleton.
```

Sources, highest priority first: function `overrides`, process environment,
explicit dotenv, TOML, defaults. Function file arguments override environment
file selectors. Environment variable names are `DESKBUDDY_<UPPERCASE_FIELD>`;
`openai_api_key` alone maps to `OPENAI_API_KEY`. Secrets use SecretStr and are
excluded from bootstrap output. A missing explicit config/env file is an error;
an absent default TOML file is allowed. TOML requires a single `[deskbuddy]` table
and unknown fields are rejected. Dotenv expansion is disabled to avoid surprising
interpolation of secrets. Examples are `.env.example` and `deskbuddy.example.toml`.

| OS | Default data | Default cache | Default config file |
| --- | --- | --- | --- |
| Windows | `%LOCALAPPDATA%/DeskBuddy` | `%LOCALAPPDATA%/DeskBuddy/Cache` | `%LOCALAPPDATA%/DeskBuddy/deskbuddy.toml` |
| Linux | `$XDG_DATA_HOME/DeskBuddy` or `~/.local/share/DeskBuddy` | `$XDG_CACHE_HOME/DeskBuddy` or `~/.cache/DeskBuddy` | `$XDG_CONFIG_HOME/DeskBuddy/deskbuddy.toml` or `~/.config/DeskBuddy/deskbuddy.toml` |
| macOS | `~/Library/Application Support/DeskBuddy` | `~/Library/Caches/DeskBuddy` | `~/Library/Application Support/DeskBuddy/deskbuddy.toml` |

Relative data_dir is appended to the default OS data directory. Other relative
path fields and image-file camera sources resolve under data_dir, independent of
the working directory or config file location. Default offload_dir is
cache_dir/offload. A stream URL or nonnegative camera index requires no device
access during validation. Explicit checkpoint/tokenizer directories and image
files must exist. Directory creation occurs only at explicit legacy runtime
startup/writes through `prepare_directories()`.
Camera stream URLs require a scheme, hostname and valid port, when supplied;
malformed URLs are rejected by bootstrap without contacting the camera.

Move existing `question_file.json`, `history_verbatim.txt` and `history_summary.txt`
from the old checkout to the configured data_dir yourself before starting the
legacy application. No automatic migration or overwrite is performed. Large mode
requires `DESKBUDDY_CHECKPOINT_DIR` and `DESKBUDDY_TOKENIZER_DIR`; neither defaults
to a checkout directory. Medium mode's saved Accelerate state also requires an
explicit checkpoint_dir. Auxiliary pretrained model identifiers now have settings
fields too, retaining their existing defaults. Persona, sampling and classifier
behavior remain unchanged. Install a spaCy model matching the selected spaCy
version separately (for the default spaCy 3.7.x, use `en_core_web_trf` 3.7.x), or
configure `spacy_model_id` to a compatible locally installed model. The historical
3.4.1 model wheel is not suitable for the new 3.7.x library.
An empty or one-entry question bank skips semantic search and returns the existing
language-model fallback signal; it never tries to embed an empty collection or
compare two missing candidates.

`device=auto` keeps CUDA-if-available/CPU fallback when the legacy runtime asks
for a device; `cpu`, `cuda`, `mps` are explicit preferences. Unsupported hardware
is rejected when the optional runtime resolves it, without importing Torch for
bootstrap. GPU/CPU budgets accept `auto` or positive integer MiB/GiB values.
Automatic CUDA placement limits reserve 20% of the detected device-0 total VRAM
from its currently free memory. A fully free 8 GiB card therefore permits at most
6.4 GiB of placement, with a lower limit when other allocations are present;
larger and smaller cards use the same headroom rule. Explicit GPU limits that
exceed the remaining safe budget fail when the optional runtime resolves hardware.
CPU-only mapping reads available RAM without probing CUDA. These limits replace
the workstation's fixed 7GiB/48GiB maps; unspecified CPU limits remain automatic.
They are placement guards, not a claim of measured peak inference headroom or a
DeepSpeed runtime rewrite.

## Legacy capabilities and boundaries

The root `start_bot.py --headless` dispatches to the packaged CLI before legacy
imports. The ordinary GUI remains a source-only compatibility entry point and
requires a working Tk environment and combined optional extras. A Linux legacy
model environment can resolve with `--extra local --extra legacy --extra cloud
--extra vision --extra speech`; it still needs model assets, spaCy's matching
`en_core_web_trf` model, distributed-launch configuration, native compilers/CUDA
where used, FFmpeg and PortAudio. Installing those components and qualifying
actual inference is separate from the small CPU-only baseline. Do not run
`--all-extras` as a general bootstrap instruction.

DeepSpeed is Linux-only and experimental because the old local checkout's version
is unknown. Its source build needs special handling documented in pyproject.toml;
`tool.uv.extra-build-variables` sets `DS_BUILD_OPS=0` to defer native op compilation
to the optional runtime instead of requiring Torch/CUDA merely to resolve the lock;
this does not pull Torch into core.

### Legacy CUDA/NCCL profiles

The retained `BotCortex.load_all_models()` dispatcher calls
`deepspeed.init_distributed("nccl")` for **Large mode with either provider** and
**Medium mode with OpenAI**. Those profiles require Linux, an NVIDIA CUDA device,
a PyTorch build with NCCL support and a configured distributed launcher. They
cannot honor `device=cpu`, `device=mps`, or `device=auto` when it resolves to CPU.
The settings schema accepts those device preferences for other capabilities; a
successful headless validation does not qualify these legacy profiles. Local
Medium selects BlenderBot without this NCCL initialization, but the legacy GUI
still imports the combined legacy dependencies and remains unqualified here.

Model-runtime migration (#8) must replace this NCCL dependency or add a
profile-specific capability guard before initialization/model loading. This is a
documented legacy limitation, not a change to model routing in the packaging
baseline. The packaged headless command and one-shot cloud diagnostic never enter
these profiles and need no CUDA/NCCL installation.

Optional local/vision/speech libraries do not initialize merely because `deskbuddy`
is imported. BotCortex's VQA weights now load
only when a configured vision question is requested. Capture, model loading and
inference run in a worker thread; a thread lock serializes the shared VQA model,
including first-load initialization. Failed loads publish no partial cache and
may be retried. Camera captures are released without OpenCV GUI cleanup, so the
headless OpenCV extra remains usable. The research-only
`dialogue_management.py`/`train_h_memory.py` are not package startup paths.
Those historical research scripts intentionally retain working-directory-based
`training_data.json` input and `trained_model.pth` output. Run them from a dedicated
training directory; they are not part of the portable production bootstrap.

The published Picoh package has an unconditional comtypes dependency and no
useful Requires-Python metadata; the `picoh` extra is therefore Windows-only until
vendor support on other OSes is qualified. Bootstrap remains portable without it.
The bundled `windows/picoh.py` is a historical custom vendor copy with import-time
device/data behavior and missing adjacent resources, not the installed package.
The vendor's `picohData` paths remain vendor-controlled; no vendor rewrite or TTS
change is attempted. `windows/buddyreqs.txt` is retired; use the optional extras.

`activateBuddy.ps1 -Headless` resolves this checkout from PSScriptRoot and runs the
bootstrap; its default legacy hardware mode uses the installed speech/Picoh extras.
It no longer activates a machine-specific virtualenv or changes to a personal path.

## Issue #4 integration contract

Robot constants in robot_client.py/buddy1.py and Windows copies read settings;
all transport operations remain owned by #4. When merging #4, retain these imports
and use `load_settings()` (or an injected Settings instance) for `robot_host`,
`robot_bind_address`, `robot_port`, `robot_auth_secret`,
`robot_connect_timeout_seconds`, and `robot_timeout_seconds`. Secret values are
accessed explicitly via `.get_secret_value()` after checking for None. #4 must
enforce missing-secret rules, authenticated transport and timeouts. #5 does not
claim that merely configuring a secret secures the old protocol.

PR #28 must adapt its transport configuration to this canonical Settings API when
rebased onto this baseline. Keep the following names and source values:

| Settings field | Transport use |
| --- | --- |
| `robot_host` | Client target address |
| `robot_bind_address` | Server bind interface, independent of the client target |
| `robot_port` | Client/server TCP port |
| `robot_auth_secret` | Authentication secret; require a nonempty value before transport startup |
| `robot_connect_timeout_seconds` | Connection establishment deadline |
| `robot_timeout_seconds` | Request/I/O deadline policy |

`DESKBUDDY_ROBOT_AUTH_SECRET`, `DESKBUDDY_ROBOT_CONNECT_TIMEOUT_SECONDS`, and
`DESKBUDDY_ROBOT_TIMEOUT_SECONDS` are the corresponding environment names.
Replace PR #28's separate `DESKBUDDY_ROBOT_SECRET` and connect/read/write/request
timeout environment parser with an adapter consuming Settings. Internal transport
configuration objects may remain, but must not read a second set of environment
variables or silently fall back on malformed values. If separate phase deadlines
remain necessary, derive them explicitly from the validated request policy; add
any additional user-facing settings here with validation and examples. The current
5-second default is not a promise that long speech or wait commands will finish
within it; PR #28 must verify timeout policy for those commands during integration.

Merge PR #28's `pytest.ini` options into `[tool.pytest.ini_options]` and remove the
redundant file so pytest does not shadow this configuration. The locked dev group
includes pytest-asyncio; run the combined transport and baseline tests after rebase.

Run `uv run --no-sync ruff check .` and `uv run --no-sync mypy` for production
checks. Ruff includes `src/`, `tests/` and the future top-level `transport/`.
Mypy discovers packages from the repository root, including `transport/` as soon
as it exists, while excluding tests, generated build artifacts, Windows vendor
copies and flat legacy scripts.
Keep the repository root a non-package (no root `__init__.py`); production packages
belong under `src/` or in the temporary `transport/` boundary. PR #28 must fix any
lint/type errors exposed by these checks rather than disabling the transport scope.

## Larynx Docker limitation

`larynx.Dockerfile` is retained byte-for-byte. It builds a Debian `.deb` under
Bullseye using dpkg/buildx and copies it into Ubuntu; it is not a DeskBuddy core
Python image. `download/`, `local/`, `licenses/`, `larynx/`, `debian/` and LICENSE
from the upstream/custom Larynx build context are absent here. Its requirements.txt
must be Larynx's own requirements, not DeskBuddy's new core export. Building it
from this repository root is not supported. Do not replace its base with a plain
Python image or claim it has been modernized/tested. TTS restoration belongs to #11.
