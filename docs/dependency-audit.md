# Historical dependency and configuration audit

This inventory preserves the dependency evidence from `requirements.txt` at source revision
`f921272ccc4096852e58560d3d3dddb1780c79bd` (2026-09-17). It contains **262 nonblank,
noncomment requirement entries** in original order. The source also had one explanatory comment
immediately before the local Triton editable entry; that annotation is retained in that row's
decision. The repository's current package metadata is an incremental baseline, not a claim that
every legacy script is installable or supported on every OS.

## Audit method and decision rules

All **12 tracked Python files** at the baseline revision were parsed with Python's AST; no syntax
errors were found. The scan records active import statements, then source references and entry
points were inspected for imported aliases, conditional imports, model loading, configuration,
and startup behavior. A missing direct import is **not** treated as proof that a package is unused:
rows called transitive/unimported are excluded as standalone pins, while a selected parent
dependency remains responsible for installing its own resolver closure. Direct imports that were
only unused aliases or commented examples are called out separately.

The old file is a workstation-wide freeze, not an application dependency definition. It mixes
app imports, transitive closure, Linux distribution packages, docs and development tools, training
experiments, CUDA builds, downloaded model artifacts, and editable/Git installs. The baseline
source has no genuinely lightweight app entry point: `start_bot.py` eagerly imports Whisper,
Tkinter, SciPy, `helpers`, and `BotCortex`, then loads models and waits on the robot server.
`BotCortex.py` imports DeepSpeed/Transformers and calls `deepspeed.init_distributed("nccl")`
at module scope. The new `deskbuddy` bootstrap is a separate configuration-only path; it does
not import these old modules, open hardware, download models, or make cloud requests.

## Compatibility and package decisions

- **Core Python:** `>=3.11,<3.13` (Python 3.11 and 3.12). This range applies to the small
  package/configuration baseline, not to all legacy hardware stacks. PyTorch 2.4.1 publishes
  CPython 3.12 wheels and its official install matrix covers CPU-only and accelerator-specific
  distributions; the PyTorch 2.4 release notes explicitly call out Python 3.12 support for
  `torch.compile`. Transformers 4.36.2 documents Python 3.8+ and keeps the legacy GPT-J APIs
  used in this checkout; tokenizers 0.15.2 publishes CPython 3.12 wheels. These facts make 3.12
  a defensible core ceiling, while not constituting a compatibility certification for every old
  import or model. The 4.36.2 metadata classifiers stop at 3.10, so keep the local-model extra
  isolated and test its imports on each claimed platform. [PyTorch 2.4.1 install matrix](https://pytorch.org/get-started/previous-versions/),
  [PyTorch 2.4 release notes](https://pytorch.org/blog/pytorch2-4/),
  [Transformers 4.36.2 package notes](https://pypi.org/project/transformers/4.36.2/),
  [tokenizers 0.15.2 wheel metadata](https://pypi.org/project/tokenizers/0.15.2/).

- **Legacy local model APIs:** use public `torch==2.4.1`, `transformers==4.36.2`,
  `accelerate>=0.25,<0.26` (the 0.25.x line) and SentencePiece only in the optional local
  environment. This retains the old APIs while separating them from core. The public Accelerate
  0.25.0 release requires Python 3.8+; the former DeskBuddy custom Accelerate fork was an SSH
  editable and must not return. [Accelerate 0.25.0](https://pypi.org/project/accelerate/0.25.0/).

- **DeepSpeed:** replace the developer-local file reference with the public `deepspeed==0.15.4`
  only in the Linux-gated experimental legacy extra. Upstream requires PyTorch before install and
  builds C++/CUDA/ROCm ops, normally JIT, using compiler tooling. Upstream's Windows support is
  partial and limited to building an inference-mode wheel; this repository's old code instead
  calls NCCL distributed initialization and its DeepSpeed inference/offload APIs. The marker is a
  DeskBuddy limitation, not a claim that upstream DeepSpeed is universally Linux-only. Do not
  describe this extra as GPU-certified. [DeepSpeed 0.15.4 release](https://pypi.org/project/deepspeed/0.15.4/).

- **Cloud API:** keep the old module-level OpenAI `ChatCompletion` API in optional `cloud` at
  `openai==0.28.1` during this foundation issue; it supports `ChatCompletion.create/acreate`
  and documents `OPENAI_API_KEY`. Never restore generic `API_KEY`. A future API modernization
  can upgrade the SDK separately. [OpenAI Python SDK 0.28.1](https://pypi.org/project/openai/0.28.1/).

- **Optional hardware/media:** local inference, speech, vision, and Picoh/vendor packages remain
  separate opt-in groups. The Picoh distribution has a universal wheel on PyPI, but this checkout's
  vendor code imports Windows COM/audio modules (`comtypes`, `winsound`) and should be treated as
  Windows-only. Keep Picoh out of headless installs. Use `picoh==1.276` only in that gated
  capability. For `playsound`, use 1.2.2 in Picoh only: its published release includes a universal
  wheel, while historical 1.3.0 is source-only; retaining 1.2.2 avoids the known clean-bootstrap
  build failure for the 1.3.0 source distribution.
  [Picoh 1.276](https://pypi.org/project/picoh/1.276/),
  [playsound 1.2.2](https://pypi.org/project/playsound/1.2.2/),
  [playsound 1.3.0 files](https://pypi.org/project/playsound/1.3.0/).

- **uv caveat:** `uv.lock` provides a universal resolution across declared platforms and Python
  versions, but optional extras are not installed by default. Use `uv sync --extra cloud`,
  `--extra local`, `--extra speech`, `--extra vision`, `--extra picoh`, or `--extra legacy` only
  when that capability is wanted. The project sets `default-groups = []`, so development tools
  are selected explicitly with `uv sync --group dev`.
  PyTorch CPU/CUDA/ROCm builds use different indexes; selecting an accelerator index and matching
  hardware runtime remains an install decision, not something the generic lockfile can certify.
  [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/),
  [uv universal resolution](https://docs.astral.sh/uv/concepts/resolution/),
  [uv PyTorch index configuration](https://docs.astral.sh/uv/concepts/projects/dependencies/).

## Startup, hardcoded settings, and follow-up contract

Baseline inspection found these portability boundaries:

- `BotCortex.py` reads generic `API_KEY`, initializes DeepSpeed/NCCL at import, pins device
  budgets to GPU 0 = `7GiB` and CPU = `48GiB`, repeats those limits in several paths, uses
  `/home/darf3/buddy/offload` / `/home/darf3/buddy/main_model/offload`, refers to
  `./checkpoint` and model IDs inline, and opens camera
  `http://192.168.0.191:56000/mjpeg`. The offload path and VRAM/RAM ceilings were specific to one
  workstation.
- `buddy1.py` binds `0.0.0.0:8888`; `robot_client.py` connects to `172.24.160.1:8888`.
  `windows/buddy1.py` and `windows/buddy2.py` also embed the server bind/port. This issue
  establishes a typed settings contract for robot host, bind address, port, auth-secret reference,
  and timeouts; it intentionally does not change framing or wire those fields deeply into transport
  code owned concurrently by Issue #4. **Issue #4 integration:** have its final transport edit read
  the shared settings values in `robot_client.py` and `buddy1.py`; preserve its protocol work.
- `windows/picoh.py` uses repository-working-directory-relative `picohData/` files and creates
  vendor data there. The PowerShell `activateBuddy.ps1` changes directory to
  `C:\Users\darf3\Documents\OhBot\buddy`, activates `buddy_venv`, and starts the legacy
  `buddy1.py`. Those are historical launcher/data expectations rather than safe application
  defaults; the new settings contract uses OS user data/cache/config directories.
- Direct imports absent from the old freeze include `openai`, `cv2` (OpenCV),
  `pyaudio`, and optional guarded `ptoled`; package metadata must add the first three only in
  cloud/vision/speech extras as appropriate. Tkinter and `wave` are standard-library/platform
  components. The old `Wave==0.0.2` requirement is unrelated to stdlib `wave`.
- Imports of `bitsandbytes as bnb`, `pandas as pd`, and `torchvision.transforms as T` have no
  live references in the baseline source (only imports or commented examples). The parent Issue #5
  change removes those unused imports instead of installing the heavy distributions.

The project core contains only settings/bootstrap dependencies. Cloud API, local model runtime,
speech, vision, Picoh, and legacy model/training dependencies must stay optional; absence of an
optional device or credential must not prevent the headless configuration check. This does not
modernize the old GUI/orchestrator/model runtime.

## Larynx Docker packaging boundary

`larynx.Dockerfile` is a separate legacy Debian package builder: it starts from Debian Bullseye,
creates a Python virtualenv, installs dependencies, builds a `.deb`, then copies the package into
an Ubuntu runtime image. Preserve those Debian/dpkg assumptions; TTS modernization belongs to
Issue #11. The root build context required by its `COPY` lines is absent in the audited checkout:
`download/`, `local/`, `licenses/`, `larynx/`, `debian/control.in`, and
`debian/larynx*`. The only tracked `larynx-master` entry is `.dockerignore`, so the Dockerfile
cannot be built from this repository snapshot without restoring its separate upstream build
context. This is a known limitation, not a reason to replace the base image speculatively.

## Historical requirements inventory

Categories and dispositions describe the source snapshot, not a blanket proof that a feature is
unneeded. Transitive/unimported rows are deliberately cautious: they are not direct pins for the
new application metadata, but absence of an import alone is not evidence a subsystem is unused.

| # | Original requirement from `requirements.txt` | Audit category | Decision |
|---:|---|---|---|
| 001 | `absl-py==1.4.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 002 | `-e git+ssh://git@github.com/dhar174/accelerate.git@bfe1e52b44491530cc56a5e9ea1ca28441c87c16#egg=accelerate` | Private editable fork — legacy local LLM | Replace the SSH/editable fork with public `accelerate>=0.25,<0.26` in the opt-in `local` extra; do not require SSH access. |
| 003 | `aiohttp==3.8.3` | Direct — robot/API networking | Keep only in the optional interactive/speech runtime set that needs the server/client; exclude from core/headless. |
| 004 | `aiosignal==1.3.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 005 | `alabaster==0.7.13` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 006 | `apex==0.1` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 007 | `appdirs==1.4.4` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 008 | `asttokens==2.2.1` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 009 | `async-timeout==4.0.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 010 | `attrs==22.2.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 011 | `autopep8==2.0.1` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 012 | `Babel==2.11.0` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 013 | `backcall==0.2.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 014 | `bitsandbytes==0.36.0.post2` | Unused import plus legacy CUDA quantization dependency | Remove the dead import and its standalone requirement; this is based on a source-reference check, not merely lack of import. |
| 015 | `bitsandbytes-cuda117==0.30.1` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 016 | `black==22.12.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 017 | `blinker==1.4` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 018 | `blis==0.7.9` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 019 | `cachetools==5.3.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 020 | `catalogue==2.0.8` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 021 | `certifi==2022.12.7` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 022 | `cffi==1.15.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 023 | `cfgv==3.3.1` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 024 | `charset-normalizer==2.1.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 025 | `clang-format==15.0.7` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 026 | `click==8.1.3` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 027 | `cmake==3.25.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 028 | `colorama==0.4.6` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 029 | `coloredlogs==15.0.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 030 | `colorful==0.5.5` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 031 | `commonmark==0.9.1` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 032 | `comtypes==1.1.14` | Direct — Windows Picoh vendor interface | Keep only as a Windows-gated Picoh dependency; never require it for headless or non-Windows installs. |
| 033 | `confection==0.0.3` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 034 | `contourpy==1.0.7` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 035 | `cryptography==3.4.8` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 036 | `cupy-cuda111==11.5.0` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 037 | `cycler==0.11.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 038 | `cymem==2.0.7` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 039 | `-e git+https://github.com/huggingface/datasets@c4a4f96ef0a4ec4b25f0872f160fa1eb9d2e711c#egg=datasets` | Public editable Git pin — legacy data/model path | Replace the commit checkout with a public PyPI range in the opt-in `legacy` extra; no editable install. |
| 040 | `dateparser==1.1.4` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 041 | `dbus-python==1.2.18` | Linux distribution/system package freeze | Do not declare as Python app dependencies; provision via the operating system only if a retained system integration needs it. |
| 042 | `decorator==5.1.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 043 | `deepspeed @ file:///home/darf3/buddy/DeepSpeed` | Developer-local path — legacy GPT-J/DeepSpeed | Replace `/home/darf3/...` with public `deepspeed==0.15.4` only in the Linux-only experimental `legacy` extra; no GPU certification. |
| 044 | `dill==0.3.6` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 045 | `distlib==0.3.4` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 046 | `distro==1.7.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 047 | `distro-info===1.1build1` | Linux distribution/system package freeze | Do not declare as Python app dependencies; provision via the operating system only if a retained system integration needs it. |
| 048 | `docker-pycreds==0.4.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 049 | `docopt==0.6.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 050 | `docutils==0.17.1` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 051 | `en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.4.1/en_core_web_trf-3.4.1-py3-none-any.whl` | Downloaded spaCy model artifact — legacy NLU | Keep out of core; make it an explicit opt-in model artifact only if the old spaCy path is kept. Installation may fetch model data. |
| 052 | `evaluate==0.4.0` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 053 | `exceptiongroup==1.1.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 054 | `execnet==1.9.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 055 | `executing==1.2.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 056 | `fastjsonschema==2.16.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 057 | `fastrlock==0.8.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 058 | `ffmpeg-python==0.2.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 059 | `filelock==3.9.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 060 | `fixup==0.0.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 061 | `flake8==6.0.0` | Development lint tooling | Keep only if the repository still runs Flake8; this baseline's new lint setup uses Ruff, so the old pin is not needed for runtime. |
| 062 | `flatbuffers==23.1.21` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 063 | `fonttools==4.38.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 064 | `frozenlist==1.3.3` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 065 | `fsspec==2023.1.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 066 | `future==0.18.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 067 | `gitdb==4.0.10` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 068 | `GitPython==3.1.30` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 069 | `google-auth==2.16.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 070 | `google-auth-oauthlib==0.4.6` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 071 | `grpcio==1.51.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 072 | `gruut==2.3.4` | Direct — Picoh phoneme generation | Keep only in the optional Picoh feature set. |
| 073 | `gruut-ipa==0.13.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 074 | `gruut-lang-en==2.0.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 075 | `gTTS==2.3.0` | Direct — Picoh speech output | Keep only in the optional Picoh/speech feature set. |
| 076 | `hf-doc-builder @ git+https://github.com/huggingface/doc-builder@c3e2c82ab738912f02cf9b0c1442a7aeef252297` | Public Git source — documentation build tool | Remove from application runtime; re-add only to an isolated docs build if documentation tooling is retained. |
| 077 | `hjson==3.1.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 078 | `httplib2==0.20.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 079 | `huggingface-hub==0.11.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 080 | `humanfriendly==10.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 081 | `identify==2.5.15` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 082 | `idna==3.4` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 083 | `imagesize==1.4.1` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 084 | `importlib-metadata==5.1.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 085 | `importlib-resources==5.10.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 086 | `iniconfig==2.0.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 087 | `InstructorEmbedding==1.0.0` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 088 | `intel-extension-for-pytorch==1.13.0` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 089 | `intel-openmp==2023.0.0` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 090 | `ipython==8.8.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 091 | `isort==5.12.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 092 | `jedi==0.18.2` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 093 | `jeepney==0.7.1` | OS credential-store/tooling dependency | Do not install in core from this freeze; keep only if a future optional credential-store integration explicitly requires it. |
| 094 | `Jinja2==3.1.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 095 | `joblib==1.2.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 096 | `jsonlines==1.2.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 097 | `jsonschema==4.17.3` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 098 | `jupyter_core==5.2.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 099 | `keyboard==0.13.5` | Direct — desktop/speech controls | Keep only in the optional interactive/speech feature set. |
| 100 | `keyring==23.5.0` | OS credential-store/tooling dependency | Do not install in core from this freeze; keep only if a future optional credential-store integration explicitly requires it. |
| 101 | `kiwisolver==1.4.4` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 102 | `langcodes==3.3.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 103 | `launchpadlib==1.10.16` | Linux distribution/system package freeze | Do not declare as Python app dependencies; provision via the operating system only if a retained system integration needs it. |
| 104 | `lazr.restfulclient==0.14.4` | Linux distribution/system package freeze | Do not declare as Python app dependencies; provision via the operating system only if a retained system integration needs it. |
| 105 | `lazr.uri==1.0.6` | Linux distribution/system package freeze | Do not declare as Python app dependencies; provision via the operating system only if a retained system integration needs it. |
| 106 | `libcst==0.4.9` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 107 | `lxml==4.9.1` | Direct — Picoh vendor XML handling | Keep only in the optional Windows Picoh feature set. |
| 108 | `Markdown==3.4.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 109 | `MarkupSafe==2.1.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 110 | `matplotlib==3.6.3` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 111 | `matplotlib-inline==0.1.6` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 112 | `mccabe==0.7.0` | Flake8 development-tool closure | Do not pin separately; if Flake8 is re-enabled, let that dev tool resolve its own dependencies. |
| 113 | `Megatron==0.5.1` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 114 | `megatron-lm @ git+https://github.com/huggingface/Megatron-LM.git@100b522bb8044d98413398f9e71563af15b83325` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 115 | `mkl==2023.0.0` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 116 | `mkl-include==2023.0.0` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 117 | `more-itertools==9.0.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 118 | `mpmath==1.2.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 119 | `multidict==6.0.4` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 120 | `multiprocess==0.70.14` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 121 | `murmurhash==1.0.9` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 122 | `mypy-extensions==0.4.3` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 123 | `nbformat==5.7.3` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 124 | `networkx==2.8.8` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 125 | `ninja==1.11.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 126 | `nltk==3.8.1` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 127 | `nodeenv==1.7.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 128 | `num2words==0.5.12` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 129 | `numpy==1.23.5` | Direct — local model/vision numerics | Keep only in optional local/legacy capability dependencies; core configuration does not need it. |
| 130 | `nvidia-cublas-cu11==11.10.3.66` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 131 | `nvidia-cuda-nvrtc-cu11==11.7.99` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 132 | `nvidia-cuda-runtime-cu11==11.7.99` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 133 | `nvidia-cudnn-cu11==8.5.0.96` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 134 | `oauthlib==3.2.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 135 | `onnx==1.12.0` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 136 | `onnxruntime-gpu==1.13.1` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 137 | `optimum==1.6.1` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 138 | `packaging==23.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 139 | `pandas==1.5.3` | Imported alias with no live use | Remove the dead import and its standalone requirement; this is based on a source-reference check, not merely lack of import. |
| 140 | `parso==0.8.3` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 141 | `pathspec==0.9.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 142 | `pathtools==0.1.2` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 143 | `pathy==0.10.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 144 | `pexpect==4.8.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 145 | `phonemes2ids==1.2.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 146 | `pickleshare==0.7.5` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 147 | `picoh==1.275` | Direct — Picoh vendor runtime | Keep only in a Windows-gated optional extra; vendor entry points are not core/headless. |
| 148 | `Pillow==9.3.0` | Direct — VQA image conversion | Keep only in optional `vision`; no image/camera setup in core. |
| 149 | `platformdirs==2.5.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 150 | `playsound==1.3.0` | Direct — Picoh audio playback | Use `playsound==1.2.2` in Windows Picoh only; historical 1.3.0 is source-only and can fail clean builds. |
| 151 | `pluggy==0.13.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 152 | `pre-commit==2.21.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 153 | `preshed==3.0.8` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 154 | `prompt-toolkit==3.0.36` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 155 | `protobuf==3.20.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 156 | `psutil==5.9.4` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 157 | `ptyprocess==0.7.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 158 | `pure-eval==0.2.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 159 | `py==1.10.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 160 | `py-cpuinfo==9.0.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 161 | `pyarrow==10.0.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 162 | `pyasn1==0.4.8` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 163 | `pyasn1-modules==0.2.8` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 164 | `pybind11==2.10.3` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 165 | `pycodestyle==2.10.0` | Flake8 development-tool closure | Do not pin separately; if Flake8 is re-enabled, let that dev tool resolve its own dependencies. |
| 166 | `pycparser==2.21` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 167 | `pydantic==1.10.4` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 168 | `pydub==0.25.1` | Direct — Picoh audio conversion | Keep only in the optional Picoh/speech feature set. |
| 169 | `pyflakes==3.0.1` | Flake8 development-tool closure | Do not pin separately; if Flake8 is re-enabled, let that dev tool resolve its own dependencies. |
| 170 | `pygame==2.1.3` | Comment-only legacy vision option | No active import found; do not retain as a runtime dependency without a real enabled feature. |
| 171 | `Pygments==2.14.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 172 | `PyGObject==3.42.1` | Linux distribution/system package freeze | Do not declare as Python app dependencies; provision via the operating system only if a retained system integration needs it. |
| 173 | `PyJWT==2.3.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 174 | `pyparsing==2.4.7` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 175 | `pyrsistent==0.19.3` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 176 | `pyserial==3.5` | Direct — Picoh serial interface | Keep only in the Windows Picoh optional set; the source imports it as `serial`. |
| 177 | `pytest==7.2.1` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 178 | `pytest-forked==1.4.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 179 | `pytest-randomly==3.12.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 180 | `pytest-xdist==3.1.0` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 181 | `python-apt==2.4.0` | Linux distribution/system package freeze | Do not declare as Python app dependencies; provision via the operating system only if a retained system integration needs it. |
| 182 | `python-crfsuite==0.9.8` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 183 | `python-dateutil==2.8.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 184 | `pytorch-triton==2.0.0+0d7e753227` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 185 | `pytz==2022.6` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 186 | `pytz-deprecation-shim==0.1.0.post0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 187 | `PyYAML==6.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 188 | `recommonmark==0.7.1` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 189 | `regex==2022.10.31` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 190 | `requests==2.28.1` | Direct — Picoh HTTP/TTS integration | Keep only in the optional Picoh capability. |
| 191 | `requests-oauthlib==1.3.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 192 | `responses==0.18.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 193 | `rsa==4.9` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 194 | `scikit-learn==1.0.2` | Direct — legacy local NLU/classifier | Keep only in the opt-in `legacy` model/runtime set; not a core dependency. |
| 195 | `scipy==1.7.3` | Direct — speech/audio helpers | Keep only in optional speech/Picoh capability dependencies. |
| 196 | `SecretStorage==3.3.1` | OS credential-store/tooling dependency | Do not install in core from this freeze; keep only if a future optional credential-store integration explicitly requires it. |
| 197 | `semantic-version==2.10.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 198 | `sentence-transformers==2.2.2` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 199 | `sentencepiece==0.1.97` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 200 | `sentry-sdk==1.14.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 201 | `setproctitle==1.3.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 202 | `setuptools-rust==1.5.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 203 | `six==1.16.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 204 | `smart-open==6.3.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 205 | `smmap==5.0.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 206 | `snowballstemmer==2.2.0` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 207 | `sounddevice==0.4.5` | Direct — Picoh audio device | Keep only in Windows Picoh optional dependencies. |
| 208 | `soundfile==0.11.0` | Direct — Picoh audio file I/O | Keep only in Windows Picoh optional dependencies. |
| 209 | `spacy==3.4.4` | Direct — legacy local NLU | Keep only in optional `legacy`; its model artifact must remain separately opt-in. |
| 210 | `spacy-alignments==0.8.6` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 211 | `spacy-legacy==3.0.10` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 212 | `spacy-loggers==1.0.4` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 213 | `spacy-transformers==1.1.8` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 214 | `Sphinx==5.3.0` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 215 | `sphinx-rtd-theme==1.1.1` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 216 | `sphinxcontrib-applehelp==1.0.4` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 217 | `sphinxcontrib-devhelp==1.0.2` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 218 | `sphinxcontrib-htmlhelp==2.0.0` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 219 | `sphinxcontrib-jsmath==1.0.1` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 220 | `sphinxcontrib-qthelp==1.0.3` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 221 | `sphinxcontrib-serializinghtml==1.1.5` | Documentation build tooling | Keep out of runtime; put in an isolated docs environment only if the Sphinx documentation build remains supported. |
| 222 | `srsly==2.4.5` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 223 | `stack-data==0.6.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 224 | `sympy==1.11.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 225 | `tabulate==0.9.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 226 | `tbb==2021.8.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 227 | `tensorboard==2.11.2` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 228 | `tensorboard-data-server==0.6.1` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 229 | `tensorboard-plugin-wit==1.8.1` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 230 | `tensorrt==8.5.2.2` | CUDA/accelerator-specific pin | Do not carry the 2023 CUDA wheel/version pin into generic metadata; let an opt-in backend select a matching platform build. |
| 231 | `thinc==8.1.5` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 232 | `threadpoolctl==3.1.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 233 | `tokenizers==0.12.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 234 | `toml==0.10.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 235 | `tomli==2.0.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 236 | `torch==1.13.1` | Direct — local model/memory runtime | Keep only in optional `local`; choose the platform wheel/index when GPU acceleration is desired. |
| 237 | `torchaudio==0.13.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 238 | `torchvision==0.15.0.dev20230125+cu117` | Unused import plus CUDA 11.7 development wheel | Remove the dead import and development wheel; it is both unreferenced and tied to the frozen CUDA 11.7 stack. |
| 239 | `tox==3.21.4` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 240 | `tqdm==4.64.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 241 | `traitlets==5.8.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 242 | `transformers==4.26.0` | Direct — legacy transformer runtime | Keep only in optional `local`, pinned to the compatible 4.36.x API until the separate runtime modernization. |
| 243 | `-e /home/darf3/buddy/triton/python` | Developer-local editable CUDA/compiler path | Remove the local editable path; let the selected PyTorch stack resolve a compatible compiler/runtime component when needed. The preceding source comment calls this `triton==1.0.0`. |
| 244 | `typer==0.7.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 245 | `typing-inspect==0.8.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 246 | `typing_extensions==4.4.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 247 | `tzdata==2022.7` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 248 | `tzlocal==4.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 249 | `unattended-upgrades==0.1` | Linux distribution/system package freeze | Do not declare as Python app dependencies; provision via the operating system only if a retained system integration needs it. |
| 250 | `urllib3==1.26.13` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 251 | `utils==1.0.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 252 | `virtualenv==20.13.0+ds` | Development/test tooling or tooling-only dependency | Move to the `dev` group only when a maintained check uses it; do not install it for end users. |
| 253 | `wadllib==1.3.6` | Linux distribution/system package freeze | Do not declare as Python app dependencies; provision via the operating system only if a retained system integration needs it. |
| 254 | `wandb==0.13.9` | Legacy GPT-J/DeepSpeed, training, or experiment stack | Not needed for minimal/cloud/headless startup; remove from core and re-add only as a separately justified optional training/runtime component. |
| 255 | `wasabi==0.10.1` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 256 | `Wave==0.0.2` | Name collision with Python standard library | Source uses stdlib `wave`; remove this unrelated PyPI package. |
| 257 | `wcwidth==0.2.6` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 258 | `Werkzeug==2.2.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 259 | `whisper @ git+https://github.com/openai/whisper.git@0b5dcfdef7ec04250b76e13f1630e32b0935ce76` | Direct — speech-to-text | Replace the source Git checkout with public `openai-whisper` only in optional `speech`. |
| 260 | `xxhash==3.2.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 261 | `yarl==1.8.2` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
| 262 | `zipp==3.11.0` | Transitive/unimported or historical environment residue | No direct import was observed in the baseline AST scan; do not keep a standalone runtime pin unless an active parent/tool requires it. Absence of a direct import is not proof the feature is unused. |
