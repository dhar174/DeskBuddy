# Issue #5 implementation plan

Scope: reproducible packaging and injectable configuration. Issue #4 owns RPC
serialization, framing, authentication enforcement and timeouts; no protocol work here.

1. Audit the 2023 freeze against imports and inspect entry points, Docker and paths.
2. Add `src/deskbuddy` for immutable typed settings and an offline CLI. Preserve
   root legacy modules for Issue #7; the built package contains only the baseline.
3. Define OS user-data/cache paths and deterministic overrides, environment,
   explicit dotenv, TOML and defaults. No implicit checkout writes or model loads.
4. Isolate cloud/local/speech/vision/Picoh/legacy dependencies and dev tooling;
   generate and check a public-index uv lock, prove clean non-editable core installs.
5. Make narrow legacy settings substitutions, defer VQA downloads, and provide a
   portable launcher. Preserve model/conversation algorithms and Larynx Debian build.
6. Add offline settings, path, optional-dependency, legacy wiring and wheel smoke tests;
   review acceptance criteria and publish a focused PR closing #5.

No GPU/model/audio/robot/live-cloud qualification is claimed by this baseline.
