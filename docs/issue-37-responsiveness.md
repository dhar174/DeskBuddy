# Issue #37 — Legacy runtime responsiveness

[Issue #37](https://github.com/dhar174/DeskBuddy/issues/37) follows the
[Issue #6 acceptance audit](issue-6-final-acceptance-audit.md). The baseline is
`fb5f2c9db5e36de2cbf5884fec7f539a30dd5681`.

## Operation inventory

The desktop loop acknowledges a transcript, generates one reply, and hands it
to robot transport before accepting the next transcript. The robot adapters
run capture, background motion, and authenticated RPC on their server loops.

| Retained path | Baseline blocking operation or existing boundary | Owner after #37 |
| --- | --- | --- |
| Desktop Fast, Small, Medium, and Large local replies | Tokenization, tensor transfers, model generation and decoding run inside async functions. Medium and Large also assemble conversation context. | Desktop worker owns each synchronous model stage. The turn stays on the event loop. |
| Desktop OpenAI Medium normal reply and early-answer rewording | Synchronous ChatCompletion calls use the legacy SDK and nonlocal message history. | Desktop worker calls the SDK with private request inputs. The loop publishes eligible history and replies. |
| Question, visual, personal, and follow-up classification | Classifier model/tokenizer calls, zero-shot NLI, or synchronous legacy Completion for cloud visual detection. | Desktop worker serializes the shared classifier resources and SDK calls. |
| Local and cloud clarification | Question generation tokenizes, generates, and decodes locally or calls ChatCompletion. | The desktop worker generates the clarification. The loop owns reply routing and any nonlocal history update. |
| Memory lookup | Atomic question-bank read is already offloaded. Embedding tokenization, model movement/forward pass, pooling, and similarity ranking run on the loop. | Preserve the bank boundary. The desktop worker owns the complete embedding search. |
| Conversation context and summary | spaCy entity extraction, summary generation, topic vectorization, and topic inference run on the loop. | Desktop worker. The loop owns turn entities and history. |
| History exports and question-bank writes | Large history summary append and verbatim overwrite perform synchronous file I/O. The question bank uses a locked, atomic offload from [PR #40](https://github.com/dhar174/DeskBuddy/pull/40). | Desktop worker writes eligible history exports in order. Bank operations keep the existing offload, lock, and recovery. |
| Visual answer | An existing offload serializes the camera, lazy processor/model cache, and inference with the vision lock. | Retain the offload, lock, complete-cache publication, and recoverable fallbacks. The originating turn controls answer delivery. |
| Robot root `buddy1`, Windows `buddy1`, Windows `buddy2` | PyAudio stream reads, PCM processing, and Whisper block the server loop. Windows `buddy2` threads only its WAV write and awaits a synchronous input-path lookup. | Each running adapter owns one capture-to-Whisper worker. Its server loop owns recording flags and transcript publication. |
| Robot idle and physical operations | Disabled motion can spin in Windows `buddy2`. The `HardwareExecutor` already owns serialized physical driver operations. | Idle loops yield. The existing hardware executor handles driver uncertainty and rejects conflicts while a physical action is unresolved. |

The [desktop entry point](../start_bot.py), [conversation code](../BotCortex.py),
[classifiers](../helpers.py), and [robot adapters](../buddy1.py) are source-only
legacy entry points. Cold model loading and the GUI chooser precede desktop
readiness. The Large startup warmup uses the same desktop generation boundary.
Robot readiness likewise follows the existing startup setup. Root `buddy1`
runs its synchronous Picoh initialization during startup, and Windows
`buddy1` retains the existing vendor setup during import.
The portable package bootstrap and one-shot cloud diagnostic have their own
entry points.

## Ownership and cancellation

The worker admits one synchronous call and one waiting caller. A cancelled
waiter never calls the dependency. If execution has started, the worker tracks
the actual completion and rejects new conflicting activity until the resource
is released. An already waiting caller can proceed after release if still live.
When the application shuts down, it closes admission, detaches pending results,
and can close transport while a synchronous call remains active. Python can
still wait for worker threads when the process exits.

Desktop state changes run on the event loop. The Large history export trigger
remains `count_index == 10`. The retained runtime does not increment that
counter automatically. When the trigger is reached, the summary append and verbatim
overwrite are authorized after an eligible generation result, in that order,
before speech. The loop consumes the checkpoint at authorization so a partly
completed export cannot cause a later turn to replay the summary append.
The initial turn retains its verbatim-only export. An authorized file write
can finish. Cancellation between the two writes can leave the summary file
ahead of the verbatim file. An abandoned inference cannot authorize an export
or send speech.

Each robot recording uses a private temporary WAV removed by the audio worker.
The adapter keeps its existing 44.1 kHz mono PCM and VAD settings, including
the longer silent-chunk threshold in Windows `buddy2`. Capture observes a
cooperative stop signal between reads and before Whisper. Once a native read
or Whisper is in progress, the worker may have to wait for it to return. The
loop drops the abandoned result before it can affect the transcript queue.
Physical commands use the existing `ExecutionOutcomeUnknown` contract with no
automatic replay. The capture prompt still waits for the adapter's physical
LED cue before starting a recording. An unresolved driver can defer a new
capture at that step while the server loop remains available for RPC and shutdown.

Provider migration, conversation-history policy, speech streaming, and robot
embodiment remain in [#8](https://github.com/dhar174/DeskBuddy/issues/8),
[#39](https://github.com/dhar174/DeskBuddy/issues/39),
[#10](https://github.com/dhar174/DeskBuddy/issues/10),
[#11](https://github.com/dhar174/DeskBuddy/issues/11), and
[#20](https://github.com/dhar174/DeskBuddy/issues/20). Coordinate retirement of
these legacy boundaries with [#7](https://github.com/dhar174/DeskBuddy/issues/7)
and trace integration with [#22](https://github.com/dhar174/DeskBuddy/issues/22).

## CPU-only comparison

The benchmark exercises production entry points with controlled synchronous
delays, temporary storage, and the authenticated loopback transport. Measure
scheduling lag, intervals between heartbeat samples, operation or turn duration,
and the timing and outcome of loopback probes. The repository's targets are
lag below 20 ms and zero blocked intervals above 50 ms. Gate 1 contracts use
event synchronization separately from these host-dependent timings.

Scheduling lag is the delay from a heartbeat's due time to its actual execution.
Heartbeat gaps use consecutive observed timestamps. Loopback results report both
scheduled-to-response time and transport round-trip time, so time spent waiting
for the loop to schedule a probe is visible. The desktop turn measurement ends at
speech handoff to a fake transport dependency. An independent authenticated
loopback connection probes responsiveness during that turn.

From the candidate checkout on Windows, use a detached baseline worktree and
the same runner for both targets. Repeat the sequence with Python 3.12.

```powershell
$candidate = (Get-Location).Path
$baseline = Join-Path (Split-Path $candidate -Parent) 'DeskBuddy-issue37-baseline'
git worktree add --detach $baseline fb5f2c9db5e36de2cbf5884fec7f539a30dd5681
$runner = Join-Path $candidate 'tests/support/measure_legacy_responsiveness.py'
$env:UV_PYTHON = '3.11'
$env:UV_PYTHON_DOWNLOADS = 'never'
uv sync --locked --group dev
uv run --no-sync python $runner --target-checkout $baseline --json-output (Join-Path $env:TEMP "deskbuddy-before-$env:UV_PYTHON.json") --warmups 3 --repetitions 20 --heartbeat-ms 5 --stall-ms 250 --watchdog-seconds 25
uv run --no-sync python $runner --target-checkout $candidate --json-output (Join-Path $env:TEMP "deskbuddy-after-$env:UV_PYTHON.json") --warmups 3 --repetitions 20 --heartbeat-ms 5 --stall-ms 250 --watchdog-seconds 25
```

Reuse the baseline worktree on the second Python version. Run the comparison
from an idle host and record any environmental load or timer outliers that
affect the reported measurements.

The recorded profiles used the same Windows 11 host, build 26340, AMD64,
with eight logical CPUs. The locked interpreters were CPython 3.11.0 and
3.12.2 with uv 0.9.26. The measured source revision was
`1b7f343263fecb759fa4c718cd2e007a8a606b55`. The Windows runner file had
SHA-256 `1d36f77bdd8e84f69d7a36e84009a55f17da7edc6b3a5718784ad93e9d534a15`.
The profiles shared lock SHA-256
`1075ff6fd201557d4ca4707a0db14be22748bcf31731a01722488ced81aa1731`.

Each profile uses three light warmups and 20 measured repetitions per case,
a 5 ms heartbeat, a 20 ms authenticated probe schedule under load, and a
25-second child watchdog. Sixteen workload cases each inject a 250 ms delay
at a production-facing fake dependency on each measured repetition.
Identical classifier fixtures send the generation cases down the same branches
at both revisions, with the delay inside the local model or cloud SDK call.
Additional recorder events have zero artificial delay in these profiles. The
calibration deliberately blocks the loop and is excluded from workload
threshold totals. Baseline Windows `buddy2` capture uses a labeled
awaitable-path shim after the driver-shaped path probe raises `TypeError`.
Its disabled motion case reaches the watchdog, so that baseline case has no
numeric idle comparison.

### Matched results

The tables use the 21 workload cases with numeric samples on both revisions.
Counts cover the recorded heartbeat or authenticated-probe windows. The
separate calibration and the unmatched Windows `buddy2` idle case are outside
the totals.

| CPython | Peak lag, ms | p99 lag, ms | Lag samples ≥20 ms | Heartbeat gaps >50 ms | p95 scheduled-to-response, ms | Missed authenticated probes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.11.0 | 310.31 → 32.26 | 248.74 → 14.82 | 356 → 55 | 340 → 0 | 276.94 → 16.04 | 2,694 → 17 |
| 3.12.2 | 309.61 → 29.70 | 249.78 → 14.58 | 336 → 42 | 329 → 0 | 307.27 → 15.92 | 3,674 → 10 |

| Scenario | Python 3.11 p95 operation or turn time, ms | Python 3.12 p95 operation or turn time, ms |
| --- | ---: | ---: |
| Desktop local turn through speech handoff | 251.25 → 258.48 | 251.27 → 254.74 |
| Desktop cloud turn through speech handoff | 251.29 → 258.64 | 251.41 → 262.03 |
| Root `buddy1` capture through transcript | 590.80 → 588.39 | 599.89 → 578.43 |
| Windows `buddy1` capture through transcript | 591.07 → 584.01 | 588.80 → 589.96 |
| Windows `buddy2` capture through transcript | 774.05 → 681.28 | 778.34 → 670.30 |

Both candidate runs recorded zero heartbeat gaps above 50 ms across all
22 workload cases. The longest gaps were 49.57 ms on Python 3.11 and
46.65 ms on Python 3.12. Candidate p99 lag came in below 20 ms on both
interpreters. Individual samples still exceeded 20 ms, so the strict
per-sample lag target was missed. Candidate authenticated probes recorded
zero timeouts and zero errors. The paired Python 3.11 baseline had nine
probe timeouts. An earlier Python 3.11 baseline attempt exceeded the
watchdog in the standalone Windows `buddy2` RPC control. The table uses the
complete rerun with the same settings. In the separate Windows `buddy2` idle case,
the candidate completed the measurement on both interpreters and recorded
no gaps over 50 ms.

## Contract verification

| Platform | CPython | Result on the measured source revision |
| --- | --- | --- |
| Windows, locked local environments | 3.11.0 and 3.12.2 | 648 passed on each |
| Ubuntu 24.04, GitHub Actions | 3.11.18 and 3.12.16 | 648 passed on each |

Both matrices passed lock consistency, Ruff lint and formatting, and mypy.
The contracts use gated fakes and authenticated localhost transport to verify
polling and shutdown while blocking work is active, admission cancellation,
late result suppression, and subsequent turn recovery. They also exercise
transport uncertainty without replay. [Hosted source revision checks](https://github.com/dhar174/DeskBuddy/actions/runs/44485774940)

## Opt-in qualification on a provisioned runtime

Live inference, GPU headroom, and physical capture and playback remain
unqualified by the CPU-only runs in this report.

1. Use the [legacy setup guide](setup.md#legacy-capabilities-and-boundaries) to
   select a supported profile and provision local weights, runtime libraries,
   and only the devices exercised. Use a dedicated data directory and cache
   settings, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and verify
   configured model and device paths before launching. Use the intended
   Linux/CUDA/NCCL profile for the Large tier. A CPU or Windows fake run
   does not establish support for that profile.
2. Invoke the selected adapter and the desktop explicitly. For example,
   from the checkout in separate terminals on a supported configuration,
   run `uv run --no-sync python -X dev buddy1.py` and
   `uv run --no-sync python -X dev start_bot.py`. Choose **Local Only** and
   the provisioned tier in the chooser. To qualify a Windows adapter, launch
   `windows/buddy1.py` or `windows/buddy2.py` in place of `buddy1.py` on
   its provisioned host. Use the expected robot bind address and an isolated
   loopback client with the same configured secret.
3. Log cold startup separately. During a controlled warm turn, capture a
   monotonic heartbeat on the exercised loop and time from authenticated
   transcript receipt to speech handoff. While the operation is active,
   issue authenticated connectivity checks from an independent client.
   The [optional loopback smoke command](testing.md#optional-integration-tests)
   checks reachability without consuming a transcript. For device
   qualification, also exercise capture with a known utterance, verify WAV and
   microphone cleanup after cancellation, verify vision with an intended camera
   or local image, and check a later turn after an abandoned operation returns.
4. Record the source revision, OS, Python, dependencies, model and device IDs,
   sample counts, operation durations, lag/blocked intervals, loopback probes,
   shutdown behavior, and any errors. For a GPU configuration record actual
   peak VRAM separately. Record each omitted path as unqualified. See
   [Gate 2 and Gate 3](../.agent/rules/deskbuddy-verification-gates.md) for
   real-runtime targets.
