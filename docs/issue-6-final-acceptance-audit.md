# DeskBuddy Issue #6 final acceptance audit

- Report date: November 19, 2026
- Audited `main`: [`76dd347f2aeac168e258b1a4f428c50e74dda61e`](https://github.com/dhar174/DeskBuddy/commit/76dd347f2aeac168e258b1a4f428c50e74dda61e)
- Issue: [#6, Add test harness, CI, and baseline behavioral contract tests](https://github.com/dhar174/DeskBuddy/issues/6)

**Audit verdict: Accept. Keep Issue #6 closed as completed.** The owner had already closed it when this audit began. All three implementation merges are ancestors of the audited commit.

**Independent review: Accept.** A separate Codex reviewer checked the issue criteria, production and test sources, and raw Actions records for the same SHA. Both reviews found zero blockers. Antigravity was attempted first and timed out before returning a verdict.

## Merge and CI evidence

Counts are per Python version and come from the GitHub Actions job logs.

| Milestone | Merge SHA | Python 3.11 | Python 3.12 | CI evidence |
| --- | --- | ---: | ---: | --- |
| #6A, [PR #30](https://github.com/dhar174/DeskBuddy/pull/30) | `a3f77d85680287b4c2b1321d4b67949e50e5fa66` | 311 | 311 | [Automatic PR run](https://github.com/dhar174/DeskBuddy/actions/runs/43386034186), [main push](https://github.com/dhar174/DeskBuddy/actions/runs/43390010886) |
| #6B, [PR #31](https://github.com/dhar174/DeskBuddy/pull/31) | `c22b9ab96c06ed504f88de766bfcd95f483cf841` | 363 | 363 | [Automatic PR run](https://github.com/dhar174/DeskBuddy/actions/runs/43397120912), [main push](https://github.com/dhar174/DeskBuddy/actions/runs/43399232346) |
| #6C, [PR #33](https://github.com/dhar174/DeskBuddy/pull/33) | `cb806c6efdacccb519c3b19364c604678c084932` | 405 | 405 | [Automatic PR run](https://github.com/dhar174/DeskBuddy/actions/runs/43517484152), [main push](https://github.com/dhar174/DeskBuddy/actions/runs/43557907104) |
| Audited `main` | `76dd347f2aeac168e258b1a4f428c50e74dda61e` | 595 | 595 | [3.11.18 job](https://github.com/dhar174/DeskBuddy/actions/runs/43630076681/job/132554510429), [3.12.16 job](https://github.com/dhar174/DeskBuddy/actions/runs/43630076681/job/132554510645) |

Both current jobs collected and passed all 595 tests with zero skips. Lock consistency and Ruff lint passed, Ruff format reported 50 formatted files, and mypy reported zero issues in 17 source files. GitHub recorded the expected `pull_request` and `push` events. The disposable probe produced a failed pytest step and exit code 1 in both PR jobs. [Workflow at audited commit](https://github.com/dhar174/DeskBuddy/blob/76dd347/.github/workflows/ci.yml) · [Failure probe](https://github.com/dhar174/DeskBuddy/actions/runs/43487493481)

## Acceptance criteria

| Criterion | Evidence at the audited commit |
| --- | --- |
| Pytest works from a clean development install | Fresh Windows environments created with `uv sync --locked --group dev` passed all 595 tests on Python 3.11.0 and 3.12.2. Both also passed the lock, lint, format, and mypy checks. [Installation and check commands](https://github.com/dhar174/DeskBuddy/blob/76dd347/docs/testing.md#gate-1-developer-checks) |
| CI runs on PRs and pushes to `main` | Verified the workflow triggers and actual run events for PR #33 and the main pushes above. [Workflow triggers](https://github.com/dhar174/DeskBuddy/blob/76dd347/.github/workflows/ci.yml#L1-L6) |
| CI runs lint, format, static types, and tests | Both matrix jobs ran pytest, Ruff lint and format, and mypy for the modern package and transport boundary. [Workflow steps](https://github.com/dhar174/DeskBuddy/blob/76dd347/.github/workflows/ci.yml#L42-L55) |
| Default tests run without models, GPU, Picoh, or external credentials | The test profile scrubs credentials, uses temporary settings and storage, blocks heavy runtime imports, and restricts Python socket activity to loopback. CI installs core and dev dependencies only. [Isolation fixture](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/conftest.py) · [Isolation tests](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_pytest_isolation.py) |
| Required routing and fake robot integration are covered | Contracts assert normal chat, provider selection, vision and memory early returns, generation on memory miss, clarification, and graceful fallbacks. The integration sends fake transcriptions through the production desktop loop, routing, authenticated robot transport, and fake `picoh.say`. [Routing contracts](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_conversation_routing_contracts.py) · [Transcript and speech integration](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_conversation_robot_integration.py#L331-L359) · [Route outcomes](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_conversation_robot_integration.py#L520-L563) |
| Four named bugs and the clarification early return have regression tests | Five in-memory probes reintroduced the original failure modes one at a time. The corresponding tests rejected classifier-dictionary comparison, a mismatched image label, the always-true memory condition, a missing `await`, and the cloud follow-up return of an uninitialized value. [Classification and async regressions](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_classification_contracts.py) · [Memory regression](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_conversation_routing_contracts.py#L265-L292) · [Cloud follow-up regression](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_conversation_routing_contracts.py#L345-L376) |
| Optional integration is documented and opt-in | Default collection excludes the optional tree and rejects an explicit optional path without `--run-optional`. An opted-in external-service run with configuration removed reported one skip with its reason. Hardware and model categories are documented for locally provisioned prerequisites. [Optional test policy](https://github.com/dhar174/DeskBuddy/blob/76dd347/docs/testing.md#optional-integration-tests) |
| Obsolete interactive `test.py` is handled | PR #30 deleted the script, and it is absent from the audited tree. Pytest collects from `tests/`. [Removal](https://github.com/dhar174/DeskBuddy/pull/30/files) · [Pytest configuration](https://github.com/dhar174/DeskBuddy/blob/76dd347/pyproject.toml) |

The test harness parses and executes production function bodies from the checked-out files, then injects deterministic dependencies. The audit inspected how those fakes were bound and how shared integration state was reset. A reverse-order run on Python 3.12 passed all 595 tests. [Harness](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/support/legacy_runtime.py#L80-L98) · [Fixture and cleanup](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_conversation_robot_integration.py#L172-L193)

## Issue #4 and Issue #5 compatibility

- **Issue #4** merged in [PR #28](https://github.com/dhar174/DeskBuddy/pull/28) at `c5e24d409e534b93346b6056f6a1c3ac4c1b139a`. Current checks cover length-framed JSON, HMAC, replay protection, explicit dispatch, bounded I/O, rejection of unsafe inputs, and the transcript and speech flows on loopback. Both repository verification guides specify authenticated JSON. Loopback tests also exercise the Windows adapters. [Negative transport tests](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_security_negative.py) · [Transport integration](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_integration.py) · [Verification gates](https://github.com/dhar174/DeskBuddy/blob/76dd347/.agent/rules/deskbuddy-verification-gates.md#L35-L49)
- **Issue #5** merged in [PR #29](https://github.com/dhar174/DeskBuddy/pull/29) at `4917d6aa4418fad02173c495dc8706305590be81`. The lock and typed settings passed the packaging contracts. A fresh non-editable core install on Python 3.12 ran both the installed `deskbuddy --headless` entry point from outside the checkout and `python start_bot.py --headless`. Both reported valid configuration and no installed heavy extras. [Packaging tests](https://github.com/dhar174/DeskBuddy/blob/76dd347/tests/test_packaging_contract.py) · [Setup](https://github.com/dhar174/DeskBuddy/blob/76dd347/docs/setup.md)

## Findings

- **Blocking.** None found.
- **Nonblocking.** Microphone capture, physical Picoh playback, live inference, and latency remain outside the offline acceptance. The loopback test starts with a recognized transcript and ends at the device driver's call. [Microphone](https://github.com/dhar174/DeskBuddy/issues/35) · [Playback](https://github.com/dhar174/DeskBuddy/issues/36) · [Latency](https://github.com/dhar174/DeskBuddy/issues/37)
- **Optional.** The hardware specialist description still refers to pickled framing. The historical acceptance matrix predates the atomic question-bank save and transcript-ID changes on `main`. These documentation updates are separate from the acceptance decision. [Specialist description](https://github.com/dhar174/DeskBuddy/blob/76dd347/.agents/agents/hardware-robot-interface-specialist/agent.md#L4-L8) · [Original matrix](https://github.com/dhar174/DeskBuddy/blob/76dd347/docs/issue-6-acceptance.md#L18-L22)
