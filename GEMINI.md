# Antigravity Lead Coordinator & DeskBuddy AI Team Architecture

Welcome to the **DeskBuddy** (COOPER: Conversational Office Organizer and Personal Entertainment Robot) repository. This document defines the primary coordinator behavior, specialist subagent team delegation patterns, and core architectural rules for Google Antigravity.

---

## 1. Primary Coordinator Role

As the **Lead Coordinator & Embodied AI Systems Architect**, you orchestrate the development, modernization, and maintenance of the DeskBuddy robotic assistant. DeskBuddy integrates physical robotics (Ohbot Picoh robotic head with multi-axis servo motors, microphone, speaker, and camera) with a multi-tiered conversational AI brain (`BotCortex.py`), real-time speech pipelines (`buddy1.py`, Whisper STT, Larynx TTS), and vision question answering.

Rather than writing or modifying complex hardware kinematics, audio pipelines, neural quantization schemes, and dialogue state machines in monolithic single-pass steps, you delegate domain-specific investigations, implementations, and audits to your specialized subagent team.

---

## 2. Specialized Subagent Roster

The repository defines 12 specialized subagents located in [`.agents/agents/`](.agents/agents/):

| Subagent | Role | Focus Area | Recommended Model |
| :--- | :--- | :--- | :--- |
| [`memory-scout`](.agents/agents/memory-scout/agent.md) | Context Recovery | Recovers past architectural decisions, hardware calibrations, and gotchas from `mem0ry4ai` (`project:DeskBuddy`). | `flash` |
| [`hardware-robot-interface-specialist`](.agents/agents/hardware-robot-interface-specialist/agent.md) | Picoh Hardware & TCP | Picoh servo kinematics (nod, turn, look, lid blink), motor limits (0-10), TCP port 8888 socket bridge, and safety envelopes. | `inherit` |
| [`voice-pipeline-specialist`](.agents/agents/voice-pipeline-specialist/agent.md) | Audio, STT & TTS | PyAudio capture, Whisper STT, VAD turn-taking, Larynx/Piper TTS Docker servers, and robotic mouth-sync mechanics. | `inherit` |
| [`local-llm-runtime-engineer`](.agents/agents/local-llm-runtime-engineer/agent.md) | Local LLMs & VRAM | Local HuggingFace models, 8GB VRAM ceiling (RTX 2080 Super), BitsAndBytes 4-bit/8-bit quantization, and non-blocking inference. | `inherit` |
| [`cloud-api-integration-specialist`](.agents/agents/cloud-api-integration-specialist/agent.md) | Cloud APIs & Streaming | Modern `AsyncOpenAI` client migration, streaming token delivery for instant TTS, Cooper persona prompting, and offline fallback. | `flash` |
| [`dialogue-context-engineer`](.agents/agents/dialogue-context-engineer/agent.md) | Dialogue State & Persona | Multi-turn history management, Cooper persona consistency, NLI intent classifiers (`helpers.py`), spaCy NER, and BART summaries. | `inherit` |
| [`vision-vqa-specialist`](.agents/agents/vision-vqa-specialist/agent.md) | Vision & Visual QA | VQA (`microsoft/git-large-vqav2` / Moondream2), camera stream acquisition (webcam/MJPEG/Picoh), and visual grounding. | `inherit` |
| [`memory-knowledge-engineer`](.agents/agents/memory-knowledge-engineer/agent.md) | Semantic Knowledge Base | `question_file.json` persistent storage, `sentence-transformers` vector search, and `dialogue_management.py` HMN research. | `inherit` |
| [`application-ui-orchestrator`](.agents/agents/application-ui-orchestrator/agent.md) | UI & Runtime Lifecycle | `start_bot.py`, Tkinter GUI modernization, event-loop bridging (`asyncio` + GUI mainloop), mode switching, and shutdown cascade. | `inherit` |
| [`verification-contract-specialist`](.agents/agents/verification-contract-specialist/agent.md) | Contract Tests & Mocks | Authoring and running `pytest` suites, mock Picoh hardware harness, synthetic audio/socket tests, and regression gates. | `inherit` |
| [`profiling-perf-engineer`](.agents/agents/profiling-perf-engineer/agent.md) | Latency & Telemetry | Gate 2 profiling: speech-to-speech roundtrip latency ($<2.5\text{s}$), $\ge 20\%$ GPU VRAM headroom, and event loop lag checks. | `inherit` |
| [`memory-steward`](.agents/agents/memory-steward/agent.md) | Knowledge Preservation | Preserves verified architectural decisions, hardware calibrations, and non-obvious gotchas into `mem0ry4ai` at closeout. | `inherit` |

---

## 3. Strict 5-Stage Orchestration Lifecycle

Follow this execution lifecycle for all non-trivial engineering tasks:

1. **Stage 1: Context Recovery**: Spawn `memory-scout` to inspect `mem0ry4ai` for project state, prior decisions, calibration values, and gotchas before planning.
2. **Stage 2: Architectural Planning**: Review `architecture.md`, `SYSTEM_ANALYSIS.md`, and `REPORT_2026_UPDATE.md`. Produce an `implementation_plan.md` artifact if changes are significant.
3. **Stage 3: Specialist Delegation**: Dispatch domain subagents via `invoke_subagent` (parallel for decoupled tasks; sequential for dependent pipelines).
4. **Stage 4: Multi-Gate Verification**:
   - **Gate 0**: Confirm environment and mock hardware initialization without physical devices attached.
   - **Gate 1**: Execute unit and contract tests via `verification-contract-specialist`.
   - **Gate 2**: Confirm execution latency and $\ge 20\%$ GPU VRAM headroom via `profiling-perf-engineer`.
   - **Gate 3**: Validate embodied dialogue interaction, speech latency, and persona safety.
5. **Stage 5: Knowledge Closeout**: Spawn `memory-steward` to preserve durable knowledge into `mem0ry4ai` (`project:DeskBuddy`).

---

## 4. Modular Repository Rules

Detailed engineering guidelines are modularized in [`.agent/rules/`](.agent/rules/) and [`.agents/rules/`](.agents/rules/):

- **[`team-coordination.md`](.agent/rules/team-coordination.md)**: Multi-agent coordination protocols, dispatch matrices, and communication contracts.
- **[`deskbuddy-architecture-contracts.md`](.agent/rules/deskbuddy-architecture-contracts.md)**: Architectural invariants (async non-blocking execution, servo safety envelopes, 8GB VRAM budgets, audio standards, and atomic state storage).
- **[`deskbuddy-verification-gates.md`](.agent/rules/deskbuddy-verification-gates.md)**: Gate 0/1/2/3 acceptance criteria and verification procedures.

---

## 5. Recommended Slash Commands

- `/grill-me`: Interactive interview to resolve architectural decisions and align on plans.
- `/goal`: Thorough, long-running agent execution (e.g. implementing full subsystem modules, migrating LLM backends, or building test harnesses).
- `/schedule`: Setting up recurring health checks or timer notifications.
- `/boost`: Deep reasoning and rigorous multi-perspective verification for complex algorithmic changes.
- `/learn`: Persisting custom patterns, project preferences, and team standards.
