# DeskBuddy Verification Gates & Acceptance Criteria

This document defines the quality and verification gates that must be satisfied before any code, configuration, or model change is accepted into the `DeskBuddy` repository.

---

## Overview of Verification Gates

```
[Gate 0: Environment & Mock Sanity]
                 │
                 ▼
[Gate 1: Contract & Unit Tests]
                 │
                 ▼
[Gate 2: Resource & Telemetry Profiling]
                 │
                 ▼
[Gate 3: Embodied Interaction & Safety Validation]
```

---

## Gate 0: Environment & Mock Sanity

**Objective**: Verify that the development environment is properly configured, dependencies import cleanly, and headless mock hardware adapters are active.

### Acceptance Criteria:
1. **Import Side-Effects**: Importing `BotCortex`, `helpers`, or `buddy1` must NOT trigger unhandled hardware errors, device lockouts, or network crashes when hardware is disconnected.
2. **Headless Mock Availability**: The mock hardware adapter (`MockPicoh`) and loopback socket bridge must initialize cleanly without physical USB controllers connected.
3. **Configuration Sanity**: Environment variables (`OPENAI_API_KEY`, etc.) are checked gracefully; missing keys trigger fallback warnings rather than hard crashes.

---

## Gate 1: Contract & Unit Tests

**Objective**: Ensure that all algorithmic, classification, networking, and memory components pass automated unit and contract tests.

### Required Checks:
```powershell
python -m pytest tests/ -v
```

### Acceptance Criteria:
1. **Classification Intent Contracts**:
   - `helpers.is_question`, `is_chat_or_command`, `is_personal`, `is_visual_question`, and `needs_follow_up` must achieve $\ge 95\%$ accuracy on standard test suites in `tests/test_classifiers.py`.
2. **Conversation History Contracts**:
   - Multi-turn buffer formatting maintains valid alternating roles (`user`, `assistant`).
   - Rolling context trimming truncates history beyond maximum turn limits without losing the system persona prompt.
3. **TCP Socket Protocol Framing**:
   - Pickled payload serialization and deserialization across client-server boundaries pass under both immediate and fragmented TCP packet delivery.
   - Client timeout handlers successfully catch socket stalls and recover without crashing.
4. **Knowledge Base Atomic Persistence**:
   - Concurrent or interrupted writes to `question_file.json` never produce corrupted or partial JSON files.

---

## Gate 2: Resource & Telemetry Profiling

**Objective**: Ensure that memory usage, event-loop responsiveness, and latency targets meet real-time interactive constraints.

### Acceptance Criteria:
1. **GPU VRAM Ceiling**:
   - Peak GPU VRAM allocated during model inference must NOT exceed **6.5 GiB** on an 8.0 GiB device.
   - Free GPU headroom must be **$\ge 20\%$ ($\ge 1.5\text{ GiB}$)** at all times.
2. **Event Loop Non-Blocking Verification**:
   - The main `asyncio` event loop must have zero blocked frames $>50\text{ ms}$.
   - Event loop lag during continuous robot background tasks (blinking, looking) must stay below **$20\text{ ms}$**.
3. **Semantic Lookup Latency**:
   - Q&A vector search in `question_file.json` must complete in **$<50\text{ ms}$**.

---

## Gate 3: Embodied Interaction & Safety Validation

**Objective**: Validate the holistic user experience, speech-to-speech roundtrip speed, robot motion boundaries, and dialogue safety.

### Acceptance Criteria:
1. **Speech-to-Speech Roundtrip Latency**:
   - End-to-end latency from the end of user speech to the start of robot voice output must be **$<2.5\text{ seconds}$** (measured across 10 sample utterances).
2. **Physical Kinematics Safety**:
   - All motor movement values commanded during conversation must remain strictly within `[0.0, 10.0]`.
   - Robot head smoothly returns to center/rest position after speech completes.
3. **Cooper Persona & Dialogue Appropriateness**:
   - Generated responses must be evaluated against the Cooper persona rubric: friendly, kid-appropriate, empathetic, humorous, and free of toxicity.
4. **Cloud Failover Gracefulness**:
   - Disconnecting internet access during an active API conversation must trigger an automatic, transparent fallback to the local model tier without terminating the session.
