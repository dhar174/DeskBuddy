# DeskBuddy Verification Gates & Acceptance Criteria

Gate 1 is the routine CPU-only pull request check. Gates 2 and 3 qualify the actual runtime, model, and hardware for changes that affect those capabilities.

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
1. **Import Side Effects**: The packaged headless entry point must validate configuration without initializing models, external services, or devices. Exercise legacy startup paths with mocked optional dependencies.
2. **Headless Mock Availability**: The mock hardware adapter (`MockPicoh`) and loopback socket bridge must initialize cleanly without physical USB controllers connected.
3. **Configuration Sanity**: Environment variables (`OPENAI_API_KEY`, etc.) are checked gracefully; missing keys trigger fallback warnings rather than hard crashes.

---

## Gate 1: Contract & Unit Tests

**Objective**: Run the locked CPU-only unit and deterministic integration tests plus lint, formatting, and type checks.

Follow the [testing guide](../../docs/testing.md) for commands, supported Python versions, check scopes, and opt-in integration tests.

### Acceptance Criteria:
1. **Classification Intent Contracts**:
   - Use deterministic classifier fakes to verify intent-label handling, routing, and fallbacks.
2. **Conversation History Contracts**:
   - Multi-turn buffer formatting maintains valid alternating roles (`user`, `assistant`).
   - Rolling context trimming truncates history beyond maximum turn limits without losing the system persona prompt.
3. **Authenticated Robot Transport (Issue #4)**:
   - Exercise typed, length-framed JSON on loopback, including fragmented frames, bidirectional HMAC authentication, and rejection of malformed, unauthorized, and replayed requests.
   - Verify timeout and disconnect handling. If execution of a physical command is uncertain, surface that outcome without blindly retrying it.
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
4. **Real Classifier Qualification**:
   - Measure the target of **at least 95% accuracy** for `helpers.is_question`, `is_chat_or_command`, `is_personal`, `is_visual_question`, and `needs_follow_up` on a labeled utterance set using provisioned model weights.

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
