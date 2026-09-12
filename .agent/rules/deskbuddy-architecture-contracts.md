# DeskBuddy Architecture & System Contracts

This document establishes the hard technical invariants, safety boundaries, and architectural contracts for `DeskBuddy` (COOPER). All code changes must adhere to these contracts.

---

## 1. Asynchronous Non-Blocking Execution Contract

- **Zero Blocking in Event Loop**:
  - The main application runs on an `asyncio` event loop. Long-running CPU operations, blocking socket calls, and GPU model inferences (`model.generate()`, Whisper transcription, PyAudio buffer reads, OpenCV frame captures) must NEVER be executed synchronously within coroutines.
  - All blocking operations MUST be wrapped in `await asyncio.to_thread(...)` or executed on background executor pools.
- **Event Loop Lag Threshold**:
  - Main event loop lag must remain $<20\text{ ms}$. If any routine delays the event loop by $>50\text{ ms}$, it is considered an architectural violation.

---

## 2. Hardware Physical Safety Envelope Contract

- **Picoh Servo Motion Clamping**:
  - All commands to `picoh.move(motor, pos)` must clamp `pos` to the calibrated physical range `[0.0, 10.0]`.
  - Servos must never be driven against mechanical stops to prevent motor overheating, gear stripping, and overcurrent resets.
- **Fail-Safe Hardware Disconnect**:
  - If the physical Picoh robot is disconnected, USB COM port drops, or socket connection fails, the system must log an error and continue running in software simulation / headless mode. Uncaught serial/socket exceptions that terminate the application are strictly prohibited.
- **Graceful Kinematic Shutdown**:
  - On application termination (`SIGINT` or window close), the robot head must be returned to neutral center coordinates (`picoh.move(picoh.TURN, 5)`, `picoh.move(picoh.NOD, 5)`), and eyes/lids reset to open positions.

---

## 3. GPU Memory & VRAM Budget Contract

- **8GB VRAM Ceiling (RTX 2080 Super)**:
  - Total allocated VRAM across all resident models (LLM, Whisper, VQA, Embeddings) must not exceed **6.5 GiB**.
  - At least **1.5 GiB ($\ge 20\%$)** of free VRAM headroom must remain available at all times for OS compositor buffers, video decoding, and peak inference activations.
- **Quantization Standard**:
  - Local models $\ge 3\text{B}$ parameters must use 4-bit NormalFloat (`nf4`) or 8-bit quantization via BitsAndBytes or GGUF quantization.
  - Avoid monolithic unquantized weights that trigger unpredictable CUDA Out-of-Memory (OOM) aborts.

---

## 4. Audio Pipeline & Turn-Taking Contract

- **Audio Stream Specification**:
  - All speech recognition input buffers must be standardized to **16,000 Hz, 16-bit signed PCM, mono**.
  - Never supply un-resampled 44.1 kHz or 48 kHz buffers directly to Whisper STT.
- **Audio Device Hygiene**:
  - PyAudio stream handles must be cleanly closed inside `finally:` blocks.
  - Use non-blocking Voice Activity Detection (VAD) with energy thresholds to determine natural conversational turn boundaries, minimizing dead air while preventing cutoffs.
- **Incremental TTS Delivery**:
  - For long generations, text must be chunked by punctuation (periods, question marks, commas) and fed incrementally to TTS and lip-sync routines so speech starts before full generation concludes.

---

## 5. Knowledge Base & State Persistence Contract

- **Atomic File Writing**:
  - Modifications to `question_file.json` or conversation state logs must NEVER write directly in-place.
  - Updates must be serialized to a temporary file (`.tmp`), flushed to disk (`os.fsync`), and atomically swapped using `os.replace`.
- **Knowledge Base Validation**:
  - All entries must pass strict schema validation (valid JSON string keys and values; non-empty answers).
- **Sub-50ms Vector Retrieval**:
  - Semantic similarity lookups against `question_file.json` must complete in $<50\text{ ms}$ with a calibrated cosine similarity threshold (default $\ge 0.75$) to prevent irrelevant matches.

---

## 6. Cooper Persona & Dialogue Safety Contract

- **Kid-Friendly Boundary**:
  - Cooper is designed as a companion for children (specifically tailored to a third-grade friend). Responses must always be age-appropriate, positive, encouraging, and free of toxicity or offensive content.
- **Humor & Character Invariants**:
  - Cooper loves the Legend of Zelda, Nintendo games, cats, telling playful jokes (including silly kid humor), and asking about school and friendship.
- **Bounded Dialogue History**:
  - Working conversation history must be bounded to a rolling window (e.g. 10-15 turns) supplemented by rolling summaries to prevent context window overflow and token waste.
