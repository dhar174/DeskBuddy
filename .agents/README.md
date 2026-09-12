# DeskBuddy Custom Agent Skills (`.agents/skills/`)

This directory contains **33 project-scoped agent skills** composed from the AAS Core catalog (`agentic-awesome-skills` v15.14.0) to provide complete assistance coverage across the DeskBuddy/COOPER codebase.

These skills are automatically discovered and progressively loaded by **Google Antigravity**, **Google Gemini**, and **Claude** via standard workspace discovery.

---

## Installed Skills Inventory

| # | Skill ID | Category | Primary Focus Area | Location |
|---|---|---|---|---|
| 1 | `async-python-patterns` | development | Python asyncio concurrency, event loops, TCP server/client | [SKILL.md](./skills/async-python-patterns/SKILL.md) |
| 2 | `python-pro` | code | Modern Python 3 standards, type hints, ruff/uv, performance | [SKILL.md](./skills/python-pro/SKILL.md) |
| 3 | `ml-engineer` | ai-ml | PyTorch 2.x systems, CUDA AMP, model serving & generation | [SKILL.md](./skills/ml-engineer/SKILL.md) |
| 4 | `hf-mem` | ai-ml | Hugging Face Safetensors/GGUF VRAM requirement calculation | [SKILL.md](./skills/hf-mem/SKILL.md) |
| 5 | `hugging-face-community-evals` | ai-ml | Local GPU evaluation with Transformers, Accelerate, vLLM | [SKILL.md](./skills/hugging-face-community-evals/SKILL.md) |
| 6 | `local-llm-expert` | data-ai | Local LLM inference, VRAM optimization, quantization (4/8-bit) | [SKILL.md](./skills/local-llm-expert/SKILL.md) |
| 7 | `huggingface-local-models` | ai-ml | CUDA model selection, memory offloading (CPU ↔ GPU) | [SKILL.md](./skills/huggingface-local-models/SKILL.md) |
| 8 | `train-sentence-transformers` | ai-ml | Sentence embeddings, zero-shot NLI, cross-encoder scoring | [SKILL.md](./skills/train-sentence-transformers/SKILL.md) |
| 9 | `scikit-learn` | ai-ml | Classical ML pipelines, topic modeling (LDA), CountVectorizer | [SKILL.md](./skills/scikit-learn/SKILL.md) |
| 10 | `llm-application-dev-ai-assistant` | ai-ml | Conversational AI architecture, multi-turn dialogue state | [SKILL.md](./skills/llm-application-dev-ai-assistant/SKILL.md) |
| 11 | `conversation-memory` | memory | Multi-tier conversation memory, entity tracking, persistence | [SKILL.md](./skills/conversation-memory/SKILL.md) |
| 12 | `context-window-management` | memory | Context summarization, history pruning, token budget management | [SKILL.md](./skills/context-window-management/SKILL.md) |
| 13 | `agent-creator` | ai-ml | Persona generation, character consistency (Cooper companion) | [SKILL.md](./skills/agent-creator/SKILL.md) |
| 14 | `voice-ai-engine-development` | ai-ml | Real-time voice engine: Mic capture → Whisper ASR → TTS | [SKILL.md](./skills/voice-ai-engine-development/SKILL.md) |
| 15 | `audio-transcriber` | voice-agents | Audio recording ingestion & transcription with OpenAI Whisper | [SKILL.md](./skills/audio-transcriber/SKILL.md) |
| 16 | `computer-vision-expert` | ai-ml | Vision Language Models (VLMs), spatial perception, VQA | [SKILL.md](./skills/computer-vision-expert/SKILL.md) |
| 17 | `hugging-face-vision-trainer` | ai-ml | Hugging Face Transformers multimodal vision models (GitForCausalLM) | [SKILL.md](./skills/hugging-face-vision-trainer/SKILL.md) |
| 18 | `docker-expert` | devops | Multi-stage Docker builds, image optimization (Larynx TTS) | [SKILL.md](./skills/docker-expert/SKILL.md) |
| 19 | `container-security-hardening` | security | Base image security, CVE scanning, non-root user execution | [SKILL.md](./skills/container-security-hardening/SKILL.md) |
| 20 | `pydantic-ai` | ai-agents | Type-safe multi-model clients (OpenAI API vs. local fallback) | [SKILL.md](./skills/pydantic-ai/SKILL.md) |
| 21 | `llm-structured-output` | ai-ml | Schema-constrained decoding, reliable JSON output parsing | [SKILL.md](./skills/llm-structured-output/SKILL.md) |
| 22 | `compile-knowledge` | productivity | Markdown knowledge base stores, atomic notes, question bank | [SKILL.md](./skills/compile-knowledge/SKILL.md) |
| 23 | `rag-engineer` | data-ai | Retrieval-Augmented Generation, vector similarity, FAISS | [SKILL.md](./skills/rag-engineer/SKILL.md) |
| 24 | `python-testing-patterns` | development | pytest test architecture, fixtures, external API/hardware mocks | [SKILL.md](./skills/python-testing-patterns/SKILL.md) |
| 25 | `pytest-skill` | testing | Production-grade pytest test generation and conftest patterns | [SKILL.md](./skills/pytest-skill/SKILL.md) |
| 26 | `git-hooks-automation` | workflow | Pre-commit lifecycle: automated formatting (black) and linting | [SKILL.md](./skills/git-hooks-automation/SKILL.md) |
| 27 | `repo-maintainer` | uncategorized | Repository hygiene audits, dependency health, artifact cleanup | [SKILL.md](./skills/repo-maintainer/SKILL.md) |
| 28 | `007` | security | Threat modeling, OWASP review, fixing insecure pickle RPC | [SKILL.md](./skills/007/SKILL.md) |
| 29 | `varlock-claude-skill` | security | Secure environment variable and API key secret protection | [SKILL.md](./skills/varlock-claude-skill/SKILL.md) |
| 30 | `documentation` | workflow-bundle | API docs, architecture diagrams, docstrings generation | [SKILL.md](./skills/documentation/SKILL.md) |
| 31 | `readme` | content | Comprehensive open-source README.md authoring | [SKILL.md](./skills/readme/SKILL.md) |
| 32 | `git-workflow-and-versioning` | workflow | Git branching, conventional commits, parallel workstreams | [SKILL.md](./skills/git-workflow-and-versioning/SKILL.md) |
| 33 | `baseline-ui` | uncategorized | Visual layout, hierarchy, and spacing for tkinter dialogs | [SKILL.md](./skills/baseline-ui/SKILL.md) |

---

## Progressive Disclosure

Each skill directory contains a `SKILL.md` file equipped with standardized YAML frontmatter (`name` and `description`). Antigravity scans these descriptions at startup and only injects the full skill instructions when relevant to a specific user request or task.
