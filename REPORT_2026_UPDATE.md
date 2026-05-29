# Report: Updating DeskBuddy to 2026 Standards

## Executive Summary
This report outlines the necessary steps to upgrade the `DeskBuddy` (also known as COOPER) project to modern 2026 standards. Currently, the application is anchored in early-2023 infrastructure, notably running Python versions under 3.11, outdated PyTorch (1.13.1), and early generations of parameter offloading infrastructure like Accelerate 0.x, DeepSpeed 0.x, and Transformers 4.26.

This upgrade proposes modernizing the Python runtime to the stable Python 3.14 branch, replacing legacy dependencies, overhauling model architecture to utilize cutting-edge efficient architectures like Mistral, Llama 3, or Phi-3 in place of Pygmalion, and adjusting Docker usage.

---

## 1. Architectural Adjustments

The current `architecture.md` depicts an ecosystem running `BotCortex.py` via `start_bot.py`, interacting via TCP with a hardware layer and whisper STT.

### A. Python Version Upgrade (Python 3.14.5)
The most recent stable version of Python as of May 2026 is **3.14.5**. Upgrading to Python 3.14 provides faster startup, significant memory optimizations (like tighter garbage collection), improved multithreading for concurrent tasks in `BotCortex.py`, and far better debugger interfaces.

* **Action**: Update local development environments and Dockerfiles (e.g., `larynx.Dockerfile`) to pull from `python:3.14-slim` or equivalent.
* **Code Adjustments**: Some APIs (like `asyncio` loop handling, typing annotations) have evolved since Python 3.9/3.10. Code will need to leverage native deferred typing and check `asyncio.TaskGroup` for concurrency in `BotCortex.py` and `robot_client.py`.

### B. LLM Upgrades (Replacing Pygmalion)
The project heavily utilizes `Pygmalion-6B` (GPT-J architecture) and `pygmalion-350m`. These models are vastly outdated compared to 2026 models in terms of reasoning, context length, and conversational alignment.
* **Large Tier**: Replace Pygmalion-6B with a model like **Llama-3-8B-Instruct** or **Mistral-7B-v0.3/0.4**. These have superior logical reasoning and native fast-attention mechanisms. Llama-3 handles persona instructions ("Cooper's Persona") effortlessly.
* **Medium/Small Tier**: Replace BlenderBot and Pygmalion-350m with **Phi-3-Mini** (3.8B) or **Qwen1.5-0.5B**. They provide immense capability at a fraction of the VRAM footprint.
* **Tooling Changes**: `transformers.models.gptj.modeling_gptj.GPTJBlock` explicitly imported in `BotCortex.py` must be removed as model architecture will change. Instead, rely on `AutoModelForCausalLM` and modern bitsandbytes quantization (e.g. 4-bit or 8-bit QLoRA variants) to avoid manually wrestling with DeepSpeed Zero-3 CPU offloading for a 6B model.

### C. Speech and Vision Upgrades
* **Whisper ASR**: The project uses an outdated git branch of `openai/whisper`. We should switch to the official pip release or better yet, `distil-whisper` or `faster-whisper`, which provide 4x to 6x speedups for real-time speech processing on consumer hardware.
* **VQA (Vision Question Answering)**: `microsoft/git-large-vqav2` is currently used. This can be replaced by newer multimodal models (e.g., `Llava-1.5` or `Qwen-VL`), which are naturally capable of handling both conversational history and image context simultaneously.

---

## 2. Dependency Modernization (`requirements.txt`)

The `requirements.txt` is heavily locked to early-2023 packages. Most importantly, Accelerate, DeepSpeed, Transformers, and PyTorch require massive version bumps.

### Core AI Libraries
* **PyTorch**: Upgrade from `1.13.1` to `torch>=2.12.0`. PyTorch 2.x introduces `torch.compile`, which provides huge inference speedups out-of-the-box, mitigating the need for some custom Accelerate loops.
* **Transformers**: Upgrade from `4.26.0` to `5.9.0` (or `>5.0.0`). The `transformers` library has deprecated many older offload methodologies in favor of seamless integration with Accelerate and BitsAndBytes.
* **Accelerate**: Move from the custom git fork (`-e git+ssh://git@github.com/dhar174/accelerate.git...`) to the stable pip release `accelerate>=1.13.0`.
* **DeepSpeed**: Upgrade from local cloned folder `@ file:///home/darf3/buddy/DeepSpeed` to official `deepspeed>=0.19.0`. DeepSpeed's APIs and `HfDeepSpeedConfig` logic have stabilized significantly. Note that native Accelerate FSDP or device_maps may completely replace the need for DeepSpeed in single-node inference.

### Quality of Life & Helper Libraries
* **Bitsandbytes**: Upgrade from `0.36.0` to latest (e.g., `>0.43.0`) for native 4-bit/8-bit support without manual compilation issues. Remove `bitsandbytes-cuda117`.
* **CUDA Toolkits**: Switch from `cu11` specific bindings (like `nvidia-cuda-runtime-cu11==11.7.99`) to PyTorch 2.12's native `cu130` wheel integrations to support Ada and Hopper generation GPUs natively.
* **Sentence-Transformers**: Used for semantic search querying. Upgrade to the latest stable release (e.g., `2.7.x` or `3.x`) for better FAISS integrations and embedding efficiency.
* **OpenAI API**: The project uses an extremely old `openai` pattern (`openai.ChatCompletion.create`). This needs to be rewritten to the modern `openai>=1.0.0` client instantiation standard (`client = OpenAI(api_key=...); client.chat.completions.create(...)`).

### Cleanup
* Remove hardcoded local paths (e.g., `-e /home/darf3/buddy/triton/python` and `/home/darf3/buddy/offload`).
* Remove `intel-extension-for-pytorch` unless explicitly running on Intel Arc hardware, as PyTorch 2.x covers CPU optimizations much better now.

---

## 3. Code Adjustments and Best Practices

### A. DeepSpeed & Accelerate Inference
`BotCortex.py` uses extremely convoluted manual device placement, `infer_auto_device_map`, and manual offloading combined with DeepSpeed `OnDevice`.
In 2026, Hugging Face `pipeline` and `AutoModelForCausalLM.from_pretrained` support `device_map="auto"` seamlessly, outperforming manual DeepSpeed Zero-3 implementations for inference.
* **Recommendation**: Strip out `set_deepspeed_activation_checkpointing` and `HfDeepSpeedConfig` for pure inference tasks. Use `accelerate` with native 4-bit quantization.

### B. OpenAI API Migration
As noted, `openai.ChatCompletion.create` will hard-crash with any OpenAI library >= 1.0.0.
* **Fix**: Update `BotCortex.py` to instantiate `client = openai.AsyncOpenAI()` and use `await client.chat.completions.create(...)`. (All legacy `openai.ChatCompletion.create` and `openai.Completion.create` calls reside in `BotCortex.py`; `helpers.py` contains no `openai` usage.)

### C. Asynchronous Programming
`BotCortex.py` heavily uses `asyncio`, but often blocks the event loop with synchronous HuggingFace model calls (e.g., `model.generate()`).
* **Fix**: Use `asyncio.to_thread(model.generate, ...)` or equivalent thread pool offloading to ensure `start_bot.py` UI and `robot_client.py` hardware pings do not freeze while the LLM generates tokens.

---

## Conclusion
The upgrade to 2026 standards represents a significant leap from the 2023 baseline. By moving to Python 3.14, replacing GPT-J architecture with modern counterparts like Llama 3/Phi-3, updating PyTorch to 2.12+, and stripping out brittle custom DeepSpeed code in favor of native Accelerate `device_map="auto"`, COOPER will see drastically improved reasoning, generation speed, and stability, with a lower VRAM overhead.