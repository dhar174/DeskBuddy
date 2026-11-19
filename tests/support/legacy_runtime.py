"""Lightweight AST-based runtime harness for testing legacy functions without heavy imports."""

from __future__ import annotations

import ast
import asyncio
import logging
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from importlib.abc import MetaPathFinder
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from deskbuddy import question_bank
from deskbuddy.legacy_worker import LegacyWorker, WorkerBusyError, WorkerClosedError
from deskbuddy.settings import Settings
from tests.support.fakes import (
    CallRecorder,
    FakeCausalLM,
    FakeCommandModel,
    FakeCountVectorizer,
    FakeEmbeddingRuntime,
    FakeEmbeddingVector,
    FakeLDA,
    FakeNLIClassifier,
    FakeNLP,
    FakeOpenAISDK,
    FakeQuestionClassifier,
    FakeTokenizer,
    FakeTorch,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOT_CORTEX_PATH = REPOSITORY_ROOT / "BotCortex.py"
HELPERS_PATH = REPOSITORY_ROOT / "helpers.py"
START_BOT_PATH = REPOSITORY_ROOT / "start_bot.py"

HEAVY_MODULES = frozenset(
    {
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "sentence_transformers",
        "deepspeed",
        "accelerate",
        "spacy",
        "PIL",
        "aiohttp",
        "psutil",
        "cv2",
        "whisper",
        "openai",
        "tkinter",
        "picoh",
        "datasets",
        "sklearn",
        "scipy",
        "keyboard",
        "pyaudio",
        "sounddevice",
    }
)


class HeavyImportBlocker(MetaPathFinder):
    """Fail immediately if any test attempts to import heavyweight runtime modules."""

    def __init__(self, blocked: frozenset[str] = HEAVY_MODULES) -> None:
        self.blocked = blocked

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        root = fullname.split(".", maxsplit=1)[0]
        if root in self.blocked:
            raise AssertionError(f"forbidden heavyweight import during contract test: {fullname}")
        return None


def extract_module_functions(
    path: Path,
    names: set[str],
    globals_: dict[str, Any],
) -> dict[str, Any]:
    """Compile and execute selected top-level functions from a legacy file in isolation."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }
    missing = names - functions.keys()
    assert not missing, f"Missing expected functions in {path.name}: {sorted(missing)}"
    module = ast.Module(body=list(functions.values()), type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"__builtins__": __builtins__, **globals_}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def install_heavy_import_guard(monkeypatch: Any) -> set[str]:
    """Activate HeavyImportBlocker and return currently loaded heavy modules baseline."""
    loaded_before = {
        name.split(".", maxsplit=1)[0]
        for name in sys.modules
        if name.split(".", maxsplit=1)[0] in HEAVY_MODULES
    }
    monkeypatch.setattr(sys, "meta_path", [HeavyImportBlocker(), *sys.meta_path])
    return loaded_before


def assert_no_new_heavy_imports(loaded_before: set[str]) -> None:
    loaded_after = {
        name.split(".", maxsplit=1)[0]
        for name in sys.modules
        if name.split(".", maxsplit=1)[0] in HEAVY_MODULES
    }
    assert loaded_after == loaded_before


def load_helpers_runtime(
    *,
    recorder: CallRecorder | None = None,
    nli_classifier: Any = None,
    isquestion_model: Any = None,
    isquestion_tokenizer: Any = None,
    local_only_mode: bool = True,
    is_requesting_sd: Callable[..., Any] | None = None,
    question_answerer: Any = None,
    command_model: Any = None,
    command_tokenizer: Any = None,
    command_classifier: Any = None,
    torch_module: Any = None,
    legacy_worker: LegacyWorker | None = None,
) -> tuple[dict[str, Any], CallRecorder, SimpleNamespace]:
    """Load real helper functions from helpers.py backed by deterministic fakes."""
    rec = recorder or CallRecorder()
    nli = nli_classifier or FakeNLIClassifier(rec)
    q_model = isquestion_model or FakeQuestionClassifier(rec, is_question=True)
    q_tok = isquestion_tokenizer or FakeTokenizer(rec, name="isquestion_tokenizer")
    cmd_model = command_model or FakeCommandModel(rec)
    cmd_tok = command_tokenizer or FakeTokenizer(rec, name="command_tokenizer")
    worker_owner = legacy_worker or LegacyWorker("desktop-fixture")

    async def run_legacy_stage(operation: Callable[[], Any], *, stop: Any = None) -> Any:
        nonlocal worker_owner
        if worker_owner.can_replace:
            worker_owner = LegacyWorker("desktop-fixture")
        worker_owner.start()
        return await worker_owner.submit(operation, stop=stop)

    async def default_requesting_sd(txt: str) -> bool:
        rec.record("BotCortex.is_requesting_SD", txt)
        return False

    fake_cortex = SimpleNamespace(
        local_only_mode=local_only_mode,
        nli_classifier=nli,
        isquestion_model=q_model,
        isquestion_tokenizer=q_tok,
        is_requesting_SD=is_requesting_sd or default_requesting_sd,
        command_classifier=command_classifier,
        command_tokenizer=cmd_tok,
        run_legacy_stage=run_legacy_stage,
        legacy_worker=worker_owner,
    )

    helper_names = {
        "_best_nli_label_sync",
        "_is_question_sync",
        "_answer_question_based_on_context_sync",
        "_needs_follow_up_sync",
        "_interpret_command_sync",
        "_command_type_sync",
        "is_question",
        "is_chat_or_command",
        "is_personal",
        "answer_question_based_on_context",
        "needs_follow_up",
        "submit_command",
        "interpret_command",
        "execute_command",
        "is_requesting_image",
        "is_visual_question",
        "command_type",
    }
    ns = extract_module_functions(
        HELPERS_PATH,
        helper_names,
        {
            "BotCortex": fake_cortex,
            "torch": torch_module or FakeTorch(),
            "question_answerer": question_answerer
            or (lambda **_kw: {"answer": "Paris", "score": 0.9, "start": 0, "end": 5}),
            "command_model": cmd_model,
            "command_tokenizer": cmd_tok,
        },
    )
    return ns, rec, fake_cortex


def load_botcortex_runtime(
    *,
    settings: Settings | None = None,
    recorder: CallRecorder | None = None,
    local_only_mode: bool = True,
    nli_classifier: Any = None,
    is_question: bool = True,
    openai_sdk: FakeOpenAISDK | None = None,
    vqa_handler: Callable[[str], Any] | None = None,
    embedding_scores: list[float] | None = None,
    qdict_data: dict[str, str] | None = None,
    qgen_output: str = "Could you clarify what you mean?",
    use_disk_question_bank: bool = False,
    legacy_worker: LegacyWorker | None = None,
) -> tuple[dict[str, Any], CallRecorder]:
    """Load real routing, memory, and talk_* functions from BotCortex.py."""
    rec = recorder or CallRecorder()
    fixture_tempdir = None
    if settings is None:
        fixture_tempdir = tempfile.TemporaryDirectory(prefix="deskbuddy-legacy-runtime-")
        data_dir = Path(fixture_tempdir.name) / "data"
        effective_settings = Settings(
            openai_model="gpt-3.5-turbo-test",
            camera_source="fixture://camera",
            data_dir=data_dir,
            cache_dir=Path(fixture_tempdir.name) / "cache",
        )
    else:
        effective_settings = settings
    desktop_worker = legacy_worker or LegacyWorker("desktop-fixture")
    worker_owners = {
        "desktop": desktop_worker,
        "question-bank": LegacyWorker("question-bank-fixture", use_default_executor=True),
        "vision": LegacyWorker("vision-fixture", use_default_executor=True),
    }
    helpers_ns, _, _ = load_helpers_runtime(
        recorder=rec,
        nli_classifier=nli_classifier,
        isquestion_model=FakeQuestionClassifier(rec, is_question=is_question),
        local_only_mode=local_only_mode,
        legacy_worker=desktop_worker,
    )
    fake_helpers = SimpleNamespace(
        is_visual_question=helpers_ns["is_visual_question"],
        needs_follow_up=helpers_ns["needs_follow_up"],
        is_question=helpers_ns["is_question"],
        is_personal=helpers_ns["is_personal"],
        is_requesting_image=helpers_ns["is_requesting_image"],
    )

    fake_openai = openai_sdk or FakeOpenAISDK(rec)
    emb_runtime = FakeEmbeddingRuntime(rec, scores=embedding_scores)
    qgen_mod = FakeCausalLM(rec, name="qgen_model")

    class _FakeQGenTokenizer(FakeTokenizer):
        def decode(self, token_ids: Any, **_kwargs: Any) -> str:
            self.recorder.record("qgen_tokenizer.decode", token_ids)
            return qgen_output

    qgen_tok = _FakeQGenTokenizer(rec, name="qgen_tokenizer")

    class _FakeChatTokenizer(FakeTokenizer):
        def decode(self, token_ids: Any, **_kwargs: Any) -> str:
            self.recorder.record("chat_tokenizer.decode", token_ids)
            tokens = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
            if tokens and isinstance(tokens[0], list):
                tokens = tokens[0]
            if 99 in tokens:
                return "LOCAL_MODEL_REPLY"
            return "PROMPT_INPUT"

    chat_tok = _FakeChatTokenizer(rec, name="chat_tokenizer")
    chat_mod = FakeCausalLM(rec, name="chat_model", output_tokens=[[10, 11, 12, 13, 99, 100]])
    sum_tok = FakeTokenizer(rec, name="sumtokenizer", decoded_prefix="SUMMARY")
    sum_mod = FakeCausalLM(rec, name="longsummarizer")
    topic_tok = FakeTokenizer(rec, name="topic_tokenizer", decoded_prefix="TOPIC")
    topic_mod = FakeCausalLM(rec, name="topic_model")

    memory_store = dict(qdict_data) if qdict_data is not None else {}

    async def fake_load_qdict(*, allow_unavailable: bool = False) -> dict[str, str]:
        _ = allow_unavailable
        rec.record("load_qdict_from_json")
        return dict(memory_store)

    async def fake_save_qdict(my_dict: dict[str, str]) -> None:
        rec.record("save_qdict_to_json", dict(my_dict))
        memory_store.clear()
        memory_store.update(my_dict)

    def fake_vqa_sync(question: str, _settings: Any) -> str:
        rec.record("answer_visual_question", question)
        if vqa_handler is not None:
            result = vqa_handler(question)
            if asyncio.iscoroutine(result):
                return asyncio.run(result)
            return result
        return "I think it is a desk lamp"

    def fake_get_embeddings_sync(texts: Any, *_args: Any) -> FakeEmbeddingVector:
        rec.record("get_embeddings", texts)
        return FakeEmbeddingVector(embedding_scores or [0.95, 0.2])

    async def fake_save_verbatim(text: Any) -> None:
        rec.record("save_history_verbatim", text)

    async def fake_save_summary(text: Any) -> None:
        rec.record("save_history_summary", text)

    cortex_names = {
        "_openai_model_name",
        "_get_legacy_worker",
        "start_workers",
        "shutdown_workers",
        "run_legacy_stage",
        "_get_turn_lock",
        "_serialize_turn",
        "_clear_question_bank_warning",
        "_warn_question_bank_unavailable",
        "_find_names_sync",
        "search_question_in_bank",
        "_search_question_sync",
        "_mean_pooling_sync",
        "_get_embeddings_sync",
        "mean_pooling",
        "get_embeddings",
        "query_memory",
        "find_names",
        "_generate_initial_summary_sync",
        "_generate_summary_sync",
        "_extract_topics_sync",
        "generate_summary",
        "extract_topics",
        "get_list_of_list_of_strings",
        "_generate_follow_up_sync",
        "_openai_chat_sync",
        "_visual_question_cloud_sync",
        "_request_image_cloud_sync",
        "_atomic_write_history_text",
        "_save_history_summary_sync",
        "_save_history_verbatim_sync",
        "_save_history_export_sync",
        "is_visual_q",
        "get_follow_up_question",
        "classify_input",
        "save_history_verbatim",
        "save_history_summary",
        "save_history_export",
        "talk_history",
        "talk_medium_API_with_history",
        "talk_medium_with_history",
        "_generate_medium_history_reply_sync",
        "talk_medium",
        "talk",
        "talk_fast",
        "talk_small",
        "_generate_reply_sync",
        "_generate_large_history_reply_sync",
        "is_requesting_SD",
        "_answer_visual_question_sync",
        "answer_visual_question",
        "load_all_models",
    }
    if use_disk_question_bank:
        cortex_names.update({"load_qdict_from_json", "save_qdict_to_json"})

    async def fake_init_fn(name: str) -> tuple[str, str]:
        rec.record(name)
        return (f"model:{name}", f"tok:{name}")

    fake_deepspeed = SimpleNamespace(
        init_distributed=lambda backend: rec.record("deepspeed.init_distributed", backend)
    )
    fake_auto_tok = SimpleNamespace(
        from_pretrained=lambda model_id, **_kw: (
            rec.record("AutoTokenizer.from_pretrained", model_id)
            or FakeTokenizer(rec, name=f"tok:{model_id}")
        )
    )
    fake_seq2seq = SimpleNamespace(
        from_pretrained=lambda model_id, **_kw: (
            rec.record("AutoModelForSeq2SeqLM.from_pretrained", model_id)
            or FakeCausalLM(rec, name=f"seq2seq:{model_id}")
        )
    )
    fake_seq_cls = SimpleNamespace(
        from_pretrained=lambda model_id, **_kw: (
            rec.record("AutoModelForSequenceClassification.from_pretrained", model_id)
            or FakeQuestionClassifier(rec)
        )
    )
    fake_spacy = SimpleNamespace(
        load=lambda model_id: rec.record("spacy.load", model_id) or FakeNLP(rec)
    )

    def fake_pipeline(**kwargs: Any) -> FakeNLIClassifier:
        rec.record("pipeline", **kwargs)
        return FakeNLIClassifier(rec)

    globals_: dict[str, Any] = {
        "asyncio": asyncio,
        "__name__": "BotCortex",
        "logging": logging,
        "question_bank": question_bank,
        "_question_bank_warning_lock": threading.Lock(),
        "_question_bank_warning_reasons": {},
        "threading": threading,
        "tempfile": __import__("tempfile"),
        "os": __import__("os"),
        "nullcontext": __import__("contextlib").nullcontext,
        "wraps": __import__("functools").wraps,
        "LegacyWorker": LegacyWorker,
        "WorkerBusyError": WorkerBusyError,
        "WorkerClosedError": WorkerClosedError,
        "_legacy_workers": worker_owners,
        "_history_export_lock": threading.Lock(),
        "_turn_lock": None,
        "_turn_lock_loop": None,
        "_runtime_shutdown": False,
        "time": SimpleNamespace(time=time.monotonic, sleep=lambda _s: None),
        "load_settings": lambda overrides=None: effective_settings,
        "torch": FakeTorch(),
        "F": emb_runtime,
        "_get_embeddings_sync": fake_get_embeddings_sync,
        "load_qdict_from_json": fake_load_qdict,
        "save_qdict_to_json": fake_save_qdict,
        "_answer_visual_question_sync": fake_vqa_sync,
        "save_history_verbatim": fake_save_verbatim,
        "save_history_summary": fake_save_summary,
        "helpers": fake_helpers,
        "openai": fake_openai,
        "deepspeed": fake_deepspeed,
        "AutoTokenizer": fake_auto_tok,
        "AutoModelForSeq2SeqLM": fake_seq2seq,
        "AutoModelForSequenceClassification": fake_seq_cls,
        "spacy": fake_spacy,
        "pipeline": fake_pipeline,
        "LatentDirichletAllocation": lambda **_kw: FakeLDA(rec),
        "CountVectorizer": FakeCountVectorizer,
        "init_fast": lambda: fake_init_fn("init_fast"),
        "init_small": lambda: fake_init_fn("init_small"),
        "init_medium": lambda: fake_init_fn("init_medium"),
        "init_API_medium": lambda: fake_init_fn("init_API_medium"),
        "init": lambda: fake_init_fn("init"),
        "local_only_mode": local_only_mode,
        "model_size": "",
        "unfound_answer": False,
        "asked_follow_up": False,
        "after_first": False,
        "after_second": False,
        "count_index": 0,
        "fp_sum": "Initial summary",
        "device": "cpu",
        "accelerator": None,
        "named_entities": [],
        "important_words": [],
        "history": [
            " Human: Hi! My name is Human. I like cats and video games!",
            "Cooper: Hello, I am Cooper. How can I help?",
        ],
        "history_nonlocal": [
            {"role": "system", "content": "Cooper's Persona: friendly kid robot."},
            {"role": "user", "content": "Hello Cooper!"},
            {"role": "assistant", "content": "Hi! Ready to chat!"},
        ],
        "model": chat_mod,
        "tokenizer": chat_tok,
        "qgen_model": qgen_mod,
        "qgen_tokenizer": qgen_tok,
        "sumtokenizer": sum_tok,
        "longsummarizer": sum_mod,
        "shortsummarizer": sum_mod,
        "titlesummarizer": sum_mod,
        "topic_tokenizer": topic_tok,
        "topic_model": topic_mod,
        "nlp": FakeNLP(rec),
        "lda": FakeLDA(rec),
        "isquestion_model": FakeQuestionClassifier(rec, is_question=is_question),
        "isquestion_tokenizer": FakeTokenizer(rec, name="isquestion_tokenizer"),
        "nli_classifier": nli_classifier or FakeNLIClassifier(rec),
        "command_tokenizer": None,
        "command_classifier": None,
        "search_tokenizer": None,
        "search_model": None,
        "optimizer": None,
        "_fake_helpers": fake_helpers,
        "cv": SimpleNamespace(__version__="fixture"),
        "__file__": str(BOT_CORTEX_PATH),
    }
    ns = extract_module_functions(BOT_CORTEX_PATH, cortex_names, globals_)
    # Keep test-only leaf models deterministic while exercising the real async
    # wrappers and their real worker admission/cancellation boundary.
    ns["_get_embeddings_sync"] = fake_get_embeddings_sync
    ns["_answer_visual_question_sync"] = fake_vqa_sync
    ns["legacy_worker"] = desktop_worker
    ns["workers"] = worker_owners
    if fixture_tempdir is not None:
        ns["_fixture_tempdir"] = fixture_tempdir

    real_save_history_export = ns["_save_history_export_sync"]

    def recorded_save_history_export(
        settings_snapshot: Settings,
        summary: str | None,
        verbatim: str,
        stop: Any,
    ) -> bool:
        accepted = real_save_history_export(settings_snapshot, summary, verbatim, stop)
        if accepted:
            if summary is not None:
                rec.record("save_history_summary", summary)
            rec.record("save_history_verbatim", verbatim)
        return accepted

    ns["_save_history_export_sync"] = recorded_save_history_export

    if use_disk_question_bank:
        real_load_qdict = ns["load_qdict_from_json"]
        real_save_qdict = ns["save_qdict_to_json"]

        async def recorded_load_qdict(*, allow_unavailable: bool = False) -> dict[str, str] | None:
            rec.record("load_qdict_from_json")
            return await real_load_qdict(allow_unavailable=allow_unavailable)

        async def recorded_save_qdict(entries: dict[str, str]) -> None:
            rec.record("save_qdict_to_json", dict(entries))
            await real_save_qdict(entries)

        ns["load_qdict_from_json"] = recorded_load_qdict
        ns["save_qdict_to_json"] = recorded_save_qdict

    # Keep a live proxy so helper calls observe routing state mutations.
    class _LiveCortexProxy:
        def __getattr__(self, item: str) -> Any:
            return ns[item]

    helpers_ns["BotCortex"] = _LiveCortexProxy()
    return ns, rec


def load_start_bot_dispatch(
    recorder: CallRecorder | None = None,
) -> tuple[Callable[..., Any], CallRecorder, SimpleNamespace]:
    """Load real start_bot.dispatch_reply with a recording fake BotCortex."""
    rec = recorder or CallRecorder()

    async def make_talk(name: str, message: str) -> str:
        rec.record(f"BotCortex.{name}", message)
        return f"REPLY:{name}:{message}"

    fake_cortex = SimpleNamespace(
        talk_history=lambda msg: make_talk("talk_history", msg),
        talk_small=lambda msg: make_talk("talk_small", msg),
        talk_medium_with_history=lambda msg: make_talk("talk_medium_with_history", msg),
        talk_medium_API_with_history=lambda msg: make_talk("talk_medium_API_with_history", msg),
        talk_fast=lambda msg: make_talk("talk_fast", msg),
    )
    ns = extract_module_functions(
        START_BOT_PATH,
        {"dispatch_reply"},
        {"BotCortex": fake_cortex},
    )
    return ns["dispatch_reply"], rec, fake_cortex


def load_start_bot_main(
    cortex: dict[str, Any],
    robot_client: Any,
    *,
    local_only: bool,
    button_text: str = "Medium",
    sleep_calls: list[float] | None = None,
    startup_calls: list[tuple[str, tuple[Any, ...]]] | None = None,
    lifecycle_calls: list[str] | None = None,
) -> Callable[..., Any]:
    """Load the legacy desktop loop without importing its GUI or runtime."""

    async def skip_cortex_startup(*args: Any) -> None:
        if startup_calls is not None:
            startup_calls.append(("cortex", args))

    async def skip_helper_startup(*args: Any) -> None:
        if startup_calls is not None:
            startup_calls.append(("helpers", args))

    def record_start_workers() -> None:
        if lifecycle_calls is not None:
            lifecycle_calls.append("start_workers")
        cortex["start_workers"]()

    def record_shutdown_workers() -> None:
        if lifecycle_calls is not None:
            lifecycle_calls.append("shutdown_workers")
        cortex["shutdown_workers"]()

    async def yield_without_delay(delay: float) -> None:
        if sleep_calls is not None:
            sleep_calls.append(delay)
        await asyncio.sleep(0)

    class LocalOnlyOption:
        def set(self, value: bool) -> None:
            self.value = value

        def get(self) -> bool:
            return self.value

    def widget(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(pack=lambda: None)

    ns = extract_module_functions(
        START_BOT_PATH,
        {"main", "dispatch_reply"},
        {
            "tk": SimpleNamespace(Button=widget, Checkbutton=widget, BooleanVar=LocalOnlyOption),
            "root": SimpleNamespace(mainloop=lambda: None),
            "BotCortex": SimpleNamespace(
                **{
                    **cortex,
                    "load_all_models": skip_cortex_startup,
                    "start_workers": record_start_workers,
                    "shutdown_workers": record_shutdown_workers,
                }
            ),
            "helpers": SimpleNamespace(load_models=skip_helper_startup),
            "rc": robot_client,
            "asyncio": SimpleNamespace(
                sleep=yield_without_delay, TimeoutError=asyncio.TimeoutError
            ),
            "local_only": local_only,
            "button_text": button_text,
        },
    )
    return ns["main"]
