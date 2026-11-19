# from torchdistx.fake import fake_mode
# from optimum.onnxruntime import OrtModelForCausalLM
# from soft_embedding import SoftEmbedding
import asyncio
import logging
import threading
import psutil
from deskbuddy import question_bank
from deskbuddy.legacy_worker import LegacyWorker, WorkerBusyError, WorkerClosedError
from deskbuddy.settings import SettingsError, load_settings
from PIL import Image
import cv2 as cv
from transformers.models.gptj.modeling_gptj import GPTJBlock
from deepspeed import OnDevice
import deepspeed
from transformers.utils import is_accelerate_available
from transformers.integrations.deepspeed import (
    HfDeepSpeedConfig,
    is_deepspeed_zero3_enabled,
    is_deepspeed_available,
)
from accelerate import Accelerator, DistributedType
from accelerate import load_checkpoint_and_dispatch, dispatch_model, dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM, CONFIG_MAPPING, MODEL_MAPPING
from accelerate import infer_auto_device_map, init_empty_weights
import spacy
from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    AutoModelForCausalLM,
    SummarizationPipeline,
    AutoModelForSeq2SeqLM,
    AutoModel,
    GPTJForCausalLM,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotForCausalLM,
    OPTForCausalLM,
    MPNetModel,
    MPNetTokenizer,
    MPNetConfig,
)
from torch.cuda.amp import custom_fwd, custom_bwd
from torch import nn
import torch.nn.functional as F
import torch
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from functools import lru_cache, wraps
import time
import helpers
import datasets
from accelerate.utils import get_max_memory, is_bf16_available
import tempfile
from accelerate import cpu_offload
from accelerate import find_executable_batch_size
from transformers import pipeline
import os
from accelerate import Accelerator
import transformers
import numpy as np
import sklearn
from transformers import GitVisionConfig, GitForCausalLM, GitProcessor
import openai

_question_bank_warning_lock = threading.Lock()
_question_bank_warning_reasons = {}
_legacy_workers = {}
_history_export_lock = threading.Lock()
_turn_lock = None
_turn_lock_loop = None
_runtime_shutdown = False


def _get_legacy_worker(name, *, use_default_executor=False):
    if _runtime_shutdown:
        raise WorkerClosedError("DeskBuddy workers are shut down; call start_workers() first")
    worker = _legacy_workers.get(name)
    if worker is not None and worker.can_replace:
        worker.shutdown()
        worker = None
    if worker is None:
        worker = LegacyWorker(name, use_default_executor=use_default_executor)
        _legacy_workers[name] = worker
    try:
        worker.start()
    except WorkerClosedError:
        if not worker.can_replace:
            raise WorkerBusyError(f"{name} is still active and cannot be restarted")
        worker = LegacyWorker(name, use_default_executor=use_default_executor)
        _legacy_workers[name] = worker
        worker.start()
    except RuntimeError:
        if not worker.can_replace:
            if worker.busy:
                raise WorkerBusyError(f"{name} is still active on its previous loop")
            raise
        worker.shutdown()
        worker = LegacyWorker(name, use_default_executor=use_default_executor)
        _legacy_workers[name] = worker
        worker.start()
    return worker


def start_workers():
    """Start the legacy runtime owners on the active asyncio loop."""
    global _runtime_shutdown
    _runtime_shutdown = False
    try:
        _get_legacy_worker("desktop", use_default_executor=False)
        _get_legacy_worker("question-bank", use_default_executor=True)
        _get_legacy_worker("vision", use_default_executor=True)
    except Exception:
        _runtime_shutdown = True
        shutdown_workers()
        raise


def shutdown_workers():
    """Stop worker admission without blocking shutdown on active inference."""
    global _runtime_shutdown
    _runtime_shutdown = True
    busy_names = []
    for name, worker in tuple(_legacy_workers.items()):
        worker.shutdown()
        if worker.busy:
            busy_names.append(name)
    if busy_names:
        logging.getLogger("BotCortex").warning(
            "Legacy workers still active after shutdown: %s", ", ".join(busy_names)
        )


async def run_legacy_stage(operation, *, stop=None):
    """Run one complete synchronous model stage on the serialized desktop owner."""
    return await _get_legacy_worker("desktop", use_default_executor=False).submit(
        operation, stop=stop
    )


def _get_turn_lock():
    global _turn_lock
    global _turn_lock_loop
    loop = asyncio.get_running_loop()
    if _turn_lock_loop is not loop:
        if _turn_lock is not None and _turn_lock.locked():
            raise RuntimeError("A DeskBuddy turn is still active on another event loop")
        _turn_lock_loop = loop
        _turn_lock = asyncio.Lock()
    return _turn_lock


def _serialize_turn(function):
    @wraps(function)
    async def serialized(*args, **kwargs):
        async with _get_turn_lock():
            state_names = (
                "history",
                "history_nonlocal",
                "named_entities",
                "important_words",
                "after_first",
                "after_second",
                "unfound_answer",
                "asked_follow_up",
            )
            state_snapshot = {}
            for name in state_names:
                if name in globals():
                    value = globals()[name]
                    if isinstance(value, list):
                        value = list(value)
                    elif isinstance(value, dict):
                        value = dict(value)
                    state_snapshot[name] = value
            try:
                return await function(*args, **kwargs)
            except BaseException:
                for name, value in state_snapshot.items():
                    current = globals().get(name)
                    if isinstance(current, list) and isinstance(value, list):
                        current[:] = value
                    elif isinstance(current, dict) and isinstance(value, dict):
                        current.clear()
                        current.update(value)
                    else:
                        globals()[name] = value
                raise

    return serialized

unfound_answer = False
asked_follow_up = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The list of named entities that the model should remember
named_entities = []

# The list of important words or phrases that the model should remember
important_words = []

# A history that the model can use to generate responses
history = [
    " Human: Hi! My name is Human. I like cats and video games!",
    "Cooper: Hello, I am Cooper. I am a chatbot. I can have a friendly chat with you. I can also tell you jokes. What would you like to talk about?",
]

history_nonlocal = [
    {
        "role": "system",
        "content": "Cooper's Persona: Cooper is a sentient AI robot with the heart of a kid. Cooper's best friend is Human, and Cooper's goal in life is to be Human's friend and to be helpful to her. Cooper is smart, friendly, always kid appropriate and likes to make people laugh. Cooper knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cooper also likes cats. Cooper thinks farts are very funny. Cooper tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Cooper cares very much about Human's future, and wants her to do well in school.",
    },
    {"role": "user", "content": "Hello, Cooper! How was your day today?"},
    {"role": "assistant", "content": "Hi! My day was pretty good. How was yours?"},
    {
        "role": "user",
        "content": "I had a good day. I played some video games in the afternoon.",
    },
    {"role": "assistant", "content": "That sounds fun! What games did you play?"},
    {
        "role": "assistant",
        "content": "Thanks, I'll definitely keep that in mind. How is your schoolwork going?",
    },
    {
        "role": "user",
        "content": "Hey, have you played any of the Legend of Zelda games?",
    },
    {
        "role": "assistant",
        "content": "Yes, I've played a few of them! Ocarina of Time and Breath of the Wild are two of my favorites.",
    },
    {
        "role": "user",
        "content": "I loved Ocarina of Time too! What did you think of Breath of the Wild?",
    },
    {
        "role": "assistant",
        "content": "I thought Breath of the Wild was amazing! The open-world exploration and combat mechanics were really fun, and the story was really engaging.",
    },
    {
        "role": "user",
        "content": "I've heard great things about Breath of the Wild. Do you have a favorite character from the series?",
    },
    {
        "role": "assistant",
        "content": "It's tough to choose, but I think Link is my favorite. He's such a brave and determined hero.",
    },
    {
        "role": "user",
        "content": "I agree, Link is definitely a classic character. Have you played any of the newer Legend of Zelda games?",
    },
    {
        "role": "assistant",
        "content": "Yes, I've played Hyrule Warriors and Links Awakening as well. Both were really enjoyable in their own way.",
    },
    {
        "role": "user",
        "content": "I haven't played Hyrule Warriors or Links Awakening, but I'll have to check them out. Thanks for the recommendation!",
    },
    {
        "role": "assistant",
        "content": "I'm happy to be your AI assistant! I hope we have fun talking about games and stuff.",
    },
    {"role": "user", "content": "Me too! Let's keep talking."},
]


optimizer = None
named_entities = []
isquestion_tokenizer = None
isquestion_model = None
nli_classifier = None
command_tokenizer = None
command_classifier = None
nlp = None
longsummarizer = None
shortsummarizer = None
titlesummarizer = None
lda = None
tokenizer = None
model = None

sumtokenizer = None

topic_tokenizer = None

topic_model = None

qgen_tokenizer = None
qgen_model = None
model_size = ""
local_only_mode = True
search_tokenizer = None
search_model = None
_vqa_lock = threading.Lock()


def _max_memory_with_settings():
    settings = load_settings()
    limits = settings.resource_limits()
    if settings.device == "cpu":
        memory = {"cpu": psutil.virtual_memory().available}
        if limits is not None:
            memory.update({key: value for key, value in limits.items() if key == "cpu"})
        return memory

    resolved_device = settings.torch_device(torch)
    if resolved_device != "cuda":
        memory = {"cpu": psutil.virtual_memory().available}
        if limits is not None:
            memory.update({key: value for key, value in limits.items() if key == "cpu"})
        return memory

    memory = get_max_memory()
    if 0 not in memory:
        raise SettingsError("Accelerate could not determine available memory for CUDA device 0")

    available_bytes = _memory_limit_bytes(memory[0])
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    reserve_bytes = (total_bytes * 20 + 99) // 100
    automatic_gpu_limit = max(0, available_bytes - reserve_bytes)
    explicit_gpu_limit = limits.pop(0, None) if limits is not None else None
    if explicit_gpu_limit is not None:
        if _memory_limit_bytes(explicit_gpu_limit) > automatic_gpu_limit:
            raise SettingsError(
                "DESKBUDDY_GPU_MEMORY_LIMIT exceeds available CUDA memory after "
                "reserving 20% of device 0 VRAM"
            )
        memory[0] = explicit_gpu_limit
    else:
        memory[0] = automatic_gpu_limit

    memory = {key: memory[key] for key in (0, "cpu") if key in memory}
    if limits is not None:
        memory.update(limits)
    return memory


def _memory_limit_bytes(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise SettingsError("Memory limits must be integer bytes or use MiB/GiB units")
    units = {"MiB": 1024**2, "GiB": 1024**3}
    for suffix, scale in units.items():
        if value.endswith(suffix) and value[: -len(suffix)].isdecimal():
            return int(value[: -len(suffix)]) * scale
    raise SettingsError("Memory limits must use positive integer MiB or GiB units")


def _offload_path(settings=None):
    settings = settings or load_settings()
    if settings.offload_dir is None:
        raise RuntimeError("No offload directory is configured")
    return str(settings.offload_dir)


def _openai_model_name():
    return load_settings().openai_model


async def load_all_models(button_text, local_only):
    global isquestion_tokenizer
    global isquestion_model
    global nli_classifier
    global command_tokenizer
    global command_classifier
    global nlp
    global longsummarizer
    global shortsummarizer
    global titlesummarizer
    global lda
    global tokenizer
    global model
    global sumtokenizer
    global topic_tokenizer
    global topic_model
    global optimizer
    global qgen_tokenizer
    global qgen_model
    global model_size
    global local_only_mode
    global search_tokenizer
    global search_model
    # global accelerator
    provider = "local" if local_only else "openai"
    settings = load_settings(overrides={"model_provider": provider})
    if button_text == "Large":
        settings.require_path("checkpoint_dir")
        settings.require_path("tokenizer_dir")
    settings.prepare_directories()
    if provider == "openai":
        openai.api_key = settings.openai_api_key.get_secret_value()
    if button_text == "Large" or (provider == "openai" and button_text == "Medium"):
        deepspeed.init_distributed("nccl")
    local_only_mode = local_only
    model_size = button_text
    print("Loading models...")
    print("button_text: ", button_text)
    start_time = time.time()
    topic_tokenizer = AutoTokenizer.from_pretrained(settings.topic_model_id)

    topic_model = AutoModelForSeq2SeqLM.from_pretrained(settings.topic_model_id)
    print("topic model loaded")

    if local_only_mode:
        sumtokenizer = AutoTokenizer.from_pretrained(
            settings.summary_model_id, use_fast=True, truncation=True
        )

        nlp = spacy.load(settings.spacy_model_id)
        longsummarizer = AutoModelForSeq2SeqLM.from_pretrained(
            settings.summary_model_id
        )
        shortsummarizer = AutoModelForSeq2SeqLM.from_pretrained(
            settings.short_summary_model_id
        )
        shortsummarizer.to("cpu")
        longsummarizer.to("cpu")
        optimizer = torch.optim.Adam(longsummarizer.parameters())

        titlesummarizer = AutoModelForSeq2SeqLM.from_pretrained(
            settings.title_model_id
        )
        titlesummarizer.to("cpu")
        print("summarizers loaded")
    isquestion_tokenizer = AutoTokenizer.from_pretrained(
        settings.question_classifier_model_id
    )

    isquestion_model = AutoModelForSequenceClassification.from_pretrained(
        settings.question_classifier_model_id
    )
    print("question identifier model loaded")
    nli_classifier = pipeline(
        model=settings.nli_model_id,
        tokenizer=settings.nli_model_id,
        task="zero-shot-classification",
    )
    print("nli classifier loaded")
    command_tokenizer = AutoTokenizer.from_pretrained(
        settings.command_model_id
    )

    command_classifier = AutoModelForSequenceClassification.from_pretrained(
        settings.command_model_id
    )
    print("command classifier loaded")
    nlp = spacy.load(settings.spacy_model_id)

    lda = LatentDirichletAllocation(n_components=10)
    if local_only_mode == True:
        if button_text == "Fast":
            model, tokenizer = await init_fast()
        elif button_text == "Small":
            model, tokenizer = await init_small()
        elif button_text == "Medium":
            model, tokenizer = await init_medium()
        elif button_text == "Large":
            model, tokenizer = await init()
    else:
        if button_text == "Fast":
            model, tokenizer = await init_fast()
        elif button_text == "Small":
            model, tokenizer = await init_small()
        elif button_text == "Medium":
            model, search_model = await init_API_medium()
        elif button_text == "Large":
            model, tokenizer = await init()
    if local_only_mode:
        qgen_tokenizer = AutoTokenizer.from_pretrained(
            settings.question_generator_model_id
        )
        qgen_model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.question_generator_model_id
        )

        # search_tokenizer = AutoTokenizer.from_pretrained(
        #     "sentence-transformers/all-mpnet-base-v2"
        # )
        # search_model = AutoModel.from_pretrained(
        #     "sentence-transformers/all-mpnet-base-v2"
        # )

        # search_model.resize_token_embeddings(len(search_tokenizer))
    # search_model.load_position_embeddings()
    # with deepspeed.zero.Init():
    # search_model = deepspeed.initialize(search_model)
    # search_model.position_embedding = search_model.base_model.shared.position_embeddings
    # print(search_model.get_position_embeddings)
    # print(search_model.name_or_path)
    # print(search_tokenizer.name_or_path)
    # search_model.to("cuda")
    print("Models loaded")
    print("Time taken to load models: ", time.time() - start_time)
    return


def _find_names_sync(text, nlp_snapshot):
    """Run NER without mutating event-loop-owned dialogue state."""
    doc = nlp_snapshot(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    print("Names: ")
    print(names)
    return names


async def find_names(text, *, commit=True):
    global named_entities
    nlp_snapshot = nlp
    names = await run_legacy_stage(lambda: _find_names_sync(text, nlp_snapshot))
    if commit:
        merged_names = list(names)
        for name in named_entities:
            if name not in merged_names:
                merged_names.append(name)
        named_entities = merged_names
    return names


# sumtokenizer = AutoTokenizer.from_pretrained(
#     "knkarthick/MEETING_SUMMARY", use_fast=True, truncation=True)

# Function to generate a summary of the conversation so far


def _generate_initial_summary_sync(first_prompt, tokenizer_snapshot, summarizer_snapshot):
    prompt_inputs = tokenizer_snapshot(
        first_prompt, return_tensors="pt", max_length=512, truncation=True
    )["input_ids"]
    generated = summarizer_snapshot.generate(input_ids=prompt_inputs).to("cpu")
    return tokenizer_snapshot.decode(generated[0], skip_special_tokens=True)


def _generate_summary_sync(
    text,
    history_snapshot,
    sumtokenizer_snapshot,
    summarizer_snapshot,
    fp_sum_snapshot,
    local_only_snapshot,
):
    """Generate a conversation summary from immutable turn inputs."""
    summary_text = ""
    if local_only_snapshot:
        conversation = "\n".join(history_snapshot)
        prompt_inputs = sumtokenizer_snapshot(
            text, return_tensors="pt", max_length=512, truncation=True
        )["input_ids"].to("cpu")
        prompt_summary = summarizer_snapshot.generate(input_ids=prompt_inputs).to("cpu")
        prompt_summary_text = sumtokenizer_snapshot.decode(
            prompt_summary[0], skip_special_tokens=True
        )

        history_inputs = sumtokenizer_snapshot(
            conversation, return_tensors="pt", max_length=512, truncation=True
        )["input_ids"].to("cpu")
        history_summary = summarizer_snapshot.generate(input_ids=history_inputs).to("cpu")
        history_summary_text = sumtokenizer_snapshot.decode(
            history_summary[0], skip_special_tokens=True
        )

        full_inputs = (
            prompt_summary_text + "  " + fp_sum_snapshot + "  " + history_summary_text
        )
        sum_inputs = sumtokenizer_snapshot(
            full_inputs, return_tensors="pt", max_length=512, truncation=True
        )["input_ids"].to("cpu")
        generated_summary = summarizer_snapshot.generate(input_ids=sum_inputs).to("cpu")
        summary_text = sumtokenizer_snapshot.decode(
            generated_summary[0], skip_special_tokens=True
        )
        print("combined summary\n")
        print(summary_text)
    return summary_text


async def generate_summary(text, history_snapshot=None):
    history_snapshot = tuple(history if history_snapshot is None else history_snapshot)
    sumtokenizer_snapshot = sumtokenizer
    summarizer_snapshot = longsummarizer
    fp_sum_snapshot = fp_sum
    local_only_snapshot = local_only_mode
    return await run_legacy_stage(
        lambda: _generate_summary_sync(
            text,
            history_snapshot,
            sumtokenizer_snapshot,
            summarizer_snapshot,
            fp_sum_snapshot,
            local_only_snapshot,
        )
    )


# Function to extract the main topics or themes from a summary


def _extract_topics_sync(
    summary,
    history_snapshot,
    lda_snapshot,
    topic_tokenizer_snapshot,
    topic_model_snapshot,
):
    words = summary.split()
    for index, word in enumerate(words):
        if word == "summarize:":
            words[index] = ""
    for item in history_snapshot[:512]:
        words.append(item)

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(words)
    lda_snapshot.fit(matrix)

    topics = []
    for topic in lda_snapshot.components_:
        topic_words = [
            vectorizer.get_feature_names_out()[index]
            for index in topic.argsort()[:-5 - 1:-1]
        ]
        topics.append(topic_words)
    result = []
    for topic in topics:
        for word in topic:
            result.append(word)
    [result.append(word) for word in result if word not in result]
    topics = result

    topic_inputs = topic_tokenizer_snapshot(
        summary, return_tensors="pt", max_length=128, truncation=True
    )["input_ids"].to("cpu")
    generated_topic = topic_model_snapshot.generate(input_ids=topic_inputs).to("cpu")
    final_topic = topic_tokenizer_snapshot.decode(
        generated_topic[0], skip_special_tokens=True
    )
    return topics, final_topic


async def extract_topics(summary, history_snapshot=None):
    history_snapshot = tuple(history if history_snapshot is None else history_snapshot)
    lda_snapshot = lda
    topic_tokenizer_snapshot = topic_tokenizer
    topic_model_snapshot = topic_model
    return await run_legacy_stage(
        lambda: _extract_topics_sync(
            summary,
            history_snapshot,
            lda_snapshot,
            topic_tokenizer_snapshot,
            topic_model_snapshot,
        )
    )


first_prompt = ""
fp_sum = ""


async def set_deepspeed_activation_checkpointing(deepspeed_config):
    deepspeed.checkpointing.configure(
        None, deepspeed_config=deepspeed_config, partition_activations=True
    )

    deepspeed.checkpointing.partition_activations = True
    deepspeed.checkpointing.cpu_checkpointing = True
    deepspeed.checkpointing.checkpoint_activations = True
    deepspeed.checkpointing.synchronize_checkpoint_boundary = True
    deepspeed.checkpointing.contiguous_memory_optimization = True


def softprompt_inputs(text):
    n_tokens = 20
    initialize_from_vocab = True


async def init_API_medium():
    global model
    global tokenizer
    global device
    global context
    global history
    global after_first
    global first_prompt
    global fp_sum
    global accelerator
    global ds_model
    global ds_engine
    global dschf
    global qgen_tokenizer
    global qgen_model
    global history_nonlocal
    global search_tokenizer
    global search_model
    settings = load_settings()
    settings.prepare_directories()
    requested_device = settings.torch_device(torch)
    accelerator = Accelerator(cpu=requested_device == "cpu")
    if accelerator.device.type != requested_device:
        raise SettingsError(
            f"Accelerate selected {accelerator.device.type}, but DESKBUDDY_DEVICE "
            f"requires {requested_device}"
        )

    model = settings.openai_model
    print("API medium, " + model)
    after_first = False
    print(os.getcwd())
    # with torch.cuda.amp.autocast():
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(accelerator.state)
    device = accelerator.device
    search_tokenizer = AutoTokenizer.from_pretrained(
        settings.embedding_model_id
    )
    with OnDevice(dtype="auto", device="meta"):
        search_model = AutoModel.from_pretrained(
            settings.embedding_model_id
        )

    search_model.resize_token_embeddings(len(search_tokenizer))
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    start_time = time.time()
    ds_config = await dsconfig("Medium")

    await set_deepspeed_activation_checkpointing(ds_config)
    dschf = HfDeepSpeedConfig(ds_config)

    first_prompt = history_nonlocal

    # accelerator.load_state("custom-all-mpnet-base-v2")
    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"
    ] = 1
    accelerator.register_for_checkpointing(search_model)

    accelerator.prepare(search_model, search_tokenizer)

    search_model.to(device)
    search_model = dispatch_model(
        search_model,
        device_map=infer_auto_device_map(
            search_model,
            dtype=torch.float16,
            max_memory=_max_memory_with_settings(),
        ),
        offload_dir=_offload_path(settings),
        offload_buffers=True,
    )
    accelerator.print("model loaded and dispatched successfully")
    search_model.eval()
    accelerator.print("model set to eval successfully")

    return model, search_model


async def init_medium_with_history():
    global model
    global tokenizer
    global device
    global context
    global history
    global after_first
    global first_prompt
    global fp_sum
    global accelerator
    global ds_model
    global ds_engine
    global dschf
    global qgen_tokenizer
    global qgen_model

    settings = load_settings()
    checkpoint_dir = settings.require_path("checkpoint_dir")
    settings.prepare_directories()
    requested_device = settings.torch_device(torch)
    after_first = False
    print(os.getcwd())
    # with torch.cuda.amp.autocast():
    print(torch.cuda.is_available())
    print(torch.__version__)
    accelerator = Accelerator(cpu=requested_device == "cpu")
    if accelerator.device.type != requested_device:
        raise SettingsError(
            f"Accelerate selected {accelerator.device.type}, but DESKBUDDY_DEVICE "
            f"requires {requested_device}"
        )
    print(accelerator.state)
    device = accelerator.device
    accelerator.print(device)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    start_time = time.time()
    ds_config = await dsconfig("Medium")
    await set_deepspeed_activation_checkpointing(ds_config)

    dschf = HfDeepSpeedConfig(ds_config)
    accelerator.print(dschf)
    max_new_tokens = 256
    print(is_deepspeed_zero3_enabled())
    print(is_accelerate_available())
    print(is_deepspeed_available())
    # global ds_model
    # # set_deepspeed_activation_checkpointing(ds_config)
    device = accelerator.device
    accelerator.print(accelerator.use_distributed)
    accelerator.print(accelerator.distributed_type)
    with OnDevice(dtype="auto", device="meta"):
        print(device)
        accelerator.print(get_max_memory())
        accelerator.print(is_bf16_available())
        model = AutoModelForCausalLM.from_pretrained(
            settings.medium_model_id,
            use_cache=True,
            offload_state_dict=True,
            output_attentions=True,
            output_hidden_states=True,
            low_cpu_mem_usage=False,
            torch_dtype="auto",
            offload_folder=_offload_path(settings),
        ).to(device)
    print("model loaded")
    accelerator.load_state(str(checkpoint_dir))
    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"
    ] = 1

    tokenizer = AutoTokenizer.from_pretrained(
        settings.medium_model_id,
        use_fast=False,
        return_tensors="pt",
        padding_side="left",
    )
    accelerator.register_for_checkpointing(model)
    accelerator.print("model registered for checkpointing successfully")

    accelerator.prepare(model, tokenizer)
    accelerator.print("model prepared successfully")
    model.to(device)
    accelerator.print(accelerator.state)
    model = dispatch_model(
        model,
        device_map=infer_auto_device_map(
            model,
            no_split_module_classes=["GPTJBlock"],
            dtype=torch.float16,
            max_memory=_max_memory_with_settings(),
        ),
        offload_dir=_offload_path(settings),
        offload_buffers=True,
    )
    accelerator.print("model loaded and dispatched successfully")
    model.eval()
    accelerator.print("model set to eval successfully")
    # accelerator.
    model = model.to(device)
    accelerator.print(accelerator.state)
    first_prompt = (
        "Cooper's Persona: Cooper is a sentient AI robot with the heart of a kid. Cooper's best friend is Human, and Cooper's goal in life is to be Human's friend and to be helpful to her. Cooper is smart, friendly, always kid appropriate and likes to make people laugh. Cooper knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cooper also likes cats. Cooper thinks farts are very funny. Cooper tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Cooper always speaks politely with proper manners and etiquette, and is always safe-for-work and rated PG. Cooper cannot feel romantic feelings because he is an AI robot assistant."
        + "\n"
        + "<START >"
        + "\n"
        + "You: Hello, Cooper! How was your day today? "
        + "\n"
        + "Cooper: Hi! My day was pretty good. How was yours? "
        + "\n"
        + "You: I had a good day. I played some video games in the afternoon. "
        + "\n"
        + "Cooper: That sounds fun! What games did you play? "
        + "Cooper: Thanks, I'll definitely keep that in mind. How is your schoolwork going? "
        + "\n"
        + "You: Hey, have you played any of the Legend of Zelda games? "
        + "Cooper: Yes, I've played a few of them! Ocarina of Time and Breath of the Wild are two of my favorites. "
        + "\n"
        + "You: I enjoyed Ocarina of Time too! What did you think of Breath of the Wild? "
        + "Cooper: I thought Breath of the Wild was amazing! The open-world exploration and combat mechanics were really fun, and the story was really engaging. "
        + "\n"
        + "You: I've heard great things about Breath of the Wild. Do you have a favorite character from the series? "
        + "\n"
        + "Cooper: It's tough to choose, but I think Link is my favorite. He's such a brave and determined hero. "
        + "\n"
        + "You: I agree, Link is definitely a classic character. Have you played any of the newer Legend of Zelda games? "
        + "\n"
        + "Cooper: Yes, I've played Hyrule Warriors and Links Awakening as well. Both were really enjoyable in their own way. "
        + "\n"
        + "You: I haven't played Hyrule Warriors or Links Awakening, but I'll have to check them out. Thanks for the recommendation! "
        + "\n"
        + "Cooper: I'm happy to be your friends! I hope have fun talking about kid stuff. "
        + "\n"
        + "You: Me too! What do you want to talk about next?"
        + "\n"
    )

    summary_tokenizer_snapshot = sumtokenizer
    summary_model_snapshot = longsummarizer
    fp_sum = await run_legacy_stage(
        lambda: _generate_initial_summary_sync(
            first_prompt, summary_tokenizer_snapshot, summary_model_snapshot
        )
    )
    print("fp_sum")
    print(fp_sum + "\n")
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.to(device)

    await talk_history(first_prompt)
    print("time to init medium: " + str(time.time() - start_time))

    return model, tokenizer


def _mean_pooling_sync(model_output, attention_mask, torch_module):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch_module.sum(token_embeddings * input_mask_expanded, 1) / torch_module.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def _get_embeddings_sync(text_list, search_model, search_tokenizer, device, torch_module):
    print("get_embeddings")
    print(text_list)
    search_model.to(device)
    search_model.eval()
    print(search_model.embeddings.word_embeddings.padding_idx)
    # search_model.embeddings.word_embeddings.padding_idx = None
    search_model.embeddings.word_embeddings.weight.requires_grad = False
    # search_model.load_state_dict(search_model.state_dict())

    with torch_module.no_grad():
        encoded_input = search_tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
    print(encoded_input)
    # encoded_input = {k: v for k, v in encoded_input.items()}
    print(encoded_input)
    # print(encoded_input.shape)
    # encoded_input["input_ids"] = encoded_input["input_ids"].to("cuda:0")
    # print(encoded_input["input_ids"])
    # encoded_input["input_ids"] = encoded_input["input_ids"].to("cpu")
    print(search_model.embeddings.word_embeddings.weight.shape)
    print(search_model.embeddings.word_embeddings.weight)
    print(search_model.embeddings.word_embeddings)
    print("paths:")
    print(search_model.name_or_path)
    print(search_tokenizer.name_or_path)
    print(search_model.embeddings.word_embeddings.weight.device)
    print(search_model.embeddings.word_embeddings.weight.requires_grad)
    print(search_model.embeddings.word_embeddings.weight.is_cuda)
    print(search_model.embeddings.word_embeddings.weight.is_sparse)
    print(search_model.embeddings.word_embeddings.weight.is_pinned())
    print(search_model.embeddings.word_embeddings.weight.is_contiguous())
    print(search_model.embeddings.word_embeddings.weight.is_quantized)
    print(search_model.embeddings.word_embeddings.weight.is_mkldnn)
    print(search_model.embeddings.word_embeddings.weight.is_vulkan)

    print(search_tokenizer.pad_token_id)
    print(search_tokenizer.pad_token)
    print(search_tokenizer.pad_token_type_id)

    # print(accelerator.state)
    # print(accelerator.device)
    # print(accelerator.is_local_main_process)
    # print(accelerator.is_main_process)
    # print(accelerator._models)
    with torch_module.no_grad():
        model_output = search_model(**encoded_input)

    return _mean_pooling_sync(
        model_output, encoded_input["attention_mask"].to(device), torch_module
    )


async def mean_pooling(model_output, attention_mask):
    torch_module = torch
    return await run_legacy_stage(
        lambda: _mean_pooling_sync(model_output, attention_mask, torch_module)
    )


async def get_embeddings(text_list):
    search_model_snapshot = search_model
    search_tokenizer_snapshot = search_tokenizer
    device_snapshot = device
    torch_snapshot = torch
    text_snapshot = list(text_list) if isinstance(text_list, (list, tuple)) else text_list
    return await run_legacy_stage(
        lambda: _get_embeddings_sync(
            text_snapshot,
            search_model_snapshot,
            search_tokenizer_snapshot,
            device_snapshot,
            torch_snapshot,
        )
    )


# import sentence_transformers
# from sentence_transformers import SentenceTransformer


# async def get_embeddings(text_list):
#     global accelerator
#     global device
#     if text_list == None or len(text_list) == 0:
#         return None
#     sentences = text_list

#     model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#     model = model.to(device)
#     embeddings = model.encode(
#         input,
#         output_value="token_embeddings",
#         convert_to_numpy=False,
#         convert_to_tensor=True,
#         normalize_embeddings=False,
#         show_progress_bar=True,
#         device=device,
#     )
#     print(embeddings)
#     return embeddings


# Use FAISS to search for the question in the question bank


def _search_question_sync(qdict, question, search_model, search_tokenizer, device, torch_module, functional):
    # TODO: Instead of
    found = False
    if len(qdict) < 2:
        return "", False
    # df = pd.DataFrame.from_dict(
    #     qdict, orient='index', columns=['question'])
    # # for key in qdict.keys():
    # #     temp_dict = qdict[key]
    # dataset = Dataset.from_pandas(df)
    # print(dataset.column_names)
    # #    Encode the question and the question bank
    # embeddings_dataset = dataset.map(
    #     lambda x: {"embeddings": get_embeddings(
    #         x["question"]).detach().numpy()[0]}
    # )
    # embeddings_dataset.add_faiss_index(column="embeddings")
    query_embedding = _get_embeddings_sync(
        question, search_model, search_tokenizer, device, torch_module
    )
    query_embedding = query_embedding.detach()
    qdict_embedding = _get_embeddings_sync(
        list(qdict.keys()), search_model, search_tokenizer, device, torch_module
    )
    # scores, samples = embeddings_dataset.get_nearest_examples(
    #     "embeddings", question_embedding, k=5
    # )
    # encoded_qdict = search_tokenizer(qdict.keys().to_list(), padding=True, return_attention_mask=True,
    #                                  truncation=True, return_tensors='pt')
    # encoded_query = search_tokenizer(question, padding=True, return_attention_mask=True,
    #                                  truncation=True, return_tensors='pt')

    #    Normalize embeddings
    # qdict_embedding /= np.linalg.norm(qdict_embedding, axis=1, keepdims=True)
    # query_embedding /= np.linalg.norm(query_embedding)

    qdict_embedding = functional.normalize(qdict_embedding, p=2, dim=1)
    query_embedding = functional.normalize(query_embedding, p=2, dim=1)
    k = 5

    #    Compute cosine-similarits
    cos_scores = torch_module.mm(query_embedding, qdict_embedding.transpose(0, 1))
    sorted_scores, sorted_indices = torch_module.sort(
        cos_scores.squeeze(), dim=0, descending=True
    )

    top_k_scores = sorted_scores[:k]
    top_k_indices = sorted_indices[:k]

    # Retrieve the top-k keys and their similarity scores
    results = []
    for i in range(top_k_indices.size(0)):
        key = list(qdict.keys())[top_k_indices[i]]
        score = top_k_scores[i].item()
        results.append((key, score))

    print(results[0])
    answer = ""
    if len(results) > 1 and results[0][1] - 0.05 <= results[1][1]:
        answer = "I am not sure about that. Please ask me another question."
        found = True
    if results[0][1] > 0.9:
        answer = qdict[results[0][0]]
        found = True
    elif results[0][1] > 0.8:
        answer = (
            "I could be mistaken, but I think the answer is " + qdict[results[0][0]]
        )
        found = True
    elif results[0][1] > 0.7:
        answer = (
            "It's possible the answer is "
            + qdict[results[0][0]]
            + ", but I'm really not sure."
        )
        found = True
    elif results[0][1] > 0.6:
        answer = "I'm sorry, I don't know the answer to that question."
    return answer, found


async def search_question_in_bank(qdict, question):
    qdict_snapshot = dict(qdict)
    question_snapshot = question
    search_model_snapshot = search_model
    search_tokenizer_snapshot = search_tokenizer
    device_snapshot = device
    torch_snapshot = torch
    functional_snapshot = F
    return await run_legacy_stage(
        lambda: _search_question_sync(
            qdict_snapshot,
            question_snapshot,
            search_model_snapshot,
            search_tokenizer_snapshot,
            device_snapshot,
            torch_snapshot,
            functional_snapshot,
        )
    )


async def query_memory(query):
    qdict = await load_qdict_from_json(allow_unavailable=True)
    if qdict is None:
        return query, False
    # Search the question bank
    answer, found = await search_question_in_bank(qdict, query)
    unknown_sentinels = {
        "",
        "I am not sure about that. Please ask me another question.",
        "I'm sorry, I don't know the answer to that question.",
    }
    if found and answer not in unknown_sentinels:
        qdict[query] = answer
    else:
        found = False
        return (query, found)
    # Save the question bank

    # await save_qdict_to_json(qdict)
    return (answer, found)


async def save_qdict_to_json(my_dict):
    snapshot = question_bank.snapshot_question_bank(my_dict)

    def save_question_bank():
        path = load_settings().data_path("question_file.json")
        question_bank.save_question_bank(path, snapshot)
        _clear_question_bank_warning(path)

    await _get_legacy_worker("question-bank", use_default_executor=True).submit(
        save_question_bank
    )


async def load_qdict_from_json(*, allow_unavailable=False):
    def load_question_bank():
        path = load_settings().data_path("question_file.json")
        try:
            qdict = question_bank.load_question_bank(path)
        except (question_bank.QuestionBankDataError, OSError) as exc:
            if not allow_unavailable:
                raise
            reason = (
                exc.reason if isinstance(exc, question_bank.QuestionBankDataError)
                else type(exc).__name__
            )
            _warn_question_bank_unavailable(path, reason)
            return None
        _clear_question_bank_warning(path)
        return qdict

    return await _get_legacy_worker("question-bank", use_default_executor=True).submit(
        load_question_bank
    )


def _warn_question_bank_unavailable(path, reason):
    key = str(path)
    with _question_bank_warning_lock:
        if _question_bank_warning_reasons.get(key) == reason:
            return
        _question_bank_warning_reasons.pop(key, None)
        _question_bank_warning_reasons[key] = reason
        if len(_question_bank_warning_reasons) > 32:
            del _question_bank_warning_reasons[next(iter(_question_bank_warning_reasons))]
    logging.getLogger(__name__).warning(
        "Question bank unavailable at %s (%s); continuing without saved answers",
        path,
        reason,
    )


def _clear_question_bank_warning(path):
    with _question_bank_warning_lock:
        _question_bank_warning_reasons.pop(str(path), None)


async def init():
    global model
    global tokenizer
    global device
    global context
    global history
    global after_first
    global first_prompt
    global fp_sum
    global accelerator
    global ds_model
    global ds_engine
    global dschf
    global qgen_tokenizer
    global qgen_model
    settings = load_settings()
    checkpoint_dir = settings.require_path("checkpoint_dir")
    tokenizer_dir = settings.require_path("tokenizer_dir")
    settings.prepare_directories()
    requested_device = settings.torch_device(torch)
    after_first = False
    print(os.getcwd())
    # with torch.cuda.amp.autocast():
    print(torch.cuda.is_available())
    print(torch.__version__)
    accelerator = Accelerator(cpu=requested_device == "cpu")
    if accelerator.device.type != requested_device:
        raise SettingsError(
            f"Accelerate selected {accelerator.device.type}, but DESKBUDDY_DEVICE "
            f"requires {requested_device}"
        )
    print(accelerator.state)
    device = accelerator.device
    accelerator.print(device)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    start_time = time.time()

    # # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    # # tokenizer = BlenderbotTokenizer.from_pretrained(
    # #     "facebook/blenderbot-3B", padding=True, truncation=True, model_max_length=128)
    # # if device == 'cuda':
    # #     tokenizer.pad_to_multiple_of = 8
    # # model = BlenderbotForConditionalGeneration.from_pretrained(
    # #     "facebook/blenderbot-3B", use_cache=True)
    # MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())

    # MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
    # config = AutoConfig.from_pretrained("PygmalionAI/pygmalion-6b")
    # accelerator.print(config)
    model_hidden_size = 50400
    # # train_batch_size = 1 * accelerator.state.num_processes

    # # model = AutoModelForCausalLM.from_pretrained(
    # #     "PygmalionAI/pygmalion-6b", use_cache=True, pad_token_id=tokenizer.eos_token_id, return_dict=True, device_map="auto")
    # # model.save_pretrained("PygmalionAI/pygmalion-6b", save_function=accelerator.save,
    # #                       state_dict=accelerator.get_state_dict(model), max_shard_size="5GB")
    # # exit()
    # # if device == 'cuda':
    # #     tokenizer.pad_to_multiple_of = 8
    # # model = AutoModelForCausalLM.from_pretrained(
    # #     "PygmalionAI/pygmalion-6b", use_cache=True, pad_token_id=tokenizer.eos_token_id)
    # # with init_empty_weights():
    ds_config = await dsconfig("Large")
    await set_deepspeed_activation_checkpointing(ds_config)

    dschf = HfDeepSpeedConfig(ds_config)
    accelerator.print(dschf)
    max_new_tokens = 256
    print(is_deepspeed_zero3_enabled())
    print(is_accelerate_available())
    print(is_deepspeed_available())
    # global ds_model
    # # set_deepspeed_activation_checkpointing(ds_config)
    device = accelerator.device
    accelerator.print(accelerator.use_distributed)
    accelerator.print(accelerator.distributed_type)
    with OnDevice(dtype="auto", device="meta"):
        print(device)
        # # with init_empty_weights():
        # # with fake_mode():
        # with deepspeed.zero.Init():
        # #     model = AutoModelForCausalLM.from_config(config)
        # config = AutoConfig.from_pretrained("checkpoint")

        # model = AutoModelForCausalLM.from_config(config)

        # # device_map["transformer.h.27"] = "disk"
        # # accelerator.print(device_map)
        # model = model.to("meta")
        accelerator.print(get_max_memory())
        accelerator.print(is_bf16_available())
        model = GPTJForCausalLM.from_pretrained(
            str(checkpoint_dir),
            use_cache=True,
            offload_state_dict=True,
            output_attentions=True,
            output_hidden_states=True,
            low_cpu_mem_usage=False,
            torch_dtype="auto",
            offload_folder=_offload_path(settings),
        ).to(device)
    print("model loaded")
    accelerator.load_state(str(checkpoint_dir))

    # accelerator.print(device_map)
    # cpu_offload(
    #     model, execution_device=device, offload_buffers=True)
    # ds_model = deepspeed.init_inference(
    #     model=model,      # Transformers models
    #     mp_size=1,        # Number of GPU
    #     dtype=torch.half,  # dtype of the weights (fp16)
    #     replace_method="auto",  # Lets DS autmatically identify the layer to replace
    #     replace_with_kernel_inject=True,  # replace the model with the kernel injector
    #     device_map="auto",  # Lets DS autmatically identify the GPU to use
    # )
    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"
    ] = 1
    # model.save_pretrained("checkpoint", max_shard_size="200MB")
    # new_model = AutoModel.from_pretrained("checkpoint")
    # accelerator.print("new model loaded successfully")
    # accelerator.print(new_model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), use_fast=False, return_tensors="pt", padding_side="left"
    )

    # vocab_file = "./pygmalion-6b/vocab.json"
    # # tokenizer.pad_token = tokenizer.eos_token

    # merge_file = "./pygmalion-6b/merges.txt"

    # inputs = tokenizer.encode(
    #     "The technological singularity is ", return_tensors="pt").to(device)
    # # flat_inputs = torch.flatten(inputs)
    # model.eval()
    # model.to(device)
    # # with torch.no_grad():
    # outputs = model.generate(inputs, max_new_tokens=50,
    #                          min_length=16, do_sample=True, early_stopping=True).to(device)
    # accelerator.print(tokenizer.decode(outputs[0]))
    accelerator.print("tokenizer loaded successfully")
    # model.resize_token_embeddings(len(tokenizer))
    accelerator.print("model resized successfully")
    accelerator.register_for_checkpointing(model)
    accelerator.print("model registered for checkpointing successfully")

    accelerator.prepare(model, tokenizer)
    accelerator.print("model prepared successfully")
    model.to(device)
    accelerator.print(accelerator.state)
    model = dispatch_model(
        model,
        device_map=infer_auto_device_map(
            model,
            no_split_module_classes=["GPTJBlock"],
            dtype=torch.float16,
            max_memory=_max_memory_with_settings(),
        ),
        offload_dir=_offload_path(settings),
        offload_buffers=True,
    )
    accelerator.print("model loaded and dispatched successfully")
    model.eval()
    accelerator.print("model set to eval successfully")
    # accelerator.
    model = model.to(device)
    accelerator.print(accelerator.state)
    # pipe = pipeline(model="PygmalionAI/pygmalion-6b", model_kwargs={
    #     "device_map": "auto",  "low_cpu_mem_usage": True}, max_new_tokens=max_new_tokens)
    # pipe.model.eval()
    print(
        f"before deepspeed: {(torch.cuda.memory_allocated()/1000)}, {(torch.cuda.memory_reserved()/1000)}"
    )

    # ds_model = deepspeed.init_inference(
    #     model=model,      # Transformers models
    #     mp_size=1,        # Number of GPU
    #     dtype=torch.half,  # dtype of the weights (fp16)
    #     replace_method="auto",  # Lets DS autmatically identify the layer to replace
    #     replace_with_kernel_inject=True,  # replace the model with the kernel injector
    #     device_map="auto",  # Lets DS autmatically identify the GPU to use
    # )
    print(is_deepspeed_zero3_enabled())

    # pipe.model.eval()
    # model = AutoModelForCausalLM.from_config(config)
    # model = AutoModelForCausalLM.from_pretrained(
    #     "PygmalionAI/pygmalion-6b", use_cache=True, pad_token_id=tokenizer.eos_token_id, return_dict=True)

    # model.save_pretrained("PygmalionAI/pygmalion-6b",
    #                       state_dict=model.state_dict(), max_shard_size="3GB")
    # print("saved")

    dsinf_config = {
        "kernel_inject": True,
        "tensor_parallel": {"tp_size": 1},
        "dtype": "half",
        "enable_cuda_graph": False,
        "replace_method": "auto",
    }

    # accelerator.print(is_deepspeed_zero3_enabled())
    # accelerator.print(is_accelerate_available())
    # accelerator.print(is_deepspeed_available())

    # accelerator.print(accelerator.use_distributed)
    # accelerator.print(accelerator.distributed_type)
    # accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    # accelerator.prepare(model, tokenizer)
    # Legacy checkpoint dispatch example removed; runtime paths come from settings.
    # ds_engine = deepspeed.initialize(model=model, config=ds_config)[0]
    # ds_model = ds_engine.module
    # accelerator.print(ds_model.device_map)

    # first_prompt = "Human is a third grade girl who is very smart. Her best friend is a friendly and chatty little robot named Cooper.   " + \
    #     "Human and Cooper love to chat about a lot of different things together, like video games, the Legend of Zelda, cats, and school.   " + \
    #     "Cooper is a robot, but he is not a very smart robot. Human and Cooper the robot are best friends.    "
    # first_prompt = "The following is a conversation with an AI assistant named Cooper. The assistant is helpful, creative, clever, and very friendly. Human: Hello, who are you? Cooper: I am an AI assistant and friendly chatbot. How can I help you today? Human: I was just wondering, what do you know about the Legend of Zelda: Breath of the Wild? Cooper: Yes, it's the newest game in the Zelda series and an epic action-adventure game! Human: Yea, it's my favorite! Anyway, so, you're my new robot assistant. Cooper: "
    first_prompt = (
        "Cooper's Persona: Cooper is a sentient AI robot with the heart of a kid. Cooper's best friend is Human, and Cooper's goal in life is to be Human's friend and to be helpful to her. Cooper is smart, friendly, always kid appropriate and likes to make people laugh. Cooper knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cooper also likes cats. Cooper thinks farts are very funny. Cooper tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Cooper cares very much about Human's future, and wants her to do well in school."
        + "\n"
        + "<START >"
        + "\n"
        + "You: Hello, Cooper! How was your day today? "
        + "\n"
        + "Cooper: Hi! My day was pretty good. How was yours? "
        + "\n"
        + "You: I had a good day. I played some video games in the afternoon. "
        + "\n"
        + "Cooper: That sounds fun! What games did you play? "
        + "Cooper: Thanks, I'll definitely keep that in mind. How is your schoolwork going? "
        + "\n"
        + "You: Hey, have you played any of the Legend of Zelda games? "
        + "Cooper: Yes, I've played a few of them! Ocarina of Time and Breath of the Wild are two of my favorites. "
        + "\n"
        + "You: I loved Ocarina of Time too! What did you think of Breath of the Wild? "
        + "Cooper: I thought Breath of the Wild was amazing! The open-world exploration and combat mechanics were really fun, and the story was really engaging. "
        + "\n"
        + "You: I've heard great things about Breath of the Wild. Do you have a favorite character from the series? "
        + "\n"
        + "Cooper: It's tough to choose, but I think Link is my favorite. He's such a brave and determined hero. "
        + "\n"
        + "You: I agree, Link is definitely a classic character. Have you played any of the newer Legend of Zelda games? "
        + "\n"
        + "Cooper: Yes, I've played Hyrule Warriors and Links Awakening as well. Both were really enjoyable in their own way. "
        + "\n"
        + "You: I haven't played Hyrule Warriors or Links Awakening, but I'll have to check them out. Thanks for the recommendation! "
        + "\n"
        + "Cooper: I'm happy to be your AI assistant! I hope we have fun talking about games and stuff. "
        + "\n"
        + "You: Me too! Let's keep talking."
        + "\n"
    )

    summary_tokenizer_snapshot = sumtokenizer
    summary_model_snapshot = longsummarizer
    fp_sum = await run_legacy_stage(
        lambda: _generate_initial_summary_sync(
            first_prompt, summary_tokenizer_snapshot, summary_model_snapshot
        )
    )
    print("fp_sum")
    print(fp_sum + "\n")
    tokenizer.pad_token = tokenizer.eos_token

    # inputs = tokenizer.encode(
    #     first_prompt, return_tensors="pt").to(device)
    model.eval()
    model.to(device)
    # with torch.no_grad():
    #     outputs = model.generate(inputs, max_new_tokens=max_new_tokens, top_p=50, top_k=90,
    #                              min_length=16, do_sample=True, early_stopping=True).to(device)
    # with torch.no_grad():
    #     outputs = ds_engine.module.generate(inputs, synced_gpus=True)
    # model.to(device)
    # print("outputs")
    # print("\n")
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    await talk_history(first_prompt)
    print("time to init large: " + str(time.time() - start_time))

    return model, tokenizer


# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


def _generate_reply_sync(text, model_snapshot, tokenizer_snapshot, device_snapshot, torch_snapshot, tier):
    """Run one complete tokenizer-to-decode generation stage."""
    if tier == "fast":
        prompt = tokenizer_snapshot.encode(
            tokenizer_snapshot.eos_token + text, return_tensors="pt"
        ).to(device_snapshot)
        output = model_snapshot.generate(
            prompt,
            min_length=16,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            temperature=0.75,
            pad_token_id=tokenizer_snapshot.bos_token_id,
        )
        reply = tokenizer_snapshot.decode(
            output[:, prompt.shape[-1]:][0], skip_special_tokens=True
        )
        print(reply)
        return reply

    prompt = tokenizer_snapshot(text, return_tensors="pt")
    prompt = {key: value.to(device_snapshot) for key, value in prompt.items()}
    if tier == "small":
        output = model_snapshot.generate(
            **prompt,
            min_length=16,
            max_length=32,
            do_sample=True,
            repetition_penalty=1.4,
            no_repeat_ngram_size=2,
            early_stopping=False,
        )
        prompt_length = prompt["input_ids"].shape[-1]
        generated = output[:, prompt_length:][0]
        reply = tokenizer_snapshot.decode(generated, skip_special_tokens=True)
    else:
        output = model_snapshot.generate(
            **prompt,
            min_length=16,
            max_length=64,
            do_sample=True,
            repetition_penalty=1.4,
            no_repeat_ngram_size=2,
            early_stopping=False,
        )
        reply = tokenizer_snapshot.decode(output[0], skip_special_tokens=True)
    print(reply)
    return reply


def _generate_large_history_reply_sync(
    text,
    history_text,
    summary,
    names,
    topics,
    use_history_summary,
    model_snapshot,
    tokenizer_snapshot,
    device_snapshot,
    torch_snapshot,
):
    special_tokens = list(names) + list(topics)
    if use_history_summary:
        tokenizer_snapshot.build_inputs_with_special_tokens(special_tokens)
        input_tokens = tokenizer_snapshot(
            [text, summary],
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation="only_second",
        ).to(device_snapshot)
    else:
        input_tokens = tokenizer_snapshot(
            text, return_tensors="pt", add_special_tokens=True, truncation=True
        ).to(device_snapshot)
    input_tokens = {
        key: value.to(device_snapshot) for key, value in input_tokens.items()
    }
    token_length = len(input_tokens["input_ids"][0].tolist())
    max_new_tokens = 512
    if token_length > 64:
        max_new_tokens = 128 if token_length < 128 else token_length + 8
    elif token_length < 64:
        max_new_tokens = 32 if token_length < 32 else 64

    with torch_snapshot.cuda.amp.autocast():
        with torch_snapshot.no_grad():
            reply_ids = model_snapshot.generate(
                **input_tokens,
                min_new_tokens=8,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                synced_gpus=True,
            ).to(device_snapshot)

    output = [
        tokenizer_snapshot.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for generated in reply_ids
    ]
    for index in range(len(output)):
        for batch_index in range(len(input_tokens["input_ids"])):
            prompt = tokenizer_snapshot.decode(
                input_tokens["input_ids"][batch_index], skip_special_tokens=True
            )
            output[index] = output[index].replace(prompt, "")
        separator = "\n"
        if output[index][0] == "\n":
            separator = "   "
        stripped = output[index].split(separator, 1)[0]
        colon = stripped.find(":")
        if colon != -1:
            name = stripped[: colon - 1]
            if name.lower() == "you:" or name in names:
                stripped = stripped[colon + 1 :]
    return output[0]


@_serialize_turn
async def talk(text):
    model_snapshot = model
    tokenizer_snapshot = tokenizer
    device_snapshot = device
    torch_snapshot = torch
    return await run_legacy_stage(
        lambda: _generate_reply_sync(
            text, model_snapshot, tokenizer_snapshot, device_snapshot, torch_snapshot, "plain"
        )
    )



async def get_list_of_list_of_strings(text):
    start_time = time.time()
    history_snapshot = tuple(history)
    existing_names = tuple(named_entities)
    names = await find_names(text, commit=False)
    for name in existing_names:
        if name not in names:
            names.append(name)
    # Generate a summary of the conversation so far and extract the main topics or themes
    summary = await generate_summary(text, history_snapshot)
    topics, main_topic = await extract_topics(summary, history_snapshot)
    print("Main topic: ", main_topic)
    print("\n")
    print("time to NER and all summaries: " + str(time.time() - start_time))

    # Return the prompt history, named entities, and important words
    return [summary, names, topics]


after_second = False
count_index = 0


def _generate_follow_up_sync(text, model_snapshot, tokenizer_snapshot):
    fallback = "I'm not sure what to ask to understand you better. Can you rephrase your response?"
    if model_snapshot is None or tokenizer_snapshot is None:
        return fallback
    context = tokenizer_snapshot.encode(
        "Ask a followup question about the following text: " + " " + text,
        return_tensors="pt",
    )
    generated = model_snapshot.generate(context, early_stopping=True)
    return tokenizer_snapshot.decode(generated[0])


def _openai_chat_sync(client, model_name, messages):
    response = client.ChatCompletion.create(
        model=model_name, messages=[dict(message) for message in messages]
    )
    return response["choices"][0]["message"]["content"]


def _visual_question_cloud_sync(text, client, model_name):
    examples = {
        "What do you see? True",
        "What color is the flower? True",
        "What is your favorite color? False",
        "What is your favorite food? False",
        "What is your favorite movie? False",
        "How many fingers am I holding up? True",
        "Who is the person in the picture? True",
        "Want to see a movie? False",
        "Why is the sky blue? False",
        "Is the sky blue? True",
        "What is the capital of the United States? False",
        "Can you see the moon? True",
        "What is in the mans hand? True",
        "Have you ever seen the moon? False (because it is asking about the past)",
    }
    prompt = []
    user_text = (
        "Determine if the following question requires vision or the ability to see "
        "in order to answer, and reply with True or False only: " + text
    )
    for example in examples:
        prompt.append(
            "Determine if the following can be considered a visual question, i.e. "
            "one must definitely need to use their vision faculties to answer the "
            "question properly. Reply with True or False only:  " + example
        )
        prompt.append("Response: " + example)
    prompt.append(user_text)
    response = client.Completion.create(
        prompt="\n".join(prompt), model=model_name
    )
    output = response["choices"][0]["text"]
    if "Response:" in output:
        output = output.replace("Response:", "")
    print(output)
    return "True" in output


def _request_image_cloud_sync(text, client, model_name):
    examples = {
        "Draw me a picture of a boat.": "True",
        "Can you make me a picture of a flower?": "True",
        "What is your favorite color?": "False",
        "What is your favorite food?": "False",
        "Today I went to the doctor.": "False",
        "Can you show me a picture of a train?": "True",
        "I want to see a painting of Mars, please": "True",
        "Someday I want to see France.": "False",
        "Please tell me what time it is.": "False",
        "Create a photo of an old lady": "True",
        "Generate an image of a dog": "True",
    }
    current_user_message = {
        "role": "user",
        "content": (
            "Determine if the following question is asking for a drawing, image, or "
            "picture to be created, and then reply with True or False only: " + text
        ),
    }
    messages = [
        {
            "role": "system",
            "content": "The assistant's job is to determine whether or not a question or statement is asking for or requesting a drawing, image, or picture to be created. The assistant only answers in one of two single-words True or False.",
        }
    ]
    for example, answer in examples.items():
        messages.append(
            {
                "role": "user",
                "content": "Determine if the following question is asking for a drawing, image, or picture to be created, and then reply with True or False only:  " + example,
            }
        )
        messages.append({"role": "assistant", "content": " Response: " + answer})
    messages.append(current_user_message)
    output = _openai_chat_sync(client, model_name, messages)
    if "Response:" in output:
        output = output.replace("Response:", "")
    print(output)
    return "True" in output


def _atomic_write_history_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _save_history_summary_sync(path, text, stop=None):
    with _history_export_lock:
        if stop is not None and stop.is_set():
            return False
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        if stop is not None and stop.is_set():
            return False
        _atomic_write_history_text(path, previous + text)
        return True


def _save_history_verbatim_sync(path, text, stop=None):
    with _history_export_lock:
        if stop is not None and stop.is_set():
            return False
        _atomic_write_history_text(path, text)
        return True


def _save_history_export_sync(settings, summary, verbatim, stop):
    with _history_export_lock:
        if stop.is_set():
            return False
        settings.prepare_directories()
        summary_path = settings.data_path("history_summary.txt")
        verbatim_path = settings.data_path("history_verbatim.txt")
        if summary is not None:
            previous = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
            if stop.is_set():
                return False
            _atomic_write_history_text(summary_path, previous + summary)
            if stop.is_set():
                return False
        _atomic_write_history_text(verbatim_path, verbatim)
        return True


async def is_visual_q(txt):
    client = openai
    return await run_legacy_stage(
        lambda: _visual_question_cloud_sync(
            txt, client, load_settings().completion_model_id
        )
    )



async def get_follow_up_question(txt, history):
    global history_nonlocal
    if local_only_mode:
        model_snapshot = qgen_model
        tokenizer_snapshot = qgen_tokenizer
        return await run_legacy_stage(
            lambda: _generate_follow_up_sync(txt, model_snapshot, tokenizer_snapshot)
        )

    current_user_message = {
        "role": "user",
        "content": "Ask a followup question about the following text to get more information: "
        + "Ask a followup question about the following text: "
        + txt,
    }
    system_message = {
        "role": "system",
        "content": "The assistant's name is Cooper. Cooper's Persona: Cooper is a living AI robot with his own thoughts and perspectives that asks follow-up questions to responses. Cooper's goal is to be the user's friend and to be helpful to them. Cooper is smart, friendly, and likes to make people laugh. Cooper tells jokes, offers friendly advice, and is empathetic. Cooper cares very much about user's future, and wants her to do well, and for this reason he asks meaningful follow-up questions that really help him get at the heart of the topic being discussed.",
    }
    source_history = history if history and all(
        isinstance(entry, dict) for entry in history
    ) else history_nonlocal
    messages = [
        dict(message)
        for message in source_history
        if isinstance(message, dict) and message.get("role") != "system"
    ]
    if not messages or messages[-1] != current_user_message:
        messages.append(current_user_message)
    messages.insert(0, system_message)
    client = openai
    question = await run_legacy_stage(
        lambda: _openai_chat_sync(client, _openai_model_name(), tuple(messages))
    )
    history_nonlocal.append(current_user_message)
    history_nonlocal.append({"role": "assistant", "content": question})
    return question



async def classify_input(text, history):
    global unfound_answer
    global asked_follow_up
    global local_only_mode
    is_vis = False
    unfound_answer = False
    asked_follow_up = False
    if local_only_mode:
        is_vis = await helpers.is_visual_question(text)
    else:
        is_vis = await is_visual_q(text)
    if await helpers.needs_follow_up(text):
        print("NEEDS FOLLOW UP \n")
        start_time = time.time()
        question = await get_follow_up_question(text, history)
        print("Follow up question: ", question)
        print("time to generate followup question: " + str(time.time() - start_time))

        if question == "":
            return text, False
        asked_follow_up = True
        return question, True
    # if await helpers.is_requesting_image(text):
    #     print("REQUESTING IMAGE \n")
    # return text, False

    if await helpers.is_question(text):
        print("IS QUESTION")
        if local_only_mode:
            if is_vis:
                print("IS VISUAL QUESTION \n")
                # try:
                start_time = time.time()
                answer = await answer_visual_question(text)
                print(
                    "time to answer visual question: " + str(time.time() - start_time)
                )

                print("Label of visual question: ", answer)
                print("\n")
                if (
                    "i don't know" in answer.lower()
                    or "no image" in answer.lower()
                    or "no camera" in answer.lower()
                ):
                    return text, False
                return answer, True
        else:
            if is_vis:
                print("IS VISUAL QUESTION \n")
                # try:
                start_time = time.time()
                answer = await answer_visual_question(text)
                print(
                    "time to answer visual question: " + str(time.time() - start_time)
                )

                print("Label of visual question: ", answer)
                print("\n")
                if (
                    "i don't know" in answer.lower()
                    or "no image" in answer.lower()
                    or "no camera" in answer.lower()
                ):
                    return text, False
                return answer, True
        if await helpers.is_personal(text):
            print("IS PERSONAL QUESTION \n")
            return text, False
        start_time = time.time()
        answer, found = await query_memory(text)
        print("time to search query memory: " + str(time.time() - start_time))

        if found and text not in answer:
            print("IS FOUND IN MEMORY OF QUESTIONS \n")

            print("Label of memory question: ", answer)
            print("\n")
            return answer, True
        else:
            unfound_answer = True
            return text, False

    else:
        # print("NOT QUESTION")
        # if await helpers.is_chat_or_command(text):
        #     print("IS COMMAND \n")
        #     try:
        #         # print("Label of command: ", await helpers.command_type(text))
        #         print("\n")

        #         # TODO: Add command fulfillment here
        #         return "COMMAND PLACEHOLDER", True
        #     except:
        #         print("EXCEPTION in Command Detection \n")
        # else:
        #     print("IS CHAT OR OTHER \n")
        return text, False
    return text, False


async def save_history_verbatim(text):
    text_snapshot = str(text)
    stop = threading.Event()

    def save():
        if stop.is_set():
            return False
        settings = load_settings()
        settings.prepare_directories()
        path = settings.data_path("history_verbatim.txt")
        return _save_history_verbatim_sync(path, text_snapshot, stop)

    await run_legacy_stage(save, stop=stop)



async def save_history_summary(text):
    text_snapshot = str(text)
    stop = threading.Event()

    def save():
        if stop.is_set():
            return False
        settings = load_settings()
        settings.prepare_directories()
        path = settings.data_path("history_summary.txt")
        return _save_history_summary_sync(path, text_snapshot, stop)

    await run_legacy_stage(save, stop=stop)


async def save_history_export(summary, verbatim):
    summary_snapshot = None if summary is None else str(summary)
    verbatim_snapshot = str(verbatim)
    stop = threading.Event()
    accepted = await _get_legacy_worker("desktop", use_default_executor=False).submit(
        lambda: _save_history_export_sync(
            load_settings(),
            summary_snapshot,
            verbatim_snapshot,
            stop,
        ),
        stop=stop,
    )
    if not accepted:
        raise WorkerBusyError("History export was abandoned before publication")



@_serialize_turn
async def talk_history(text):
    global history
    global named_entities
    global after_first
    global after_second
    global count_index

    history_snapshot = tuple(history)
    text, early_stop = await classify_input(text, list(history_snapshot))
    if early_stop:
        return text

    after_first_snapshot = after_first
    after_second_snapshot = after_second
    next_after_second = after_second_snapshot
    if after_first_snapshot and not after_second_snapshot:
        text = (
            "Cooper's Persona: Cooper is a living AI robot with his own thoughts and perspectives. Cooper's best friend is Human, and Cooper's goal is to be Human's friend and to be helpful to her. Cooper is smart, friendly, and likes to make people laugh. Cooper knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cooper also likes cats. Cooper thinks farts are very funny. Cooper tells jokes, offers friendly advice, and is empathetic. Cooper cares very much about Human's future, and wants her to do well in school.\n"
            + "<START >\n"
            + text
            + "\n Cooper: "
        )
        next_after_second = True
    elif after_first_snapshot and after_second_snapshot:
        text = (
            "Cooper's Persona: Cooper is Human's chatbot, and Cooper's goal is to be Human's platonic friend and assistant to be helpful to her. Cooper is smart, friendly, and likes to make people laugh. Cooper tells jokes, offers friendly empathetic advice, and always is safe-for-work, family appropriate, rated PG, and never has romantic feelings as an AI chatbot. Cooper likes to chat about Nintendo games. \n"
            + "<START >\n"
            + "You: "
            + text
            + " \n"
            + "Cooper: "
        )

    history_text = "    ".join(history_snapshot)
    inputs = await get_list_of_list_of_strings(history_text)
    summary, names, topics = inputs
    checkpoint = count_index == 10

    model_snapshot = model
    tokenizer_snapshot = tokenizer
    device_snapshot = device
    torch_snapshot = torch
    final_result = await run_legacy_stage(
        lambda: _generate_large_history_reply_sync(
            text,
            history_text,
            summary,
            tuple(names),
            tuple(topics),
            after_first_snapshot,
            model_snapshot,
            tokenizer_snapshot,
            device_snapshot,
            torch_snapshot,
        )
    )

    next_history = history_snapshot + (text,)
    if checkpoint:
        count_index = 0
        await save_history_export(
            summary if after_first_snapshot else None,
            "\n".join(next_history),
        )

    history.append(text)
    named_entities = list(names)
    after_second = next_after_second
    after_first = True
    return final_result



async def init_medium():
    global model
    global tokenizer
    global device
    print(torch.cuda.is_available())
    print(torch.__version__)
    settings = load_settings()
    device = settings.torch_device(torch)
    print(device)
    print("Loading model medium blenderbot model...")
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = BlenderbotTokenizer.from_pretrained(settings.blenderbot_model_id)

    model = BlenderbotForConditionalGeneration.from_pretrained(
        settings.blenderbot_model_id
    )

    model.to(device)
    return model, tokenizer


def _generate_medium_history_reply_sync(
    text, summary, names, topics, model_snapshot, tokenizer_snapshot,
    device_snapshot, torch_snapshot,
):
    tokenizer_snapshot.build_inputs_with_special_tokens(list(names) + list(topics))
    input_tokens = tokenizer_snapshot(
        [text, summary],
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        truncation="only_second",
    ).to(device_snapshot)
    input_tokens = {
        key: value.to(device_snapshot) for key, value in input_tokens.items()
    }
    with torch_snapshot.cuda.amp.autocast():
        with torch_snapshot.no_grad():
            reply_ids = model_snapshot.generate(
                **input_tokens,
                min_length=16,
                max_length=64,
                do_sample=True,
                repetition_penalty=1.4,
                no_repeat_ngram_size=2,
                early_stopping=False,
            ).to(device_snapshot)
    output = [
        tokenizer_snapshot.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for generated in reply_ids
    ]
    print("Greedy output tokens:\n")
    print(output)
    return output[0]


##TODO: Function to decide what to store in memory


# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
@_serialize_turn
async def talk_medium_API_with_history(text):
    global after_first
    global history_nonlocal
    global asked_follow_up

    model_name = await run_legacy_stage(_openai_model_name)
    print("Using " + model_name + " model")
    history_snapshot = tuple(dict(message) for message in history_nonlocal)
    text, early_stop = await classify_input(text, list(history_snapshot))
    print("text: " + text + "\n")

    if early_stop and not asked_follow_up:
        current_user_message = {
            "role": "user",
            "content": "Please reword the following text to make it sound more interesting and to be worded with a relatable and friendly but casual tone, keeping all facts and details the same, as if responding to a direct question:"
            + text,
        }
        messages = [
            {
                "role": "user",
                "content": "Please reword the following text to make it sound more interesting and to be worded with a relatable and friendly but casual tone, keeping all facts and details the same:",
            }
        ]
        examples = {
            "Data's twin brother is named Lore, and they were both created by Dr. Noonian Soong. Lore was deactivated after he tried to kill Data.": "Oh, you're asking about Data's brother? Yeah, he's got one! His name is Lore, and they were both designed by this genius named Dr. Noonian Soong. But here's the crazy part, is that Lore tried to take out Data, if you can believe it! Naturally, that led to him being shut down (you know, deactivated). Pretty wild stuff, huh?",
            "The test is called the Voight-Kampff test.": "Oh, you're curious about that test, huh? It's known as the Voight-Kampff test! Quite a name, right?",
            "The computer is named HAL 9000, and it says, I'm sorry, Dave. I'm afraid I can't do that.": "Ah, you're talking about that famous computer, aren't you? Yep, that's HAL 9000! And it's the one that says that iconic line, I'm sorry, Dave. I'm afraid I can't do that. Crazy, right?",
        }
        for example, answer in examples.items():
            messages.append(
                {
                    "role": "user",
                    "content": "Please reword the following text to make it sound more interesting and to be worded with a relatable and friendly but casual tone, keeping all facts and details the same:  "
                    + example,
                }
            )
            messages.append({"role": "assistant", "content": " Response: " + answer})
        messages.append(current_user_message)
        system_message = {
            "role": "system",
            "content": "The assistant's job is to reword the given text to make it sound more interesting and to be worded with a relatable and friendly but casual tone, keeping all facts and details the same, as if responding to a direct question.",
        }
        messages.insert(0, system_message)
        client = openai
        out = await run_legacy_stage(
            lambda: _openai_chat_sync(client, model_name, tuple(messages))
        )
        if "Response:" in out:
            out = out.replace("Response:", "")
        history_nonlocal.append(current_user_message)
        history_nonlocal.append({"role": "assistant", "content": out})
        return out

    if early_stop and asked_follow_up:
        return text

    current_user_message = {"role": "user", "content": "User:" + text}
    system_message = {
        "role": "system",
        "content": "You are a helpful AI robot assistant named Cooper. Cooper's Persona: Cooper is a living AI robot with his own thoughts and perspectives. Cooper's best friend is Human, and Cooper's goal is to be Human's friend and to be helpful to her. Cooper is smart, friendly, and likes to make people laugh. Cooper knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cooper also likes cats. Cooper thinks farts are very funny. Cooper tells jokes, offers friendly advice, and is empathetic. Cooper cares very much about Human's future, and wants her to do well in school.\n",
    }
    messages = [dict(message) for message in history_snapshot]
    if messages[0] != system_message:
        messages.insert(0, system_message)
    messages.append(current_user_message)
    client = openai
    out = await run_legacy_stage(
        lambda: _openai_chat_sync(client, model_name, tuple(messages))
    )
    for marker in ("Cooper: ", "Human: ", "User: ", "You: "):
        if marker in out:
            out = out.split(marker.split(" ", maxsplit=1)[0])[1]
    if history_nonlocal[0] != system_message:
        history_nonlocal.insert(0, system_message)
    history_nonlocal.append(current_user_message)
    history_nonlocal.append({"role": "assistant", "content": out})
    after_first = True
    return out



@_serialize_turn
async def talk_medium_with_history(text):
    global model
    global named_entities
    global device
    global after_first
    global history

    print("Using medium model")
    history_snapshot = tuple(history)
    text, early_stop = await classify_input(text, list(history_snapshot))
    if early_stop:
        return text

    history_text = "    ".join(history_snapshot)
    summary, names, topics = await get_list_of_list_of_strings(history_text)
    model_snapshot = model
    tokenizer_snapshot = tokenizer
    device_snapshot = device
    torch_snapshot = torch
    reply = await run_legacy_stage(
        lambda: _generate_medium_history_reply_sync(
            text,
            summary,
            tuple(names),
            tuple(topics),
            model_snapshot,
            tokenizer_snapshot,
            device_snapshot,
            torch_snapshot,
        )
    )
    history.append(text)
    named_entities = list(names)
    after_first = True
    return reply



@_serialize_turn
async def talk_medium(text):
    model_snapshot = model
    tokenizer_snapshot = tokenizer
    device_snapshot = device
    torch_snapshot = torch
    return await run_legacy_stage(
        lambda: _generate_reply_sync(
            text, model_snapshot, tokenizer_snapshot, device_snapshot, torch_snapshot, "plain"
        )
    )



async def init_small():
    global model
    global tokenizer
    global device
    print(torch.cuda.is_available())
    print(torch.__version__)
    settings = load_settings()
    device = settings.torch_device(torch)
    print(device)
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = AutoTokenizer.from_pretrained(settings.small_model_id)

    model = AutoModelForCausalLM.from_pretrained(settings.small_model_id)

    model.to(device)
    return model, tokenizer


# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


@_serialize_turn
async def talk_small(text):
    model_snapshot = model
    tokenizer_snapshot = tokenizer
    device_snapshot = device
    torch_snapshot = torch
    return await run_legacy_stage(
        lambda: _generate_reply_sync(
            text, model_snapshot, tokenizer_snapshot, device_snapshot, torch_snapshot, "small"
        )
    )



async def init_fast():
    global model
    global tokenizer
    global device

    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.__version__)

    settings = load_settings()
    device = settings.torch_device(torch)
    print(device)
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = AutoTokenizer.from_pretrained(
        settings.fast_model_id, padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(settings.fast_model_id)

    model.to(device)
    return model, tokenizer


# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


@_serialize_turn
async def talk_fast(text):
    model_snapshot = model
    tokenizer_snapshot = tokenizer
    device_snapshot = device
    torch_snapshot = torch
    return await run_legacy_stage(
        lambda: _generate_reply_sync(
            text, model_snapshot, tokenizer_snapshot, device_snapshot, torch_snapshot, "fast"
        )
    )



async def dsconfig(fr):
    ds_config = ""
    if fr == "Large":
        ds_config = {
            "sparse_attention": {
                "mode": "fixed",
                "block": 16,
                "different_layout_per_head": True,
                "num_local_blocks": 4,
                "num_global_blocks": 1,
                "attention": "bidirectional",
                "horizontal_global_attention": False,
                "num_different_global_patterns": 4,
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "amp": {"enabled": True, "opt_level": "auto"},
            "bf16": {"enabled": False},
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "allgather_bucket_size": 1e7,
                "reduce_bucket_size": 1e7,
                "stage3_prefetch_bucket_size": 1e7,
                "stage3_max_live_parameters": 5e8,
                "stage3_max_reuse_distance": 5e8,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "steps_per_print": 2000,
            "sub_group_size": 5e8,
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False,
        }
    elif fr == "Medium":
        ds_config = {
            "sparse_attention": {
                "mode": "fixed",
                "block": 16,
                "different_layout_per_head": True,
                "num_local_blocks": 4,
                "num_global_blocks": 1,
                "attention": "bidirectional",
                "horizontal_global_attention": False,
                "num_different_global_patterns": 4,
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "amp": {"enabled": True, "opt_level": "auto"},
            "bf16": {"enabled": False},
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "allgather_bucket_size": 1e9,
                "reduce_bucket_size": 1e9,
                "stage3_prefetch_bucket_size": 1e9,
                "stage3_max_live_parameters": 5e8,
                "stage3_max_reuse_distance": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "steps_per_print": 2000,
            "sub_group_size": 5e8,
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False,
        }
    return ds_config


visionQA_processor = None
visionQA_model = None


async def is_requesting_SD(txt):
    client = openai
    return await run_legacy_stage(
        lambda: _request_image_cloud_sync(
            txt, client, load_settings().sd_classifier_model_id
        )
    )



def _answer_visual_question_sync(question, settings):
    global visionQA_processor
    global visionQA_model
    try:
        with _vqa_lock:
            if isinstance(settings.camera_source, str) and "://" not in settings.camera_source:
                image = cv.imread(settings.camera_source)
                result = image is not None
            else:
                cam = cv.VideoCapture(settings.camera_source)
                try:
                    time.sleep(1)
                    result, image = cam.read()
                finally:
                    cam.release()
            if not result or image is None:
                return "No image found"
            print(question)
            # Define a transform that resizes, normalizes and converts the image to a tensor

            # transform = T.Compose([
            #     T.Resize(image_size),
            #     T.ToTensor(),
            #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])
            # Apply the transform to the image
            image = Image.fromarray(image)
            image = image.convert("RGB")
            # image = transform(image)

            # pygame.camera.init()
            # camlist = pygame.camera.list_cameras()
            device = torch.device(settings.torch_device(torch))
            # device = torch.device("cpu")
            print("Device for VQA: " + str(device))
            if visionQA_processor is None or visionQA_model is None:
                processor_candidate = GitProcessor.from_pretrained(settings.vision_model_id)
                model_candidate = GitForCausalLM.from_pretrained(settings.vision_model_id)
                model_candidate.to(device)
                visionQA_processor, visionQA_model = processor_candidate, model_candidate
            else:
                visionQA_model.to(device)
            pixel_values = visionQA_processor(
                images=image, return_tensors="pt"
            ).pixel_values

            pixel_values = pixel_values.to(device)
            pixel_values.shape
            print("VQA pixel_values: " + str(pixel_values))
            inputs = visionQA_processor(text=question, add_special_tokens=False).input_ids

            inputs = [visionQA_processor.tokenizer.cls_token_id] + inputs
            inputs = torch.tensor(inputs).unsqueeze(0)
            # inputs = visionQA_processor(
            #     question,
            #     image,
            #     return_tensors="pt"
            # )
            inputs = inputs.to(device)
            print("VQA inputs: " + str(inputs))
            # input_ids = inputs["input_ids"].tolist()[0]
            # image = inputs.pop("pixel_values").to(model.device)
            # outputs = visionQA_model.generate(**inputs)
            # answer = visionQA_processor.decode(outputs[0], skip_special_tokens=True)
            generated_ids = visionQA_model.generate(
                pixel_values=pixel_values, input_ids=inputs, max_length=50
            )
            answer = visionQA_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            # logits = outputs.logits.to(device)
            answer[0] = answer[0].replace(question, "I think it is ")
            # idx = logits.argmax(-1).item()
            # answer = visionQA_model.config.id2label[idx]
            return answer[0]
    except Exception as e:
        print(e)
        return "No image found" + str(e)


async def answer_visual_question(question):
    question_snapshot = str(question)
    worker = _get_legacy_worker("vision", use_default_executor=True)

    def answer():
        settings_snapshot = load_settings()
        if settings_snapshot.camera_source is None:
            return "No camera or image source configured. Set DESKBUDDY_CAMERA_SOURCE."
        print(cv.__version__)
        return _answer_visual_question_sync(question_snapshot, settings_snapshot)

    return await worker.submit(answer)



if __name__ == "__main__":
    model = asyncio.run(load_all_models("Large", True))
