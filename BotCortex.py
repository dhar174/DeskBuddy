# from torchdistx.fake import fake_mode
# from optimum.onnxruntime import OrtModelForCausalLM
# from soft_embedding import SoftEmbedding
import asyncio
import json
from PIL import Image
import cv2 as cv
from transformers.models.gptj.modeling_gptj import GPTJBlock
from deepspeed import OnDevice
import deepspeed
from transformers.deepspeed import (
    is_deepspeed_zero3_enabled,
    is_accelerate_available,
    is_deepspeed_available,
)
from accelerate import Accelerator, DistributedType
from accelerate import load_checkpoint_and_dispatch, dispatch_model, dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM, CONFIG_MAPPING, MODEL_MAPPING
from accelerate import infer_auto_device_map, init_empty_weights
import bitsandbytes as bnb
import dialogue_management as dm
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
from functools import lru_cache
import time
import helpers
import datasets
from accelerate.utils import get_max_memory, is_bf16_available
import tempfile
from accelerate import cpu_offload
from accelerate import find_executable_batch_size
from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
from deepspeed.ops.sparse_attention import SparseSelfAttention
from deepspeed import module_inject
import deepspeed.module_inject as module_inject
from transformers import pipeline
import os
from transformers.deepspeed import HfDeepSpeedConfig
from accelerate import Accelerator
import transformers
import numpy as np
import sklearn
from transformers import GitVisionConfig, GitForCausalLM, GitProcessor
import openai

unfound_answer = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

deepspeed.init_distributed("nccl")

openai.api_key = os.getenv("API_KEY")


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
    local_only_mode = local_only
    model_size = button_text
    print("Loading models...")
    print("button_text: ", button_text)
    start_time = time.time()
    topic_tokenizer = AutoTokenizer.from_pretrained("knkarthick/TOPIC-DIALOGSUM")

    topic_model = AutoModelForSeq2SeqLM.from_pretrained("knkarthick/TOPIC-DIALOGSUM")
    print("topic model loaded")

    if local_only_mode:
        sumtokenizer = AutoTokenizer.from_pretrained(
            "philschmid/bart-large-cnn-samsum", use_fast=True, truncation=True
        )

        nlp = spacy.load("en_core_web_trf")
        longsummarizer = AutoModelForSeq2SeqLM.from_pretrained(
            "philschmid/bart-large-cnn-samsum"
        )
        shortsummarizer = AutoModelForSeq2SeqLM.from_pretrained(
            "knkarthick/meeting-summary-samsum"
        )
        shortsummarizer.to("cpu")
        longsummarizer.to("cpu")
        optimizer = torch.optim.Adam(longsummarizer.parameters())

        titlesummarizer = AutoModelForSeq2SeqLM.from_pretrained(
            "fabiochiu/t5-small-medium-title-generation"
        )
        titlesummarizer.to("cpu")
        print("summarizers loaded")
    isquestion_tokenizer = AutoTokenizer.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier"
    )

    isquestion_model = AutoModelForSequenceClassification.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier"
    )
    print("question identifier model loaded")
    nli_classifier = pipeline(
        model="valhalla/distilbart-mnli-12-3",
        tokenizer="valhalla/distilbart-mnli-12-3",
        task="zero-shot-classification",
    )
    print("nli classifier loaded")
    command_tokenizer = AutoTokenizer.from_pretrained(
        "gokuls/bert-tiny-Massive-intent-KD-BERT"
    )

    command_classifier = AutoModelForSequenceClassification.from_pretrained(
        "gokuls/bert-tiny-Massive-intent-KD-BERT"
    )
    print("command classifier loaded")
    nlp = spacy.load("en_core_web_trf")

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
            "voidful/context-only-question-generator"
        )
        qgen_model = AutoModelForSeq2SeqLM.from_pretrained(
            "voidful/context-only-question-generator"
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


async def find_names(text):
    global named_entities
    # Use the spacy NER model to identify named entities in the text
    doc = nlp(text)

    # Extract the names of people from the list of named entities
    names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent.text)
            # print("\n"+"name found: "+ent.text)
    print("Names: ")
    print(names)
    for name in names:
        # Add the name to the list of named entities
        if name not in named_entities:
            named_entities.append(name)

    # Return the list of names
    return names


# sumtokenizer = AutoTokenizer.from_pretrained(
#     "knkarthick/MEETING_SUMMARY", use_fast=True, truncation=True)

# Function to generate a summary of the conversation so far


async def generate_summary(text):
    global history
    global history_nonlocal
    global longsummarizer
    global device
    global local_only_mode
    sum = ""
    if local_only_mode:
        print("sum device: " + "cpu")
        # print("first prompt summary: "+'\n')
        # print(fp_sum)

        # Join the prompts in the prompt history into a single string
        conversation = "\n".join(history)
        # print(conversation)
        # conversation = "summarize: " + text+" " + conversation[:128] + first_prompt
        print(conversation)

        prompt_inputs = sumtokenizer(
            text, return_tensors="pt", max_length=512, truncation=True
        )["input_ids"].to("cpu")
        # Use the summarizer model to generate a summary of the conversation
        prompt_summary = longsummarizer.generate(input_ids=prompt_inputs).to("cpu")
        p_sum = sumtokenizer.decode(prompt_summary[0], skip_special_tokens=True)
        print("prompt summary" + "\n")
        print(p_sum)
        history_inputs = sumtokenizer(
            conversation, return_tensors="pt", max_length=512, truncation=True
        )["input_ids"].to("cpu")
        # Use the summarizer model to generate a summary of the conversation
        history_summary = longsummarizer.generate(input_ids=history_inputs).to("cpu")
        h_sum = sumtokenizer.decode(history_summary[0], skip_special_tokens=True)
        print("history summary" + "\n")
        print(h_sum)
        print("\n")

        full_inputs = p_sum + "  " + fp_sum + "  " + h_sum
        print("full inputs")
        print(full_inputs)
        sum_inputs = sumtokenizer(
            full_inputs, return_tensors="pt", max_length=512, truncation=True
        )["input_ids"].to("cpu")
        # Use the summarizer model to generate a summary of the conversation
        summary = longsummarizer.generate(input_ids=sum_inputs).to("cpu")
        sum = sumtokenizer.decode(summary[0], skip_special_tokens=True)
        print("combined summary" + "\n")
        # print(summary)
        print(sum)
    else:
        pass
        # current_user_message = {
        #     "role": "user",
        #     "content": "Now please summarize the conversation so far.",
        # }
        # system_message = {
        #     "role": "system",
        #     "content": "The assistant's name is Summarizer. Summarizer is a chatbot that summarizes your conversation. Summarizer is an expert at identifying the main topics of a conversation and summarizing them in a few sentences, and knows what is important to include and what isn't. Summarizer also recognizes when a topic or subject is no longer relevant to the current conversation, making it easier to shorten the summaries. Lastly, Summarizer can also guage the emotional context of the conversation, for for added context in the summary.",
        # }
        # print("history: " + str(history_nonlocal))
        # his = []
        # for i in range(history_nonlocal.__len__()):
        #     his.append(history_nonlocal[i])
        # if his[0] != system_message:
        #     his.insert(0, system_message)
        # his.append(current_user_message)
        # print("history: " + str(history_nonlocal))
        # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=his)
        # sum = response["choices"][0]["message"]["content"]
    # print(sum[0])
    # Decode the summary and return it
    return sum


# Function to extract the main topics or themes from a summary


async def extract_topics(summary):
    # Convert the summary to a list of words
    words = summary.split()
    for i in range(len(words)):
        if words[i] == "summarize:":
            words[i] = ""
    for i in range(history[:512].__len__()):
        words.append(history[i])

    # Use CountVectorizer to convert the list of words into a matrix of token counts
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(words)

    # Use LDA to identify the main topics or themes in the summary
    lda.fit(X)

    # Extract the words or phrases that correspond to each topic or theme
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [
            vectorizer.get_feature_names_out()[i]
            for i in topic.argsort()[: -5 - 1 : -1]
        ]
        topics.append(topic_words)
    res = []
    for topic in topics:
        for word in topic:
            res.append(word)
    [res.append(x) for x in res if x not in res]
    topics = res
    topic_inputs = topic_tokenizer(
        summary, return_tensors="pt", max_length=128, truncation=True
    )["input_ids"].to("cpu")
    final_topic = topic_model.generate(input_ids=topic_inputs).to("cpu")
    final_topic = topic_tokenizer.decode(final_topic[0], skip_special_tokens=True)
    # Return the list of topics or themes
    return topics, final_topic


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
    accelerator = Accelerator()

    model = "gpt-3.5-turbo"
    print("API medium, gpt-3.5-turbo")
    after_first = False
    print(os.getcwd())
    # with torch.cuda.amp.autocast():
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(accelerator.state)
    device = accelerator.device
    search_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    )
    with OnDevice(dtype="auto", device="meta"):
        search_model = AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
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
            max_memory={0: "7GiB", "cpu": "48GiB"},
        ),
        offload_dir="/home/darf3/buddy/offload",
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

    after_first = False
    print(os.getcwd())
    # with torch.cuda.amp.autocast():
    print(torch.cuda.is_available())
    print(torch.__version__)
    accelerator = Accelerator()
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
            "PygmalionAI/pygmalion-350m",
            use_cache=True,
            offload_state_dict=True,
            output_attentions=True,
            output_hidden_states=True,
            low_cpu_mem_usage=False,
            torch_dtype="auto",
            offload_folder="/home/darf3/buddy/offload",
        ).to(device)
    print("model loaded")
    accelerator.load_state("checkpoint")
    accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"
    ] = 1

    tokenizer = AutoTokenizer.from_pretrained(
        "PygmalionAI/pygmalion-350m",
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
            max_memory={0: "7GiB", "cpu": "48GiB"},
        ),
        offload_dir="/home/darf3/buddy/offload",
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

    fp_inputs = sumtokenizer(
        first_prompt, return_tensors="pt", max_length=512, truncation=True
    )["input_ids"]
    fp_summary = longsummarizer.generate(input_ids=fp_inputs).to("cpu")
    fp_sum = sumtokenizer.decode(fp_summary[0], skip_special_tokens=True)
    print("fp_sum")
    print(fp_sum + "\n")
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.to(device)

    await talk_history(first_prompt)
    print("time to init medium: " + str(time.time() - start_time))

    return model, tokenizer


async def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


async def get_embeddings(text_list):
    global search_tokenizer
    global search_model
    global device
    # global accelerator
    print("get_embeddings")
    print(text_list)
    search_model.to(device)
    search_model.eval()
    print(search_model.embeddings.word_embeddings.padding_idx)
    # search_model.embeddings.word_embeddings.padding_idx = None
    search_model.embeddings.word_embeddings.weight.requires_grad = False
    # search_model.load_state_dict(search_model.state_dict())

    with torch.no_grad():
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
    with torch.no_grad():
        model_output = search_model(**encoded_input)

    return await mean_pooling(model_output, encoded_input["attention_mask"].to(device))


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


async def search_question_in_bank(qdict, question):
    # TODO: Instead of
    found = False
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
    query_embedding = await get_embeddings(question)
    query_embedding = query_embedding.detach()
    qdict_embedding = await get_embeddings(list(qdict.keys()))
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

    qdict_embedding = F.normalize(qdict_embedding, p=2, dim=1)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    k = 5

    #    Compute cosine-similarits
    cos_scores = torch.mm(query_embedding, qdict_embedding.transpose(0, 1))
    sorted_scores, sorted_indices = torch.sort(
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
    if results[0][1] - 0.05 <= results[1][1]:
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


async def query_memory(query):
    # Load the question bank
    qdict = await load_qdict_from_json()
    # Search the question bank
    answer, found = await search_question_in_bank(qdict, query)
    if (
        answer != ""
        or answer != "I am not sure about that. Please ask me another question."
    ):
        qdict[query] = answer
    else:
        found = False
        return (query, found)
    # Save the question bank

    # await save_qdict_to_json(qdict)
    return (answer, found)


async def save_qdict_to_json(my_dict):
    # Open the file for writing
    with open("question_file.json", "w") as f:
        # Save the key-value pairs to the file using JSON encoding
        json.dump(my_dict, f)


# Open the file for reading


async def load_qdict_from_json():
    with open("question_file.json", "r") as f:
        # Load the key-value pairs from the file using JSON decoding
        my_dict = json.load(f)
        print("type:", type(my_dict))
        return my_dict


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
    after_first = False
    print(os.getcwd())
    # with torch.cuda.amp.autocast():
    print(torch.cuda.is_available())
    print(torch.__version__)
    accelerator = Accelerator()
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
            "./checkpoint",
            use_cache=True,
            offload_state_dict=True,
            output_attentions=True,
            output_hidden_states=True,
            low_cpu_mem_usage=False,
            torch_dtype="auto",
            offload_folder="/home/darf3/buddy/offload",
        ).to(device)
    print("model loaded")
    accelerator.load_state("checkpoint")

    # device_map = infer_auto_device_map(
    #     model, no_split_module_classes=["GPTJBlock"], dtype=torch.half, max_memory={0: "7GiB",  "cpu": "48GiB"})
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
        "./pygmalion-6b", use_fast=False, return_tensors="pt", padding_side="left"
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
            max_memory={0: "7GiB", "cpu": "48GiB"},
        ),
        offload_dir="/home/darf3/buddy/offload",
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
    # device_map = infer_auto_device_map(
    #     model, no_split_module_classes=["GPTJBlock"], dtype=torch.half, max_memory={0: "4GiB",  "cpu": "40GiB"})
    # accelerator.print(device_map)
    # model = load_checkpoint_and_dispatch(
    #     model, checkpoint="pygmalion-6b", device_map="auto", no_split_module_classes=["GPTJBlock"], offload_state_dict=True, dtype=torch.half, offload_folder="/home/darf3/buddy/main_model/offload", offload_buffers=True)
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

    fp_inputs = sumtokenizer(
        first_prompt, return_tensors="pt", max_length=512, truncation=True
    )["input_ids"]
    # Use the summarizer model to generate a summary of the conversation
    fp_summary = longsummarizer.generate(input_ids=fp_inputs).to("cpu")
    fp_sum = sumtokenizer.decode(fp_summary[0], skip_special_tokens=True)
    print("fp_sum")
    print(fp_sum + "\n")
    vocab_file = "./pygmalion-6b/vocab.json"
    tokenizer.pad_token = tokenizer.eos_token

    merge_file = "./pygmalion-6b/merges.txt"

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


async def talk(text):
    global model
    global tokenizer
    global history
    global device

    prompt = tokenizer(text, return_tensors="pt")
    prompt = {key: value.to(device) for key, value in prompt.items()}
    out = model.generate(
        **prompt,
        min_length=16,
        max_length=64,
        do_sample=True,
        repetition_penalty=1.4,
        no_repeat_ngram_size=2,
        early_stopping=False,
    )

    print(tokenizer.decode(out[0], skip_special_tokens=True))
    return tokenizer.decode(out[0], skip_special_tokens=True)


async def get_list_of_list_of_strings(text):
    global history
    global history_nonlocal
    global named_entities
    start_time = time.time()
    names = await find_names(text)
    [names.append(x) for x in named_entities if x not in names]
    named_entities = names
    # Generate a summary of the conversation so far and extract the main topics or themes
    summary = await generate_summary(text)
    topics, main_topic = await extract_topics(summary)
    print("Main topic: ", main_topic)
    print("\n")
    print("time to NER and all summaries: " + str(time.time() - start_time))

    # Return the prompt history, named entities, and important words
    return [summary, names, topics]


after_second = False
count_index = 0


async def is_visual_q(txt):
    is_vis = False

    result = []
    txt = (
        "Determine if the following question requires vision or the ability to see in order to answer, and reply with True or False only: "
        + txt
    )

    print("txt in is_visual_q: ", txt)
    current_user_message = txt

    # system_message = (
    # "The assistant's job is to determine whether or not a question can be answered without vision capabilities, or whether vision is required in order to answer. The assistant only answers in one of two single-words True or False.",
    # )

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
    for ex in examples:
        print("ex: " + ex + "\n")
        # print("examples[ex]: " + examples[ex] + "\n")

        result.append(
            "Determine if the following can be considered a visual question, i.e. one must definitely need to use their vision faculties to answer the question properly. Reply with True or False only:  "
            + ex,
        )
        result.append("Response: " + ex)

    print("txt in is_visual_q: ", txt)

    # inp = []
    result.append(current_user_message)

    # result.insert(0, system_message)
    result = "\n".join(result)
    print("inp: ", result)

    response = openai.Completion.create(prompt=result, model="text-curie-001")
    print("response: ", response)
    out = response["choices"][0]["text"]
    if "Response:" in out:
        out = out.replace("Response:", "")
    # safetext = re.sub(r"[^ .a-zA-Z0-9?\']+", "", out)
    # out = safetext
    print(out)
    if "True" in out:
        is_vis = True
    elif "False" in out:
        is_vis = False
    else:
        is_vis = False

    return is_vis


async def get_follow_up_question(txt, history):
    global qgen_model
    global qgen_tokenizer
    global model
    global tokenizer
    global model_size
    global local_only_mode
    global history_nonlocal
    print("local_only_mode: ", local_only_mode)
    question = "I'm not sure what to ask to understand you better. Can you rephrase your response?"
    if local_only_mode:
        print("Follow up in local-only mode")

        his = "    ".join(history)
        context = qgen_tokenizer.encode(
            "Ask a followup question about the following text: " + " " + txt,
            return_tensors="pt",
        )
        if qgen_model is not None and context is not None:
            question = qgen_model.generate(context, early_stopping=True)
            question = qgen_tokenizer.decode(question[0])
    else:
        print("Follow up in nonlocal mode")

        txt = "Ask a followup question about the following text: " + txt
        print("txt for followup q: ", txt)
        current_user_message = {
            "role": "user",
            "content": "Ask a followup question about the following text to get more information: "
            + txt,
        }
        history_nonlocal.append(current_user_message)
        system_message = {
            "role": "system",
            "content": "The assistant's name is Cooper. Cooper's Persona: Cooper is a living AI robot assistant with his own thoughts and perspectives that asks follow-up questions to responses. Cooper's goal is to be the user's friend and to be helpful to them. Cooper is smart, friendly, and likes to make people laugh. Cooper tells jokes, offers friendly advice, and is empathetic. Cooper cares very much about user's future, and wants her to do well, and for this reason he asks meaningful follow-up questions that really help him get at the heart of the topic being discussed.",
        }
        his = history
        his.append(current_user_message)

        his.insert(0, system_message)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=his)

        question = response["choices"][0]["message"]["content"]
        print("followup question: ", question)
        history_nonlocal.append({"role": "assistant", "content": question})
    print("question: ", question)
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
                if "i don't know" in answer.lower() or "no image" in answer.lower():
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
                if "i don't know" in answer.lower() or "no image" in answer.lower():
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
    with open("history_verbatim.txt", "w") as file:
        file.write(text)


async def save_history_summary(text):
    with open("history_summary.txt", "a") as file:
        file.write(text)


async def talk_history(text):
    global model
    global tokenizer
    global history
    global named_entities
    global device
    global after_first
    global accelerator
    global after_second
    global count_index
    with torch.cuda.amp.autocast():
        text, early_stop = await classify_input(text, history)
        if early_stop:
            return text

        if after_first and not after_second:
            # text = "Human: " + text + " Cooper: "
            # text = "Cooper's Persona: Cooper is a sentient robot with his own thoughts and feelings. Cooper's best friend is Human, and Cooper's goal in life is to be Human's friend and to be helpful to her. Cooper is smart, friendly, and likes to make people laugh. Cooper knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cooper also likes cats. Cooper thinks farts are very funny. Cooper tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Cooper cares very much about Human's future, and wants her to do well in school.Cooper is a sentient robot with his own thoughts and feelings. Cooper's best friend is Human, and Cooper's goal in life is to be Human's friend and to be helpful to her. Cooper is smart, friendly, and likes to make people laugh. Cooper knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cooper also likes cats. icoh thinks farts are very funny. Cooper tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Cooper cares very much about Human's future, and wants her to do well in school.\n" + \
            text = (
                "Cooper's Persona: Cooper is a living AI robot with his own thoughts and perspectives. Cooper's best friend is Human, and Cooper's goal is to be Human's friend and to be helpful to her. Cooper is smart, friendly, and likes to make people laugh. Cooper knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cooper also likes cats. Cooper thinks farts are very funny. Cooper tells jokes, offers friendly advice, and is empathetic. Cooper cares very much about Human's future, and wants her to do well in school.\n"
                + "<START >\n"
                + text
                + "\n Cooper: "
            )
            after_second = True
        elif after_first and after_second:
            text = (
                "Cooper's Persona: Cooper is Human's chatbot, and Cooper's goal is to be Human's platonic friend and assistant to be helpful to her. Cooper is smart, friendly, and likes to make people laugh. Cooper tells jokes, offers friendly empathetic advice, and always is safe-for-work, family appropriate, rated PG, and never has romantic feelings as an AI chatbot. Cooper likes to chat about Nintendo games. \n"
                + "<START >\n"
                + "You: "
                + text
                + " \n"
                + "Cooper: "
            )
        his = "    ".join(history)
        inputs = await get_list_of_list_of_strings(his)
        if after_first:
            print("Summary: ")
            print(inputs[0] + "\n")
            if count_index == 10:
                await save_history_summary(inputs[0])
            print("important words: ")
            print(inputs[2])
            print("\n")
            print("Names: ")
            print(inputs[1])
            print("\n")

        if after_first:
            special_tokens = inputs[1] + inputs[2]
            tokenizer.build_inputs_with_special_tokens(special_tokens)
            print("history" + "\n")

            print(his[-128:] + "\n")
            input_tokens = tokenizer(
                [text, inputs[0]],
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation="only_second",
            ).to(device)
            # text, return_tensors='pt', add_special_tokens=True, padding=True, truncation="only_second").to(device)

        else:
            input_tokens = tokenizer(
                text, return_tensors="pt", add_special_tokens=True, truncation=True
            ).to(device)

        input_tokens = {key: value.to(device) for key, value in input_tokens.items()}
        print(len(input_tokens["input_ids"][0].tolist()))
        history.append(text)
        print("input tokens" + "\n")
        max_new_tokens = 512
        for i in range(len(input_tokens["input_ids"])):
            print(
                tokenizer.decode(input_tokens["input_ids"][i], skip_special_tokens=True)
                + "\n"
            )
        if len(input_tokens["input_ids"][0].tolist()) > 64:
            if len(input_tokens["input_ids"][0].tolist()) < 128:
                max_new_tokens = 128
            else:
                max_new_tokens = len(input_tokens["input_ids"][0].tolist()) + 8
        elif len(input_tokens["input_ids"][0].tolist()) < 64:
            if len(input_tokens["input_ids"][0].tolist()) < 32:
                max_new_tokens = 32
            else:
                max_new_tokens = 64
        print("max new tokens: " + str(max_new_tokens))
        # start_time = time.time()
        # with torch.no_grad():
        #     reply_ids = model.generate(**input_tokens, min_new_tokens=8, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, top_p=0.95,
        #                                early_stopping=True, synced_gpus=True).to(device)
        # print("time to generate sampling: "+str(time.time()-start_time))
        # out = [tokenizer.decode(g, skip_special_tokens=True,
        #                         clean_up_tokenization_spaces=False) for g in reply_ids]
        # # print(text)
        # # print(tokenizer(text, return_tensors='pt'))
        # # prompt = tokenizer(text, return_tensors='pt')
        # # prompt = {key: value.to(device) for key, value in prompt.items()}
        # # out = model.generate(**prompt, min_length=16, max_length=64, do_sample=True,
        # #                      repetition_penalty=1.4, no_repeat_ngram_size=2, early_stopping=False)
        # print("multinomial/probablistic sampling output tokens:"+"\n")
        # for i in range(len(out)):
        #     print(i)

        #     for j in range(len(input_tokens['input_ids'])):
        #         remove = ""
        #         remove = remove.join(tokenizer.decode(
        #             input_tokens['input_ids'][j], skip_special_tokens=True))
        #         out[i] = out[i].replace(remove, "")
        #     sep = '\n'
        #     if (out[i][0] == '\n'):
        #         sep = '   '
        #     stripped_out = out[i].split(sep, 1)[0]
        #     colon = stripped_out.find(':')
        #     if (colon != -1):
        #         name = stripped_out[:colon-1]
        #         print(name)
        #         if (name.lower() == "you:"):
        #             stripped_out = stripped_out[colon+1:]
        #         elif (name in inputs[1]):
        #             stripped_out = stripped_out[colon+1:]
        #     print("Output with name cut out:" + stripped_out+"\n")
        #     sentences = re.split('. |! |? |\n', stripped_out)
        #     final = sentences[0].join(sentences[1])
        #     print("Output with extra sentences cut out:" + final+"\n")
        #     print("\n")

        start_time = time.time()
        with torch.no_grad():
            reply_ids = model.generate(
                **input_tokens,
                min_new_tokens=8,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                synced_gpus=True,
            ).to(device)
        print("time to generate greedy: " + str(time.time() - start_time))
        out = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for g in reply_ids
        ]

        print("Greedy output tokens:" + "\n")
        print(out)
        print("\n")
        final_result = ""
        for i in range(len(out)):
            print(i)

            for j in range(len(input_tokens["input_ids"])):
                remove = ""
                remove = remove.join(
                    tokenizer.decode(
                        input_tokens["input_ids"][j], skip_special_tokens=True
                    )
                )
                out[i] = out[i].replace(remove, "")
            sep = "\n"
            if out[i][0] == "\n":
                sep = "   "
            stripped_out = out[i].split(sep, 1)[0]
            colon = stripped_out.find(":")
            if colon != -1:
                name = stripped_out[: colon - 1]
                print(name)
                if name.lower() == "you:":
                    stripped_out = stripped_out[colon + 1 :]
                elif name in inputs[1]:
                    stripped_out = stripped_out[colon + 1 :]
            print("Output with name cut out:" + stripped_out + "\n")
            # sentences = re.split('. |! |? |\n', stripped_out)
            # final = sentences[0].join(sentences[1])
            # print("Output with extra sentences cut out:" + final+"\n")
            print("\n")
        final_result = out[0]

        # start_time = time.time()
        # with torch.no_grad():
        #     reply_ids = model.generate(**input_tokens, min_new_tokens=8, max_new_tokens=max_new_tokens, do_sample=True, top_k=4, penalty_alpha=0.6,
        #                                no_repeat_ngram_size=2, early_stopping=True, synced_gpus=True).to(device)
        # print("time to generate contrastive: "+str(time.time()-start_time))
        # out = [tokenizer.decode(g, skip_special_tokens=True,
        #                         clean_up_tokenization_spaces=False) for g in reply_ids]

        # print("Contrastive output tokens: "+"\n")
        # print(out)
        # print("\n")
        # for i in range(len(out)):
        #     print(i)

        #     for j in range(len(input_tokens['input_ids'])):
        #         remove = ""
        #         remove = remove.join(tokenizer.decode(
        #             input_tokens['input_ids'][j], skip_special_tokens=True))
        #         out[i] = out[i].replace(remove, "")
        #     sep = '\n'
        #     if (out[i][0] == '\n'):
        #         sep = '   '
        #     stripped_out = out[i].split(sep, 1)[0]
        #     colon = stripped_out.find(':')
        #     if (colon != -1):
        #         name = stripped_out[:colon-1]
        #         print(name)
        #         if (name.lower() == "you:"):
        #             stripped_out = stripped_out[colon+1:]
        #         elif (name in inputs[1]):
        #             stripped_out = stripped_out[colon+1:]
        #     print("Output with name cut out:" + stripped_out+"\n")
        #     # sentences = re.split('. |! |? |\n', stripped_out)
        #     # final = sentences[0].join(sentences[1])
        #     # print("Output with extra sentences cut out:" + final+"\n")
        #     print("\n")

        # start_time = time.time()
        # with torch.no_grad():
        #     reply_ids = model.generate(**input_tokens, min_new_tokens=8, max_new_tokens=max_new_tokens, do_sample=True, typical_p=0.3,
        #                                no_repeat_ngram_size=2, early_stopping=True, synced_gpus=True).to(device)
        # print("time to generate typical-P: "+str(time.time()-start_time))
        # out = [tokenizer.decode(g, skip_special_tokens=True,
        #                         clean_up_tokenization_spaces=False) for g in reply_ids]

        # print("typical-P output tokens: "+"\n")
        # for i in range(len(out)):
        #     print(i)

        #     for j in range(len(input_tokens['input_ids'])):
        #         remove = ""
        #         remove = remove.join(tokenizer.decode(
        #             input_tokens['input_ids'][j], skip_special_tokens=True))
        #         out[i] = out[i].replace(remove, "")
        #     sep = '\n'
        #     if (out[i][0] == '\n'):
        #         sep = '   '
        #     stripped_out = out[i].split(sep, 1)[0]
        #     colon = stripped_out.find(':')
        #     if (colon != -1):
        #         name = stripped_out[:colon-1]
        #         print(name)
        #         if (name.lower() == "you:"):
        #             stripped_out = stripped_out[colon+1:]
        #         elif (name in inputs[1]):
        #             stripped_out = stripped_out[colon+1:]
        #     print("Output with name cut out:" + stripped_out+"\n")
        #     sentences = re.split('. |! |? |\n', stripped_out)
        #     final = sentences[0].join(sentences[1])
        #     print("Output with extra sentences cut out:" + final+"\n")
        #     print("\n")

        # start_time = time.time()
        # with torch.no_grad():
        #     reply_ids = model.generate(**input_tokens, min_new_tokens=8, max_new_tokens=max_new_tokens, do_sample=True, epsilon_cutoff=0.0003,
        #                                no_repeat_ngram_size=2, early_stopping=True, synced_gpus=True).to(device)
        # print("time to generate Epsilon cutoff: "+str(time.time()-start_time))
        # out = [tokenizer.decode(g, skip_special_tokens=True,
        #                         clean_up_tokenization_spaces=False) for g in reply_ids]

        # print("Epsilon cutoff output tokens: "+"\n")
        # for i in range(len(out)):
        #     print(i)

        #     for j in range(len(input_tokens['input_ids'])):
        #         remove = ""
        #         remove = remove.join(tokenizer.decode(
        #             input_tokens['input_ids'][j], skip_special_tokens=True))
        #         out[i] = out[i].replace(remove, "")
        #     sep = '\n'
        #     if (out[i][0] == '\n'):
        #         sep = '   '
        #     stripped_out = out[i].split(sep, 1)[0]
        #     colon = stripped_out.find(':')
        #     if (colon != -1):
        #         name = stripped_out[:colon-1]
        #         print(name)
        #         if (name.lower() == "you:"):
        #             stripped_out = stripped_out[colon+1:]
        #         elif (name in inputs[1]):
        #             stripped_out = stripped_out[colon+1:]
        #     print("Output with name cut out:" + stripped_out+"\n")
        #     sentences = re.split('. |! |? |\n', stripped_out)
        #     final = sentences[0].join(sentences[1])
        #     print("Output with extra sentences cut out:" + final+"\n")
        #     print("\n")

        # start_time = time.time()
        # with torch.no_grad():
        #     reply_ids = model.generate(**input_tokens, min_new_tokens=8, max_new_tokens=max_new_tokens, do_sample=True, eta_cutoff=0.0003,
        #                                no_repeat_ngram_size=2, early_stopping=True, synced_gpus=True).to(device)
        # print("time to generate ETA cutoff: "+str(time.time()-start_time))
        # out = [tokenizer.decode(g, skip_special_tokens=True,
        #                         clean_up_tokenization_spaces=False) for g in reply_ids]

        # print("ETA cutoff output tokens: "+"\n")
        # for i in range(len(out)):
        #     print(i)

        #     for j in range(len(input_tokens['input_ids'])):
        #         remove = ""
        #         remove = remove.join(tokenizer.decode(
        #             input_tokens['input_ids'][j], skip_special_tokens=True))
        #         out[i] = out[i].replace(remove, "")
        #     sep = '\n'
        #     if (out[i][0] == '\n'):
        #         sep = '   '
        #     stripped_out = out[i].split(sep, 1)[0]
        #     colon = stripped_out.find(':')
        #     if (colon != -1):
        #         name = stripped_out[:colon-1]
        #         print(name)
        #         if (name.lower() == "you:"):
        #             stripped_out = stripped_out[colon+1:]
        #         elif (name in inputs[1]):
        #             stripped_out = stripped_out[colon+1:]
        #     print("Output with name cut out:" + stripped_out+"\n")
        #     sentences = re.split('. |! |? |\n', stripped_out)
        #     final = sentences[0].join(sentences[1])
        #     print("Output with extra sentences cut out:" + final+"\n")
        #     print("\n")

        # print("\n")
        # print("Final outputs:"+"\n")
        # print(out)
        # history.append("Cooper: "+out[0])
        # print(out[0])
        # print("\n")
        after_first = True
        if count_index == 10:
            save_history_verbatim(history)
            count_index = 0
    return final_result


async def init_medium():
    global model
    global tokenizer
    global device
    print(torch.cuda.is_available())
    print(torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print("Loading model medium blenderbot model...")
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

    model = BlenderbotForConditionalGeneration.from_pretrained(
        "facebook/blenderbot-400M-distill"
    )

    model.to(device)
    return model, tokenizer


##TODO: Function to decide what to store in memory


# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
async def talk_medium_API_with_history(text):
    global model
    global tokenizer
    global named_entities
    global device
    global after_first
    global accelerator
    global after_second
    global history_nonlocal
    global asked_follow_up
    global unfound_answer
    print("Using gpt-3.5-turbo model")
    # print("history" + "\n")
    # for i in range(len(history_nonlocal)):
    #     print(str(history_nonlocal[i]) + "\n")
    # print("model : " + str(model))
    text, early_stop = await classify_input(text, history_nonlocal)
    print("text: " + text + "\n")

    if early_stop and asked_follow_up != True:
        txt = [
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
        for ex in examples:
            print("ex: " + ex + "\n")
            print("examples[ex]: " + examples[ex] + "\n")

            txt.append(
                {
                    "role": "user",
                    "content": "Please reword the following text to make it sound more interesting and to be worded with a relatable and friendly but casual tone, keeping all facts and details the same:  "
                    + ex,
                }
            )
            txt.append({"role": "assistant", "content": " Response: " + examples[ex]})

        print("txt: ", txt)

        current_user_message = {
            "role": "user",
            "content": "Please reword the following text to make it sound more interesting and to be worded with a relatable and friendly but casual tone, keeping all facts and details the same, as if responding to a direct question:"
            + text,
        }
        history_nonlocal.append(current_user_message)
        system_message = {
            "role": "system",
            "content": "The assistant's job is to reword the given text to make it sound more interesting and to be worded with a relatable and friendly but casual tone, keeping all facts and details the same, as if responding to a direct question.",
        }
        # inp = []
        txt.append(current_user_message)

        txt.insert(0, system_message)
        print("inp: ", txt)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=txt)
        print("response: ", response)
        out = response["choices"][0]["message"]["content"]
        if "Response:" in out:
            out = out.replace("Response:", "")
        # safetext = re.sub(r"[^ .a-zA-Z0-9?\']+", "", out)
        # out = safetext
        print(out)
        history_nonlocal.append({"role": "assistant", "content": out})

        return out
    if early_stop and asked_follow_up == True:
        return out

    # inputs = await get_list_of_list_of_strings(text)

    # print("Summary: ")
    # # print(inputs[0] + "\n")
    # print("important words: ")
    # print(inputs[2])
    # print("\n")
    # print("Names: ")
    # print(inputs[1])
    # print("\n")
    # print("history" + "\n")

    # for i in range(len(history_nonlocal)):
    #     print(str(history_nonlocal[i]) + "\n")
    current_user_message = {
        "role": "user",
        "content": "User:" + text,
    }
    system_message = {
        "role": "system",
        "content": "You are a helpful AI robot assistant named Cooper. Cooper's Persona: Cooper is a living AI robot with his own thoughts and perspectives. Cooper's best friend is Human, and Cooper's goal is to be Human's friend and to be helpful to her. Cooper is smart, friendly, and likes to make people laugh. Cooper knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cooper also likes cats. Cooper thinks farts are very funny. Cooper tells jokes, offers friendly advice, and is empathetic. Cooper cares very much about Human's future, and wants her to do well in school.\n",
    }

    if history_nonlocal[0] != system_message:
        history_nonlocal.insert(0, system_message)
    history_nonlocal.append(current_user_message)
    # print(history_nonlocal)
    print("\n")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=history_nonlocal
    )
    print(response)
    out = response["choices"][0]["message"]["content"]
    if "Cooper: " in out:
        out = out.split("Cooper:")[1]
    if "Human: " in out:
        out = out.split("Human:")[1]
    if "User: " in out:
        out = out.split("User:")[1]
    if "You: " in out:
        out = out.split("You:")[1]
    print("Cooper: " + out)
    history_nonlocal.append({"role": "assistant", "content": out})
    after_first = True
    return out


async def talk_medium_with_history(text):
    global model
    global tokenizer
    global named_entities
    global device
    global after_first
    global accelerator
    global after_second
    global history
    print("Using medium model")
    # print("model : " + str(model))
    with torch.cuda.amp.autocast():
        text, early_stop = await classify_input(text, history)
        if early_stop:
            return text

        his = "    ".join(history)
        inputs = await get_list_of_list_of_strings(his)

        print("Summary: ")
        print(inputs[0] + "\n")
        print("important words: ")
        print(inputs[2])
        print("\n")
        print("Names: ")
        print(inputs[1])
        print("\n")
        special_tokens = inputs[1] + inputs[2]
        tokenizer.build_inputs_with_special_tokens(special_tokens)
        print("history" + "\n")

        print(his[-128:] + "\n")
        input_tokens = tokenizer(
            [text, inputs[0]],
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation="only_second",
        ).to(device)
        # text, return_tensors='pt', add_special_tokens=True, padding=True, truncation="only_second").to(device)

        input_tokens = {key: value.to(device) for key, value in input_tokens.items()}
        print(len(input_tokens["input_ids"][0].tolist()))
        history.append(text)
        max_new_tokens = 64

        start_time = time.time()
        with torch.no_grad():
            reply_ids = model.generate(
                **input_tokens,
                min_length=16,
                max_length=64,
                do_sample=True,
                repetition_penalty=1.4,
                no_repeat_ngram_size=2,
                early_stopping=False,
            ).to(device)
            print("time to generate greedy: " + str(time.time() - start_time))

        out = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for g in reply_ids
        ]

        print("Greedy output tokens:" + "\n")
        print(out)
        print("\n")
        after_first = True
        return out[0]


async def talk_medium(text):
    global model
    global tokenizer

    # print(text)
    # print(tokenizer(text, return_tensors='pt'))
    prompt = tokenizer(text, return_tensors="pt")
    prompt = {key: value.to(device) for key, value in prompt.items()}
    out = model.generate(
        **prompt,
        min_length=16,
        max_length=64,
        do_sample=True,
        repetition_penalty=1.4,
        no_repeat_ngram_size=2,
        early_stopping=False,
    )
    # print(out)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    return tokenizer.decode(out[0], skip_special_tokens=True)


async def init_small():
    global model
    global tokenizer
    global device
    print(torch.cuda.is_available())
    print(torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

    model = AutoModelForCausalLM.from_pretrained("facebook/blenderbot_small-90M")

    model.to(device)
    return model, tokenizer


# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


async def talk_small(text):
    global model
    global tokenizer

    # print(text)
    # print(tokenizer(text, return_tensors='pt'))
    prompt = tokenizer(text, return_tensors="pt")
    prompt = {key: value.to(device) for key, value in prompt.items()}
    out = model.generate(
        **prompt,
        min_length=16,
        max_length=32,
        do_sample=True,
        repetition_penalty=1.4,
        no_repeat_ngram_size=2,
        early_stopping=False,
    )
    # print(out)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    # print(out)
    print(tokenizer.decode(out[0]), skip_special_tokens=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)


async def init_fast():
    global model
    global tokenizer
    global device

    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.__version__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-large", padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

    model.to(device)
    return model, tokenizer


# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


async def talk_fast(text):
    global model
    global tokenizer
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(text)
    # print(tokenizer(text, return_tensors='pt'))
    prompt = tokenizer.encode(tokenizer.eos_token + text, return_tensors="pt")
    # prompt = {key: value.to(device) for key, value in prompt.items()}
    print(device)
    prompt = prompt.to(device)
    out = model.generate(
        prompt,
        min_length=16,
        max_length=64,
        do_sample=True,
        top_p=0.95,
        top_k=0,
        temperature=0.75,
        pad_token_id=tokenizer.bos_token_id,
    )
    # print(out)
    print(tokenizer.decode(out[:, prompt.shape[-1] :][0], skip_special_tokens=True))
    return tokenizer.decode(out[:, prompt.shape[-1] :][0], skip_special_tokens=True)


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


visionQA_processor = GitProcessor.from_pretrained("microsoft/git-large-vqav2")

visionQA_model = GitForCausalLM.from_pretrained("microsoft/git-large-vqav2")


async def is_requesting_SD(txt):
    is_vis = False

    result = []
    txt = (
        "Determine if the following question is asking for a drawing, image, or picture to be created, and then reply with True or False only: "
        + txt
    )
    print("txt: ", txt)
    current_user_message = {"role": "user", "content": txt}

    system_message = {
        "role": "system",
        "content": "The assistant's job is to determine whether or not a question or statement is asking for or requesting a drawing, image, or picture to be created. The assistant only answers in one of two single-words True or False.",
    }

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
    for ex in examples:
        print("ex: " + ex + "\n")
        print("examples[ex]: " + examples[ex] + "\n")

        txt.append(
            {
                "role": "user",
                "content": "Determine if the following question is asking for a drawing, image, or picture to be created, and then reply with True or False only:  "
                + ex,
            }
        )
        txt.append({"role": "assistant", "content": " Response: " + examples[ex]})

    print("txt: ", txt)

    # inp = []
    result.append(current_user_message)

    result.insert(0, system_message)
    print("inp: ", result)
    response = openai.ChatCompletion.create(model="curie", messages=result)
    print("response: ", response)
    out = response["choices"][0]["message"]["content"]
    if "Response:" in out:
        out = out.replace("Response:", "")
    # safetext = re.sub(r"[^ .a-zA-Z0-9?\']+", "", out)
    # out = safetext
    print(out)
    if "True" in out:
        is_vis = True
    elif "False" in out:
        is_vis = False
    else:
        is_vis = False

    return is_vis


async def answer_visual_question(question):
    global accelerator
    print(cv.__version__)
    try:
        cam = cv.VideoCapture("http://192.168.0.191:56000/mjpeg")
        await asyncio.sleep(1)

        result, image = cam.read()
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

        cam.release()
        # Destroy all the windows
        cv.destroyAllWindows()
        # pygame.camera.init()
        # camlist = pygame.camera.list_cameras()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print("Device for VQA: " + str(device))
        visionQA_model.to(device)
        # Split the image into patches by reshaping and permuting the dimensions
        # patches = image.unfold(1, patch_size, patch_size).unfold(
        #     2, patch_size, patch_size).to(device)
        # patches = patches.contiguous().view(
        #     1, -1, 3 * patch_size * patch_size).permute(0, 2, 1).to(device)

        # # Add position embeddings to each patch (you can use any learnable embedding layer here)
        # position_embeddings = torch.nn.Embedding(patches.size(1), patches.size(2)).to(
        #     device)
        # patches += position_embeddings.weight.to(device)

        # # Optionally add a class token at the beginning of the sequence of patches
        # class_token = torch.zeros(1, 1, patches.size(2)).to(device)
        # patches = torch.cat([class_token, patches], dim=1).to(device)

        # Check if cuda is available and move the image and model to cuda if yes

        # image = image.to(device)
        if result:
            pass
        else:
            return "No image found"
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


if __name__ == "__main__":
    model = load_all_models("Large")
