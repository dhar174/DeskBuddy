# from torchdistx.fake import fake_mode
# from optimum.onnxruntime import OrtModelForCausalLM
# from soft_embedding import SoftEmbedding
from transformers.models.gptj.modeling_gptj import GPTJBlock
from deepspeed import OnDevice
import deepspeed
from transformers.deepspeed import is_deepspeed_zero3_enabled,  is_accelerate_available, is_deepspeed_available
from accelerate import Accelerator, DistributedType
from accelerate import load_checkpoint_and_dispatch, dispatch_model, dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM, CONFIG_MAPPING,  MODEL_MAPPING
from accelerate import infer_auto_device_map, init_empty_weights
import bitsandbytes as bnb
import dialogue_management as dm
import spacy
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoModelForCausalLM, SummarizationPipeline, AutoModelForSeq2SeqLM, AutoModel, GPTJForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BlenderbotForCausalLM, OPTForCausalLM
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


os.environ["TOKENIZERS_PARALLELISM"] = "false"

deepspeed.init_distributed("nccl")

# The list of named entities that the model should remember
named_entities = []

# The list of important words or phrases that the model should remember
important_words = []

# A history that the model can use to generate responses
history = [" Emma: Hi! My name is Emma. I love cats and video games!",
           "Cid: Hello, I am Cid. I am a chatbot. I can have a friendly chat with you. I can also tell you jokes. What would you like to talk about?"]


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


async def load_all_models(button_text):
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
    print("Loading models...")
    print("button_text: ", button_text)
    start_time = time.time()
    topic_tokenizer = AutoTokenizer.from_pretrained(
        "knkarthick/TOPIC-DIALOGSUM")

    topic_model = AutoModelForSeq2SeqLM.from_pretrained(
        "knkarthick/TOPIC-DIALOGSUM")

    sumtokenizer = AutoTokenizer.from_pretrained(
        "philschmid/bart-large-cnn-samsum", use_fast=True, truncation=True)

    nlp = spacy.load("en_core_web_trf")
    longsummarizer = AutoModelForSeq2SeqLM.from_pretrained(
        "philschmid/bart-large-cnn-samsum")
    shortsummarizer = AutoModelForSeq2SeqLM.from_pretrained(
        "knkarthick/meeting-summary-samsum")
    shortsummarizer.to('cpu')
    longsummarizer.to('cpu')
    optimizer = torch.optim.Adam(longsummarizer.parameters())

    titlesummarizer = AutoModelForSeq2SeqLM.from_pretrained(
        "fabiochiu/t5-small-medium-title-generation")
    titlesummarizer.to('cpu')

    isquestion_tokenizer = AutoTokenizer.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier")

    isquestion_model = AutoModelForSequenceClassification.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier")

    nli_classifier = pipeline(
        model="valhalla/distilbart-mnli-12-3", tokenizer="valhalla/distilbart-mnli-12-3", task="zero-shot-classification")

    command_tokenizer = AutoTokenizer.from_pretrained(
        "gokuls/bert-tiny-Massive-intent-KD-BERT")

    command_classifier = AutoModelForSequenceClassification.from_pretrained(
        "gokuls/bert-tiny-Massive-intent-KD-BERT")
    nlp = spacy.load("en_core_web_trf")
    longsummarizer = AutoModelForSeq2SeqLM.from_pretrained(
        "philschmid/bart-large-cnn-samsum")
    shortsummarizer = AutoModelForSeq2SeqLM.from_pretrained(
        "knkarthick/meeting-summary-samsum")
    shortsummarizer.to('cpu')
    longsummarizer.to('cpu')
    titlesummarizer = AutoModelForSeq2SeqLM.from_pretrained(
        "fabiochiu/t5-small-medium-title-generation")
    titlesummarizer.to('cpu')
    lda = LatentDirichletAllocation(n_components=10)
    if (button_text == "Fast"):
        model, tokenizer = await init_fast()
    elif (button_text == "Small"):
        model, tokenizer = await init_small()
    elif (button_text == "Medium"):
        model, tokenizer = await init_medium()
    elif (button_text == "Large"):
        model, tokenizer = await init()

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
    global longsummarizer
    global device
    print("sum device: "+"cpu")
    # print("first prompt summary: "+'\n')
    # print(fp_sum)

    # Join the prompts in the prompt history into a single string
    conversation = "\n".join(history)
    # print(conversation)
    # conversation = "summarize: " + text+" " + conversation[:128] + first_prompt
    print(conversation)

    prompt_inputs = sumtokenizer(text, return_tensors="pt",
                                 max_length=512, truncation=True)["input_ids"].to("cpu")
    # Use the summarizer model to generate a summary of the conversation
    prompt_summary = longsummarizer.generate(
        input_ids=prompt_inputs).to("cpu")
    p_sum = sumtokenizer.decode(prompt_summary[0], skip_special_tokens=True)
    print("prompt summary"+'\n')
    print(p_sum)
    history_inputs = sumtokenizer(conversation, return_tensors="pt",
                                  max_length=512, truncation=True)["input_ids"].to("cpu")
    # Use the summarizer model to generate a summary of the conversation
    history_summary = longsummarizer.generate(
        input_ids=history_inputs).to("cpu")
    h_sum = sumtokenizer.decode(history_summary[0], skip_special_tokens=True)
    print("history summary"+'\n')
    print(h_sum)
    print("\n")

    full_inputs = p_sum + "  "+fp_sum+"  " + h_sum
    print("full inputs")
    print(full_inputs)
    sum_inputs = sumtokenizer(full_inputs, return_tensors="pt",
                              max_length=512, truncation=True)["input_ids"].to("cpu")
    # Use the summarizer model to generate a summary of the conversation
    summary = longsummarizer.generate(
        input_ids=sum_inputs).to("cpu")
    sum = sumtokenizer.decode(summary[0], skip_special_tokens=True)
    print("combined summary"+'\n')
    # print(summary)
    print(sum)
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
        topic_words = [vectorizer.get_feature_names_out()[i]
                       for i in topic.argsort()[:-5 - 1:-1]]
        topics.append(topic_words)
    res = []
    for topic in topics:
        for word in topic:
            res.append(word)
    [res.append(x) for x in res if x not in res]
    topics = res
    topic_inputs = topic_tokenizer(summary, return_tensors="pt", max_length=128,
                                   truncation=True)["input_ids"].to("cpu")
    final_topic = topic_model.generate(input_ids=topic_inputs).to("cpu")
    final_topic = topic_tokenizer.decode(
        final_topic[0], skip_special_tokens=True)
    # Return the list of topics or themes
    return topics, final_topic


first_prompt = ""
fp_sum = ""


async def set_deepspeed_activation_checkpointing(deepspeed_config):

    deepspeed.checkpointing.configure(
        None, deepspeed_config=deepspeed_config, partition_activations=True)

    deepspeed.checkpointing.partition_activations = True
    deepspeed.checkpointing.cpu_checkpointing = True
    deepspeed.checkpointing.checkpoint_activations = True
    deepspeed.checkpointing.synchronize_checkpoint_boundary = True
    deepspeed.checkpointing.contiguous_memory_optimization = True


def softprompt_inputs(text):
    n_tokens = 20
    initialize_from_vocab = True


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
    ds_config = await dsconfig()
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
            "PygmalionAI/pygmalion-350m", use_cache=True, offload_state_dict=True, output_attentions=True, output_hidden_states=True,  low_cpu_mem_usage=False, torch_dtype="auto", offload_folder="/home/darf3/buddy/offload").to(device)
    print("model loaded")
    accelerator.load_state("checkpoint")
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    tokenizer = AutoTokenizer.from_pretrained(
        "PygmalionAI/pygmalion-350m",  use_fast=False, return_tensors="pt", padding_side='left')
    accelerator.register_for_checkpointing(model)
    accelerator.print("model registered for checkpointing successfully")

    accelerator.prepare(model, tokenizer)
    accelerator.print("model prepared successfully")
    model.to(device)
    accelerator.print(accelerator.state)
    model = dispatch_model(
        model,  device_map=infer_auto_device_map(model, no_split_module_classes=["GPTJBlock"], dtype=torch.float16, max_memory={0: "7GiB",  "cpu": "48GiB"}), offload_dir="/home/darf3/buddy/offload", offload_buffers=True)
    accelerator.print("model loaded and dispatched successfully")
    model.eval()
    accelerator.print("model set to eval successfully")
    # accelerator.
    model = model.to(device)
    accelerator.print(accelerator.state)
    first_prompt = "Cid's Persona: Cid is a sentient AI robot with the heart of a kid. Cid's best friend is Emma, and Cid's goal in life is to be Emma's friend and to be helpful to her. Cid is smart, friendly, always kid appropriate and loves to make people laugh. Cid knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cid also loves cats. Cid thinks farts are very funny. Cid tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Cid cares very much about Emma's future, and wants her to do well in school." + "\n" + \
        "<START >" + "\n" + \
        "You: Hello, Cid! How was your day today? " + "\n" + \
        "Cid: Hi! My day was pretty good. How was yours? " + "\n" + \
        "You: I had a good day. I played some video games in the afternoon. " + "\n" + \
        "Cid: That sounds fun! What games did you play? " + \
        "Cid: Thanks, I'll definitely keep that in mind. How is your schoolwork going? " + "\n" + \
        "You: Hey, have you played any of the Legend of Zelda games? " + \
        "Cid: Yes, I've played a few of them! Ocarina of Time and Breath of the Wild are two of my favorites. " + "\n" + \
        "You: I loved Ocarina of Time too! What did you think of Breath of the Wild? " + \
        "Cid: I thought Breath of the Wild was amazing! The open-world exploration and combat mechanics were really fun, and the story was really engaging. " + "\n" + \
        "You: I've heard great things about Breath of the Wild. Do you have a favorite character from the series? " + "\n" + \
        "Cid: It's tough to choose, but I think Link is my favorite. He's such a brave and determined hero. " + "\n" + \
        "You: I agree, Link is definitely a classic character. Have you played any of the newer Legend of Zelda games? " + "\n" + \
        "Cid: Yes, I've played Hyrule Warriors and Links Awakening as well. Both were really enjoyable in their own way. " + "\n" + \
        "You: I haven't played Hyrule Warriors or Links Awakening, but I'll have to check them out. Thanks for the recommendation! " + "\n" +\
        "Cid: I'm happy to be your friends! I hope have fun talking about kid stuff. " + "\n" + \
        "You: Me too! What do you want to talk about next?" + "\n"

    fp_inputs = sumtokenizer(first_prompt, return_tensors="pt",
                             max_length=512, truncation=True)["input_ids"]
    fp_summary = longsummarizer.generate(
        input_ids=fp_inputs).to("cpu")
    fp_sum = sumtokenizer.decode(fp_summary[0], skip_special_tokens=True)
    print("fp_sum")
    print(fp_sum+"\n")
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.to(device)

    talk_history(first_prompt)
    print("time to init medium: "+str(time.time()-start_time))

    return model, tokenizer


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
    ds_config = await dsconfig()
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
            "./checkpoint", use_cache=True, offload_state_dict=True, output_attentions=True, output_hidden_states=True,  low_cpu_mem_usage=False, torch_dtype="auto", offload_folder="/home/darf3/buddy/offload").to(device)
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
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    # model.save_pretrained("checkpoint", max_shard_size="200MB")
    # new_model = AutoModel.from_pretrained("checkpoint")
    # accelerator.print("new model loaded successfully")
    # accelerator.print(new_model)
    tokenizer = AutoTokenizer.from_pretrained(
        "./pygmalion-6b",  use_fast=False, return_tensors="pt", padding_side='left')

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
        model,  device_map=infer_auto_device_map(model, no_split_module_classes=["GPTJBlock"], dtype=torch.float16, max_memory={0: "7GiB",  "cpu": "48GiB"}), offload_dir="/home/darf3/buddy/offload", offload_buffers=True)
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
        f'before deepspeed: {(torch.cuda.memory_allocated()/1000)}, {(torch.cuda.memory_reserved()/1000)}'
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

    # first_prompt = "Emma is a third grade girl who is very smart. Her best friend is a friendly and chatty little robot named Cid.   " + \
    #     "Emma and Cid love to chat about a lot of different things together, like video games, the Legend of Zelda, cats, and school.   " + \
    #     "Cid is a robot, but he is not a very smart robot. Emma and Cid the robot are best friends.    "
    # first_prompt = "The following is a conversation with an AI assistant named Cid. The assistant is helpful, creative, clever, and very friendly. Emma: Hello, who are you? Cid: I am an AI assistant and friendly chatbot. How can I help you today? Emma: I was just wondering, what do you know about the Legend of Zelda: Breath of the Wild? Cid: Yes, it's the newest game in the Zelda series and an epic action-adventure game! Emma: Yea, it's my favorite! Anyway, so, you're my new robot assistant. Cid: "
    first_prompt = "Cid's Persona: Cid is a sentient AI robot with the heart of a kid. Cid's best friend is Emma, and Cid's goal in life is to be Emma's friend and to be helpful to her. Cid is smart, friendly, always kid appropriate and loves to make people laugh. Cid knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cid also loves cats. Cid thinks farts are very funny. Cid tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Cid cares very much about Emma's future, and wants her to do well in school." + "\n" + \
        "<START >" + "\n" + \
        "You: Hello, Cid! How was your day today? " + "\n" + \
        "Cid: Hi! My day was pretty good. How was yours? " + "\n" + \
        "You: I had a good day. I played some video games in the afternoon. " + "\n" + \
        "Cid: That sounds fun! What games did you play? " + \
        "Cid: Thanks, I'll definitely keep that in mind. How is your schoolwork going? " + "\n" + \
        "You: Hey, have you played any of the Legend of Zelda games? " + \
        "Cid: Yes, I've played a few of them! Ocarina of Time and Breath of the Wild are two of my favorites. " + "\n" + \
        "You: I loved Ocarina of Time too! What did you think of Breath of the Wild? " + \
        "Cid: I thought Breath of the Wild was amazing! The open-world exploration and combat mechanics were really fun, and the story was really engaging. " + "\n" + \
        "You: I've heard great things about Breath of the Wild. Do you have a favorite character from the series? " + "\n" + \
        "Cid: It's tough to choose, but I think Link is my favorite. He's such a brave and determined hero. " + "\n" + \
        "You: I agree, Link is definitely a classic character. Have you played any of the newer Legend of Zelda games? " + "\n" + \
        "Cid: Yes, I've played Hyrule Warriors and Links Awakening as well. Both were really enjoyable in their own way. " + "\n" + \
        "You: I haven't played Hyrule Warriors or Links Awakening, but I'll have to check them out. Thanks for the recommendation! " + "\n" +\
        "Cid: I'm happy to be your friends! I hope have fun talking about kid stuff. " + "\n" + \
        "You: Me too! What do you want to talk about next?" + "\n"

    fp_inputs = sumtokenizer(first_prompt, return_tensors="pt",
                             max_length=512, truncation=True)["input_ids"]
    # Use the summarizer model to generate a summary of the conversation
    fp_summary = longsummarizer.generate(
        input_ids=fp_inputs).to("cpu")
    fp_sum = sumtokenizer.decode(fp_summary[0], skip_special_tokens=True)
    print("fp_sum")
    print(fp_sum+"\n")
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
    talk_history(first_prompt)
    print("time to init large: "+str(time.time()-start_time))

    return model, tokenizer


# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


async def talk(text):
    global model
    global tokenizer
    global history
    global device

    prompt = tokenizer(text, return_tensors='pt')
    prompt = {key: value.to(device) for key, value in prompt.items()}
    out = model.generate(**prompt, min_length=16, max_length=64, do_sample=True,
                         repetition_penalty=1.4, no_repeat_ngram_size=2, early_stopping=False)

    print(tokenizer.decode(out[0], skip_special_tokens=True))
    return tokenizer.decode(out[0], skip_special_tokens=True)


async def get_list_of_list_of_strings(text):
    global history
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
    print("time to NER and all summaries: "+str(time.time()-start_time))

    # Return the prompt history, named entities, and important words
    return [summary, names, topics]


after_second = False
count_index = 0


async def classify_input(text):

    if (await helpers.needs_follow_up(text)):
        print("NEEDS FOLLOW UP \n")
        start_time = time.time()
        question = await helpers.get_follow_up_question(text)
        print("time to generate followup question: "+str(time.time()-start_time))

        if (question == ""):
            return text, False
        return question, False
    if (await helpers.is_question(text)):
        print("IS QUESTION")
        if (await helpers.is_visual_question(text)):
            print("IS VISUAL QUESTION \n")
            # try:
            start_time = time.time()
            answer = await helpers.answer_visual_question(text)
            print("time to answer visual question: " +
                  str(time.time()-start_time))

            print("Label of visual question: ",
                  answer)
            print("\n")
            return answer, True
        if (await helpers.is_personal(text)):
            print("IS PERSONAL QUESTION \n")
            return text, False
        start_time = time.time()
        answer, found = await helpers.query_memory(text)
        print("time to search query memory: "+str(time.time()-start_time))

        if (found):
            print("IS FOUND IN MEMORY OF QUESTIONS \n")

            print("Label of memory question: ", answer)
            print("\n")
            return answer, True
        else:
            pass

    else:
        print("NOT QUESTION")
        if (await helpers.is_chat_or_command(text)):
            print("IS COMMAND \n")
            try:
                # print("Label of command: ", await helpers.command_type(text))
                print("\n")

                # TODO: Add command fulfillment here
                return "COMMAND PLACEHOLDER", True
            except:
                print("EXCEPTION in Command Detection \n")
        else:
            print("IS CHAT OR OTHER \n")
            return text, False


async def save_history_verbatim(text):
    with open('history_verbatim.txt', 'w') as file:
        file.write(text)


async def save_history_summary(text):
    with open('history_summary.txt', 'a') as file:
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

        text, early_stop = await classify_input(text)
        if early_stop:
            return text

        if after_first and not after_second:
            # text = "Emma: " + text + " Cid: "
            # text = "Cid's Persona: Cid is a sentient robot with his own thoughts and feelings. Cid's best friend is Emma, and Cid's goal in life is to be Emma's friend and to be helpful to her. Cid is smart, friendly, and loves to make people laugh. Cid knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cid also loves cats. Cid thinks farts are very funny. Cid tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Cid cares very much about Emma's future, and wants her to do well in school.Cid is a sentient robot with his own thoughts and feelings. Cid's best friend is Emma, and Cid's goal in life is to be Emma's friend and to be helpful to her. Cid is smart, friendly, and loves to make people laugh. Cid knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cid also loves cats. icoh thinks farts are very funny. Cid tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Cid cares very much about Emma's future, and wants her to do well in school.\n" + \
            text = "Cid's Persona: Cid is a sentient AI robot with his own thoughts and perspectives. Cid's best friend is Emma, and Cid's goal is to be Emma's friend and to be helpful to her. Cid is smart, friendly, and loves to make people laugh. Cid knows a lot about video games, Nintendo, and especially the Legend of Zelda. Cid also loves cats. Cid thinks farts are very funny. Cid tells jokes, offers friendly advice, and is empathetic. Cid cares very much about Emma's future, and wants her to do well in school.\n" + \
                "<START >\n" + \
                text + "\n Cid: "
            after_second = True
        elif after_first and after_second:
            text = "Cid's Persona: Cid is Emma's chatbot, and Cid's goal is to be Emma's friend and to be helpful to her. Cid is smart, friendly, and loves to make people laugh. Cid tells jokes, offers friendly empathetic advice, and chats about Nintendo games. \n" + \
                "<START >\n" + \
                "You: " + text + " \n" + \
                "Cid: "
        his = "    ".join(history)
        inputs = await get_list_of_list_of_strings(his)
        if after_first:

            print("Summary: ")
            print(inputs[0]+"\n")
            if (count_index == 10):
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
            print("history"+"\n")

            print(his[-128:]+"\n")
            input_tokens = tokenizer(
                [text, inputs[0]], return_tensors='pt', add_special_tokens=True, padding=True, truncation="only_second").to(device)
            # text, return_tensors='pt', add_special_tokens=True, padding=True, truncation="only_second").to(device)

        else:
            input_tokens = tokenizer(
                text, return_tensors='pt', add_special_tokens=True, truncation=True).to(device)

        input_tokens = {key: value.to(device)
                        for key, value in input_tokens.items()}
        print(len(input_tokens['input_ids'][0].tolist()))
        history.append(text)
        print("input tokens"+"\n")
        max_new_tokens = 512
        for i in range(len(input_tokens['input_ids'])):
            print(tokenizer.decode(
                input_tokens['input_ids'][i], skip_special_tokens=True)+"\n")
        if (len(input_tokens['input_ids'][0].tolist()) > 64):
            if (len(input_tokens['input_ids'][0].tolist()) < 128):
                max_new_tokens = 128
            else:
                max_new_tokens = len(input_tokens['input_ids'][0].tolist())+8
        elif (len(input_tokens['input_ids'][0].tolist()) < 64):
            if (len(input_tokens['input_ids'][0].tolist()) < 32):
                max_new_tokens = 32
            else:
                max_new_tokens = 64
        print("max new tokens: "+str(max_new_tokens))
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
            reply_ids = model.generate(**input_tokens, min_new_tokens=8, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                       no_repeat_ngram_size=2, early_stopping=True, synced_gpus=True).to(device)
        print("time to generate greedy: "+str(time.time()-start_time))
        out = [tokenizer.decode(g, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False) for g in reply_ids]

        print("Greedy output tokens:"+"\n")
        print(out)
        print("\n")
        for i in range(len(out)):
            print(i)

            for j in range(len(input_tokens['input_ids'])):
                remove = ""
                remove = remove.join(tokenizer.decode(
                    input_tokens['input_ids'][j], skip_special_tokens=True))
                out[i] = out[i].replace(remove, "")
            sep = '\n'
            if (out[i][0] == '\n'):
                sep = '   '
            stripped_out = out[i].split(sep, 1)[0]
            colon = stripped_out.find(':')
            if (colon != -1):
                name = stripped_out[:colon-1]
                print(name)
                if (name.lower() == "you:"):
                    stripped_out = stripped_out[colon+1:]
                elif (name in inputs[1]):
                    stripped_out = stripped_out[colon+1:]
            print("Output with name cut out:" + stripped_out+"\n")
            # sentences = re.split('. |! |? |\n', stripped_out)
            # final = sentences[0].join(sentences[1])
            # print("Output with extra sentences cut out:" + final+"\n")
            print("\n")

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
        # history.append("Cid: "+out[0])
        # print(out[0])
        # print("\n")
        after_first = True
        if (count_index == 10):
            save_history_verbatim(history)
            count_index = 0
    return stripped_out


async def init_medium():
    global model
    global tokenizer
    global device
    print(torch.cuda.is_available())
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = BlenderbotTokenizer.from_pretrained(
        "facebook/blenderbot-400M-distill")

    model = BlenderbotForConditionalGeneration.from_pretrained(
        "facebook/blenderbot-400M-distill")

    model.to(device)
    return model, tokenizer
# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


async def talk_medium_with_history(text):
    global model
    global tokenizer
    global named_entities
    global device
    global after_first
    global accelerator
    global after_second
    global history
    with torch.cuda.amp.autocast():

        text, early_stop = await classify_input(text)
        if early_stop:
            return text

        his = "    ".join(history)
        inputs = await get_list_of_list_of_strings(his)

        print("Summary: ")
        print(inputs[0]+"\n")
        print("important words: ")
        print(inputs[2])
        print("\n")
        print("Names: ")
        print(inputs[1])
        print("\n")
        special_tokens = inputs[1] + inputs[2]
        tokenizer.build_inputs_with_special_tokens(special_tokens)
        print("history"+"\n")

        print(his[-128:]+"\n")
        input_tokens = tokenizer(
            [text, inputs[0]], return_tensors='pt', add_special_tokens=True, padding=True, truncation="only_second").to(device)
        # text, return_tensors='pt', add_special_tokens=True, padding=True, truncation="only_second").to(device)

        input_tokens = {key: value.to(device)
                        for key, value in input_tokens.items()}
        print(len(input_tokens['input_ids'][0].tolist()))
        history.append(text)
        max_new_tokens = 64

        start_time = time.time()
        with torch.no_grad():
            reply_ids = model.generate(**input_tokens, min_length=16, max_length=64, do_sample=True,
                                       repetition_penalty=1.4, no_repeat_ngram_size=2, early_stopping=False).to(device)
            print("time to generate greedy: "+str(time.time()-start_time))

        out = [tokenizer.decode(g, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False) for g in reply_ids]

        print("Greedy output tokens:"+"\n")
        print(out)
        print("\n")
        after_first = True
        return out[0]


async def talk_medium(text):
    global model
    global tokenizer

    # print(text)
    # print(tokenizer(text, return_tensors='pt'))
    prompt = tokenizer(text, return_tensors='pt')
    prompt = {key: value.to(device) for key, value in prompt.items()}
    out = model.generate(**prompt, min_length=16, max_length=64, do_sample=True,
                         repetition_penalty=1.4, no_repeat_ngram_size=2, early_stopping=False)
    # print(out)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    return tokenizer.decode(out[0], skip_special_tokens=True)


def init_small():
    global model
    global tokenizer
    global device
    print(torch.cuda.is_available())
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/blenderbot_small-90M")

    model = AutoModelForCausalLM.from_pretrained(
        "facebook/blenderbot_small-90M")

    model.to(device)
    return model, tokenizer
# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


def talk_small(text):
    global model
    global tokenizer

    # print(text)
    # print(tokenizer(text, return_tensors='pt'))
    prompt = tokenizer(text, return_tensors='pt')
    prompt = {key: value.to(device) for key, value in prompt.items()}
    out = model.generate(**prompt, min_length=16, max_length=32, do_sample=True,
                         repetition_penalty=1.4, no_repeat_ngram_size=2, early_stopping=False)
    # print(out)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    # print(out)
    print(tokenizer.decode(out[0]), skip_special_tokens=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def init_fast():
    global model
    global tokenizer
    global device

    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.__version__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-large", padding_side='left')

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-large")

    model.to(device)
    return model, tokenizer
# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


def talk_fast(text):
    global model
    global tokenizer
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(text)
    # print(tokenizer(text, return_tensors='pt'))
    prompt = tokenizer.encode(
        tokenizer.eos_token+text, return_tensors="pt")
    # prompt = {key: value.to(device) for key, value in prompt.items()}
    print(device)
    prompt = prompt.to(device)
    out = model.generate(prompt, min_length=16, max_length=64, do_sample=True,
                         top_p=0.95,
                         top_k=0,
                         temperature=0.75,
                         pad_token_id=tokenizer.bos_token_id)
    # print(out)
    print(tokenizer.decode(out[:, prompt.shape[-1]:]
          [0], skip_special_tokens=True))
    return tokenizer.decode(out[:, prompt.shape[-1]:][0], skip_special_tokens=True)


async def dsconfig(fr):
    ds_config = ""
    if (fr == "Large"):
        ds_config = {
            "sparse_attention": {
                "mode": "fixed",
                "block": 16,
                "different_layout_per_head": True,
                "num_local_blocks": 4,
                "num_global_blocks": 1,
                "attention": "bidirectional",
                "horizontal_global_attention": False,
                "num_different_global_patterns": 4
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "amp": {
                "enabled": True,
                "opt_level": "auto"
            },
            "bf16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
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
    elif (fr == "Medium"):
        ds_config = {
            "sparse_attention": {
                "mode": "fixed",
                "block": 16,
                "different_layout_per_head": True,
                "num_local_blocks": 4,
                "num_global_blocks": 1,
                "attention": "bidirectional",
                "horizontal_global_attention": False,
                "num_different_global_patterns": 4
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "amp": {
                "enabled": True,
                "opt_level": "auto"
            },
            "bf16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
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


if __name__ == "__main__":
    model = load_all_models("Large")
