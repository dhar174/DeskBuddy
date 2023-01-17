import transformers
import numpy as np
import sklearn


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, BlenderbotForCausalLM, OPTForCausalLM
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoModelForCausalLM, SummarizationPipeline, AutoModelForSeq2SeqLM
import spacy
import dialogue_management as dm
# Define the optimizer
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"
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
# fabiochiu/t5-small-medium-title-generation

# The list of named entities that the model should remember
named_entities = []

# The list of important words or phrases that the model should remember
important_words = []

# A history that the model can use to generate responses
history = [" Emma: Hi! My name is Emma. I love cats and video games!    Picoh: Hello, I am Picoh. I am a chatbot. I can have a friendly chat with you. I can also tell you jokes. What would you like to talk about?"]
# from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
# from .autonotebook import tqdm as notebook_tqdm
# from tqdm.auto import tqdm

lda = LatentDirichletAllocation(n_components=10)

# Define the memory network or graph network
# model = dm.MemoryNetwork(input_size, output_size,
#                          memory_size, memory_vector_dim)
# # or:
# model = dm.GraphNetwork(input_size, output_size, hidden_size,
#                         num_layers, num_relations)

optimizer = torch.optim.Adam(longsummarizer.parameters())
named_entities = []


def find_names(text):
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


sumtokenizer = AutoTokenizer.from_pretrained(
    "philschmid/bart-large-cnn-samsum", use_fast=True, truncation=True)
# sumtokenizer = AutoTokenizer.from_pretrained(
#     "knkarthick/MEETING_SUMMARY", use_fast=True, truncation=True)

# Function to generate a summary of the conversation so far


def generate_summary(text):
    global history
    global longsummarizer
    global device
    print("sum device: "+"cpu")
    print("first prompt summary: "+'\n')
    print(fp_sum)

    # Join the prompts in the prompt history into a single string
    conversation = "\n".join(history)
    # print(conversation)
    # conversation = "summarize: " + text+" " + conversation[:128] + first_prompt
    print(conversation)

    prompt_inputs = sumtokenizer(text, return_tensors="pt",
                                 max_length=128, truncation=True)["input_ids"].to("cpu")
    # Use the summarizer model to generate a summary of the conversation
    prompt_summary = longsummarizer.generate(
        input_ids=prompt_inputs).to("cpu")
    p_sum = sumtokenizer.decode(prompt_summary[0], skip_special_tokens=True)
    print("prompt summary"+'\n')
    print(p_sum)
    history_inputs = sumtokenizer(conversation, return_tensors="pt",
                                  max_length=128, truncation=True)["input_ids"].to("cpu")
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
                              max_length=128, truncation=True)["input_ids"].to("cpu")
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


topic_tokenizer = AutoTokenizer.from_pretrained("knkarthick/TOPIC-DIALOGSUM")

topic_model = AutoModelForSeq2SeqLM.from_pretrained(
    "knkarthick/TOPIC-DIALOGSUM")

# Function to extract the main topics or themes from a summary


def extract_topics(summary):
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


def init():
    global model
    global tokenizer
    global device
    global context
    global history
    global after_first
    global first_prompt
    global fp_sum
    after_first = False

    print(torch.cuda.is_available())
    print(torch.__version__)

    print(device)
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")
    # tokenizer = BlenderbotTokenizer.from_pretrained(
    #     "facebook/blenderbot-3B", padding=True, truncation=True, model_max_length=128)
    # if device == 'cuda':
    #     tokenizer.pad_to_multiple_of = 8
    # model = BlenderbotForConditionalGeneration.from_pretrained(
    #     "facebook/blenderbot-3B", use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "PygmalionAI/pygmalion-6b", padding=True, truncation=True, model_max_length=1024, use_fast=False)
    if device == 'cuda':
        tokenizer.pad_to_multiple_of = 8
    model = AutoModelForCausalLM.from_pretrained(
        "PygmalionAI/pygmalion-6b", use_cache=True, pad_token_id=tokenizer.eos_token_id)

    # first_prompt = "Emma is a third grade girl who is very smart. Her best friend is a friendly and chatty little robot named Picoh.   " + \
    #     "Emma and Picoh love to chat about a lot of different things together, like video games, the Legend of Zelda, cats, and school.   " + \
    #     "Picoh is a robot, but he is not a very smart robot. Emma and Picoh the robot are best friends.    "
    # first_prompt = "The following is a conversation with an AI assistant named Picoh. The assistant is helpful, creative, clever, and very friendly. Emma: Hello, who are you? Picoh: I am an AI assistant and friendly chatbot. How can I help you today? Emma: I was just wondering, what do you know about the Legend of Zelda: Breath of the Wild? Picoh: Yes, it's the newest game in the Zelda series and an epic action-adventure game! Emma: Yea, it's my favorite! Anyway, so, you're my new robot assistant. Picoh: "
    first_prompt = "Picoh's Persona: Picoh is a sentient robot with his own thoughts and feelings. Picohs best friend is Emma, and Picohs goal in life is to be Emma's friend and to be helpful to her. Picoh is smart, friendly, and loves to make people laugh. Picoh knows a lot about video games, Nintendo, and especially the Legend of Zelda. Picoh also loves cats. icoh thinks farts are very funny. Picoh tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Picoh cares very much about Emma's future, and wants her to do well in school.Picoh is a sentient robot with his own thoughts and feelings. Picohs best friend is Emma, and Picohs goal in life is to be Emma's friend and to be helpful to her. Picoh is smart, friendly, and loves to make people laugh. Picoh knows a lot about video games, Nintendo, and especially the Legend of Zelda. Picoh also loves cats. icoh thinks farts are very funny. Picoh tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Picoh cares very much about Emma's future, and wants her to do well in school." + \
        "<START >" + \
        "Emma: Hello, Picoh! How was your day today?" + \
        "Picoh: Hi! My day was pretty good. How was yours?" + \
        "Emma: I had a good day. I played some video games in the afternoon." + \
        "Picoh: That sounds fun! What games did you play?" + \
        "Picoh: Thanks, I'll definitely keep that in mind. How is your schoolwork going?" + \
        "Emma: Hey, have you played any of the Legend of Zelda games?" + \
        "Picoh: Yes, I've played a few of them! Ocarina of Time and Breath of the Wild are two of my favorites." + \
        "Emma: I loved Ocarina of Time too! What did you think of Breath of the Wild?" + \
        "Picoh: I thought Breath of the Wild was amazing! The open-world exploration and combat mechanics were really fun, and the story was really engaging." + \
        "Emma: I've heard great things about Breath of the Wild. Do you have a favorite character from the series?" + \
        "Picoh: It's tough to choose, but I think Link is my favorite. He's such a brave and determined hero." + \
        "Emma: I agree, Link is definitely a classic character. Have you played any of the newer Legend of Zelda games?" + \
        "Picoh: Yes, I've played Hyrule Warriors and Links Awakening as well. Both were really enjoyable in their own way." + \
        "Emma: I haven't played Hyrule Warriors or Links Awakening, but I'll have to check them out. Thanks for the recommendation!" +\
        "Emma: I'm so happy you are my robot friend!" + \
        "Picoh: I'm happy to be your friend too! I hope we can be friends for a long time." + \
        "Emma: Me too! I love you, Picoh!" + \
        "Picoh: I love you too, Emma!" + \
        "Emma: What do you want to talk about next?"

    fp_inputs = sumtokenizer(first_prompt, return_tensors="pt",
                             max_length=512, truncation=True)["input_ids"].to("cpu")
    # Use the summarizer model to generate a summary of the conversation
    fp_summary = longsummarizer.generate(
        input_ids=fp_inputs).to("cpu")
    fp_sum = sumtokenizer.decode(fp_summary[0], skip_special_tokens=True)
    print("fp_sum")
    print(fp_sum+"\n")
    model.to(device)
    # print(first_prompt)
    talk_history(first_prompt)


# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


def talk(text):
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


def get_list_of_list_of_strings(text):
    global history
    global named_entities
    names = find_names(text)
    [names.append(x) for x in named_entities if x not in names]
    named_entities = names
    # Generate a summary of the conversation so far and extract the main topics or themes
    summary = generate_summary(text)
    topics, main_topic = extract_topics(summary)
    print("Main topic: ", main_topic)
    print("\n")
    # Return the prompt history, named entities, and important words
    return [summary, named_entities, topics]


def talk_history(text):
    global model
    global tokenizer
    global history
    global named_entities
    global device
    global after_first
    if after_first:
        # text = "Emma: " + text + " Picoh: "
        text = "Picoh's Persona: Picoh is a sentient robot with his own thoughts and feelings. Picohs best friend is Emma, and Picohs goal in life is to be Emma's friend and to be helpful to her. Picoh is smart, friendly, and loves to make people laugh. Picoh knows a lot about video games, Nintendo, and especially the Legend of Zelda. Picoh also loves cats. icoh thinks farts are very funny. Picoh tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Picoh cares very much about Emma's future, and wants her to do well in school.Picoh is a sentient robot with his own thoughts and feelings. Picohs best friend is Emma, and Picohs goal in life is to be Emma's friend and to be helpful to her. Picoh is smart, friendly, and loves to make people laugh. Picoh knows a lot about video games, Nintendo, and especially the Legend of Zelda. Picoh also loves cats. icoh thinks farts are very funny. Picoh tells jokes, offers friendly advice, and understands what it is like to be in the shoes of a third grader. Picoh cares very much about Emma's future, and wants her to do well in school.\n" + \
               "<START >\n" + \
               "Emma: " + text + "\n Picoh: "
    inputs = get_list_of_list_of_strings(text)
    if after_first:

        print("Summary: ")
        print(inputs[0]+"\n")
        print("important words: ")
        print(inputs[2])
        print("\n")
        print("Names: ")
        print(inputs[1])
        print("\n")
        # history.pop(0)

    # inputs = ""
    # if len(history) == 0:
    #     inputs = text
    # for i in range(len(history)):
    #     inputs = '    '.join(inputs+history[i])
    # print(history)
    # print(inputs)
    # print(type(history))
    # for i, dialogue in enumerate(history):
    #     # Loop over the utterances in the dialogue
    #     for j, utterance in enumerate(dialogue):
    #         # Convert the utterance to a tensor
    #         inputs = torch.tensor(utterance)

    #         # If this is the first utterance in the dialogue, initialize the memory or graph
    #         if j == 0:
    #             edges = torch.zeros(0, 2)
    #             nodes = torch.zeros(0, input_size)
    #          # Use the model to generate a response
    #         response = model(inputs, edges, nodes)
    #         history[i][j + 1] = response

    if after_first:
        special_tokens = inputs[1] + inputs[2]
        tokenizer.build_inputs_with_special_tokens(special_tokens)
        print("history"+"\n")
        # print(history)

        his = "    ".join(history)
        print(his[:128]+"\n")
        input_tokens = tokenizer(
            [text, inputs[0]], return_tensors='pt', add_special_tokens=True, padding=True, truncation="only_second").to(device)

    else:
        input_tokens = tokenizer(
            text, return_tensors='pt', add_special_tokens=True, truncation=True).to(device)

    input_tokens = {key: value.to(device)
                    for key, value in input_tokens.items()}
    history.append(text)
    for i in range(len(input_tokens['input_ids'])):
        print(tokenizer.decode(
            input_tokens['input_ids'][i], skip_special_tokens=True)+"\n")

    reply_ids = model.generate(**input_tokens, min_length=16, max_length=1024, do_sample=True,
                               repetition_penalty=1.1, no_repeat_ngram_size=2, early_stopping=True).to(device)
    out = [tokenizer.decode(g, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False) for g in reply_ids]
    # print(text)
    # print(tokenizer(text, return_tensors='pt'))
    # prompt = tokenizer(text, return_tensors='pt')
    # prompt = {key: value.to(device) for key, value in prompt.items()}
    # out = model.generate(**prompt, min_length=16, max_length=64, do_sample=True,
    #                      repetition_penalty=1.4, no_repeat_ngram_size=2, early_stopping=False)
    print(out)
    print("\n")
    history.append("Picoh: "+out[0])
    print(out[0])
    print("\n")
    after_first = True
    return out[0]


def init_medium():
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
# config = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill")


def talk_medium(text):
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