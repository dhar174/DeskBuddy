from transformers import pipeline
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import cv2 as cv
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
import json
import BotCortex
import torchvision.transforms as T

# from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import (
    GitVisionConfig,
    GitForCausalLM,
    GitProcessor,
    AutoModelForQuestionAnswering,
    DistilBertModel,
    DistilBertTokenizer,
    AutoModelForSeq2SeqLM,
)
import asyncio

accelerator = BotCortex.Accelerator
# Define the arguments to the function
question_answerer = None
visionQA_processor = None

visionQA_model = None

command_model = None
command_tokenizer = None
search_tokenizer = None
search_model = None
qgen_tokenizer = None
qgen_model = None


async def load_models():
    global question_answerer
    global visionQA_processor
    global visionQA_model
    global command_model
    global command_tokenizer
    global search_tokenizer
    global search_model
    # global qgen_tokenizer
    # global qgen_model
    global accelerator

    print("loading helper models")
    question_answerer = pipeline(
        "question-answering", model="distilbert-base-cased-distilled-squad"
    )
    # visionQA_processor = GitProcessor.from_pretrained(
    #     "microsoft/git-large-vqav2")

    # visionQA_model = GitForCausalLM.from_pretrained(
    #     "microsoft/git-large-vqav2")

    command_model = AutoModelForSequenceClassification.from_pretrained(
        "gokuls/bert-tiny-Massive-intent-KD-BERT"
    )
    command_tokenizer = AutoTokenizer.from_pretrained(
        "gokuls/bert-tiny-Massive-intent-KD-BERT"
    )
    # search_tokenizer = AutoTokenizer.from_pretrained(
    #     "sentence-transformers/all-mpnet-base-v2"
    # )
    # search_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    # print("Pad token ID:", search_tokenizer.pad_token_id)
    # print("Vocab size:", search_tokenizer.vocab_size)

    # qgen_tokenizer = AutoTokenizer.from_pretrained(
    #     "voidful/context-only-question-generator"
    # )

    # qgen_model = AutoModelForSeq2SeqLM.from_pretrained(
    #     "voidful/context-only-question-generator"
    # )
    return


async def is_question(txt):
    is_q = False
    logits = BotCortex.isquestion_model(
        BotCortex.isquestion_tokenizer.encode(txt, return_tensors="pt", max_length=512)
    )
    print("text for isquestion: " + txt)
    # label.logits = label.logits.detach().numpy()
    # labels_decoded = BotCortex.isquestion_tokenizer.decode(
    #     label.logits.argmax(axis=1))
    logits = logits["logits"].detach().cpu().numpy()
    logit = logits.argmax(axis=1)
    print(logit)
    if logit == 1:
        is_q = True
    return is_q


async def is_chat_or_command(txt):
    is_com = False
    candidate_labels = ["command", "conversation", "request", "other"]

    labels = BotCortex.nli_classifier(txt, candidate_labels)
    for label, val in labels.items():
        print("label: " + label)
        print("val: " + str(val))
    label = labels["labels"][labels["scores"].index(max(labels["scores"]))]
    print(label)
    if label == "request" or label == "command":
        is_com = True
    return is_com


async def is_personal(txt):
    if (
        str("your name") in txt
        or str("can you do") in txt
        or str("your favorite") in txt
        or str("what are you") in txt
        or str("What are you") in txt
    ):
        return False
    is_pers = False
    candidate_labels = ["personal question", "knowledge question"]

    labels = BotCortex.nli_classifier(txt, candidate_labels)
    for label, val in labels.items():
        print("label: " + label)
        print("val: " + str(val))
    label = labels["labels"][labels["scores"].index(max(labels["scores"]))]
    print(label)
    if label == "personal question":
        is_pers = True

    return is_pers


# answer_tokenizer = DistilBertTokenizer.from_pretrained(
#     "distilbert-base-cased-distilled-squad", return_tensors="pt")

# answer_model = DistilBertModel.from_pretrained(
#     "distilbert-base-cased-distilled-squad")


async def determine_question_context():
    # TODO: determine the context of the question or if question requires a context
    return


# The function below is an important one that will be referenced for many types of questions. It takes in a question and a context and returns the answer to the question based on the context. The context is the text that the question is based on. For example, if the question is "What is the capital of France?" and the context is "France is a country in Europe. The capital of France is Paris.", the answer will be "Paris". The function below uses the DistilBert model to answer the question. The model is trained on the SQuAD dataset, which is a dataset of questions and answers. The model is trained to answer questions based on the context of the question. The model is not trained to answer questions that are not in the context of the question. For example, if the question is "What is the capital of France?" and the context is "France is a country in Europe. The capital of France is Paris. The capital of Germany is Berlin.", the model will not be able to answer the question because the context does not contain the answer to the question. The model will return a score that indicates how confident it is in the answer. The score is a number between 0 and 1. The higher the score, the more confident the model is in the answer. The function below returns the answer and the score. If the score is less than 0.5, the model is not confident in the answer and the function returns "I'm not sure I understand the question. Can you rephrase it?" instead of the answer. If the score is greater than 0.5, the model is confident in the answer and the function returns the answer. The function below also returns a boolean value that indicates whether the model is confident in the answer. If the score is less than 0.5, the boolean value is False. If the score is greater than 0.5, the boolean value is True.


async def answer_question_based_on_context(txt, context):
    # inputs = answer_tokenizer(txt, context, return_tensors="pt")

    # with torch.no_grad():

    #     answer = answer_model(**inputs)
    result = question_answerer(question=txt, context=context)
    print(
        f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
    )

    if result["score"] < 0.5:
        return "I'm not sure I understand the question. Can you rephrase it?", False
    else:
        return result["answer"], True
    # print(f"The predicted class is {answer_model.config.id2label[pred]}")


async def needs_follow_up(txt):
    needs_fu = False
    candidate_labels = ["needs a follow-up question", "needs a statement response"]

    labels = BotCortex.nli_classifier(txt, candidate_labels)
    for label, val in labels.items():
        print("label: " + label)
        print("val: " + str(val))
    label = labels["labels"][labels["scores"].index(max(labels["scores"]))]
    print(label)
    candidate_labelsb = [
        "there is enough info here to reply just fine",
        "more info is needed here to reply appropriately",
    ]

    labelsb = BotCortex.nli_classifier(txt, candidate_labelsb)
    for label, val in labelsb.items():
        print("label: " + label)
        print("val: " + str(val))
    labelb = labelsb["labels"][labelsb["scores"].index(max(labelsb["scores"]))]
    print(labelsb)
    candidate_labelsc = [
        "I think I need to ask a question to understand",
        "this sentence makes sense on its own",
    ]

    labelsc = BotCortex.nli_classifier(txt, candidate_labelsc)

    for label, val in labelsc.items():
        print("label: " + label)
        print("val: " + str(val))
    labelc = labelsc["labels"][labelsc["scores"].index(max(labelsc["scores"]))]
    print(label)
    if label == "needs a follow-up question":
        if labelsb == "more info is needed here to reply appropriately":
            if labelsc == "I think I need to ask a question to understand":
                needs_fu = True

    return needs_fu


async def submit_command(text):
    command = interpret_command(text)
    execute_command(command)


async def interpret_command(text):
    inputs = command_tokenizer(text, return_tensors="pt")
    outputs = command_model(**inputs)
    pred = torch.argmax(outputs[0]).item()
    print(f"The predicted class is {command_model.config.id2label[pred]}")
    return command_model.config.id2label[pred]


async def execute_command(command):
    return "No commands available yet"
    # if command = "datetime_query"
    # "iot_hue_lightchange",
    #  "qa_currency",
    # "transport_traffic",
    # "general_quirky",
    # "weather_query",
    # "audio_volume_up",
    # "email_addcontact",
    # "takeaway_order",
    # "email_querycontact",
    # "iot_hue_lightup",
    # "recommendation_locations",
    # "transport_ticket",
    # "play_audiobook",
    # "lists_createoradd",
    # "news_query",
    # "alarm_query",
    # "iot_wemo_on",
    # "general_joke",
    # "qa_definition",
    # "social_query",
    # "music_settings",
    # "audio_volume_other",
    # "takeaway_query",
    # "calendar_remove",
    # "iot_hue_lightdim",
    # "calendar_query",
    # "email_sendemail",
    # "iot_cleaning",
    # "audio_volume_down",
    # "play_radio",
    # "cooking_query",
    # "datetime_convert",
    # "qa_maths",
    # "qa_stock",
    # "iot_hue_lightoff",
    # "iot_hue_lighton",
    # "transport_query",
    # "music_likeness",
    # "email_query",
    # "play_music",
    # "audio_volume_mute",
    # "social_post",
    # "alarm_set",
    # "qa_factoid",
    # "general_greet",
    # "calendar_set",
    # "play_game",
    # "alarm_remove",
    # "lists_remove",
    # "transport_taxi",
    # "recommendation_movies",
    # "iot_coffee",
    # "music_query",
    # "play_podcasts",
    # "lists_query",
    # "recommendation_events",
    # "music_dislikeness",
    # "iot_wemo_off",
    # "cooking_recipe"


image_size = 224  # pixels
patch_size = 16  # pixels


async def is_requesting_image(txt):
    is_requesting_image = False
    if BotCortex.local_only_mode:
        candidate_labels = [
            "a request for a drawing or image",
            "something else",
        ]

        labels = BotCortex.nli_classifier(txt, candidate_labels)
        for label, val in labels.items():
            print("label: " + label)
            print("val: " + str(val))
        label = labels["labels"][labels["scores"].index(max(labels["scores"]))]
        print(label)
        if label == "question requiring vision to answer":
            is_requesting_image = True
    else:
        is_requesting_image = await BotCortex.is_requesting_SD(txt)
    return is_requesting_image


async def is_visual_question(txt):
    is_vis = False
    # if BotCortex.local_only_mode:
    candidate_labels = [
        "question requiring vision to answer",
        "question not requiring vision to answer",
    ]

    labels = BotCortex.nli_classifier(txt, candidate_labels)
    for label, val in labels.items():
        print("label: " + label)
        print("val: " + str(val))
    label = labels["labels"][labels["scores"].index(max(labels["scores"]))]
    print(label)
    if label == "question requiring vision to answer":
        is_vis = True
    # else:
    #     is_vis = await BotCortex.is_visual_q(txt)
    return is_vis


# async def mean_pooling(model_output, attention_mask):
#     # First element of model_output contains all token embeddings
#     token_embeddings = model_output[0]
#     input_mask_expanded = (
#         attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     )
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#         input_mask_expanded.sum(1), min=1e-9
#     )


# async def get_embeddings(text_list):
#     global search_tokenizer, search_model
#     print("get_embeddings")
#     print(text_list)
#     encoded_input = search_tokenizer(
#         text_list,
#         padding=True,
#         truncation=True,
#         return_tensors="pt",
#         return_attention_mask=True,
#     )
#     encoded_input = {k: v for k, v in encoded_input.items()}
#     print(encoded_input)
#     model_output = search_model(**encoded_input)
#     return await mean_pooling(model_output, encoded_input["attention_mask"])


# # Use FAISS to search for the question in the question bank


# async def search_question_in_bank(qdict, question):
#     # TODO: Instead of
#     found = False
#     # df = pd.DataFrame.from_dict(
#     #     qdict, orient='index', columns=['question'])
#     # # for key in qdict.keys():
#     # #     temp_dict = qdict[key]
#     # dataset = Dataset.from_pandas(df)
#     # print(dataset.column_names)
#     # #    Encode the question and the question bank
#     # embeddings_dataset = dataset.map(
#     #     lambda x: {"embeddings": get_embeddings(
#     #         x["question"]).detach().numpy()[0]}
#     # )
#     # embeddings_dataset.add_faiss_index(column="embeddings")
#     query_embedding = await get_embeddings(question)
#     query_embedding = query_embedding.detach()
#     qdict_embedding = await get_embeddings(list(qdict.keys()))
#     # scores, samples = embeddings_dataset.get_nearest_examples(
#     #     "embeddings", question_embedding, k=5
#     # )
#     # encoded_qdict = search_tokenizer(qdict.keys().to_list(), padding=True, return_attention_mask=True,
#     #                                  truncation=True, return_tensors='pt')
#     # encoded_query = search_tokenizer(question, padding=True, return_attention_mask=True,
#     #                                  truncation=True, return_tensors='pt')

#     #    Normalize embeddings
#     # qdict_embedding /= np.linalg.norm(qdict_embedding, axis=1, keepdims=True)
#     # query_embedding /= np.linalg.norm(query_embedding)

#     qdict_embedding = F.normalize(qdict_embedding, p=2, dim=1)
#     query_embedding = F.normalize(query_embedding, p=2, dim=1)
#     k = 5

#     #    Compute cosine-similarits
#     cos_scores = torch.mm(query_embedding, qdict_embedding.transpose(0, 1))
#     sorted_scores, sorted_indices = torch.sort(
#         cos_scores.squeeze(), dim=0, descending=True
#     )

#     top_k_scores = sorted_scores[:k]
#     top_k_indices = sorted_indices[:k]

#     # Retrieve the top-k keys and their similarity scores
#     results = []
#     for i in range(top_k_indices.size(0)):
#         key = list(qdict.keys())[top_k_indices[i]]
#         score = top_k_scores[i].item()
#         results.append((key, score))

#     print(results[0])
#     answer = ""
#     if results[0][1] - 0.05 <= results[1][1]:
#         answer = "I am not sure about that. Please ask me another question."
#         found = True
#     if results[0][1] > 0.9:
#         answer = qdict[results[0][0]]
#         found = True
#     elif results[0][1] > 0.8:
#         answer = (
#             "I could be mistaken, but I think the answer is " + qdict[results[0][0]]
#         )
#         found = True
#     elif results[0][1] > 0.7:
#         answer = (
#             "It's possible the answer is "
#             + qdict[results[0][0]]
#             + ", but I'm really not sure."
#         )
#         found = True
#     elif results[0][1] > 0.6:
#         answer = "I'm sorry, I don't know the answer to that question."
#     return answer, found


# async def query_memory(query):
#     # Load the question bank
#     qdict = await load_qdict_from_json()
#     # Search the question bank
#     answer, found = await search_question_in_bank(qdict, query)
#     if (
#         answer != ""
#         or answer == "I am not sure about that. Please ask me another question."
#     ):
#         qdict[query] = answer
#     # Save the question bank
#     await save_qdict_to_json(qdict)
#     return (answer, found)


# async def save_qdict_to_json(my_dict):
#     # Open the file for writing
#     with open("question_file.json", "w") as f:
#         # Save the key-value pairs to the file using JSON encoding
#         json.dump(my_dict, f)


# # Open the file for reading


# async def load_qdict_from_json():
#     with open("question_file.json", "r") as f:
#         # Load the key-value pairs from the file using JSON decoding
#         my_dict = json.load(f)
#         print("type:", type(my_dict))
#         return my_dict


async def command_type(txt):
    candidate_labels = ["command", "conversation", "request", "other"]

    labels = await BotCortex.command_classifier(
        BotCortex.command_tokenizer.encode(txt, return_tensors="pt", max_length=512),
        candidate_labels,
    )
    label = labels["labels"][labels["scores"].index(max(labels["scores"]))]
    return label


async def start():
    await load_models()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start())
