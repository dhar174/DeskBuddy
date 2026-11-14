"""Behavioral contract and regression tests for DeskBuddy conversation routing."""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

import pytest

from deskbuddy.settings import Settings
from tests.support.fakes import CallRecorder, FakeNLIClassifier, FakeOpenAISDK, nli_routes
from tests.support.legacy_runtime import (
    assert_no_new_heavy_imports,
    install_heavy_import_guard,
    load_botcortex_runtime,
    load_start_bot_dispatch,
)

# ---------------------------------------------------------------------------
# A. Normal conversation contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "talk_fn", ["talk_history", "talk_medium_with_history", "talk_fast", "talk_small"]
)
def test_normal_conversation_routes_to_selected_local_chat_model(
    talk_fn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(recorder, routes=nli_routes())
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=True,
        nli_classifier=nli,
        is_question=False,
    )
    reply = asyncio.run(ns[talk_fn]("I played Zelda today"))

    assert reply == "LOCAL_MODEL_REPLY"
    if talk_fn == "talk_small":
        (decode_call,) = recorder.for_target("chat_tokenizer.decode")
        assert decode_call.args[0].tolist() == [99, 100]
    assert recorder.count("chat_model.generate") == 1
    assert recorder.count("answer_visual_question") == 0
    assert recorder.count("load_qdict_from_json") == 0
    assert recorder.count("qgen_model.generate") == 0
    assert recorder.count("openai.ChatCompletion.create") == 0
    assert_no_new_heavy_imports(loaded_before)


def test_normal_conversation_routes_to_openai_chat_in_medium_api_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    openai_sdk = FakeOpenAISDK(
        recorder,
        chat_reply="Cooper: That Zelda dungeon sounds awesome!",
        completion_reply="False",
    )
    nli = FakeNLIClassifier(recorder, routes=nli_routes())
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=False,
        nli_classifier=nli,
        is_question=False,
        openai_sdk=openai_sdk,
    )

    reply = asyncio.run(ns["talk_medium_API_with_history"]("I played Zelda today"))

    assert reply == " That Zelda dungeon sounds awesome!"
    chat_calls = recorder.for_target("openai.ChatCompletion.create")
    assert len(chat_calls) == 1
    assert chat_calls[0].kwargs["model"] == "gpt-3.5-turbo-test"
    assert chat_calls[0].kwargs["messages"][-1] == {
        "role": "user",
        "content": "User:I played Zelda today",
    }
    assert recorder.count("answer_visual_question") == 0
    assert recorder.count("load_qdict_from_json") == 0
    assert recorder.count("chat_model.generate") == 0
    assert_no_new_heavy_imports(loaded_before)


# ---------------------------------------------------------------------------
# B. Visual questions (early return + recognized unavailable fallback)
# ---------------------------------------------------------------------------


def test_visual_question_with_useful_answer_returns_early_without_memory_or_local_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(recorder, routes=nli_routes(visual=True))
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=True,
        nli_classifier=nli,
        is_question=True,
        vqa_handler=lambda _q: "I think it is a red coffee mug",
    )

    reply = asyncio.run(ns["talk_history"]("What am I holding?"))

    assert reply == "I think it is a red coffee mug"
    assert recorder.count("answer_visual_question") == 1
    assert recorder.count("load_qdict_from_json") == 0
    assert recorder.count("qgen_model.generate") == 0
    assert recorder.count("chat_model.generate") == 0
    assert_no_new_heavy_imports(loaded_before)


def test_visual_question_in_medium_api_mode_rewords_vqa_answer_without_memory_or_normal_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    openai_sdk = FakeOpenAISDK(
        recorder,
        completion_reply="True",
        chat_reply="Response: Looks like you're holding a bright red mug!",
    )
    nli = FakeNLIClassifier(recorder, routes=nli_routes(visual=True))
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=False,
        nli_classifier=nli,
        is_question=True,
        openai_sdk=openai_sdk,
        vqa_handler=lambda _q: "I think it is a red mug",
    )

    reply = asyncio.run(ns["talk_medium_API_with_history"]("What am I holding?"))

    assert reply == " Looks like you're holding a bright red mug!"
    assert recorder.count("answer_visual_question") == 1
    assert recorder.count("load_qdict_from_json") == 0
    chat_calls = recorder.for_target("openai.ChatCompletion.create")
    assert len(chat_calls) == 1
    # Rewording branch sends reword instruction rather than normal User: prompt
    assert "Please reword the following text" in chat_calls[0].kwargs["messages"][-1]["content"]
    assert_no_new_heavy_imports(loaded_before)


@pytest.mark.parametrize(
    "vqa_unavailable_reply",
    [
        "No image found",
        "No image found: camera disconnected",
        "I don't know what is in the picture",
        "No camera or image source configured. Set DESKBUDDY_CAMERA_SOURCE.",
    ],
)
def test_visual_question_falls_back_to_generation_when_vqa_is_unavailable_or_unknown(
    vqa_unavailable_reply: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(recorder, routes=nli_routes(visual=True))
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=True,
        nli_classifier=nli,
        is_question=True,
        vqa_handler=lambda _q: vqa_unavailable_reply,
    )

    reply = asyncio.run(ns["talk_history"]("What is on my desk?"))

    assert reply == "LOCAL_MODEL_REPLY"
    assert recorder.count("answer_visual_question") == 1
    assert recorder.count("load_qdict_from_json") == 0
    assert recorder.count("chat_model.generate") == 1
    assert_no_new_heavy_imports(loaded_before)


# ---------------------------------------------------------------------------
# C & D. Memory hit / miss + Regression 3 (query_memory boolean logic)
# ---------------------------------------------------------------------------


def test_memory_hit_returns_early_without_invoking_local_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(recorder, routes=nli_routes())
    qdict = {
        "What is the capital of France?": "Paris is the capital of France",
        "Is the moon made of cheese?": "No, it is rock and dust.",
    }
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=True,
        nli_classifier=nli,
        is_question=True,
        qdict_data=qdict,
        embedding_scores=[0.95, 0.2],
    )

    reply = asyncio.run(ns["talk_medium_with_history"]("Tell me the capital of France"))

    assert reply == "Paris is the capital of France"
    assert recorder.count("load_qdict_from_json") == 1
    assert recorder.count("get_embeddings") == 2
    assert recorder.count("chat_model.generate") == 0
    assert recorder.count("answer_visual_question") == 0
    assert recorder.count("save_qdict_to_json") == 0
    assert_no_new_heavy_imports(loaded_before)


def test_memory_miss_falls_back_to_generation_and_sets_unfound_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(recorder, routes=nli_routes())
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=True,
        nli_classifier=nli,
        is_question=True,
        qdict_data={},
    )

    reply = asyncio.run(ns["talk_history"]("What is the distance to Mars?"))

    assert reply == "LOCAL_MODEL_REPLY"
    assert ns["unfound_answer"] is True
    assert recorder.count("load_qdict_from_json") == 1
    assert recorder.count("get_embeddings") == 0
    assert recorder.count("chat_model.generate") == 1
    assert_no_new_heavy_imports(loaded_before)


@pytest.mark.parametrize(
    ("search_result", "expected_result", "should_insert"),
    [
        (("Paris is the capital", True), ("Paris is the capital", True), True),
        (("", False), ("unknown question?", False), False),
        (
            ("I am not sure about that. Please ask me another question.", True),
            ("unknown question?", False),
            False,
        ),
        (
            ("I'm sorry, I don't know the answer to that question.", False),
            ("unknown question?", False),
            False,
        ),
        (
            ("I'm sorry, I don't know the answer to that question.", True),
            ("unknown question?", False),
            False,
        ),
    ],
)
def test_query_memory_regression_rejects_empty_and_unknown_sentinels_without_inserting(
    search_result: tuple[str, bool],
    expected_result: tuple[str, bool],
    should_insert: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    ns, _ = load_botcortex_runtime(recorder=recorder)
    observed_qdict: dict[str, str] = {"q1": "a1", "q2": "a2"}

    async def fake_load() -> dict[str, str]:
        recorder.record("load_qdict_from_json")
        return observed_qdict

    async def fake_search(_qdict: dict[str, str], _query: str) -> tuple[str, bool]:
        recorder.record("search_question_in_bank")
        return search_result

    ns["load_qdict_from_json"] = fake_load
    ns["search_question_in_bank"] = fake_search

    result = asyncio.run(ns["query_memory"]("unknown question?"))

    assert result == expected_result
    assert ("unknown question?" in observed_qdict) is should_insert
    assert recorder.count("save_qdict_to_json") == 0
    assert_no_new_heavy_imports(loaded_before)


def test_search_question_in_bank_skips_embeddings_under_two_entries_and_scores_valid_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    ns, _ = load_botcortex_runtime(recorder=recorder, embedding_scores=[0.85, 0.2])

    assert asyncio.run(ns["search_question_in_bank"]({}, "q")) == ("", False)
    assert asyncio.run(ns["search_question_in_bank"]({"only": "one"}, "q")) == ("", False)
    assert recorder.count("get_embeddings") == 0

    answer, found = asyncio.run(
        ns["search_question_in_bank"]({"q1": "Blue", "q2": "Red"}, "favorite?")
    )
    assert found is True
    assert answer == "I could be mistaken, but I think the answer is Blue"
    assert recorder.count("get_embeddings") == 2
    assert_no_new_heavy_imports(loaded_before)


# ---------------------------------------------------------------------------
# E. Clarification early-return + Regression 5 (talk_medium_API_with_history)
# ---------------------------------------------------------------------------


def test_clarification_returns_follow_up_early_without_local_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(recorder, routes=nli_routes(follow_up=True))
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=True,
        nli_classifier=nli,
        qgen_output="Which Zelda game do you mean?",
    )

    reply = asyncio.run(ns["talk_history"]("I love that game"))

    assert reply == "Which Zelda game do you mean?"
    assert ns["asked_follow_up"] is True
    assert recorder.count("qgen_model.generate") == 1
    assert recorder.count("isquestion_model") == 0
    assert recorder.count("answer_visual_question") == 0
    assert recorder.count("load_qdict_from_json") == 0
    assert recorder.count("chat_model.generate") == 0
    assert_no_new_heavy_imports(loaded_before)


def test_talk_medium_api_clarification_returns_followup_without_unbound_out_or_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    api_calls = 0

    def chat_reply(kwargs: dict[str, Any]) -> str:
        nonlocal api_calls
        api_calls += 1
        messages = kwargs.get("messages", [])
        if any("Ask a followup question" in str(msg) for msg in messages):
            return "Which part of that did you mean?"
        raise AssertionError("No post-clarification ChatCompletion call should occur")

    openai_sdk = FakeOpenAISDK(recorder, chat_reply=chat_reply, completion_reply="False")
    nli = FakeNLIClassifier(recorder, routes=nli_routes(follow_up=True))
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=False,
        nli_classifier=nli,
        openai_sdk=openai_sdk,
    )

    reply = asyncio.run(ns["talk_medium_API_with_history"]("It was really cool"))

    assert reply == "Which part of that did you mean?"
    assert ns["asked_follow_up"] is True
    # Only the follow-up generation call inside get_follow_up_question ran; no reword/chat call
    assert api_calls == 1
    assert recorder.count("openai.ChatCompletion.create") == 1
    assert_no_new_heavy_imports(loaded_before)


def test_talk_history_nonlocal_clarification_does_not_corrupt_local_history_on_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    openai_sdk = FakeOpenAISDK(
        recorder,
        chat_reply="Which app crashed?",
        completion_reply="False",
    )
    nli = FakeNLIClassifier(recorder, routes=nli_routes(follow_up=True))
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=False,
        nli_classifier=nli,
        is_question=False,
        openai_sdk=openai_sdk,
    )

    reply1 = asyncio.run(ns["talk_history"]("It crashed"))
    assert reply1 == "Which app crashed?"
    assert ns["asked_follow_up"] is True
    assert all(isinstance(entry, str) for entry in ns["history"])
    chat_calls = recorder.for_target("openai.ChatCompletion.create")
    assert len(chat_calls) == 1
    assert all(isinstance(msg, dict) for msg in chat_calls[0].kwargs["messages"])

    nli.routes = nli_routes(follow_up=False)
    reply2 = asyncio.run(ns["talk_history"]("My browser crashed"))
    assert reply2 == "LOCAL_MODEL_REPLY"
    assert recorder.count("chat_model.generate") == 1
    assert_no_new_heavy_imports(loaded_before)


# ---------------------------------------------------------------------------
# F. Local / cloud selection contracts (start_bot.dispatch_reply & load_all_models)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("local_only", "button_text", "expected_target"),
    [
        (True, "Fast", "talk_fast"),
        (True, "Small", "talk_small"),
        (True, "Medium", "talk_medium_with_history"),
        (True, "Large", "talk_history"),
        (True, "", "talk_fast"),
        (False, "Fast", "talk_fast"),
        (False, "Small", "talk_small"),
        (False, "Medium", "talk_medium_API_with_history"),
        (False, "Large", "talk_history"),
        (False, "Unexpected", "talk_fast"),
    ],
)
def test_start_bot_dispatch_reply_matches_local_only_and_model_size_contract(
    local_only: bool,
    button_text: str,
    expected_target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    dispatch_reply, recorder, _ = load_start_bot_dispatch()

    reply = asyncio.run(dispatch_reply("hello cooper", button_text, local_only))

    assert reply == f"REPLY:{expected_target}:hello cooper"
    assert recorder.targets() == [f"BotCortex.{expected_target}"]
    assert_no_new_heavy_imports(loaded_before)


@pytest.mark.parametrize(
    ("local_only", "button_text", "expected_init", "expects_nccl"),
    [
        (True, "Fast", "init_fast", False),
        (True, "Small", "init_small", False),
        (True, "Medium", "init_medium", False),
        (True, "Large", "init", True),
        (False, "Fast", "init_fast", False),
        (False, "Small", "init_small", False),
        (False, "Medium", "init_API_medium", True),
        (False, "Large", "init", True),
    ],
)
def test_load_all_models_dispatches_expected_initializer_and_nccl_guard(
    tmp_path: Any,
    local_only: bool,
    button_text: str,
    expected_init: str,
    expects_nccl: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    ckpt_dir = tmp_path / "checkpoint"
    tok_dir = tmp_path / "tokenizer"
    ckpt_dir.mkdir()
    tok_dir.mkdir()
    settings = Settings(
        model_provider="local" if local_only else "openai",
        openai_api_key="test-key" if not local_only else None,
        checkpoint_dir=ckpt_dir,
        tokenizer_dir=tok_dir,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    recorder = CallRecorder()
    ns, _ = load_botcortex_runtime(settings=settings, recorder=recorder)

    asyncio.run(ns["load_all_models"](button_text, local_only))

    assert recorder.count(expected_init) == 1
    assert (recorder.count("deepspeed.init_distributed") == 1) is expects_nccl
    assert ns["local_only_mode"] is local_only
    assert ns["model_size"] == button_text
    assert_no_new_heavy_imports(loaded_before)


# ---------------------------------------------------------------------------
# G & H. Graceful degradation and awaited async calls
# ---------------------------------------------------------------------------


def test_get_follow_up_question_degrades_gracefully_when_qgen_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    ns, _ = load_botcortex_runtime(local_only_mode=True)
    ns["qgen_model"] = None
    ns["qgen_tokenizer"] = None

    question = asyncio.run(ns["get_follow_up_question"]("vague prompt", ["Human: hi"]))

    assert (
        question
        == "I'm not sure what to ask to understand you better. Can you rephrase your response?"
    )
    assert_no_new_heavy_imports(loaded_before)


def test_async_entrypoints_complete_without_unawaited_coroutine_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(recorder, routes=nli_routes())
    ns, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=True,
        nli_classifier=nli,
        is_question=False,
    )
    ns["count_index"] = 10

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        result = asyncio.run(ns["talk_history"]("Let's talk about Nintendo"))

    assert result == "LOCAL_MODEL_REPLY"
    assert recorder.count("save_history_verbatim") == 1
    saved_arg = recorder.for_target("save_history_verbatim")[0].args[0]
    assert isinstance(saved_arg, str)
    assert "Let's talk about Nintendo" in saved_arg
    assert ns["count_index"] == 0
    assert caught == []
    assert_no_new_heavy_imports(loaded_before)
