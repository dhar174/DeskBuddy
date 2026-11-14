"""Behavioral contract and regression tests for helpers.py classifiers and async commands."""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

import pytest

from deskbuddy.settings import Settings
from tests.support.fakes import (
    CallRecorder,
    FakeCommandModel,
    FakeNLIClassifier,
    FakeOpenAISDK,
    FakeQuestionClassifier,
    FakeTokenizer,
)
from tests.support.legacy_runtime import (
    assert_no_new_heavy_imports,
    install_heavy_import_guard,
    load_botcortex_runtime,
    load_helpers_runtime,
)

FOLLOW_UP_LABELS_1 = ("needs a follow-up question", "needs a statement response")
FOLLOW_UP_LABELS_2 = (
    "there is enough info here to reply just fine",
    "more info is needed here to reply appropriately",
)
FOLLOW_UP_LABELS_3 = (
    "I think I need to ask a question to understand",
    "this sentence makes sense on its own",
)
IMAGE_REQ_LABELS = ("a request for a drawing or image", "something else")
VISUAL_Q_LABELS = (
    "question requiring vision to answer",
    "question not requiring vision to answer",
)


def test_needs_follow_up_returns_true_when_all_three_classifiers_select_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(
        recorder,
        routes={
            FOLLOW_UP_LABELS_1: [0.91, 0.09],
            FOLLOW_UP_LABELS_2: [0.12, 0.88],  # index 1 must win
            FOLLOW_UP_LABELS_3: [0.85, 0.15],
        },
    )
    ns, _, _ = load_helpers_runtime(recorder=recorder, nli_classifier=nli)

    result = asyncio.run(ns["needs_follow_up"]("Tell me about it"))

    assert result is True
    nli_calls = recorder.for_target("nli_classifier")
    assert [call.args[1] for call in nli_calls] == [
        FOLLOW_UP_LABELS_1,
        FOLLOW_UP_LABELS_2,
        FOLLOW_UP_LABELS_3,
    ]
    assert_no_new_heavy_imports(loaded_before)


@pytest.mark.parametrize(
    ("failing_stage", "routes"),
    [
        (
            "first",
            {
                FOLLOW_UP_LABELS_1: [0.2, 0.8],
                FOLLOW_UP_LABELS_2: [0.1, 0.9],
                FOLLOW_UP_LABELS_3: [0.9, 0.1],
            },
        ),
        (
            "second",
            {
                FOLLOW_UP_LABELS_1: [0.9, 0.1],
                FOLLOW_UP_LABELS_2: [0.8, 0.2],
                FOLLOW_UP_LABELS_3: [0.9, 0.1],
            },
        ),
        (
            "third",
            {
                FOLLOW_UP_LABELS_1: [0.9, 0.1],
                FOLLOW_UP_LABELS_2: [0.1, 0.9],
                FOLLOW_UP_LABELS_3: [0.2, 0.8],
            },
        ),
    ],
)
def test_needs_follow_up_returns_false_when_any_individual_classifier_rejects_clarification(
    failing_stage: str,
    routes: dict[tuple[str, ...], list[float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(recorder, routes=routes)
    ns, _, _ = load_helpers_runtime(recorder=recorder, nli_classifier=nli)

    result = asyncio.run(ns["needs_follow_up"](f"case-{failing_stage}"))

    assert result is False
    assert recorder.count("nli_classifier") == 3
    assert_no_new_heavy_imports(loaded_before)


def test_needs_follow_up_is_deterministic_and_independent_of_dict_identity_or_key_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)

    class ShuffledFreshDictNLI:
        def __init__(self, reverse: bool) -> None:
            self.reverse = reverse
            self.seen_ids: list[int] = []

        def __call__(self, text: str, candidate_labels: list[str]) -> dict[str, Any]:
            # Scores favor clarification even when label order in returned dict is non-trivial
            if candidate_labels[0] == FOLLOW_UP_LABELS_2[0]:
                labels = [candidate_labels[1], candidate_labels[0]]
                scores = [0.87, 0.13]
            else:
                labels = [candidate_labels[1], candidate_labels[0]]
                scores = [0.14, 0.86]
            payload = (
                {"scores": scores, "labels": labels, "sequence": text}
                if self.reverse
                else {"sequence": text, "labels": labels, "scores": scores}
            )
            self.seen_ids.append(id(payload))
            return payload

    nli_forward = ShuffledFreshDictNLI(reverse=False)
    nli_reverse = ShuffledFreshDictNLI(reverse=True)
    ns_forward, _, _ = load_helpers_runtime(nli_classifier=nli_forward)
    ns_reverse, _, _ = load_helpers_runtime(nli_classifier=nli_reverse)

    res_forward = asyncio.run(ns_forward["needs_follow_up"]("That thing broke"))
    res_reverse = asyncio.run(ns_reverse["needs_follow_up"]("That thing broke"))

    assert res_forward is True
    assert res_reverse is True
    assert len(set(nli_forward.seen_ids)) == 3
    assert len(set(nli_reverse.seen_ids)) == 3
    assert_no_new_heavy_imports(loaded_before)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.92, 0.08], True),
        ([0.15, 0.85], False),
    ],
)
def test_is_requesting_image_matches_image_request_label_in_local_mode(
    scores: list[float],
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(recorder, routes={IMAGE_REQ_LABELS: scores})
    ns, _, _ = load_helpers_runtime(recorder=recorder, nli_classifier=nli, local_only_mode=True)

    result = asyncio.run(ns["is_requesting_image"]("Draw me a cat"))

    assert result is expected
    assert recorder.count("nli_classifier") == 1
    assert recorder.for_target("nli_classifier")[0].args[1] == IMAGE_REQ_LABELS
    assert recorder.count("BotCortex.is_requesting_SD") == 0
    assert_no_new_heavy_imports(loaded_before)


def test_is_requesting_image_preserves_separation_from_visual_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    # Suppose visual question classifier says True, while image generation says False
    nli = FakeNLIClassifier(
        recorder,
        routes={
            IMAGE_REQ_LABELS: [0.1, 0.9],
            VISUAL_Q_LABELS: [0.95, 0.05],
        },
    )
    ns, _, _ = load_helpers_runtime(recorder=recorder, nli_classifier=nli, local_only_mode=True)

    wants_image = asyncio.run(ns["is_requesting_image"]("What am I holding?"))
    wants_vqa = asyncio.run(ns["is_visual_question"]("What am I holding?"))

    assert wants_image is False
    assert wants_vqa is True
    assert [c.args[1] for c in recorder.for_target("nli_classifier")] == [
        IMAGE_REQ_LABELS,
        VISUAL_Q_LABELS,
    ]
    assert_no_new_heavy_imports(loaded_before)


def test_is_requesting_image_delegates_to_is_requesting_sd_when_not_local_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()

    async def fake_sd(txt: str) -> bool:
        recorder.record("BotCortex.is_requesting_SD", txt)
        return True

    ns, _, _ = load_helpers_runtime(
        recorder=recorder,
        local_only_mode=False,
        is_requesting_sd=fake_sd,
    )

    result = asyncio.run(ns["is_requesting_image"]("Draw a spaceship"))

    assert result is True
    assert recorder.count("BotCortex.is_requesting_SD") == 1
    assert recorder.count("nli_classifier") == 0
    assert_no_new_heavy_imports(loaded_before)


@pytest.mark.parametrize(
    ("model_reply", "expected"),
    [("Response: True", True), ("Response: False", False)],
)
def test_real_is_requesting_sd_builds_valid_messages_and_classifies(
    model_reply: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    sdk = FakeOpenAISDK(recorder, chat_reply=model_reply)
    cortex, _ = load_botcortex_runtime(
        recorder=recorder,
        local_only_mode=False,
        openai_sdk=sdk,
        settings=Settings(sd_classifier_model_id="integration-sd-model"),
    )

    result = asyncio.run(cortex["is_requesting_SD"]("Draw a spaceship"))

    assert result is expected
    (call,) = recorder.for_target("openai.ChatCompletion.create")
    assert call.kwargs["model"] == "integration-sd-model"
    messages = call.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert all(isinstance(message, dict) for message in messages)
    assert messages[-1] == {
        "role": "user",
        "content": (
            "Determine if the following question is asking for a drawing, image, "
            "or picture to be created, and then reply with True or False only: Draw a spaceship"
        ),
    }
    assert len(messages) == 24
    assert_no_new_heavy_imports(loaded_before)


def test_submit_command_awaits_interpret_and_execute_once_without_coroutine_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    cmd_model = FakeCommandModel(recorder, pred_index=1, id2label={0: "noop", 1: "weather_query"})
    cmd_tok = FakeTokenizer(recorder, name="command_tokenizer")
    ns, _, _ = load_helpers_runtime(
        recorder=recorder,
        command_model=cmd_model,
        command_tokenizer=cmd_tok,
    )

    orig_interpret = ns["interpret_command"]
    orig_execute = ns["execute_command"]
    interpret_calls: list[str] = []
    execute_calls: list[str] = []

    async def spy_interpret(text: str) -> str:
        interpret_calls.append(text)
        return await orig_interpret(text)

    async def spy_execute(command: str) -> str:
        execute_calls.append(command)
        return await orig_execute(command)

    ns["interpret_command"] = spy_interpret
    ns["execute_command"] = spy_execute

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        result = asyncio.run(ns["submit_command"]("What is the weather?"))

    assert result == "No commands available yet"
    assert interpret_calls == ["What is the weather?"]
    assert execute_calls == ["weather_query"]
    assert recorder.count("command_model") == 1
    assert caught == []
    assert_no_new_heavy_imports(loaded_before)


def test_supporting_helpers_classifiers_and_qa_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_before = install_heavy_import_guard(monkeypatch)
    recorder = CallRecorder()
    nli = FakeNLIClassifier(
        recorder,
        routes={
            ("personal question", "knowledge question"): [0.85, 0.15],
            ("command", "conversation", "request", "other"): [0.05, 0.85, 0.05, 0.05],
        },
    )
    ns, _, _ = load_helpers_runtime(
        recorder=recorder,
        nli_classifier=nli,
        isquestion_model=FakeQuestionClassifier(recorder, is_question=True),
        question_answerer=lambda question, context: {
            "answer": "Paris",
            "score": 0.82 if "France" in context else 0.2,
            "start": 0,
            "end": 5,
        },
    )

    assert asyncio.run(ns["is_question"]("Where is France?")) is True
    # Hardcoded bot-identity phrases short-circuit before NLI
    assert asyncio.run(ns["is_personal"]("What is your name?")) is False
    assert recorder.count("nli_classifier") == 0
    assert asyncio.run(ns["is_personal"]("Do you feel happy today?")) is True
    assert asyncio.run(ns["is_chat_or_command"]("Let's chat")) is False
    assert asyncio.run(
        ns["answer_question_based_on_context"]("Capital?", "France capital is Paris.")
    ) == ("Paris", True)
    assert asyncio.run(
        ns["answer_question_based_on_context"]("Capital?", "Unrelated context.")
    ) == ("I'm not sure I understand the question. Can you rephrase it?", False)
    assert_no_new_heavy_imports(loaded_before)
