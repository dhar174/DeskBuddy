"""Run the desktop conversation loop against authenticated robot loopback."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

import buddy1
import robot_client
from deskbuddy.settings import Settings
from tests.support.fakes import CallRecorder, FakeNLIClassifier, FakeOpenAISDK, nli_routes
from tests.support.legacy_runtime import load_botcortex_runtime, load_start_bot_main
from transport import ExecutionOutcomeUnknown, TransportConfig, TransportError, TransportErrorCode
from transport.hardware import reset_default_hardware_executor


class PollScriptComplete(Exception):
    """Stop the production loop before it issues an unscripted poll."""


class Poll(Enum):
    NO_NEW_UTTERANCE = auto()
    EMPTY_RPC_RESULT = auto()
    NULL_RPC_RESULT = auto()


@dataclass
class LoopbackConversation:
    recorder: CallRecorder = field(default_factory=CallRecorder)
    requests: list[str] = field(default_factory=list)
    received_replies: list[str] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)
    unknown: list[ExecutionOutcomeUnknown] = field(default_factory=list)
    delivery_errors: list[TransportError] = field(default_factory=list)
    transcriptions: list[str | None] = field(default_factory=list)
    drop_reply_ack: bool = False
    fail_before_reply_transmission: bool = False
    poll: str | Poll = Poll.NO_NEW_UTTERANCE

    async def run(
        self,
        turns: list[str | Poll],
        *,
        local_only: bool = True,
        button_text: str = "Medium",
        missing_qgen: bool = False,
        **options: Any,
    ) -> None:
        pending = iter(turns)
        cortex, _ = load_botcortex_runtime(
            recorder=self.recorder, local_only_mode=local_only, **options
        )
        if missing_qgen:
            cortex["qgen_model"] = None
            cortex["qgen_tokenizer"] = None

        async def request(**kwargs: Any) -> Any:
            assert kwargs == {"function_name": "check_for_response"}
            try:
                self.poll = next(pending)
            except StopIteration:
                raise PollScriptComplete from None
            self.requests.append("check_for_response")
            result = await robot_client.request_to_server(**kwargs)
            self.transcriptions.append(result)
            return result

        async def send_reply(message: str) -> None:
            self.requests.append("reply")
            try:
                await robot_client.send_reply(message)
            except ExecutionOutcomeUnknown as exc:
                self.unknown.append(exc)
                raise
            except TransportError as exc:
                self.delivery_errors.append(exc)
                raise

        main = load_start_bot_main(
            cortex,
            SimpleNamespace(
                request_to_server=request,
                send_reply=send_reply,
                ExecutionOutcomeUnknown=ExecutionOutcomeUnknown,
                TransportError=TransportError,
            ),
            local_only=local_only,
            button_text=button_text,
        )
        async with asyncio.timeout(10):
            try:
                await main()
            except PollScriptComplete:
                pass


@pytest_asyncio.fixture
async def conversation(monkeypatch: pytest.MonkeyPatch, ephemeral_server_config: TransportConfig):
    reset_default_hardware_executor()
    try:
        async with conversation_server(monkeypatch, ephemeral_server_config) as flow:
            yield flow
    finally:
        reset_default_hardware_executor()


@asynccontextmanager
async def conversation_server(
    monkeypatch: pytest.MonkeyPatch, ephemeral_server_config: TransportConfig
):
    flow = LoopbackConversation()
    real_poll = buddy1.MyFunctions.check_for_response
    real_reply = buddy1.MyFunctions.reply

    async def scripted_poll():
        if isinstance(flow.poll, str):
            await buddy1.MyFunctions.send_response(flow.poll)
        result = await real_poll()
        # These are explicit RPC fault cases, not legacy server output.
        if flow.poll is Poll.EMPTY_RPC_RESULT:
            return ""
        if flow.poll is Poll.NULL_RPC_RESULT:
            return None
        return result

    async def no_device_call(*args):
        pass

    def record_speech(text, *args, **kwargs):
        flow.spoken.append(text)

    async def record_reply(text):
        flow.received_replies.append(text)
        return await real_reply(text)

    for name in (
        "lastTranscription",
        "priorTranscription",
        "lastReplyMessage",
        "priorReplyMessage",
    ):
        monkeypatch.setattr(buddy1, name, "")
    for name, initial in (("recording", False), ("recorded", False), ("result", None)):
        monkeypatch.setattr(buddy1, name, initial, raising=False)
    monkeypatch.setattr(buddy1.MyFunctions, "check_for_response", scripted_poll)
    monkeypatch.setattr(buddy1.MyFunctions, "reply", record_reply)
    monkeypatch.setattr(buddy1.picoh, "say", record_speech)
    monkeypatch.setattr(buddy1.MyFunctions, "baseCol", no_device_call)
    monkeypatch.setattr(buddy1.MyFunctions, "picoh_wait", no_device_call)
    server = buddy1.create_robot_server(ephemeral_server_config)
    send_response = server._send_response
    dropped = False

    async def maybe_drop_ack(writer, response):
        nonlocal dropped
        if flow.drop_reply_ack and flow.spoken and not dropped:
            dropped = True
            writer.close()
            await writer.wait_closed()
        else:
            await send_response(writer, response)

    monkeypatch.setattr(server, "_send_response", maybe_drop_ack)
    await server.start()
    try:
        monkeypatch.setenv("DESKBUDDY_ROBOT_HOST", "127.0.0.1")
        monkeypatch.setenv("DESKBUDDY_ROBOT_PORT", str(server.port))
        monkeypatch.setenv("DESKBUDDY_ROBOT_AUTH_SECRET", ephemeral_server_config.secret)
        await robot_client.close_client()
        client = await robot_client.get_robot_client()
        ensure_connected = client._ensure_connected

        async def maybe_fail_before_transmission():
            if flow.fail_before_reply_transmission and flow.requests[-1] == "reply":
                flow.fail_before_reply_transmission = False
                raise TransportError(TransportErrorCode.TIMEOUT, "test failure before transmission")
            return await ensure_connected()

        monkeypatch.setattr(client, "_ensure_connected", maybe_fail_before_transmission)
        yield flow
    finally:
        try:
            await robot_client.close_client()
        finally:
            await server.stop()


@pytest.mark.parametrize("local_only", [True, False], ids=["local-medium", "cloud-medium"])
async def test_transcription_reaches_selected_model_and_robot_speech(conversation, local_only):
    transcript = "I played Zelda today"
    rec = conversation.recorder
    await conversation.run(
        [transcript],
        local_only=local_only,
        is_question=False,
        nli_classifier=FakeNLIClassifier(rec, routes=nli_routes()),
        openai_sdk=FakeOpenAISDK(rec, chat_reply=f"Cloud reply\ncontinues: {transcript}"),
        settings=Settings(openai_model="integration-test-model"),
    )

    if local_only:
        prompt = rec.for_target("chat_tokenizer.__call__")[0].args[0]
        assert transcript in prompt
        assert rec.count("chat_model.generate") == 1
        assert rec.count("openai.ChatCompletion.create") == 0
        assert conversation.spoken == ["LOCAL_MODEL_REPLY"]
    else:
        (call,) = rec.for_target("openai.ChatCompletion.create")
        assert call.kwargs["model"] == "integration-test-model"
        assert call.kwargs["messages"][-1]["content"] == f"User:{transcript}"
        assert rec.count("chat_model.generate") == 0
        assert conversation.spoken == [f"Cloud reply continues: {transcript}"]
    assert conversation.requests == ["check_for_response", "reply"]


@pytest.mark.parametrize(
    ("tier", "prompt_call", "length_parameter", "length"),
    [
        ("Fast", "chat_tokenizer.encode", "max_length", 64),
        ("Small", "chat_tokenizer.__call__", "max_length", 32),
        ("Large", "chat_tokenizer.__call__", "max_new_tokens", 32),
    ],
)
async def test_other_local_tiers_reach_speech(
    conversation, tier, prompt_call, length_parameter, length
):
    transcript = "What a day"
    rec = conversation.recorder
    await conversation.run(
        [transcript],
        button_text=tier,
        is_question=False,
        nli_classifier=FakeNLIClassifier(rec, routes=nli_routes()),
    )
    assert any(transcript in str(call.args[0]) for call in rec.for_target(prompt_call))
    (generation,) = rec.for_target("chat_model.generate")
    assert generation.kwargs[length_parameter] == length
    assert rec.count("openai.ChatCompletion.create") == 0
    assert conversation.requests == ["check_for_response", "reply"]
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"]


@pytest.mark.parametrize(
    ("route", "expected", "generation_calls"),
    [
        ("vision", "I think it is a desk lamp", 0),
        ("memory", "Paris is the capital of France", 0),
        ("clarify", "Could you clarify what you mean?", 0),
        ("no-memory", "LOCAL_MODEL_REPLY", 1),
        ("no-vision", "LOCAL_MODEL_REPLY", 1),
        (
            "no-qgen",
            "I'm not sure what to ask to understand you better. Can you rephrase your response?",
            0,
        ),
    ],
)
async def test_route_outcomes_reach_robot_speech(conversation, route, expected, generation_calls):
    rec = conversation.recorder
    visual = route in {"vision", "no-vision"}
    follow_up = route in {"clarify", "no-qgen"}
    await conversation.run(
        ["Tell me about this"],
        missing_qgen=route == "no-qgen",
        nli_classifier=FakeNLIClassifier(
            rec, routes=nli_routes(visual=visual, follow_up=follow_up)
        ),
        qdict_data=(
            {
                "What is the capital of France?": "Paris is the capital of France",
                "Other": "Other answer",
            }
            if route == "memory"
            else {}
        ),
        embedding_scores=[0.95, 0.2],
        vqa_handler=(lambda _: "No camera configured") if route == "no-vision" else None,
    )

    assert conversation.requests == ["check_for_response", "reply"]
    assert conversation.spoken == [expected]
    assert rec.count("chat_model.generate") == generation_calls
    assert rec.count("answer_visual_question") == int(visual)
    assert rec.count("load_qdict_from_json") == int(route in {"memory", "no-memory"})
    assert rec.count("qgen_model.generate") == int(route == "clarify")
    assert rec.count("openai.ChatCompletion.create") == 0


@pytest.mark.parametrize("visual", [True, False], ids=["vision", "memory"])
async def test_cloud_early_answer_only_rewords_before_speech(conversation, visual):
    rec = conversation.recorder
    await conversation.run(
        ["What am I holding?"],
        local_only=False,
        nli_classifier=FakeNLIClassifier(rec, routes=nli_routes()),
        qdict_data={"What is in my hand?": "A desk lamp", "Other": "Other answer"},
        embedding_scores=[0.95, 0.2],
        openai_sdk=FakeOpenAISDK(
            rec, completion_reply=str(visual), chat_reply="Response: A desk lamp."
        ),
    )

    (call,) = rec.for_target("openai.ChatCompletion.create")
    assert "Please reword the following text" in call.kwargs["messages"][-1]["content"]
    assert rec.count("answer_visual_question") == int(visual)
    assert rec.count("load_qdict_from_json") == int(not visual)
    assert rec.count("chat_model.generate") == 0
    assert conversation.requests == ["check_for_response", "reply"]
    assert conversation.spoken == ["A desk lamp."]


async def test_absent_and_repeated_transcriptions_do_not_generate_or_speak(conversation):
    rec = conversation.recorder
    await conversation.run(
        [
            Poll.NO_NEW_UTTERANCE,
            Poll.EMPTY_RPC_RESULT,
            Poll.NULL_RPC_RESULT,
            "First",
            Poll.NO_NEW_UTTERANCE,
            "First",
            "Second",
        ],
        local_only=False,
        is_question=False,
        nli_classifier=FakeNLIClassifier(rec, routes=nli_routes()),
        openai_sdk=FakeOpenAISDK(rec, chat_reply=lambda kwargs: f"Reply {len(kwargs['messages'])}"),
    )

    assert conversation.requests == [
        "check_for_response",
        "check_for_response",
        "check_for_response",
        "check_for_response",
        "reply",
        "check_for_response",
        "check_for_response",
        "check_for_response",
        "reply",
    ]
    assert conversation.transcriptions == [
        "No transcription",
        "",
        None,
        "First",
        "First",
        "No transcription",
        "Second",
    ]
    calls = rec.for_target("openai.ChatCompletion.create")
    assert [call.kwargs["messages"][-1]["content"] for call in calls] == [
        "User:First",
        "User:Second",
    ]
    assert conversation.spoken == [f"Reply {len(call.kwargs['messages'])}" for call in calls]


@pytest.mark.parametrize("result", [None, "", "\n \t"], ids=["error", "empty", "whitespace"])
async def test_unusable_provider_result_skips_speech_and_next_turn_recovers(
    conversation, result, capsys
):
    rec = conversation.recorder

    def chat_reply(kwargs):
        if kwargs["messages"][-1]["content"] == "User:Hello":
            if result is None:
                raise RuntimeError("fake model failed")
            return result
        return "I can help now."

    await conversation.run(
        ["Hello", "I"],
        local_only=False,
        is_question=False,
        nli_classifier=FakeNLIClassifier(rec, routes=nli_routes()),
        openai_sdk=FakeOpenAISDK(rec, chat_reply=chat_reply),
    )

    assert conversation.requests == ["check_for_response", "check_for_response", "reply"]
    assert conversation.spoken == ["I can help now."]
    assert rec.count("openai.ChatCompletion.create") == 2
    if result is None:
        assert "Could not generate a reply (RuntimeError)" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["before-transmission", "server-rejection"])
async def test_speech_failure_is_reported_and_polling_continues(conversation, capsys, failure):
    conversation.fail_before_reply_transmission = failure == "before-transmission"
    rec = conversation.recorder

    def reply(kwargs):
        if kwargs["messages"][-1]["content"] == "User:Hello":
            # An oversize message is rejected by the server's reply validator.
            return "x" * 10001 if failure == "server-rejection" else "Lost reply"
        return "Second reply"

    await conversation.run(
        ["Hello", "Next"],
        local_only=False,
        is_question=False,
        nli_classifier=FakeNLIClassifier(rec, routes=nli_routes()),
        openai_sdk=FakeOpenAISDK(rec, chat_reply=reply),
    )
    (error,) = conversation.delivery_errors
    expected_code = (
        TransportErrorCode.TIMEOUT
        if failure == "before-transmission"
        else TransportErrorCode.INVALID_ARGUMENTS
    )
    assert error.code == expected_code
    assert conversation.unknown == []
    assert conversation.requests == ["check_for_response", "reply", "check_for_response", "reply"]
    assert conversation.received_replies == conversation.spoken == ["Second reply"]
    assert f"Could not confirm speech delivery ({expected_code})" in capsys.readouterr().out


async def test_identical_replies_to_distinct_turns_both_reach_speech(conversation):
    await conversation.run(
        ["Hello", "Next"],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )
    assert conversation.received_replies == conversation.spoken == ["LOCAL_MODEL_REPLY"] * 2


async def test_non_text_model_result_is_not_spoken(conversation, monkeypatch, capsys):
    real_loader = load_botcortex_runtime

    def load_with_non_text_result(**options):
        cortex, recorder = real_loader(**options)

        async def unusable_model(_message):
            return None

        cortex["talk_medium_with_history"] = unusable_model
        return cortex, recorder

    monkeypatch.setattr(f"{__name__}.load_botcortex_runtime", load_with_non_text_result)
    await conversation.run(["Hello", Poll.NO_NEW_UTTERANCE])
    assert conversation.requests == ["check_for_response"] * 2
    assert conversation.received_replies == conversation.spoken == []
    assert "Could not generate a text reply" in capsys.readouterr().out


async def test_corrupt_question_bank_falls_through_to_generation(conversation, tmp_path, caplog):
    bank = tmp_path / "question_file.json"
    damaged = b'{"private-memory-fragment":'
    bank.write_bytes(damaged)
    rec = conversation.recorder

    await conversation.run(
        ["What is the capital of France?"],
        settings=Settings(data_dir=tmp_path, cache_dir=tmp_path / "cache"),
        use_disk_question_bank=True,
        nli_classifier=FakeNLIClassifier(rec, routes=nli_routes()),
    )

    assert conversation.received_replies == conversation.spoken == ["LOCAL_MODEL_REPLY"]
    assert rec.count("chat_model.generate") == 1
    assert rec.count("get_embeddings") == 0
    assert bank.read_bytes() == damaged
    warnings = [record.getMessage() for record in caplog.records if record.name == "BotCortex"]
    (warning,) = warnings
    assert str(bank) in warning
    assert "malformed-file" in warning
    assert "private-memory-fragment" not in caplog.text


async def test_lost_speech_acknowledgement_is_reported_without_retry(conversation, capsys):
    conversation.drop_reply_ack = True
    await conversation.run(
        ["Hello", Poll.NO_NEW_UTTERANCE],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )

    (unknown,) = conversation.unknown
    assert unknown.command == "reply"
    assert unknown.request_id
    assert conversation.requests == ["check_for_response", "reply", "check_for_response"]
    assert conversation.received_replies == ["LOCAL_MODEL_REPLY"]
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"]
    assert "Will not automatically retry physical speech action" in capsys.readouterr().out
