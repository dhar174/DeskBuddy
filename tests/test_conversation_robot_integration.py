"""Run the desktop conversation loop against authenticated robot loopback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

import buddy1
import robot_client
from deskbuddy.settings import Settings
from tests.support.fakes import CallRecorder, FakeNLIClassifier, FakeOpenAISDK
from tests.support.legacy_runtime import load_botcortex_runtime, load_start_bot_main
from tests.test_conversation_routing_contracts import _nli_routes
from transport import ExecutionOutcomeUnknown, TransportConfig


class PollScriptComplete(Exception):
    """Stop the production loop before it issues an unscripted poll."""


@dataclass
class LoopbackConversation:
    recorder: CallRecorder = field(default_factory=CallRecorder)
    requests: list[str] = field(default_factory=list)
    received_replies: list[str] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)
    unknown: list[ExecutionOutcomeUnknown] = field(default_factory=list)
    drop_reply_ack: bool = False
    poll_result: str | None = "No transcription"

    async def run(
        self,
        turns: list[str | None],
        *,
        local_only: bool = True,
        missing_qgen: bool = False,
        **options: Any,
    ) -> None:
        pending = iter(turns)
        cortex, _ = load_botcortex_runtime(
            recorder=self.recorder, local_only_mode=local_only, **options
        )
        if missing_qgen:
            cortex["qgen_model"] = None

        async def request(**kwargs: Any) -> Any:
            command = kwargs["function_name"]
            if command == "check_for_response":
                try:
                    self.poll_result = next(pending)
                except StopIteration:
                    raise PollScriptComplete from None
            self.requests.append(command)
            try:
                return await robot_client.request_to_server(**kwargs)
            except ExecutionOutcomeUnknown as exc:
                self.unknown.append(exc)
                raise

        main = load_start_bot_main(
            cortex,
            SimpleNamespace(
                request_to_server=request, ExecutionOutcomeUnknown=ExecutionOutcomeUnknown
            ),
            local_only=local_only,
        )
        async with asyncio.timeout(10):
            try:
                await main()
            except PollScriptComplete:
                pass


@pytest_asyncio.fixture
async def conversation(monkeypatch: pytest.MonkeyPatch, ephemeral_server_config: TransportConfig):
    flow = LoopbackConversation()
    real_poll = buddy1.MyFunctions.check_for_response
    real_reply = buddy1.MyFunctions.reply

    async def scripted_poll():
        # The legacy server emits "No transcription"; inject the other two
        # supported empty results through the same authenticated RPC response.
        if flow.poll_result in ("", None):
            return flow.poll_result
        buddy1.lastTranscription = (
            "" if flow.poll_result == "No transcription" else flow.poll_result
        )
        return await real_poll()

    async def no_device_call(*args):
        pass

    async def record_speech(text):
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
    monkeypatch.setattr(buddy1.MyFunctions, "check_for_response", scripted_poll)
    monkeypatch.setattr(buddy1.MyFunctions, "reply", record_reply)
    monkeypatch.setattr(buddy1.MyFunctions, "picoh_say", record_speech)
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
        nli_classifier=FakeNLIClassifier(rec, routes=_nli_routes()),
        openai_sdk=FakeOpenAISDK(rec, chat_reply=f"{transcript}Cloud reply\ncontinues"),
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
        assert conversation.spoken == ["Cloud reply continues"]
    assert conversation.requests == ["check_for_response", "reply"]


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
            rec, routes=_nli_routes(visual=visual, follow_up=follow_up)
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
        nli_classifier=FakeNLIClassifier(rec, routes=_nli_routes()),
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
    assert conversation.spoken == [" A desk lamp."]


async def test_absent_and_repeated_transcriptions_do_not_generate_or_speak(conversation):
    rec = conversation.recorder
    await conversation.run(
        ["No transcription", "", None, "First", "First", "Second"],
        local_only=False,
        is_question=False,
        nli_classifier=FakeNLIClassifier(rec, routes=_nli_routes()),
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
        "reply",
    ]
    calls = rec.for_target("openai.ChatCompletion.create")
    assert [call.kwargs["messages"][-1]["content"] for call in calls] == [
        "User:First",
        "User:Second",
    ]
    assert conversation.spoken == [f"Reply {len(call.kwargs['messages'])}" for call in calls]


@pytest.mark.parametrize("provider_raises", [True, False], ids=["error", "empty"])
async def test_unusable_provider_result_produces_no_speech(conversation, provider_raises):
    rec = conversation.recorder

    def fail(_):
        raise RuntimeError("fake model failed")

    async def run():
        await conversation.run(
            ["Hello"],
            local_only=False,
            is_question=False,
            nli_classifier=FakeNLIClassifier(rec, routes=_nli_routes()),
            openai_sdk=FakeOpenAISDK(rec, chat_reply=fail if provider_raises else ""),
        )

    if provider_raises:
        with pytest.raises(RuntimeError, match="fake model failed"):
            await run()
        assert conversation.requests == ["check_for_response"]
    else:
        await run()
        assert conversation.requests == ["check_for_response", "reply"]
    assert conversation.spoken == []
    assert rec.count("openai.ChatCompletion.create") == 1


async def test_lost_speech_acknowledgement_is_reported_without_retry(conversation, capsys):
    conversation.drop_reply_ack = True
    await conversation.run(
        ["Hello", "No transcription"],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=_nli_routes()),
    )

    (unknown,) = conversation.unknown
    assert unknown.command == "reply"
    assert unknown.request_id
    assert conversation.requests == ["check_for_response", "reply", "check_for_response"]
    assert conversation.received_replies == ["LOCAL_MODEL_REPLY"]
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"]
    assert "Will not automatically retry physical speech action" in capsys.readouterr().out
