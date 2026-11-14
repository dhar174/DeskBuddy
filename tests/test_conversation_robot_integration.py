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
from deskbuddy.settings import Settings, SettingsError
from tests.support.fakes import CallRecorder, FakeNLIClassifier, FakeOpenAISDK, nli_routes
from tests.support.legacy_runtime import load_botcortex_runtime, load_start_bot_main
from transport import (
    ExecutionOutcomeUnknown,
    ProtocolViolationError,
    TransportConfig,
    TransportError,
    TransportErrorCode,
)
from transport.hardware import reset_default_hardware_executor
from transport.transcripts import register_transcript_commands


class PollScriptComplete(BaseException):
    """Stop the production loop before it issues an unscripted poll."""


class Poll(Enum):
    NO_NEW_UTTERANCE = auto()
    EMPTY_RPC_RESULT = auto()
    NULL_RPC_RESULT = auto()


class DropTranscriptResponse(Enum):
    POLL = auto()
    ACK_BEFORE_REMOVAL = auto()
    ACK_AFTER_REMOVAL = auto()
    LEGACY = auto()


@dataclass
class LoopbackConversation:
    recorder: CallRecorder = field(default_factory=CallRecorder)
    requests: list[str] = field(default_factory=list)
    received_replies: list[str] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)
    unknown: list[ExecutionOutcomeUnknown] = field(default_factory=list)
    delivery_errors: list[TransportError] = field(default_factory=list)
    polling_errors: list[Exception] = field(default_factory=list)
    transcriptions: list[str | None] = field(default_factory=list)
    poll_generations: list[int] = field(default_factory=list)
    reply_generations: list[int] = field(default_factory=list)
    server_commands: list[str] = field(default_factory=list)
    server_writers: list[Any] = field(default_factory=list)
    sleep_calls: list[float] = field(default_factory=list)
    startup_calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    close_client_calls: int = 0
    poller_instances: int = 0
    drop_reply_ack: bool = False
    fail_before_reply_transmission: bool = False
    legacy_server: bool = False
    drop_transcript_response: DropTranscriptResponse | None = None
    malformed_poll_responses: int = 0
    malformed_transcript_ack: bool = False
    reject_transcript_ack: bool = False
    poll: str | Poll = Poll.NO_NEW_UTTERANCE
    server: Any = field(default=None, repr=False)

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

        poller: robot_client.TranscriptPoller | None = None

        async def next_transcript() -> str | None:
            try:
                self.poll = next(pending)
            except StopIteration:
                raise PollScriptComplete from None
            if not self.legacy_server and isinstance(self.poll, str):
                await buddy1.MyFunctions.send_response(self.poll)
            self.requests.append("poll")
            if poller is None:
                raise AssertionError("The production loop must create its poller before polling")
            try:
                result = await poller.next_transcript()
            except Exception as exc:
                self.polling_errors.append(exc)
                raise
            finally:
                client = await robot_client.get_robot_client()
                self.poll_generations.append(client.connection_generation)
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
            finally:
                client = await robot_client.get_robot_client()
                self.reply_generations.append(client.connection_generation)

        async def close_client() -> None:
            self.close_client_calls += 1
            await robot_client.close_client()

        def create_poller() -> SimpleNamespace:
            nonlocal poller
            self.poller_instances += 1
            poller = robot_client.TranscriptPoller()
            return SimpleNamespace(next_transcript=next_transcript)

        main = load_start_bot_main(
            cortex,
            SimpleNamespace(
                TranscriptPoller=create_poller,
                send_reply=send_reply,
                close_client=close_client,
                ExecutionOutcomeUnknown=robot_client.ExecutionOutcomeUnknown,
                TransportError=robot_client.TransportError,
                SettingsError=robot_client.SettingsError,
                TRANSCRIPT_MODE_MISMATCH_MESSAGE=robot_client.TRANSCRIPT_MODE_MISMATCH_MESSAGE,
            ),
            local_only=local_only,
            button_text=button_text,
            sleep_calls=self.sleep_calls,
            startup_calls=self.startup_calls,
        )
        async with asyncio.timeout(10):
            try:
                await main()
            except PollScriptComplete:
                pass


@pytest_asyncio.fixture
async def conversation(
    monkeypatch: pytest.MonkeyPatch, ephemeral_server_config: TransportConfig, request
):
    reset_default_hardware_executor()
    try:
        async with conversation_server(
            monkeypatch, ephemeral_server_config, legacy_server=getattr(request, "param", False)
        ) as flow:
            yield flow
    finally:
        reset_default_hardware_executor()


@asynccontextmanager
async def conversation_server(
    monkeypatch: pytest.MonkeyPatch,
    ephemeral_server_config: TransportConfig,
    *,
    legacy_server: bool,
):
    flow = LoopbackConversation(legacy_server=legacy_server)
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
    flow.server = server
    if legacy_server:
        server.registry.unregister("poll_transcript")
        server.registry.unregister("ack_transcript")
    send_response = server._send_response
    process_payload = server._process_request_payload
    pending_drop_request_ids: set[str] = set()
    dropped = False
    transcript_queue = server._transcript_queue
    assert transcript_queue is not None
    original_ack = transcript_queue.ack

    def ack_with_scripted_rejection(transcript_id: object) -> bool:
        if flow.reject_transcript_ack:
            flow.reject_transcript_ack = False
            return False
        return original_ack(transcript_id)

    monkeypatch.setattr(transcript_queue, "ack", ack_with_scripted_rejection)

    async def maybe_drop_transcript(payload):
        from transport.protocol import RobotRequest, RobotResponse

        try:
            request = RobotRequest.from_json(payload)
        except ProtocolViolationError:
            return await process_payload(payload)
        flow.server_commands.append(request.command)
        if flow.malformed_poll_responses and request.command == "poll_transcript":
            flow.malformed_poll_responses -= 1
            response = await process_payload(payload)
            if response.ok:
                response.result = {"id": "invalid", "text": "Unusable transcript"}
            return response
        failure = flow.drop_transcript_response
        if request.command == "poll_transcript" and failure is DropTranscriptResponse.POLL:
            flow.drop_transcript_response = None
            pending_drop_request_ids.add(request.request_id)
        if request.command == "check_for_response" and failure is DropTranscriptResponse.LEGACY:
            flow.drop_transcript_response = None
            pending_drop_request_ids.add(request.request_id)
        if request.command == "ack_transcript":
            if failure is DropTranscriptResponse.ACK_BEFORE_REMOVAL:
                flow.drop_transcript_response = None
                pending_drop_request_ids.add(request.request_id)
                # Model transport loss before the acknowledgement reaches dispatch.
                return RobotResponse.success(request.request_id)
            if failure is DropTranscriptResponse.ACK_AFTER_REMOVAL:
                flow.drop_transcript_response = None
                pending_drop_request_ids.add(request.request_id)
        response = await process_payload(payload)
        if request.command == "ack_transcript" and response.ok:
            if flow.malformed_transcript_ack:
                flow.malformed_transcript_ack = False
                response.result = 1  # A truthy integer is not a confirmed acknowledgement.
        return response

    async def maybe_drop_ack(writer, response):
        nonlocal dropped
        if not flow.server_writers or flow.server_writers[-1] is not writer:
            flow.server_writers.append(writer)
        if response.request_id in pending_drop_request_ids:
            pending_drop_request_ids.remove(response.request_id)
            writer.close()
            await writer.wait_closed()
        elif flow.drop_reply_ack and flow.spoken and not dropped:
            dropped = True
            writer.close()
            await writer.wait_closed()
        else:
            await send_response(writer, response)

    monkeypatch.setattr(server, "_send_response", maybe_drop_ack)
    monkeypatch.setattr(server, "_process_request_payload", maybe_drop_transcript)
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
    assert conversation.requests == ["poll", "reply"]
    assert conversation.close_client_calls == 1
    assert conversation.poller_instances == 1
    assert conversation.sleep_calls == []


async def test_desktop_poll_and_speech_use_connect_budget_but_ack_keeps_default_timeout(
    conversation, monkeypatch
):
    client = await robot_client.get_robot_client()
    real_request = client.request
    observed_requests: list[tuple[str, dict[str, object]]] = []

    async def record_request(command, *args, **kwargs):
        observed_requests.append((command, dict(kwargs)))
        return await real_request(command, *args, **kwargs)

    monkeypatch.setattr(client, "request", record_request)
    await conversation.run(
        ["Hello"],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )

    assert [command for command, _ in observed_requests] == [
        "poll_transcript",
        "ack_transcript",
        "reply",
    ]
    poll_options = observed_requests[0][1]
    ack_options = observed_requests[1][1]
    reply_options = observed_requests[2][1]
    assert poll_options["connect_budget"] == client.config.connect_timeout
    assert ack_options["expected_connection_generation"] == 1
    assert "timeout" not in ack_options
    assert reply_options["connect_budget"] == client.config.connect_timeout
    assert reply_options["side_effects"] is True
    assert conversation.reply_generations == [1]


async def test_idle_poll_sleeps_only_after_an_empty_result(conversation):
    await conversation.run(
        [Poll.NO_NEW_UTTERANCE],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )

    assert conversation.requests == ["poll"]
    assert conversation.sleep_calls == [0.1]
    assert conversation.close_client_calls == 1


async def test_desktop_close_before_selection_skips_startup_and_polling(conversation):
    await conversation.run([], button_text="")

    assert conversation.startup_calls == []
    assert conversation.poller_instances == 0
    assert conversation.requests == []
    assert conversation.sleep_calls == []
    assert conversation.close_client_calls == 1


async def test_slow_model_turn_refreshes_stale_socket_before_speech(conversation, monkeypatch):
    client = await robot_client.get_robot_client()
    real_loader = load_botcortex_runtime
    acknowledged_request_starts: list[float] = []

    def load_with_slow_model(**options):
        cortex, recorder = real_loader(**options)
        model = cortex["model"]
        generate = model.generate

        def generate_after_slow_turn(*args, **kwargs):
            result = generate(*args, **kwargs)
            request_started_at = client._last_request_started_at
            assert request_started_at is not None
            acknowledged_request_starts.append(request_started_at)
            client._last_request_started_at = (
                asyncio.get_running_loop().time() - 2 * client.config.read_timeout
            )
            return result

        model.generate = generate_after_slow_turn
        return cortex, recorder

    monkeypatch.setattr(f"{__name__}.load_botcortex_runtime", load_with_slow_model)
    await conversation.run(
        ["Hello"],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )

    assert acknowledged_request_starts
    assert conversation.poll_generations == [1]
    assert conversation.reply_generations == [2]
    assert conversation.received_replies == conversation.spoken == ["LOCAL_MODEL_REPLY"]
    assert conversation.requests == ["poll", "reply"]
    assert conversation.recorder.count("chat_model.generate") == 1


async def test_slow_model_turn_reuses_socket_below_refresh_threshold(conversation, monkeypatch):
    client = await robot_client.get_robot_client()
    real_loader = load_botcortex_runtime
    acknowledged_request_starts: list[float] = []

    def load_with_nearly_stale_model(**options):
        cortex, recorder = real_loader(**options)
        model = cortex["model"]
        generate = model.generate

        def generate_before_refresh_threshold(*args, **kwargs):
            result = generate(*args, **kwargs)
            request_started_at = client._last_request_started_at
            assert request_started_at is not None
            acknowledged_request_starts.append(request_started_at)
            refresh_after = client.config.read_timeout - min(1.0, client.config.read_timeout * 0.2)
            client._last_request_started_at = asyncio.get_running_loop().time() - (
                refresh_after - 0.25
            )
            return result

        model.generate = generate_before_refresh_threshold
        return cortex, recorder

    monkeypatch.setattr(f"{__name__}.load_botcortex_runtime", load_with_nearly_stale_model)
    await conversation.run(
        ["Hello"],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )

    assert acknowledged_request_starts
    assert conversation.poll_generations == [1]
    assert conversation.reply_generations == [1]
    assert conversation.received_replies == conversation.spoken == ["LOCAL_MODEL_REPLY"]


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
    assert conversation.requests == ["poll", "reply"]
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

    assert conversation.requests == ["poll", "reply"]
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
    assert conversation.requests == ["poll", "reply"]
    assert conversation.spoken == ["A desk lamp."]


@pytest.mark.parametrize("conversation", [True], indirect=True, ids=["legacy-server"])
async def test_legacy_text_polling_retains_deduplication(conversation, caplog):
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
        "poll",
        "poll",
        "poll",
        "poll",
        "reply",
        "poll",
        "poll",
        "poll",
        "reply",
    ]
    assert conversation.transcriptions == [None, None, None, "First", None, None, "Second"]
    calls = rec.for_target("openai.ChatCompletion.create")
    assert [call.kwargs["messages"][-1]["content"] for call in calls] == [
        "User:First",
        "User:Second",
    ]
    assert conversation.spoken == [f"Reply {len(call.kwargs['messages'])}" for call in calls]
    assert caplog.text.count("Robot server lacks utterance IDs") == 1


async def test_identical_utterances_already_queued_produce_two_turns(conversation):
    await buddy1.MyFunctions.send_response("Yes")
    await buddy1.MyFunctions.send_response("Yes")
    await conversation.run(
        [Poll.NO_NEW_UTTERANCE] * 3,
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )
    assert conversation.transcriptions == ["Yes", "Yes", None]
    assert conversation.recorder.count("chat_model.generate") == 2
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"] * 2


@pytest.mark.parametrize(
    ("failure", "expected_texts"),
    [
        (DropTranscriptResponse.POLL, ["First", "Second"]),
        (DropTranscriptResponse.ACK_BEFORE_REMOVAL, ["First", "Second"]),
        (DropTranscriptResponse.ACK_AFTER_REMOVAL, ["Second"]),
    ],
)
async def test_transcript_response_loss_and_reconnect(
    conversation, failure, expected_texts, capsys
):
    await buddy1.MyFunctions.send_response("First")
    await buddy1.MyFunctions.send_response("Second")
    conversation.drop_transcript_response = failure
    await conversation.run(
        [Poll.NO_NEW_UTTERANCE] * 4,
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )
    assert [text for text in conversation.transcriptions if text is not None] == expected_texts
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"] * len(expected_texts)
    assert conversation.recorder.count("chat_model.generate") == len(expected_texts)
    output = capsys.readouterr().out.lower()
    assert len(conversation.polling_errors) == 1
    assert "transcript polling failed (" in output
    if failure is DropTranscriptResponse.POLL:
        assert "acknowledgement outcome is unknown" not in output
    else:
        assert isinstance(conversation.polling_errors[0], ExecutionOutcomeUnknown)
        assert "acknowledgement outcome is unknown" in output


async def test_desktop_restart_does_not_replay_accepted_transcript(conversation):
    await buddy1.MyFunctions.send_response("Accepted just before desktop exit")
    await buddy1.MyFunctions.send_response("Still pending")
    assert (
        await robot_client.TranscriptPoller().next_transcript()
        == "Accepted just before desktop exit"
    )
    await robot_client.close_client()

    await conversation.run(
        [Poll.NO_NEW_UTTERANCE] * 2,
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )
    assert conversation.transcriptions == ["Still pending", None]
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"]


async def test_malformed_authenticated_poll_response_does_not_enable_legacy(
    conversation, monkeypatch, caplog, capsys
):
    async def unexpected_legacy_poll():
        pytest.fail("Legacy polling must not run after an invalid response")

    monkeypatch.setattr(buddy1.MyFunctions, "check_for_response", unexpected_legacy_poll)
    conversation.malformed_poll_responses = 1
    await conversation.run(
        ["First", Poll.NO_NEW_UTTERANCE],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )
    assert conversation.requests == ["poll", "poll", "reply"]
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"]
    assert len(conversation.polling_errors) == 1
    assert isinstance(conversation.polling_errors[0], ProtocolViolationError)
    assert conversation.server_commands == [
        "poll_transcript",
        "poll_transcript",
        "ack_transcript",
        "reply",
    ]
    assert "transcript polling failed (malformed_request)" in capsys.readouterr().out.lower()
    assert "using legacy text polling" not in caplog.text


async def test_repeated_poll_faults_back_off_once_per_reason_then_resume(conversation, capsys):
    await buddy1.MyFunctions.send_response("Eventually delivered")
    conversation.malformed_poll_responses = 2
    await conversation.run(
        [Poll.NO_NEW_UTTERANCE] * 3,
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )

    assert len(conversation.polling_errors) == 2
    assert all(isinstance(error, ProtocolViolationError) for error in conversation.polling_errors)
    assert conversation.transcriptions == ["Eventually delivered"]
    assert conversation.requests == ["poll", "poll", "poll", "reply"]
    assert conversation.sleep_calls == [0.1, 0.2]
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"]
    output = capsys.readouterr().out.lower()
    assert output.count("transcript polling failed (malformed_request)") == 1


async def test_startup_loop_handles_settings_error_caps_backoff_and_resets(capsys):
    outcomes: list[str | None | BaseException] = [
        *[SettingsError("Injected settings fault") for _ in range(6)],
        SettingsError("Different settings fault"),
        None,
        SettingsError("Injected settings fault after recovery"),
        PollScriptComplete(),
    ]
    sleep_calls: list[float] = []
    startup_calls: list[tuple[str, tuple[Any, ...]]] = []
    close_calls = 0

    async def next_transcript() -> str | None:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close_client() -> None:
        nonlocal close_calls
        close_calls += 1

    async def unused_reply(_message: str) -> None:
        pytest.fail("The scripted polls must not generate replies")

    main = load_start_bot_main(
        {},
        SimpleNamespace(
            TranscriptPoller=lambda: SimpleNamespace(next_transcript=next_transcript),
            close_client=close_client,
            send_reply=unused_reply,
            ExecutionOutcomeUnknown=robot_client.ExecutionOutcomeUnknown,
            TransportError=robot_client.TransportError,
            SettingsError=robot_client.SettingsError,
            TRANSCRIPT_MODE_MISMATCH_MESSAGE=robot_client.TRANSCRIPT_MODE_MISMATCH_MESSAGE,
        ),
        local_only=True,
        sleep_calls=sleep_calls,
        startup_calls=startup_calls,
    )
    with pytest.raises(PollScriptComplete):
        await main()

    assert sleep_calls == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 5.0, 0.1, 0.1])
    assert startup_calls == [("cortex", ("Medium", True)), ("helpers", ())]
    assert close_calls == 1
    output = capsys.readouterr().out.lower()
    assert output.count("transcript polling failed (settingserror)") == 3
    assert "different settings fault" in output
    assert "injected settings fault" in output
    assert "injected settings fault after recovery" in output
    assert "transcript acknowledgement outcome is unknown" not in output


async def test_wrong_secret_does_not_enable_legacy(conversation, monkeypatch, caplog):
    await robot_client.close_client()
    monkeypatch.setenv("DESKBUDDY_ROBOT_AUTH_SECRET", "wrong-test-secret-not-used-by-server")
    await conversation.run([Poll.NO_NEW_UTTERANCE])
    assert conversation.requests == ["poll"]
    assert not conversation.spoken
    assert "using legacy text polling" not in caplog.text


async def test_false_transcript_ack_keeps_pending_text_for_next_poll(conversation):
    await buddy1.MyFunctions.send_response("First")
    conversation.reject_transcript_ack = True
    await conversation.run(
        [Poll.NO_NEW_UTTERANCE, Poll.NO_NEW_UTTERANCE],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )

    assert conversation.transcriptions == [None, "First"]
    assert conversation.requests == ["poll", "poll", "reply"]
    assert conversation.server_commands == [
        "poll_transcript",
        "ack_transcript",
        "poll_transcript",
        "ack_transcript",
        "reply",
    ]
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"]
    assert conversation.polling_errors == []
    assert conversation.sleep_calls == [0.1]


async def test_malformed_ack_is_logged_and_following_transcript_resumes(conversation, capsys):
    await buddy1.MyFunctions.send_response("Claimed but not confirmed")
    await buddy1.MyFunctions.send_response("Next accepted transcript")
    conversation.malformed_transcript_ack = True
    await conversation.run(
        [Poll.NO_NEW_UTTERANCE, Poll.NO_NEW_UTTERANCE],
        is_question=False,
        nli_classifier=FakeNLIClassifier(conversation.recorder, routes=nli_routes()),
    )

    assert conversation.transcriptions == ["Next accepted transcript"]
    assert conversation.requests == ["poll", "poll", "reply"]
    assert len(conversation.polling_errors) == 1
    assert isinstance(conversation.polling_errors[0], ProtocolViolationError)
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"]
    assert conversation.recorder.count("chat_model.generate") == 1
    assert "transcript polling failed (malformed_request)" in capsys.readouterr().out.lower()
    assert conversation.sleep_calls == [0.1]


async def test_legacy_connection_is_reprobed_after_disconnect(
    monkeypatch, ephemeral_server_config, caplog
):
    async with conversation_server(
        monkeypatch, ephemeral_server_config, legacy_server=True
    ) as conversation:
        poller = robot_client.TranscriptPoller()
        conversation.poll = "Legacy turn"
        assert await poller.next_transcript() == "Legacy turn"
        client = await robot_client.get_robot_client()
        await client.close()
        # Model a second recognition of the exact same text after the new
        # connection has reported that the server still lacks transcript IDs.
        buddy1.lastTranscription = ""
        buddy1.priorTranscription = ""
        conversation.poll = "Legacy turn"
        assert await poller.next_transcript() == "Legacy turn"
        assert caplog.text.count("Robot server lacks utterance IDs") == 2


async def test_legacy_fallback_does_not_reconnect_after_probe_disconnect(
    monkeypatch, ephemeral_server_config
):
    async with conversation_server(
        monkeypatch, ephemeral_server_config, legacy_server=True
    ) as conversation:
        conversation.poll = "Legacy turn"
        client = await robot_client.get_robot_client()
        poller = robot_client.TranscriptPoller()
        real_request = client.request
        disconnect_after_probe = True

        async def disconnect_before_legacy_poll(command, *args, **kwargs):
            nonlocal disconnect_after_probe
            response = await real_request(command, *args, **kwargs)
            if command == "poll_transcript" and disconnect_after_probe:
                disconnect_after_probe = False
                await client.close()
            return response

        monkeypatch.setattr(client, "request", disconnect_before_legacy_poll)
        with pytest.raises(TransportError) as failure:
            await poller.next_transcript()
        assert failure.value.code == TransportErrorCode.INTERNAL_ERROR
        assert client.connection_generation == 1
        assert conversation.server_commands == ["poll_transcript"]

        assert await poller.next_transcript() == "Legacy turn"
        assert client.connection_generation == 2
        assert conversation.server_commands == [
            "poll_transcript",
            "poll_transcript",
            "check_for_response",
        ]


async def test_legacy_poll_disconnect_after_transmission_is_not_retried(
    monkeypatch, ephemeral_server_config
):
    async with conversation_server(
        monkeypatch, ephemeral_server_config, legacy_server=True
    ) as conversation:
        conversation.poll = "Legacy turn"
        conversation.drop_transcript_response = DropTranscriptResponse.LEGACY
        client = await robot_client.get_robot_client()
        poller = robot_client.TranscriptPoller()

        with pytest.raises(TransportError) as failure:
            await poller.next_transcript()

        assert failure.value.code == TransportErrorCode.INTERNAL_ERROR
        assert client.connection_generation == 1
        assert conversation.server_commands == ["poll_transcript", "check_for_response"]

        buddy1.lastTranscription = ""
        buddy1.priorTranscription = ""
        conversation.poll = "Legacy turn"
        assert await poller.next_transcript() == "Legacy turn"
        assert client.connection_generation == 2
        assert conversation.server_commands == [
            "poll_transcript",
            "check_for_response",
            "poll_transcript",
            "check_for_response",
        ]


async def test_legacy_disconnect_then_server_upgrade_uses_new_generation(
    monkeypatch, ephemeral_server_config, caplog
):
    async with conversation_server(
        monkeypatch, ephemeral_server_config, legacy_server=True
    ) as conversation:
        poller = robot_client.TranscriptPoller()
        conversation.poll = "Legacy turn"
        assert await poller.next_transcript() == "Legacy turn"

        client = await robot_client.get_robot_client()
        legacy_generation = client.connection_generation
        legacy_reader = client._reader
        assert legacy_reader is not None
        server_writer = conversation.server_writers[-1]
        server_writer.close()
        await server_writer.wait_closed()
        assert await asyncio.wait_for(legacy_reader.read(), timeout=2) == b""
        assert not client.is_connected

        transcript_queue = conversation.server._transcript_queue
        assert transcript_queue is not None
        transcript_queue.reset(
            max_payload_size=ephemeral_server_config.max_payload_size,
            secret=ephemeral_server_config.secret,
        )
        register_transcript_commands(conversation.server.registry, transcript_queue)
        await buddy1.MyFunctions.send_response("Upgraded turn")

        assert await poller.next_transcript() == "Upgraded turn"
        assert client.connection_generation == legacy_generation + 1
        assert conversation.server_commands[-4:] == [
            "poll_transcript",
            "check_for_response",
            "poll_transcript",
            "ack_transcript",
        ]
        assert caplog.text.count("Robot server lacks utterance IDs") == 1


async def test_ack_rejects_a_different_active_connection_generation(
    monkeypatch, ephemeral_server_config
):
    async with conversation_server(
        monkeypatch, ephemeral_server_config, legacy_server=False
    ) as conversation:
        await buddy1.MyFunctions.send_response("Still pending after stale ACK socket")
        client = await robot_client.get_robot_client()
        poller = robot_client.TranscriptPoller()
        real_request = client.request
        reconnect_before_ack = True

        async def reconnect_before_pinned_ack(command, *args, **kwargs):
            nonlocal reconnect_before_ack
            if command == "ack_transcript" and reconnect_before_ack:
                reconnect_before_ack = False
                await client.close()
                await real_request("ping")
            return await real_request(command, *args, **kwargs)

        monkeypatch.setattr(client, "request", reconnect_before_pinned_ack)
        with pytest.raises(TransportError) as failure:
            await poller.next_transcript()
        assert failure.value.code == TransportErrorCode.INTERNAL_ERROR
        assert client.connection_generation == 2
        assert conversation.server_commands == ["poll_transcript", "ping"]

        assert await poller.next_transcript() == "Still pending after stale ACK socket"
        assert client.connection_generation == 2
        assert conversation.server_commands == [
            "poll_transcript",
            "ping",
            "poll_transcript",
            "ack_transcript",
        ]


async def test_transcript_ack_lost_after_dispatch_is_not_retried_when_side_effects_false(
    monkeypatch, ephemeral_server_config
):
    async with conversation_server(
        monkeypatch, ephemeral_server_config, legacy_server=False
    ) as conversation:
        await buddy1.MyFunctions.send_response("Claimed exactly once")
        client = await robot_client.get_robot_client()
        polled = await client.request("poll_transcript")
        assert polled.ok and isinstance(polled.result, dict)
        transcript_id = polled.result["id"]
        conversation.drop_transcript_response = DropTranscriptResponse.ACK_AFTER_REMOVAL

        with pytest.raises(ExecutionOutcomeUnknown) as failure:
            await client.request("ack_transcript", {"id": transcript_id}, side_effects=False)

        assert failure.value.command == "ack_transcript"
        assert conversation.server_commands == ["poll_transcript", "ack_transcript"]
        assert conversation.server._transcript_queue.poll() is None


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

    assert conversation.requests == ["poll", "poll", "reply"]
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
    assert conversation.requests == ["poll", "reply", "poll", "reply"]
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
    assert conversation.requests == ["poll"] * 2
    assert conversation.received_replies == conversation.spoken == []
    assert "Could not generate a text reply" in capsys.readouterr().out


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
    assert conversation.requests == ["poll", "reply", "poll"]
    assert conversation.received_replies == ["LOCAL_MODEL_REPLY"]
    assert conversation.spoken == ["LOCAL_MODEL_REPLY"]
    assert "Will not automatically retry physical speech action" in capsys.readouterr().out
