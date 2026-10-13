"""Integration tests exercising real robot server and client communication flows."""

import pytest

import buddy1
import robot_client as rc
from transport import (
    RobotClient,
    TransportConfig,
)


@pytest.mark.asyncio
async def test_integration_transcription_polling_and_speech_reply(ephemeral_server_config: TransportConfig):
    # Set simulated transcription and reply messages in buddy1 module
    buddy1.lastTranscription = "Hello DeskBuddy, what is the weather?"
    buddy1.priorTranscription = ""
    spoken_phrases: list[str] = []

    # Intercept speech output
    original_picoh_say = buddy1.MyFunctions.picoh_say

    async def mock_picoh_say(text):
        spoken_phrases.append(str(text))

    buddy1.MyFunctions.picoh_say = mock_picoh_say

    server = buddy1.create_robot_server(ephemeral_server_config)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=3.0,
        )

        async with RobotClient(client_cfg) as client:
            # 1. Connectivity check
            is_alive = await client.ping()
            assert is_alive is True

            # 2. Transcription polling flow (as start_bot.py performs)
            transcription = await client.check_for_response()
            assert transcription == "Hello DeskBuddy, what is the weather?"

            # Polling again with no new speech returns "No transcription"
            buddy1.priorTranscription = buddy1.lastTranscription
            no_transcription = await client.check_for_response()
            assert no_transcription == "No transcription"

            # 3. Speech response flow (as start_bot.py performs)
            await client.reply("It is bright and sunny outside!")
            assert "It is bright and sunny outside!" in spoken_phrases

            # 4. LED color setting flow
            await client.base_col(5, 5, 5)

            # 5. Servo move flow (within safety range [0, 10])
            await client.move(motor=1, position=6.5)

    finally:
        buddy1.MyFunctions.picoh_say = original_picoh_say
        await server.stop()


@pytest.mark.asyncio
async def test_integration_legacy_request_to_server_compatibility(
    ephemeral_server_config: TransportConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test start_bot.py calling convention: rc.request_to_server(**kwargs)."""
    buddy1.lastTranscription = "User voice query"
    buddy1.priorTranscription = ""
    spoken_messages: list[str] = []

    original_picoh_say = buddy1.MyFunctions.picoh_say

    async def mock_picoh_say(text):
        spoken_messages.append(str(text))

    buddy1.MyFunctions.picoh_say = mock_picoh_say

    server = buddy1.create_robot_server(ephemeral_server_config)
    await server.start()

    # Configure environment via monkeypatch so changes are restored on teardown
    monkeypatch.setenv("DESKBUDDY_ROBOT_HOST", "127.0.0.1")
    monkeypatch.setenv("DESKBUDDY_ROBOT_PORT", str(server.port))
    monkeypatch.setenv("DESKBUDDY_ROBOT_AUTH_SECRET", ephemeral_server_config.secret)

    try:
        # Reset shared client to pick up environment variables
        await rc.close_client()

        # Check for transcription via legacy kwargs
        kwargs_check = {"function_name": "check_for_response"}
        result_transcription = await rc.request_to_server(**kwargs_check)
        assert result_transcription == "User voice query"

        # Send speech reply via legacy kwargs
        kwargs_reply = {"function_name": "reply", "message": "Picoh speaks this answer"}
        await rc.request_to_server(**kwargs_reply)
        assert "Picoh speaks this answer" in spoken_messages

    finally:
        await rc.close_client()
        buddy1.MyFunctions.picoh_say = original_picoh_say
        await server.stop()


@pytest.mark.asyncio
async def test_windows_buddy1_server_integration(ephemeral_server_config: TransportConfig):
    from transport.client import RobotClient
    from transport.protocol import TransportError, TransportErrorCode
    from windows import buddy1 as win_buddy1

    win_buddy1.lastTranscription = "Transcription from Windows buddy1"
    win_buddy1.priorTranscription = ""

    server = win_buddy1.create_robot_server(ephemeral_server_config)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=3.0,
        )
        async with RobotClient(client_cfg) as client:
            # 1. Ping
            assert await client.ping() is True

            # 2. Transcription polling
            transcription = await client.check_for_response()
            assert transcription == "Transcription from Windows buddy1"

            # 3. Speech reply
            await client.reply("Hello from Windows buddy1 test")

            # 4. LED color
            await client.base_col(1, 2, 3)

            # 5. Servo movements (int, named, signed string, whitespace-padded string)
            await client.move(motor=0, position=5.0)
            await client.move(motor="HEADNOD", position=6.0)
            await client.move(motor="+1", position=4.5)
            await client.move(motor=" 2 ", position=5.5)

            # 6. Out-of-bounds rejection (motor index >= 7, position > 10.0, channel > 10)
            with pytest.raises(TransportError) as exc_info:
                await client.move(motor=7, position=5.0)
            assert exc_info.value.code == TransportErrorCode.INVALID_ARGUMENTS

            with pytest.raises(TransportError) as exc_info:
                await client.move(motor=0, position=12.0)
            assert exc_info.value.code == TransportErrorCode.INVALID_ARGUMENTS

            with pytest.raises(TransportError) as exc_info:
                await client.base_col(11, 0, 0)
            assert exc_info.value.code == TransportErrorCode.INVALID_ARGUMENTS

            # 7. Non-blocking wait
            await client.wait(0.05)
    finally:
        win_buddy1.lastTranscription = ""
        win_buddy1.priorTranscription = ""
        await server.stop()


@pytest.mark.asyncio
async def test_windows_buddy2_server_integration(ephemeral_server_config: TransportConfig):
    from transport.client import RobotClient
    from transport.protocol import TransportError, TransportErrorCode
    from windows import buddy2 as win_buddy2

    win_buddy2.lastTranscription = "Transcription from Windows buddy2"
    win_buddy2.priorTranscription = ""

    server = win_buddy2.create_robot_server(ephemeral_server_config)
    await server.start()

    try:
        client_cfg = TransportConfig(
            host="127.0.0.1",
            port=server.port,
            secret=ephemeral_server_config.secret,
            request_timeout=3.0,
        )
        async with RobotClient(client_cfg) as client:
            # 1. Ping
            assert await client.ping() is True

            # 2. Transcription polling
            transcription = await client.check_for_response()
            assert transcription == "Transcription from Windows buddy2"

            # 3. Speech reply
            await client.reply("Hello from Windows buddy2 test")

            # 4. LED color
            await client.base_col(4, 5, 6)

            # 5. Servo movements (int, named, signed string, whitespace-padded string)
            await client.move(motor=0, position=5.0)
            await client.move(motor="HEADNOD", position=6.0)
            await client.move(motor="+1", position=4.5)
            await client.move(motor=" 2 ", position=5.5)

            # 6. Out-of-bounds rejection
            with pytest.raises(TransportError) as exc_info:
                await client.move(motor=8, position=5.0)
            assert exc_info.value.code == TransportErrorCode.INVALID_ARGUMENTS

            with pytest.raises(TransportError) as exc_info:
                await client.move(motor=0, position=-1.0)
            assert exc_info.value.code == TransportErrorCode.INVALID_ARGUMENTS

            with pytest.raises(TransportError) as exc_info:
                await client.base_col(0, 15, 0)
            assert exc_info.value.code == TransportErrorCode.INVALID_ARGUMENTS

            # 7. Non-blocking wait
            await client.wait(0.05)
    finally:
        win_buddy2.lastTranscription = ""
        win_buddy2.priorTranscription = ""
        await server.stop()


@pytest.mark.asyncio
async def test_request_to_server_custom_timeout_and_wait_scaling(
    ephemeral_server_config: TransportConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify request_to_server extracts timeout cleanly without corrupting command arguments."""
    server = buddy1.create_robot_server(ephemeral_server_config)
    await server.start()

    monkeypatch.setenv("DESKBUDDY_ROBOT_HOST", "127.0.0.1")
    monkeypatch.setenv("DESKBUDDY_ROBOT_PORT", str(server.port))
    monkeypatch.setenv("DESKBUDDY_ROBOT_AUTH_SECRET", ephemeral_server_config.secret)

    try:
        await rc.close_client()

        # Sending timeout=5.0 with reply must succeed and not trigger invalid_arguments
        res = await rc.request_to_server(function_name="reply", message="Testing timeout param", timeout=5.0)
        assert res is None  # reply returns None on success

        # Wait command via request_to_server
        res_wait = await rc.request_to_server(function_name="wait", seconds=0.1)
        assert res_wait is None
    finally:
        await rc.close_client()
        await server.stop()
