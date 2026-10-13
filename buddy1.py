
import asyncio
import concurrent.futures
import os
import socket
import tkinter as tk
import threading
from random import *
import time
from sys import byteorder
from array import array
from struct import pack
import signal
import sys
from deskbuddy.settings import load_settings
from typing import Any

try:
    from scipy.io.wavfile import write
except ImportError:
    write = None

try:
    import keyboard
except ImportError:
    keyboard = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    import pyaudio
    import wave
except ImportError:
    pyaudio = None
    wave = None

_ALLOW_MOCK = (
    os.getenv("DESKBUDDY_MOCK_HARDWARE", "").lower() in ("1", "true", "yes")
    or os.getenv("DESKBUDDY_HEADLESS", "").lower() in ("1", "true", "yes")
    or "PYTEST_CURRENT_TEST" in os.environ
    or "pytest" in sys.modules
)

_USING_MOCK_PICOH = False

try:
    from picoh import picoh
except ImportError as exc:
    if not _ALLOW_MOCK:
        raise RuntimeError(
            "Picoh robot hardware module could not be imported. "
            "To run in headless or mock mode for testing, set DESKBUDDY_MOCK_HARDWARE=1 or DESKBUDDY_HEADLESS=1."
        ) from exc
    _USING_MOCK_PICOH = True

    class MockPicoh:
        HEADNOD = 0
        HEADTURN = 1
        EYETURN = 2
        LIDBLINK = 3
        TOPLIP = 4
        BOTTOMLIP = 5
        EYETILT = 6

        def reset(self): pass
        def wait(self, s): pass
        def setEyeShape(self, shape): pass
        def setEyeBrightness(self, val): pass
        def say(self, text, untilDone=True): pass
        def move(self, motor, pos): pass
        def setBaseColour(self, r, g, b): pass
        def get_input_path(self): return "input.wav"

    picoh = MockPicoh()

from transport.hardware import (
    get_default_hardware_executor,
    run_hardware_sync,
)

from transport import (
    RobotServer,
    TransportConfig,
    CommandRegistry,
    validate_empty_arguments,
    validate_reply_arguments,
    validate_base_col_arguments,
    validate_say_arguments,
    validate_move_arguments,
    validate_wait_arguments,
)

# State tracking globals
lastReplyMessage = ""
priorReplyMessage = ""
lastTranscription = ""
priorTranscription = ""


class MyFunctions:
    _public_methods_ = ['simple_test', 'button_clicked', "send_response", "listen", "baseCol", "randomNod",
                        "randomTurn", "randomLook", "blinkLids", "record_to_file", "record", "add_silence", "trim", "normalize", "is_silent", "button_clicked"]  # COM exposed methods

    async def simple_test():
        print("Hello")
        return "Hello"

    async def check_for_response():
        global lastTranscription
        global priorTranscription
        print("Checking for response")

        if lastTranscription != "" and lastTranscription != priorTranscription:
            print(priorTranscription)
            print(lastTranscription)
            return lastTranscription
        else:

            return "No transcription"

    async def blinkLids():
        global blinking

        # While True - Loop forever.
        while True:
            for i in range(10):
                await asyncio.sleep(1)
            if blinking:
                # for the numbers 10 to 0 set the lidblink position.
                for x in range(10, 0, -1):
                    await run_hardware_sync(picoh.move, getattr(picoh, "LIDBLINK", 3), x)
                    await asyncio.sleep(0.01)

                # for the numbers 0 to 10 set the lidblink position.
                for x in range(0, 10):
                    await run_hardware_sync(picoh.move, getattr(picoh, "LIDBLINK", 3), x)
                    await asyncio.sleep(0.01)

                # wait for a random amount of time for realistic blinking
                await asyncio.sleep(random() * 6)
            await asyncio.sleep(1)

    async def randomLook():
        global moving
        while True:
            for i in range(10):
                await asyncio.sleep(1)
            # if moving is True.
            if moving:
                # Look in a random direction.
                await run_hardware_sync(picoh.move, getattr(picoh, "EYETILT", 6), randint(2, 8))
                await run_hardware_sync(picoh.move, getattr(picoh, "EYETURN", 2), randint(2, 8))

                # Wait for between 0 and 5 seconds.
                await asyncio.sleep(random() * 5)
            await asyncio.sleep(1)

    async def randomTurn():
        global moving
        while True:
            for i in range(10):
                await asyncio.sleep(1)
            if moving:
                # Move Picoh's HEADTURN motor to a random position between 3 and 7.
                await run_hardware_sync(picoh.move, getattr(picoh, "HEADTURN", 1), randint(3, 7))

                # wait for a random amount of time before moving again.
                await asyncio.sleep(random() * 4)
            await asyncio.sleep(1)

    async def randomNod():
        global moving
        while True:
            for i in range(10):
                await asyncio.sleep(1)
            if moving:

                # Move Picoh's HEADNOD motor to a random position between 4 and 7.
                await run_hardware_sync(picoh.move, getattr(picoh, "HEADNOD", 0), randint(4, 7))

                # wait for a random amount of time before moving again.
                await asyncio.sleep(random() * 4)
            await asyncio.sleep(1)

    async def baseCol(r, g, b):
        # Set the base to rgb values between 0 and 10.
        await run_hardware_sync(picoh.setBaseColour, r, g, b)
        return

    async def picoh_say(text):
        print("in picoh_say")
        print(str(text))
        await run_hardware_sync(picoh.say, str(text))
        return

    async def picoh_move(motor, position):
        await run_hardware_sync(picoh.move, motor, position)
        return

    async def picoh_wait(seconds):
        await asyncio.sleep(float(seconds))
        return

    async def picoh_get_input_path():
        return picoh.get_input_path()
    THRESHOLD = 500

    async def send_response(text):

        global recording
        global recorded
        global result
        global lastTranscription
        global priorTranscription
        global lastReplyMessage
        global priorReplyMessage
        print("in send_response")
        recorded = False
        recording = False
        priorTranscription = lastTranscription
        lastTranscription = text
        result = None
        return

    async def reply(text):
        global lastReplyMessage
        global priorReplyMessage
        print("Replying: " + text)
        if text != lastReplyMessage and text != "" and text is not None:
            print(text)
            priorReplyMessage = lastReplyMessage
            lastReplyMessage = text
            await MyFunctions.baseCol(8, 0, 0)
            await MyFunctions.picoh_wait(0.1)
            print("about to speak")
            await MyFunctions.picoh_say(text)
            await MyFunctions.picoh_wait(0.1)
        return

    async def is_silent(snd_data):
        # "Returns 'True' if below the 'silent' threshold"
        return max(snd_data) < THRESHOLD

    async def normalize(snd_data):
        "Average the volume out"
        MAXIMUM = 16384
        times = float(MAXIMUM)/max(abs(i) for i in snd_data)

        r = array('h')
        for i in snd_data:
            r.append(int(i*times))
        return r

    async def trim(snd_data):
        "Trim the blank spots at the start and end"
        def _trim(snd_data):
            snd_started = False
            r = array('h')

            for i in snd_data:
                if not snd_started and abs(i) > THRESHOLD:
                    snd_started = True
                    r.append(i)

                elif snd_started:
                    r.append(i)
            return r

        # Trim to the left
        snd_data = _trim(snd_data)

        # Trim to the right
        snd_data.reverse()
        snd_data = _trim(snd_data)
        snd_data.reverse()
        return snd_data

    async def add_silence(snd_data, seconds):
        "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
        silence = [0] * int(seconds * RATE)
        r = array('h', silence)
        r.extend(snd_data)
        r.extend(silence)
        return r

    async def record():
        """
        Record a word or words from the microphone and
        return the data as an array of signed shorts.

        Normalizes the audio, trims silence from the
        start and end, and pads with 0.5 seconds of
        blank sound to make sure VLC et al can play
        it without getting chopped off.
        """

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=1, rate=RATE,
                        input=True, output=True,
                        frames_per_buffer=CHUNK_SIZE)

        num_silent = 0
        snd_started = False

        r = array('h')

        while 1:
            # little endian, signed short
            snd_data = array('h', stream.read(CHUNK_SIZE))
            if byteorder == 'big':
                snd_data.byteswap()
            r.extend(snd_data)
            silent = await MyFunctions.is_silent(snd_data)
            if silent and snd_started:
                num_silent += 1
            elif not silent and not snd_started:
                snd_started = True
            if snd_started and num_silent > 30:
                break

        sample_width = p.get_sample_size(FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()

        r = await MyFunctions.normalize(r)
        r = await MyFunctions.trim(r)
        r = await MyFunctions.add_silence(r, 0.5)
        return sample_width, r

    async def record_to_file(path):
        "Records from the microphone and outputs the resulting data to 'path'"
        global recording
        global recorded
        recording = True
        try:
            sample_width, data = await MyFunctions.record()
            data = pack('<' + ('h'*len(data)), *data)

            wf = wave.open(path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(RATE)
            wf.writeframes(data)
            wf.close()
        except:
            print("Recording failed")
            recording = False
            recorded = False
            return
        recorded = True
        recording = False
        await MyFunctions.baseCol(0, 0, 5)
        print("Recording stopped")
        return None

    async def listen():
        global listening
        global recording
        global recorded
        global result
        global lastReplyMessage
        recorded = False
        recording = False
        result = None
        while True:
            try:
                if listening:
                    fs = 44100
                    if keyboard.is_pressed("space") and recording == False:
                        await MyFunctions.baseCol(0, 8, 0)
                        print("Recording started")
                        if keyboard.is_pressed("space") and recorded == False:
                            await MyFunctions.record_to_file(picoh.get_input_path())
                # else:
                #     time.sleep(0.1)
                if (recording != True and recorded == True):
                    print("Transcribing...")
                    # wav = wave.open(picoh.get_input_path(), 'rb')
                    result = model.transcribe(
                        await MyFunctions.picoh_get_input_path())
                    await MyFunctions.picoh_wait(0.25)
                    print(result["text"])
                if (result != None and result["text"].replace(" ", "") != ""):
                    resp = await MyFunctions.send_response(
                        result["text"])
                else:
                    recorded = False
                    recording = False
                    result = None
                    await MyFunctions.baseCol(0, 0, 5)
            except Exception as e:
                print(e)
                pass
            for i in range(10):
                if (i == 9):
                    await asyncio.sleep(0.1)
        return None  # This is required to stop the loop


# Define a TCP server configuration
_settings = load_settings()
HOST = _settings.robot_bind_address
PORT = _settings.robot_port


def create_robot_server(config: TransportConfig | None = None) -> RobotServer:
    """Create and configure a RobotServer with explicit command allowlisting."""
    cfg = config or TransportConfig.from_settings(load_settings(), role="server")
    registry = CommandRegistry()

    # Register allowed DeskBuddy operations with strict validators
    async def _handle_check_for_response(args: dict) -> Any:
        return await MyFunctions.check_for_response()

    registry.register(
        "check_for_response",
        _handle_check_for_response,
        validate_empty_arguments,
    )

    async def _handle_reply(args: dict) -> Any:
        msg = args.get("message")
        if msg is None:
            msg = args.get("text", "")
        return await MyFunctions.reply(str(msg))

    registry.register("reply", _handle_reply, validate_reply_arguments)

    async def _handle_simple_test(args: dict) -> Any:
        return await MyFunctions.simple_test()

    registry.register("simple_test", _handle_simple_test, validate_empty_arguments)
    registry.register("ping", _handle_simple_test, validate_empty_arguments)

    async def _handle_base_col(args: dict) -> Any:
        return await MyFunctions.baseCol(args["r"], args["g"], args["b"])

    registry.register("base_col", _handle_base_col, validate_base_col_arguments)
    registry.register("baseCol", _handle_base_col, validate_base_col_arguments)

    async def _handle_say(args: dict) -> Any:
        return await MyFunctions.picoh_say(args["text"])

    registry.register("say", _handle_say, validate_say_arguments)
    registry.register("picoh_say", _handle_say, validate_say_arguments)

    async def _handle_move(args: dict) -> Any:
        motor = args["motor"]
        if isinstance(motor, str):
            motor_map = {
                "HEADNOD": getattr(picoh, "HEADNOD", 0),
                "HEADTURN": getattr(picoh, "HEADTURN", 1),
                "EYETURN": getattr(picoh, "EYETURN", 2),
                "LIDBLINK": getattr(picoh, "LIDBLINK", 3),
                "TOPLIP": getattr(picoh, "TOPLIP", 4),
                "BOTTOMLIP": getattr(picoh, "BOTTOMLIP", 5),
                "EYETILT": getattr(picoh, "EYETILT", 6),
            }
            motor_clean = motor.strip().upper()
            if motor_clean in motor_map:
                motor = motor_map[motor_clean]
            else:
                motor = int(motor)
        return await MyFunctions.picoh_move(motor, float(args["position"]))

    registry.register("move", _handle_move, validate_move_arguments)
    registry.register("picoh_move", _handle_move, validate_move_arguments)

    async def _handle_wait(args: dict) -> Any:
        return await MyFunctions.picoh_wait(float(args["seconds"]))

    registry.register("wait", _handle_wait, validate_wait_arguments)
    registry.register("picoh_wait", _handle_wait, validate_wait_arguments)

    return RobotServer(config=cfg, registry=registry, hardware_executor=get_default_hardware_executor())


async def start_server(config: TransportConfig | None = None, server: RobotServer | None = None) -> None:
    """Start the authenticated, framed robot server. Propagates startup failures."""
    if server is None:
        server = create_robot_server(config)
    await server.start()
    print(f"Server started on {server.host}:{server.port}")
    await server.serve_forever()



# Create the main window

# Create global variables to enable movement and blinking to be turned on and off.
global moving, blinking

# Set these global variables to False for the time being.
moving = False
blinking = False

# Set a default eye shape.
defaultEyeshape = "Eyeball"

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16 if pyaudio is not None else 2

RATE = 44100


# def record(fs):
#     global recording
#     global recorded
#     recorded = False
#     recording = True
#     # Set the recording parameters
#     # Sample rate

#     # Create an empty list to store the recorded audio
#     my_recording = []

#     # Start the recording
#     print("Press and hold the space key to start recording")

#     while True:
#         # Start recording when the space key is pressed

#         # Continue recording while the space key is pressed
#         while keyboard.is_pressed("space"):
#             my_recording.extend(
#                 sd.rec(int(fs / 10), samplerate=fs, channels=2))
#             print("Recording...")
#             sd.wait()  # Wait until recording is finished
#         print("Recording stopped")
#         recording = False
#         recorded = True
#         baseCol(0, 0, 5)
#         return my_recording

#     # Save the recorded audio as a WAV file


def init_hardware():
    """Initialize physical Picoh hardware or mock adapter safely."""
    global moving, blinking, listening
    if _USING_MOCK_PICOH and not _ALLOW_MOCK:
        raise RuntimeError("Running with MockPicoh is not permitted in production mode.")
    try:
        picoh.reset()
        picoh.wait(1)
        picoh.setEyeShape(defaultEyeshape)
        moving = True
        blinking = True
        listening = True
        picoh.say("Hello my name is Cooper. Please give me a few moments to start up.")
    except Exception as exc:
        print(f"Hardware initialization failed or running in headless/mock mode: {exc}")
        moving = False
        blinking = False
        listening = False



def signal_handler(signal, frame):
    loop.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


async def main():
    global servertask
    global t0
    global t1
    global t2
    global t3
    global t4
    global t5
    global picoh
    global lastReplyMessage
    global priorReplyMessage
    global lastTranscription
    global priorTranscription
    global CHUNK_SIZE
    global FORMAT
    global RATE
    global listening
    global model
    global moving
    global blinking
    global defaultEyeshape
    global THRESHOLD
    global recording
    global recorded

    # asyncio.ensure_future(start_server())
    # loop.create_task(start_server())
    # asyncio.ensure_future(MyFunctions.listen())
    # asyncio.ensure_future(MyFunctions.blinkLids())
    # asyncio.ensure_future(MyFunctions.randomLook())
    # asyncio.ensure_future(MyFunctions.randomNod())
    # asyncio.ensure_future(MyFunctions.randomTurn())
    # loop.run_forever()
    init_hardware()
    server_task = asyncio.create_task(start_server())
    listen_task = asyncio.create_task(MyFunctions.listen())
    blink_task = asyncio.create_task(MyFunctions.blinkLids())
    look_task = asyncio.create_task(MyFunctions.randomLook())
    nod_task = asyncio.create_task(MyFunctions.randomNod())
    turn_task = asyncio.create_task(MyFunctions.randomTurn())

    await asyncio.gather(server_task, listen_task)


if __name__ == "__main__":
    if whisper is not None:
        model = whisper.load_model("base")
    loop = asyncio.get_event_loop()

    lastReplyMessage = ""
    priorReplyMessage = ""
    lastTranscription = ""
    priorTranscription = ""
    CHUNK_SIZE = 1024
    if pyaudio is not None:
        FORMAT = pyaudio.paInt16
    RATE = 44100

    asyncio.run(main())
    loop.run_forever()

