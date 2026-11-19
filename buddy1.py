
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
from deskbuddy.legacy_audio import (
    CaptureStopped,
    CaptureStopToken,
    capture_and_transcribe,
    capture_pcm,
    submit_until_stopped,
    write_wav,
)
from deskbuddy.legacy_worker import LegacyWorker, WorkerBusyError, WorkerClosedError
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
from transport.transcripts import TranscriptQueue, is_valid_transcript_text

# State tracking globals
lastReplyMessage = ""
priorReplyMessage = ""
lastTranscription = ""
priorTranscription = ""
transcript_queue = TranscriptQueue()
listening = False
_runtime_stop_event: asyncio.Event | None = None
_shared_audio_worker: LegacyWorker | None = None
_shared_audio_users = 0
_shared_audio_worker_owned = False


def _audio_worker_for_call(
    supplied: LegacyWorker | None,
) -> tuple[LegacyWorker, bool]:
    global _shared_audio_worker, _shared_audio_users, _shared_audio_worker_owned
    if supplied is not None:
        supplied.start()
        if _shared_audio_worker is None or _shared_audio_worker.can_replace:
            _shared_audio_worker = supplied
            _shared_audio_worker_owned = False
        elif _shared_audio_worker is not supplied:
            raise WorkerBusyError("another audio worker owns this adapter")
        return supplied, False
    if _shared_audio_worker is None or _shared_audio_worker.can_replace:
        _shared_audio_worker = LegacyWorker("deskbuddy-legacy-audio")
        _shared_audio_worker_owned = True
    worker = _shared_audio_worker
    worker.start()
    _shared_audio_users += 1
    return worker, True


def _release_audio_worker(worker: LegacyWorker, leased: bool) -> None:
    global _shared_audio_worker, _shared_audio_users, _shared_audio_worker_owned
    if not leased or _shared_audio_worker is not worker:
        return
    _shared_audio_users = max(0, _shared_audio_users - 1)
    if _shared_audio_users == 0 and _shared_audio_worker_owned:
        worker.shutdown()
        if worker.can_replace:
            _shared_audio_worker = None
            _shared_audio_worker_owned = False


def _bind_runtime_audio_worker(worker: LegacyWorker) -> None:
    global _shared_audio_worker, _shared_audio_users, _shared_audio_worker_owned
    if _shared_audio_worker is not None and not _shared_audio_worker.can_replace:
        if _shared_audio_worker is not worker:
            raise WorkerBusyError("another audio worker owns this adapter")
    _shared_audio_worker = worker
    _shared_audio_users = 0
    _shared_audio_worker_owned = False


def _shutdown_runtime_audio_worker(worker: LegacyWorker) -> None:
    global _shared_audio_worker, _shared_audio_users, _shared_audio_worker_owned
    worker.shutdown()
    if _shared_audio_worker is worker and worker.can_replace:
        _shared_audio_worker = None
        _shared_audio_users = 0
        _shared_audio_worker_owned = False


def _capture_operation(
    stop: CaptureStopToken | threading.Event,
    silence_chunks: int,
) -> Any:
    return capture_and_transcribe(
        pyaudio_module=pyaudio,
        whisper_model=model,
        audio_format=FORMAT,
        rate=RATE,
        chunk_size=CHUNK_SIZE,
        threshold=THRESHOLD,
        silence_chunks=silence_chunks,
        stop=stop,
    )


def _capture_to_path(path: str, stop: threading.Event, silence_chunks: int) -> tuple[int, array]:
    sample_width, samples = capture_pcm(
        pyaudio_module=pyaudio,
        audio_format=FORMAT,
        rate=RATE,
        chunk_size=CHUNK_SIZE,
        threshold=THRESHOLD,
        silence_chunks=silence_chunks,
        stop=stop,
    )
    if stop.is_set():
        raise CaptureStopped("audio capture stopped before WAV write")
    write_wav(path, sample_width, samples, rate=RATE)
    return sample_width, samples


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
        if not is_valid_transcript_text(text):
            recorded = False
            recording = False
            result = None
            transcript_queue.enqueue(text)
            return
        recorded = False
        recording = False
        priorTranscription = lastTranscription
        lastTranscription = text
        result = None
        transcript_queue.enqueue(text)
        if transcript_queue.polling_mode == "legacy":
            transcript_queue.fits_legacy_response(text)
        return

    async def reply(text):
        global lastReplyMessage
        global priorReplyMessage
        print("Replying: " + text)
        if text is not None and text.strip():
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
        """Capture one VAD-delimited recording without blocking the event loop."""
        stop = threading.Event()
        worker, owns_worker = _audio_worker_for_call(None)
        try:
            return await worker.submit(
                lambda: capture_pcm(
                    pyaudio_module=pyaudio,
                    audio_format=FORMAT,
                    rate=RATE,
                    chunk_size=CHUNK_SIZE,
                    threshold=THRESHOLD,
                    silence_chunks=30,
                    stop=stop,
                ),
                stop=stop,
            )
        finally:
            _release_audio_worker(worker, owns_worker)

    async def record_to_file(path: str):
        """Capture a recording and write it to the requested legacy path."""
        global recording, recorded
        stop = threading.Event()
        worker, owns_worker = _audio_worker_for_call(None)
        recording = True
        recorded = False
        try:
            result = await worker.submit(
                lambda: _capture_to_path(path, stop, 30),
                stop=stop,
            )
        except Exception as exc:
            print(f"Recording failed: {exc}")
            recording = False
            recorded = False
            raise
        finally:
            recording = False
            _release_audio_worker(worker, owns_worker)
        await MyFunctions.baseCol(0, 0, 5)
        recorded = True
        print("Recording stopped")
        return result

    async def listen(
        stop_event: asyncio.Event | None = None,
        audio_worker: LegacyWorker | None = None,
        capture_stop: threading.Event | None = None,
    ):
        global listening, recording, recorded, result
        stop = stop_event if stop_event is not None else asyncio.Event()
        audio_stop = capture_stop if capture_stop is not None else threading.Event()
        worker, owns_worker = _audio_worker_for_call(audio_worker)
        recorded = False
        recording = False
        result = None
        try:
            while not stop.is_set():
                try:
                    pressed = (
                        listening
                        and keyboard is not None
                        and keyboard.is_pressed("space")
                        and not recording
                    )
                    if pressed and not recorded:
                        await MyFunctions.baseCol(0, 8, 0)
                        if stop.is_set():
                            break
                        print("Recording started")
                        recording = True
                        capture_token = CaptureStopToken(audio_stop)
                        has_result, captured = await submit_until_stopped(
                            worker,
                            lambda: _capture_operation(capture_token, 30),
                            stop_event=stop,
                            capture_stop=capture_token,
                        )
                        recording = False
                        if (
                            not has_result
                            or stop.is_set()
                            or capture_token.is_set()
                            or audio_stop.is_set()
                        ):
                            recorded = False
                            result = None
                            if stop.is_set() or audio_stop.is_set():
                                break
                        else:
                            result = captured
                            recorded = True
                            await MyFunctions.picoh_wait(0.25)
                            if stop.is_set() or capture_token.is_set() or audio_stop.is_set():
                                recorded = False
                                result = None
                                if stop.is_set() or audio_stop.is_set():
                                    break
                                continue
                            transcript_text = (
                                captured.get("text") if isinstance(captured, dict) else None
                            )
                            if (
                                not stop.is_set()
                                and not capture_token.is_set()
                                and not audio_stop.is_set()
                                and is_valid_transcript_text(transcript_text)
                            ):
                                await MyFunctions.send_response(transcript_text)
                            else:
                                recorded = False
                                recording = False
                                result = None
                                if not stop.is_set():
                                    await MyFunctions.baseCol(0, 0, 5)
                except asyncio.CancelledError:
                    audio_stop.set()
                    raise
                except (CaptureStopped, WorkerBusyError, WorkerClosedError) as exc:
                    print(f"Audio turn unavailable: {exc}")
                    recording = False
                    recorded = False
                    result = None
                    if stop.is_set() or audio_stop.is_set():
                        break
                except Exception as exc:
                    print(exc)
                    recording = False
                    recorded = False
                    result = None
                await asyncio.sleep(0.1)
        finally:
            audio_stop.set()
            recording = False
            recorded = False
            result = None
            listening = False
            _release_audio_worker(worker, owns_worker)
        return None


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
        transcript_queue.select_polling_mode("legacy")
        text = await MyFunctions.check_for_response()
        if text == "No transcription":
            return text
        if is_valid_transcript_text(text) and transcript_queue.fits_legacy_response(text) is not True:
            return "No transcription"
        return text

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

    return RobotServer(
        config=cfg,
        registry=registry,
        hardware_executor=get_default_hardware_executor(),
        transcript_queue=transcript_queue,
    )


async def start_server(
    config: TransportConfig | None = None,
    server: RobotServer | None = None,
) -> None:
    """Start the authenticated, framed robot server. Propagates startup failures."""
    if server is None:
        server = create_robot_server(config)
    try:
        await server.start()
        print(f"Server started on {server.host}:{server.port}")
        await server.serve_forever()
    finally:
        await server.stop()



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
    stop = _runtime_stop_event
    if stop is not None:
        stop.set()
    else:
        raise KeyboardInterrupt


def _install_runtime_sigint(
    stop: asyncio.Event,
) -> tuple[asyncio.AbstractEventLoop | None, Any | None]:
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, stop.set)
        return loop, None
    except (NotImplementedError, OSError, RuntimeError):
        try:
            previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal_handler)
            return None, previous
        except (OSError, ValueError):
            return None, None


def _restore_runtime_sigint(
    loop: asyncio.AbstractEventLoop | None,
    previous: Any | None,
) -> None:
    if loop is not None:
        loop.remove_signal_handler(signal.SIGINT)
    elif previous is not None:
        try:
            signal.signal(signal.SIGINT, previous)
        except (OSError, ValueError):
            pass


async def main(
    config: TransportConfig | None = None,
    server: RobotServer | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Own server, listener, motion tasks, and the bounded audio worker."""
    global _runtime_stop_event
    stop = stop_event if stop_event is not None else asyncio.Event()
    previous_stop = _runtime_stop_event
    _runtime_stop_event = stop
    signal_loop, previous_sigint = _install_runtime_sigint(stop)
    audio_worker = LegacyWorker("deskbuddy-legacy-audio")
    active_server: RobotServer | None = None
    tasks: list[asyncio.Task[Any]] = []
    stop_task: asyncio.Task[bool] | None = None
    try:
        audio_worker.start()
        _bind_runtime_audio_worker(audio_worker)
        active_server = server if server is not None else create_robot_server(config)
        stop_task = asyncio.create_task(stop.wait(), name="buddy1-stop")
        if stop.is_set():
            return
        init_hardware()
        tasks = [
            asyncio.create_task(
                start_server(config, active_server), name="buddy1-server"
            ),
            asyncio.create_task(
                MyFunctions.listen(stop, audio_worker), name="buddy1-listener"
            ),
            asyncio.create_task(MyFunctions.blinkLids(), name="buddy1-blink"),
            asyncio.create_task(MyFunctions.randomLook(), name="buddy1-look"),
            asyncio.create_task(MyFunctions.randomNod(), name="buddy1-nod"),
            asyncio.create_task(MyFunctions.randomTurn(), name="buddy1-turn"),
        ]
        done, _ = await asyncio.wait(
            (*tasks, stop_task), return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task not in done:
            for task in done:
                if task.cancelled():
                    continue
                if (error := task.exception()) is not None:
                    raise error
            if not stop.is_set():
                raise RuntimeError("a robot runtime task stopped unexpectedly")
    finally:
        stop.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
        cleanup_tasks: list[asyncio.Task[Any]] = tasks[:]
        if stop_task is not None:
            cleanup_tasks.append(stop_task)
        try:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            if active_server is not None:
                await active_server.stop()
        finally:
            _shutdown_runtime_audio_worker(audio_worker)
            _restore_runtime_sigint(signal_loop, previous_sigint)
            _runtime_stop_event = previous_stop


if __name__ == "__main__":
    if whisper is not None:
        model = whisper.load_model("base")
    lastReplyMessage = ""
    priorReplyMessage = ""
    lastTranscription = ""
    priorTranscription = ""
    CHUNK_SIZE = 1024
    if pyaudio is not None:
        FORMAT = pyaudio.paInt16
    RATE = 44100

    asyncio.run(main())

