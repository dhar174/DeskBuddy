from concurrent.futures import ThreadPoolExecutor
from sys import byteorder
import asyncio
import concurrent.futures
import inspect
import os
from pathlib import Path
import signal
import sys
import threading
from typing import Any
import wave
from array import array
from random import randint, random
from struct import pack

from deskbuddy.settings import load_settings

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

# Ensure repo root is on sys.path for transport package
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from transport import (
    CommandRegistry,
    RobotServer,
    TransportConfig,
    validate_base_col_arguments,
    validate_empty_arguments,
    validate_move_arguments,
    validate_reply_arguments,
    validate_say_arguments,
    validate_wait_arguments,
)
from transport.transcripts import TranscriptQueue, is_valid_transcript_text
from deskbuddy.legacy_audio import (
    CaptureStopped,
    CaptureStopToken,
    capture_and_transcribe,
    capture_pcm,
    submit_until_stopped,
    write_wav,
)
from deskbuddy.legacy_worker import LegacyWorker, WorkerBusyError, WorkerClosedError

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
    run_hardware_op,
)


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
        _shared_audio_worker = LegacyWorker("deskbuddy-windows-buddy2-audio")
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


def _capture_operation(stop: CaptureStopToken | threading.Event, silence_chunks: int) -> Any:
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


class MyFunctions:
    _public_methods_ = [
        "simple_test",
        "button_clicked",
        "send_response",
        "listen",
        "baseCol",
        "randomNod",
        "randomTurn",
        "randomLook",
        "blinkLids",
        "record_to_file",
        "record",
        "add_silence",
        "trim",
        "normalize",
        "is_silent",
        "button_clicked",
    ]

    async def simple_test():
        print("Hello")
        return "Hello"

    async def check_for_response():
        global lastTranscription
        global priorTranscription
        # print("Checking for response")

        if lastTranscription != "" and lastTranscription != priorTranscription:
            return lastTranscription
        else:
            return "No transcription"

    async def blinkLids(stop_event: asyncio.Event | None = None):
        global blinking

        while stop_event is None or not stop_event.is_set():
            # for i in range(10):
            #     await asyncio.sleep(1)
            if blinking:
                for x in range(10, 0, -1):
                    await run_hardware_op(picoh.move, getattr(picoh, "LIDBLINK", 3), x)
                    await asyncio.sleep(0.01)

                for x in range(0, 10):
                    await run_hardware_op(picoh.move, getattr(picoh, "LIDBLINK", 3), x)
                    await asyncio.sleep(0.01)

                await asyncio.sleep(random() * 6)
            await asyncio.sleep(0.1)

    async def randomLook(stop_event: asyncio.Event | None = None):
        global moving
        while stop_event is None or not stop_event.is_set():
            # for i in range(10):
            #     await asyncio.sleep(1)
            if moving:
                await run_hardware_op(picoh.move, getattr(picoh, "EYETILT", 6), randint(2, 8))
                await run_hardware_op(picoh.move, getattr(picoh, "EYETURN", 2), randint(2, 8))

                await asyncio.sleep(random() * 5)
            await asyncio.sleep(0.1)

    async def randomTurn(stop_event: asyncio.Event | None = None):
        global moving
        while stop_event is None or not stop_event.is_set():
            # for i in range(10):
            # await asyncio.sleep(1)
            if moving:
                await run_hardware_op(picoh.move, getattr(picoh, "HEADTURN", 1), randint(3, 7))

                await asyncio.sleep(random() * 4)
            await asyncio.sleep(0.1)

    async def randomNod(stop_event: asyncio.Event | None = None):
        global moving
        while stop_event is None or not stop_event.is_set():
            # for i in range(10):
            #     await asyncio.sleep(1)
            if moving:
                await run_hardware_op(picoh.move, getattr(picoh, "HEADNOD", 0), randint(4, 7))

                await asyncio.sleep(random() * 4)
            await asyncio.sleep(0.1)

    async def baseCol(r, g, b):
        await run_hardware_op(picoh.setBaseColour, r, g, b)
        return

    async def picoh_say(text):
        print("in picoh_say")
        print(str(text))
        await run_hardware_op(picoh.say, str(text))
        return

    async def picoh_move(motor, position):
        await run_hardware_op(picoh.move, motor, position)
        return

    async def picoh_wait(seconds):
        await asyncio.sleep(float(seconds))
        return

    async def picoh_get_input_path():
        return await run_hardware_op(picoh.get_input_path)

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

    async def check_key_pressed(key, delay=0.1, stop_event=None):
        while stop_event is None or not stop_event.is_set():
            if keyboard is not None and keyboard.is_pressed(key):
                return True
            await asyncio.sleep(delay)
        return False

    @staticmethod
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
        waiting_for_reply = False
        try:
            while not stop.is_set():
                try:
                    pressed = (
                        listening
                        and not waiting_for_reply
                        and not recording
                        and keyboard is not None
                        and keyboard.is_pressed("space")
                    )
                    if pressed and not recorded:
                        await MyFunctions.baseCol(0, 8, 0)
                        if stop.is_set():
                            break
                        print("Recording started")
                        waiting_for_reply = True
                        recording = True
                        capture_token = CaptureStopToken(audio_stop)
                        has_result, captured = await submit_until_stopped(
                            worker,
                            lambda: _capture_operation(capture_token, 80),
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
                            waiting_for_reply = False
                        else:
                            result = captured
                            recorded = True
                            await MyFunctions.picoh_wait(0.25)
                            if stop.is_set() or capture_token.is_set() or audio_stop.is_set():
                                recorded = False
                                result = None
                                if stop.is_set() or audio_stop.is_set():
                                    break
                                waiting_for_reply = False
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
                            waiting_for_reply = False
                except asyncio.CancelledError:
                    audio_stop.set()
                    raise
                except (CaptureStopped, WorkerBusyError, WorkerClosedError) as exc:
                    print(f"Audio turn unavailable: {exc}")
                    recording = False
                    recorded = False
                    result = None
                    waiting_for_reply = False
                    if stop.is_set() or audio_stop.is_set():
                        break
                except Exception as exc:
                    print(exc)
                    recording = False
                    recorded = False
                    result = None
                    waiting_for_reply = False
                await asyncio.sleep(0.1)
        finally:
            audio_stop.set()
            recording = False
            recorded = False
            result = None
            listening = False
            _release_audio_worker(worker, owns_worker)
        return None

    def is_silent(snd_data):
        # "Returns 'True' if below the 'silent' threshold"
        return max(snd_data) < THRESHOLD

    def normalize(snd_data):
        "Average the volume out"
        MAXIMUM = 16384
        times = float(MAXIMUM) / max(abs(i) for i in snd_data)

        r = array("h")
        for i in snd_data:
            r.append(int(i * times))
        return r

    def trim(snd_data):
        # "Trim the blank spots at the start and end"
        def _trim(snd_data):
            snd_started = False
            r = array("h")

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

    def add_silence(snd_data, seconds):
        "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
        silence = [0] * int(seconds * RATE)
        r = array("h", silence)
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
                    silence_chunks=80,
                    stop=stop,
                ),
                stop=stop,
            )
        finally:
            _release_audio_worker(worker, owns_worker)

    async def record_to_file(path, sample_width, data):
        """Write signed PCM to the path expected by the retained COM method."""
        global recording, recorded
        worker, owns_worker = _audio_worker_for_call(None)
        stop = threading.Event()
        recording = True
        recorded = False
        try:
            await worker.submit(
                lambda: write_wav(path, sample_width, array("h", data), rate=RATE),
                stop=stop,
            )
        except Exception as exc:
            print(f"Error saving the file: {exc}")
            recorded = False
            raise
        finally:
            recording = False
            _release_audio_worker(worker, owns_worker)
        await MyFunctions.baseCol(0, 0, 5)
        recorded = True
        print("Recording stopped")
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


# Set the moving and blinking global variables to True.


moving = True
blinking = True
listening = True

# Set these global variables to False for the time being.
moving = False
blinking = False

# Set a default eye shape.
defaultEyeshape = "Eyeball"

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = getattr(pyaudio, "paInt16", 16) if pyaudio else 16
RATE = 44100


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
    global _runtime_stop_event, blinking, moving
    stop = stop_event if stop_event is not None else asyncio.Event()
    previous_stop = _runtime_stop_event
    _runtime_stop_event = stop
    signal_loop, previous_sigint = _install_runtime_sigint(stop)
    audio_worker = LegacyWorker("deskbuddy-windows-buddy2-audio")
    active_server: RobotServer | None = None
    tasks: list[asyncio.Task[Any]] = []
    stop_task: asyncio.Task[bool] | None = None
    try:
        audio_worker.start()
        _bind_runtime_audio_worker(audio_worker)
        active_server = server if server is not None else create_robot_server(config)
        stop_task = asyncio.create_task(stop.wait(), name="windows-buddy2-stop")
        if stop.is_set():
            return
        blinking = True
        moving = True
        if _USING_MOCK_PICOH and not _ALLOW_MOCK:
            raise RuntimeError("Running with MockPicoh is not permitted in production mode.")
        if globals().get("firstLoop", False):
            await run_hardware_op(picoh.reset)
            await asyncio.sleep(1)
            await run_hardware_op(picoh.setEyeBrightness, 1)
            await run_hardware_op(picoh.setEyeShape, defaultEyeshape)

        await run_hardware_op(
            picoh.say,
            "Hello my name is Cooper. Cooper stands for Conversational Office Organizer and Personal Entertainment Robot. Please give me a few moments to start up.",
        )
        if stop.is_set():
            return
        print("Starting up...")
        tasks = [
            asyncio.create_task(
                start_server(config, active_server), name="windows-buddy2-server"
            ),
            asyncio.create_task(
                MyFunctions.listen(stop, audio_worker), name="windows-buddy2-listener"
            ),
            asyncio.create_task(
                MyFunctions.blinkLids(stop), name="windows-buddy2-blink"
            ),
            asyncio.create_task(
                MyFunctions.randomLook(stop), name="windows-buddy2-look"
            ),
            asyncio.create_task(
                MyFunctions.randomNod(stop), name="windows-buddy2-nod"
            ),
            asyncio.create_task(
                MyFunctions.randomTurn(stop), name="windows-buddy2-turn"
            ),
        ]
        print("Tasks starting")
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
    firstLoop = True

    if whisper is not None:
        model = whisper.load_model("base")
    # loop = asyncio.get_event_loop()
    lastReplyMessage = ""
    priorReplyMessage = ""
    lastTranscription = ""
    priorTranscription = ""
    CHUNK_SIZE = 1024
    if pyaudio is not None:
        FORMAT = pyaudio.paInt16
    RATE = 44100
    asyncio.run(main())
