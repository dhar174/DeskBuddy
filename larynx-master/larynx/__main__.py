#!/usr/bin/env python3
import argparse
import io
import logging
import os
import platform
import shlex
import string
import subprocess
import sys
import threading
import time
import typing
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from queue import Queue

from larynx.constants import InferenceBackend
from larynx.utils import (
    DEFAULT_VOICE_URL_FORMAT,
    VOCODER_DIR_NAMES,
    get_runtime_dir,
    get_voices_dirs,
    valid_voice_dir,
)

_DIR = Path(__file__).parent

_LOGGER = logging.getLogger("larynx")

# -----------------------------------------------------------------------------


class OutputNaming(str, Enum):
    """Format used for output file names"""

    TEXT = "text"
    TIME = "time"
    ID = "id"


class StdinFormat(str, Enum):
    """Format of standard input"""

    AUTO = "auto"
    """Choose based on SSML state"""

    LINES = "lines"
    """Each line is a separate sentence/document"""

    DOCUMENT = "document"
    """Entire input is one document"""


# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    args = get_args()

    if args.cuda:
        import torch

        args.cuda = torch.cuda.is_available()
        if not args.cuda:
            args.half = False
            _LOGGER.warning("CUDA is not available")

    # Handle optimizations.
    # onnxruntime crashes on armv7l if optimizations are enabled.
    setattr(args, "no_optimizations", False)
    if args.optimizations == "off":
        args.no_optimizations = True
    elif args.optimizations == "auto":
        if platform.machine() == "armv7l":
            # Enabling optimizations on 32-bit ARM crashes
            args.no_optimizations = True

    backend: typing.Optional[InferenceBackend] = None
    if args.backend:
        backend = InferenceBackend(args.backend)

    # -------------------------------------------------------------------------
    # Daemon
    # -------------------------------------------------------------------------

    if args.daemon:
        runtime_dir = get_runtime_dir()
        pidfile_path = runtime_dir / "daemon.pidfile"
        _LOGGER.debug("Trying to start daemon on port %s", args.daemon_port)

        daemon_cmd = [
            "python3",
            "-m",
            "larynx.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.daemon_port),
            "--pidfile",
            str(pidfile_path),
            "--logfile",
            str(runtime_dir / "daemon.log"),
        ]

        _LOGGER.debug(daemon_cmd)

        # pylint: disable=consider-using-with
        subprocess.Popen(
            daemon_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        _LOGGER.debug("Waiting for daemon to start...")
        while (not pidfile_path.is_file()) or (pidfile_path.stat().st_size == 0):
            time.sleep(0.1)

        daemon_pid = int(pidfile_path.read_text().strip())
        _LOGGER.info("Daemon running (pid=%s)", daemon_pid)

        if args.text:
            text = " ".join(args.text)
        else:
            text = sys.stdin.read().encode()

        vocoder = f"{args.vocoder_model_type}/{args.vocoder_model}"
        values = {"voice": args.voice, "text": text, "vocoder": vocoder}
        url = f"http://localhost:{args.daemon_port}/api/tts?" + urllib.parse.urlencode(
            values
        )

        _LOGGER.debug(url)

        start_time = time.perf_counter()
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            wav_data = response.read()
            end_time = time.perf_counter()
            _LOGGER.debug(
                "Got %s byte(s) of WAV data in %s second(s)",
                len(wav_data),
                end_time - start_time,
            )
            sys.stdout.buffer.write(wav_data)
            sys.stdout.flush()

        return

    # -------------------------------------------------------------------------
    # No Daemon
    # -------------------------------------------------------------------------

    import numpy as np

    # Create output directory
    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Open file for writing the names from <mark> tags in SSML.
    # Each name is printed on a single line.
    mark_writer: typing.Optional[typing.TextIO] = None
    if args.mark_file:
        args.mark_file = Path(args.mark_file)
        args.mark_file.parent.mkdir(parents=True, exist_ok=True)
        mark_writer = open(  # pylint: disable=consider-using-with
            args.mark_file, "w", encoding="utf-8"
        )

    if args.seed is not None:
        _LOGGER.debug("Setting random seed to %s", args.seed)
        np.random.seed(args.seed)

    if args.csv:
        args.output_naming = "id"

    # Read text from stdin or arguments
    if args.text:
        # Use arguments
        texts = args.text
    else:
        # Use stdin
        stdin_format = StdinFormat.LINES

        if (args.stdin_format == StdinFormat.AUTO) and args.ssml:
            # Assume SSML input is entire document
            stdin_format = StdinFormat.DOCUMENT

        if stdin_format == StdinFormat.DOCUMENT:
            # One big line
            texts = [sys.stdin.read()]
        else:
            # Multiple lines
            texts = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    if args.process_on_blank_line:

        # Combine text until a blank line is encountered.
        # Good for line-wrapped books where
        # sentences are broken
        # up across multiple
        # lines.
        def process_on_blank_line(lines):
            text = ""
            for line in lines:
                line = line.strip()
                if not line:
                    if text:
                        yield text

                    text = ""
                    continue

                text += " " + line

        texts = process_on_blank_line(texts)

    # -------------------------------------------------------------------------

    from larynx import text_to_speech
    from larynx.wavfile import write as wav_write

    max_thread_workers: typing.Optional[int] = None

    if args.max_thread_workers is not None:
        max_thread_workers = (
            None if args.max_thread_workers < 1 else args.max_thread_workers
        )
    elif args.raw_stream:
        # Faster time to first audio
        max_thread_workers = 2

    executor = ThreadPoolExecutor(max_workers=max_thread_workers)

    if os.isatty(sys.stdout.fileno()):
        if (not args.output_dir) and (not args.raw_stream):
            # No where else for the audio to go
            args.interactive = True

    # Raw stream queue
    raw_queue: typing.Optional["Queue[bytes]"] = None
    raw_stream_thread: typing.Optional[threading.Thread] = None

    if args.raw_stream:
        # Output in a separate thread to avoid blocking audio processing
        raw_queue = Queue(maxsize=args.raw_stream_queue_size)

        def output_raw_stream():
            while True:
                audio = raw_queue.get()
                if audio is None:
                    break

                _LOGGER.debug(
                    "Writing %s byte(s) of 16-bit 22050Hz mono PCM to stdout",
                    len(audio),
                )
                sys.stdout.buffer.write(audio)
                sys.stdout.buffer.flush()

        raw_stream_thread = threading.Thread(target=output_raw_stream, daemon=True)
        raw_stream_thread.start()

    all_audios: typing.List[np.ndarray] = []
    sample_rate: int = 22050
    wav_data: typing.Optional[bytes] = None
    play_command = shlex.split(args.play_command)

    # Settings for TTS and vocoder
    tts_settings: typing.Dict[str, typing.Any] = {
        "noise_scale": args.noise_scale,
        "length_scale": args.length_scale,
    }
    vocoder_settings: typing.Dict[str, typing.Any] = {
        "denoiser_strength": args.denoiser_strength,
    }

    # -------------------
    # Process input lines
    # -------------------
    start_time_to_first_audio = time.perf_counter()

    try:
        for line in texts:
            line_id = ""
            line = line.strip()
            if not line:
                continue

            if args.output_naming == OutputNaming.ID:
                # Line has the format id|text instead of just text
                line_id, line = line.split(args.id_delimiter, maxsplit=1)

            tts_results = text_to_speech(
                text=line,
                voice_or_lang=args.voice,
                ssml=args.ssml,
                vocoder_or_quality=args.quality,
                backend=backend,
                use_cuda=args.cuda,
                half=args.half,
                denoiser_strength=args.denoiser_strength,
                executor=executor,
                tts_settings=tts_settings,
                vocoder_settings=vocoder_settings,
                custom_voices_dir=args.voices_dir,
                url_format=args.url_format,
            )

            text_id = ""

            for result_idx, result in enumerate(tts_results):
                text = result.text

                if result_idx == 0:
                    end_time_to_first_audio = time.perf_counter()
                    _LOGGER.debug(
                        "Seconds to first audio: %s",
                        end_time_to_first_audio - start_time_to_first_audio,
                    )

                sample_rate = result.sample_rate

                # Write before marks
                if result.marks_before and mark_writer:
                    for mark_name in result.marks_before:
                        print(mark_name, file=mark_writer)

                if args.raw_stream:
                    assert raw_queue is not None
                    raw_queue.put(result.audio.tobytes())
                elif args.interactive or args.output_dir:
                    # Convert to WAV audio
                    with io.BytesIO() as wav_io:
                        wav_write(wav_io, result.sample_rate, result.audio)
                        wav_data = wav_io.getvalue()

                    assert wav_data is not None

                    if args.interactive:

                        # Play audio
                        _LOGGER.debug("Playing audio with play command")
                        try:
                            subprocess.run(
                                play_command,
                                input=wav_data,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=True,
                            )
                        except FileNotFoundError:
                            _LOGGER.error(
                                "Unable to play audio with command '%s'. set with --play-command or redirect stdout",
                                args.play_command,
                            )
                            with open("output.wav", "wb") as output_file:
                                output_file.write(wav_data)

                            _LOGGER.warning(
                                "stdout not redirected. Wrote audio to output.wav."
                            )

                    if args.output_dir:
                        # Determine file name
                        if args.output_naming == OutputNaming.TEXT:
                            # Use text itself
                            file_name = text.strip().replace(" ", "_")
                            file_name = file_name.translate(
                                str.maketrans(
                                    "", "", string.punctuation.replace("_", "")
                                )
                            )
                        elif args.output_naming == OutputNaming.TIME:
                            # Use timestamp
                            file_name = str(time.time())
                        elif args.output_naming == OutputNaming.ID:
                            if not text_id:
                                text_id = line_id
                            else:
                                text_id = f"{line_id}_{result_idx + 1}"

                            file_name = text_id

                        assert file_name, f"No file name for text: {text}"
                        wav_path = args.output_dir / (file_name + ".wav")
                        with open(wav_path, "wb") as wav_file:
                            wav_write(wav_file, sample_rate, result.audio)

                        _LOGGER.debug("Wrote %s", wav_path)
                else:
                    # Combine all audio and output to stdout at the end
                    all_audios.append(result.audio)

                # Write after marks
                if result.marks_after and mark_writer:
                    for mark_name in result.marks_after:
                        print(mark_name, file=mark_writer)

    except KeyboardInterrupt:
        if raw_queue is not None:
            # Draw audio playback queue
            while not raw_queue.empty():
                raw_queue.get()
    finally:
        # Wait for raw stream to finish
        if raw_queue is not None:
            raw_queue.put(None)

        if raw_stream_thread is not None:
            raw_stream_thread.join()

    # -------------------------------------------------------------------------

    # Write combined audio to stdout
    if all_audios:
        with io.BytesIO() as wav_io:
            wav_write(wav_io, sample_rate, np.concatenate(all_audios))
            wav_data = wav_io.getvalue()

        _LOGGER.debug("Writing WAV audio to stdout")
        sys.stdout.buffer.write(wav_data)
        sys.stdout.buffer.flush()


# -----------------------------------------------------------------------------


def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="larynx")
    parser.add_argument(
        "--language", help="Gruut language for text input (en-us, etc.)"
    )
    parser.add_argument(
        "text", nargs="*", help="Text to convert to speech (default: stdin)"
    )
    parser.add_argument(
        "--stdin-format",
        choices=[str(v.value) for v in StdinFormat],
        default=StdinFormat.AUTO,
        help="Format of stdin text (default: auto)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        default="en-us",
        help="Name of voice (expected in <voices-dir>/<language>)",
    )
    parser.add_argument(
        "--voices-dir",
        help="Directory with voices (format is <language>/<name_model-type>)",
    )
    parser.add_argument(
        "--quality",
        "-q",
        choices=["high", "medium", "low"],
        default="high",
        help="Vocoder quality (default: high)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available voices/vocoders"
    )
    parser.add_argument(
        "--config", help="Path to JSON configuration file with audio settings"
    )
    parser.add_argument("--output-dir", help="Directory to write WAV file(s)")
    parser.add_argument(
        "--output-naming",
        choices=[v.value for v in OutputNaming],
        default="text",
        help="Naming scheme for output WAV files (requires --output-dir)",
    )
    parser.add_argument(
        "--id-delimiter",
        default="|",
        help="Delimiter between id and text in lines (default: |). Requires --output-naming id",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Play audio after each input line (see --play-command)",
    )
    parser.add_argument("--csv", action="store_true", help="Input format is id|text")
    parser.add_argument(
        "--mark-file",
        help="File to write mark names to as they're encountered (--ssml only)",
    )

    # GlowTTS setttings
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.667,
        help="Noise scale (default: 0.667, GlowTTS only)",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Length scale (default: 1.0, GlowTTS only)",
    )

    # Vocoder settings
    parser.add_argument(
        "--denoiser-strength",
        type=float,
        default=0.005,
        help="Strength of denoiser, if available (default: 0 = disabled)",
    )

    # Miscellaneous
    parser.add_argument(
        "--max-thread-workers",
        type=int,
        help="Maximum number of threads to concurrently load models and run sentences through TTS/Vocoder",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't automatically download voices or vocoders",
    )
    parser.add_argument(
        "--url-format",
        default=DEFAULT_VOICE_URL_FORMAT,
        help="Format string for download URLs (accepts {voice})",
    )
    parser.add_argument(
        "--play-command",
        default="play -",
        help="Shell command used to play audio in interactive model (default: play -)",
    )
    parser.add_argument(
        "--raw-stream",
        action="store_true",
        help="Stream raw 16-bit 22050Hz mono PCM audio to stdout",
    )
    parser.add_argument(
        "--raw-stream-queue-size",
        default=5,
        help="Maximum number of sentences to maintain in output queue with --raw-stream (default: 5)",
    )
    parser.add_argument(
        "--process-on-blank-line",
        action="store_true",
        help="Process text only after encountering a blank line",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Connect to or run a background HTTP server for TTS functionality",
    )
    parser.add_argument(
        "--daemon-port",
        type=int,
        default=15002,
        help="Port to run daemon HTTP server on (default: 15002)",
    )
    parser.add_argument(
        "--stop-daemon",
        action="store_true",
        help="Try to stop the currently running Larynx daemon and exit",
    )
    parser.add_argument("--ssml", action="store_true", help="Input text is SSML")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use faster FP16 for inference (requires --cuda)",
    )
    parser.add_argument(
        "--optimizations",
        choices=["auto", "on", "off"],
        default="auto",
        help="Enable/disable Onnx optimizations (auto=disable on armv7l)",
    )

    parser.add_argument(
        "--backend",
        choices=[v.value for v in InferenceBackend],
        help="Force use of specific inference backend (default: prefer onnx)",
    )

    parser.add_argument("--seed", type=int, help="Set random seed (default: not set)")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # -------------------------------------------------------------------------

    if args.version:
        # Print version and exit
        from larynx import __version__

        print(__version__)
        sys.exit(0)

    if args.stop_daemon:
        # Try to stop daemon and exit
        import psutil

        runtime_dir = get_runtime_dir()
        pidfile = runtime_dir / "daemon.pidfile"
        if pidfile.is_file():
            # Get pid from file and block until terminated
            daemon_pid = int(pidfile.read_text().strip())
            proc = psutil.Process(daemon_pid)
            if proc.is_running():
                _LOGGER.debug("Trying to stop daemon (pid=%s)", daemon_pid)
                proc.terminate()
                psutil.wait_procs([proc])

            _LOGGER.info("Stopped daemon")
        else:
            _LOGGER.info("No daemon running")

        sys.exit(0)

    # -------------------------------------------------------------------------

    # Directories to search for voices
    voices_dirs = get_voices_dirs(args.voices_dir)

    def list_voices_vocoders():
        """Print all vocoders and voices"""
        # (type, name) -> location
        local_info = {}

        # Search for downloaded voices/vocoders
        for voices_dir in voices_dirs:
            if not voices_dir.is_dir():
                continue

            for voice_dir in voices_dir.iterdir():
                if not voice_dir.is_dir():
                    continue

                if voice_dir.name in VOCODER_DIR_NAMES:
                    # Vocoder
                    for vocoder_model_dir in voice_dir.iterdir():
                        if not valid_voice_dir(vocoder_model_dir):
                            continue

                        full_vocoder_name = f"{voice_dir.name}-{vocoder_model_dir.name}"
                        local_info[("vocoder", full_vocoder_name)] = str(
                            vocoder_model_dir
                        )
                else:
                    # Voice
                    voice_lang = voice_dir.name
                    for voice_model_dir in voice_dir.iterdir():
                        if not valid_voice_dir(voice_model_dir):
                            continue

                        local_info[("voice", voice_model_dir.name)] = str(
                            voice_model_dir
                        )

        # (type, lang, name, downloaded, aliases, location)
        voices_and_vocoders = []
        with open(_DIR / "VOCODERS", "r", encoding="utf-8") as vocoders_file:
            for line in vocoders_file:
                line = line.strip()
                if not line:
                    continue

                *vocoder_aliases, full_vocoder_name = line.split()
                downloaded = False

                location = local_info.get(("vocoder", full_vocoder_name), "")
                if location:
                    downloaded = True

                voices_and_vocoders.append(
                    (
                        "vocoder",
                        " ",
                        "*" if downloaded else " ",
                        full_vocoder_name,
                        ",".join(vocoder_aliases),
                        location,
                    )
                )

        with open(_DIR / "VOICES", "r", encoding="utf-8") as voices_file:
            for line in voices_file:
                line = line.strip()
                if not line:
                    continue

                *voice_aliases, full_voice_name, download_name = line.split()
                voice_lang = download_name.split("_", maxsplit=1)[0]

                downloaded = False

                location = local_info.get(("voice", full_voice_name), "")
                if location:
                    downloaded = True

                voices_and_vocoders.append(
                    (
                        "voice",
                        voice_lang,
                        "*" if downloaded else " ",
                        full_voice_name,
                        ",".join(voice_aliases),
                        location,
                    )
                )

        headers = ("TYPE", "LANG", "LOCAL", "NAME", "ALIASES", "LOCATION")

        # Get widths of columns
        col_widths = [0] * len(voices_and_vocoders[0])
        for item in voices_and_vocoders:
            for col in range(len(col_widths)):
                col_widths[col] = max(
                    col_widths[col], len(item[col]) + 1, len(headers[col]) + 1
                )

        # Print results
        print(*(h.ljust(col_widths[col]) for col, h in enumerate(headers)))

        for item in sorted(voices_and_vocoders):
            print(*(v.ljust(col_widths[col]) for col, v in enumerate(item)))

    if args.list:
        list_voices_vocoders()
        sys.exit(0)

    return args


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
