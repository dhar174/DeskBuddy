"""Synchronous audio capture used by the retained robot adapters.

The legacy robot scripts run these functions through ``LegacyWorker`` so device
reads, WAV I/O, and Whisper inference never execute on the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import wave
from array import array
from collections.abc import Callable
from contextlib import ExitStack, suppress
from pathlib import Path
from sys import byteorder
from threading import Event
from typing import Any

from deskbuddy.legacy_worker import LegacyWorker


class CaptureStopped(RuntimeError):
    """Raised when a capture is asked to stop between blocking reads."""


class CaptureStopToken(Event):
    """A fresh per-attempt stop token optionally linked to a runtime stop event."""

    def __init__(self, external: Event | None = None) -> None:
        super().__init__()
        self._external = external

    def is_set(self) -> bool:
        return super().is_set() or (self._external is not None and self._external.is_set())


def _raise_if_stopped(stop: Event | CaptureStopToken | None) -> None:
    if stop is not None and stop.is_set():
        raise CaptureStopped("audio capture stopped")


def _normalize(samples: array) -> array:
    if not samples:
        return array("h")
    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        return array("h", samples)
    multiplier = 16384.0 / peak
    return array("h", (int(sample * multiplier) for sample in samples))


def _trim(samples: array, threshold: int) -> array:
    start = next(
        (index for index, sample in enumerate(samples) if abs(sample) > threshold),
        None,
    )
    if start is None:
        return array("h")

    end = len(samples) - 1
    while end >= start and abs(samples[end]) <= threshold:
        end -= 1
    return array("h", samples[start : end + 1])


def capture_pcm(
    *,
    pyaudio_module: Any,
    audio_format: int,
    rate: int,
    chunk_size: int,
    threshold: int,
    silence_chunks: int,
    stop: Event | CaptureStopToken | None = None,
) -> tuple[int, array]:
    """Capture one VAD-delimited utterance and return normalized signed PCM."""
    if pyaudio_module is None:
        raise RuntimeError("PyAudio is unavailable")
    _raise_if_stopped(stop)

    audio = pyaudio_module.PyAudio()
    with ExitStack() as cleanup:
        cleanup.callback(audio.terminate)
        stream = audio.open(
            format=audio_format,
            channels=1,
            rate=rate,
            input=True,
            output=True,
            frames_per_buffer=chunk_size,
        )
        cleanup.callback(stream.close)
        cleanup.callback(stream.stop_stream)
        silent_chunks = 0
        speech_started = False
        samples = array("h")

        while True:
            _raise_if_stopped(stop)
            chunk = array("h", stream.read(chunk_size))
            if byteorder == "big":
                chunk.byteswap()
            _raise_if_stopped(stop)
            samples.extend(chunk)

            # This intentionally matches the old adapters' max(sample) VAD.
            silent = max(chunk) < threshold
            if silent and speech_started:
                silent_chunks += 1
            elif not silent and not speech_started:
                speech_started = True

            if speech_started and silent_chunks > silence_chunks:
                break

        _raise_if_stopped(stop)
        sample_width = audio.get_sample_size(audio_format)

    normalized = _normalize(samples)
    trimmed = _trim(normalized, threshold)
    padding = array("h", [0]) * int(0.5 * rate)
    result = array("h", padding)
    result.extend(trimmed)
    result.extend(padding)
    _raise_if_stopped(stop)
    return sample_width, result


def write_wav(
    path: str | os.PathLike[str],
    sample_width: int,
    samples: array,
    *,
    rate: int,
) -> None:
    """Write signed mono PCM in the little-endian format expected by WAV."""
    pcm = array("h", samples)
    if byteorder == "big":
        pcm.byteswap()
    with wave.open(os.fspath(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(sample_width)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def capture_and_transcribe(
    *,
    pyaudio_module: Any,
    whisper_model: Any,
    audio_format: int,
    rate: int,
    chunk_size: int,
    threshold: int,
    silence_chunks: int,
    stop: Event | CaptureStopToken | None = None,
    temp_dir: str | os.PathLike[str] | None = None,
) -> Any:
    """Capture, normalize, trim, transcribe, and remove a private attempt WAV."""
    sample_width, samples = capture_pcm(
        pyaudio_module=pyaudio_module,
        audio_format=audio_format,
        rate=rate,
        chunk_size=chunk_size,
        threshold=threshold,
        silence_chunks=silence_chunks,
        stop=stop,
    )

    _raise_if_stopped(stop)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".wav", dir=temp_dir)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        write_wav(temporary_path, sample_width, samples, rate=rate)
        _raise_if_stopped(stop)
        if whisper_model is None:
            raise RuntimeError("Whisper model is unavailable")
        return whisper_model.transcribe(os.fspath(temporary_path))
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


async def submit_until_stopped(
    worker: LegacyWorker,
    operation: Callable[[], Any],
    *,
    stop_event: asyncio.Event,
    capture_stop: Event | CaptureStopToken,
) -> tuple[bool, Any]:
    """Await audio work while allowing the loop to stop before it finishes."""
    operation_task = asyncio.create_task(worker.submit(operation, stop=capture_stop))
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, _ = await asyncio.wait(
            (operation_task, stop_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task in done and stop_event.is_set():
            capture_stop.set()
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            return False, None
        return True, await operation_task
    except asyncio.CancelledError:
        capture_stop.set()
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise
    finally:
        if not stop_task.done():
            stop_task.cancel()
            with suppress(asyncio.CancelledError):
                await stop_task
