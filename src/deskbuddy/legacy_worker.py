"""Bounded, cancellation-aware execution for retained synchronous resources."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class WorkerBusyError(RuntimeError):
    """The worker has no safe admission slot for another operation."""


class WorkerClosedError(RuntimeError):
    """The owner stopped accepting work or abandoned its result during shutdown."""


@dataclass(eq=False)
class _Operation(Generic[T]):
    stop: Event | None
    result: Future[T] = field(default_factory=Future)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    abandoned: bool = False

    def run(self, operation: Callable[[], T]) -> None:
        # This is the atomic handoff between asyncio cancellation and worker entry.
        if not self.result.set_running_or_notify_cancel():
            return
        try:
            value = operation()
        except BaseException as exc:
            self.result.set_exception(exc)
        else:
            self.result.set_result(value)

    def abandon(self) -> None:
        self.abandoned = True
        if self.stop is not None:
            self.stop.set()
        self.result.cancel()


class LegacyWorker:
    """Own one synchronous resource, with one waiting caller and no worker backlog.

    Start and submit on the same event loop. Shut down on that loop or after it
    closes. Completion belongs to the callable, even if nobody awaits it.
    """

    def __init__(self, name: str, *, use_default_executor: bool = False) -> None:
        self._name = name
        self._use_default_executor = use_default_executor
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._active: _Operation[Any] | None = None
        self._waiting: asyncio.Event | None = None
        self._offload_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def start(self) -> None:
        """Bind the owner to the running loop and prepare its execution boundary."""
        loop = asyncio.get_running_loop()
        if self._closed:
            raise WorkerClosedError(f"{self._name} is closed")
        if self._loop is loop:
            return
        if self._loop is not None:
            raise RuntimeError(f"{self._name} belongs to another event loop")
        self._loop = loop
        if not self._use_default_executor:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._name)

    @property
    def busy(self) -> bool:
        return self._active is not None and not self._active.result.done()

    @property
    def abandoned(self) -> bool:
        active = self._active
        return active is not None and active.abandoned and not active.result.done()

    @property
    def can_replace(self) -> bool:
        """A retired owner can be replaced only after its active work finishes."""
        loop = self._loop
        return not self.busy and (self._closed or (loop is not None and loop.is_closed()))

    def _require_open(self) -> asyncio.AbstractEventLoop:
        if self._closed:
            raise WorkerClosedError(f"{self._name} is closed")
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            raise RuntimeError(f"{self._name} must be started on this event loop")
        return loop

    async def submit(self, operation: Callable[[], T], *, stop: Event | None = None) -> T:
        """Run a callable off-loop; cancellation abandons delivery, not ownership."""
        loop = self._require_open()
        if self.abandoned or self._waiting is not None:
            raise WorkerBusyError(f"{self._name} is busy")

        if self.busy:
            waiting = asyncio.Event()
            self._waiting = waiting
            try:
                await waiting.wait()
            finally:
                self._waiting = None
            self._require_open()

        job: _Operation[T] = _Operation(stop)
        self._active = job

        def notify(_: Future[T]) -> None:
            try:
                loop.call_soon_threadsafe(self._complete, job)
            except RuntimeError:
                # A worker may outlive the event loop. Its real result is already
                # complete, so busy is false even when delivery is impossible.
                if not loop.is_closed():
                    raise

        job.result.add_done_callback(notify)
        self._launch(job, operation)
        try:
            await job.ready.wait()
        except asyncio.CancelledError:
            job.abandon()
            raise
        self._require_open()
        return job.result.result()

    def _launch(self, job: _Operation[T], operation: Callable[[], T]) -> None:
        if self._executor is not None:
            try:
                self._executor.submit(job.run, operation)
            except RuntimeError as exc:
                job.result.set_exception(exc)
            return
        task = asyncio.create_task(asyncio.to_thread(job.run, operation))
        self._offload_tasks.add(task)

        def observe(offload: asyncio.Task[None]) -> None:
            self._offload_tasks.discard(offload)
            if offload.cancelled():
                job.result.cancel()
            elif (error := offload.exception()) is not None and not job.result.done():
                job.result.set_exception(error)

        task.add_done_callback(observe)

    def _complete(self, job: _Operation[Any]) -> None:
        if self._active is job:
            self._active = None
            if self._waiting is not None:
                self._waiting.set()
        job.ready.set()

    def shutdown(self) -> None:
        """Close admission and wake callers while retaining any running operation."""
        if self._closed:
            return
        loop = self._loop
        if loop is not None and loop.is_running() and asyncio.get_running_loop() is not loop:
            raise RuntimeError(f"{self._name} belongs to another event loop")
        self._closed = True
        try:
            active = self._active
            if active is not None:
                active.abandon()
            if loop is not None and not loop.is_closed():
                if self._waiting is not None:
                    self._waiting.set()
                if active is not None:
                    active.ready.set()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
