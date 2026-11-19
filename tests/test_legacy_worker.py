import asyncio
from threading import Event

import pytest

from deskbuddy import legacy_worker
from deskbuddy.legacy_worker import LegacyWorker, WorkerBusyError, WorkerClosedError


class BlockedCall:
    def __init__(self, *, error: bool = False) -> None:
        self.loop = asyncio.get_running_loop()
        self.started = asyncio.Event()
        self.release = Event()
        self.error = error

    def __call__(self) -> str:
        self.loop.call_soon_threadsafe(self.started.set)
        if not self.release.wait(10):
            raise AssertionError("the test did not release the worker")
        if self.error:
            raise ValueError("late result")
        return "first"


@pytest.fixture(params=[False, True], ids=["owned-thread", "existing-offload"])
async def worker(request: pytest.FixtureRequest):
    owner = LegacyWorker("legacy-test", use_default_executor=request.param)
    owner.start()
    try:
        yield owner
    finally:
        owner.shutdown()


async def wait_for_event(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), 5)


async def test_serializes_and_rejects_work_past_the_waiting_slot(worker: LegacyWorker) -> None:
    first = BlockedCall()
    entered = asyncio.Event()
    taken: list[str] = []

    async def queue_second() -> str:
        entered.set()
        return await worker.submit(lambda: taken.append("second") or "second")

    active = asyncio.create_task(worker.submit(first))
    try:
        await wait_for_event(first.started)
        waiting = asyncio.create_task(queue_second())
        await wait_for_event(entered)
        assert worker.busy
        assert not taken
        with pytest.raises(WorkerBusyError):
            await worker.submit(lambda: taken.append("third"))
        first.release.set()
        assert await asyncio.wait_for(active, 5) == "first"
        assert await asyncio.wait_for(waiting, 5) == "second"
        assert taken == ["second"]
        assert not worker.busy
    finally:
        first.release.set()


async def test_cancelled_waiter_never_enters_worker(worker: LegacyWorker) -> None:
    first = BlockedCall()
    entered = asyncio.Event()
    taken = Event()

    async def queue_cancelled() -> None:
        entered.set()
        await worker.submit(taken.set)

    active = asyncio.create_task(worker.submit(first))
    try:
        await wait_for_event(first.started)
        waiting = asyncio.create_task(queue_cancelled())
        await wait_for_event(entered)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        first.release.set()
        assert await asyncio.wait_for(active, 5) == "first"
        assert await worker.submit(lambda: "after cancellation") == "after cancellation"
        assert not taken.is_set()
    finally:
        first.release.set()


async def test_timeout_keeps_failed_abandoned_call_owned_until_real_completion(
    worker: LegacyWorker,
) -> None:
    first = BlockedCall(error=True)
    stop = Event()
    entered = asyncio.Event()
    timeout = asyncio.timeout(None)

    async def time_out() -> str:
        async with timeout:
            return await worker.submit(first, stop=stop)

    async def queue_before_timeout() -> str:
        entered.set()
        return await worker.submit(lambda: "second")

    active = asyncio.create_task(time_out())
    try:
        await wait_for_event(first.started)
        waiting = asyncio.create_task(queue_before_timeout())
        await wait_for_event(entered)
        timeout.reschedule(asyncio.get_running_loop().time())
        with pytest.raises(TimeoutError):
            await active
        assert stop.is_set()
        assert worker.busy and worker.abandoned
        with pytest.raises(WorkerBusyError):
            await worker.submit(lambda: "new conflicting work")
        assert not waiting.done()
        first.release.set()
        assert await asyncio.wait_for(waiting, 5) == "second"
        assert not worker.busy and not worker.abandoned
        assert await worker.submit(lambda: "third") == "third"
    finally:
        first.release.set()


async def test_shutdown_detaches_caller_and_keeps_resource_busy(worker: LegacyWorker) -> None:
    first = BlockedCall()
    stop = Event()
    waiting_started = asyncio.Event()
    called = Event()

    async def queue_second() -> None:
        waiting_started.set()
        await worker.submit(called.set)

    active = asyncio.create_task(worker.submit(first, stop=stop))
    try:
        await wait_for_event(first.started)
        waiting = asyncio.create_task(queue_second())
        await wait_for_event(waiting_started)
        worker.shutdown()
        for caller in (active, waiting):
            with pytest.raises(WorkerClosedError):
                await asyncio.wait_for(caller, 5)
        assert stop.is_set()
        assert worker.busy and worker.abandoned
        assert not worker.can_replace
        with pytest.raises(WorkerClosedError):
            await worker.submit(called.set)
        assert not called.is_set()
    finally:
        first.release.set()


async def test_cancel_between_submission_and_callable_entry(
    worker: LegacyWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    approaching_entry = asyncio.Event()
    continue_entry = Event()
    callable_started = Event()
    original = legacy_worker._Operation.run

    def delayed_entry(job, operation) -> None:
        loop.call_soon_threadsafe(approaching_entry.set)
        if not continue_entry.wait(10):
            raise AssertionError("the test did not release worker entry")
        original(job, operation)

    monkeypatch.setattr(legacy_worker._Operation, "run", delayed_entry)
    called = asyncio.create_task(worker.submit(callable_started.set))
    try:
        await wait_for_event(approaching_entry)
        called.cancel()
        with pytest.raises(asyncio.CancelledError):
            await called
        continue_entry.set()
        assert await asyncio.wait_for(worker.submit(lambda: "next"), 5) == "next"
        assert not callable_started.is_set()
    finally:
        continue_entry.set()


async def test_cancel_after_completion_before_delivery_discards_result(
    worker: LegacyWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion_pending = asyncio.Event()
    retained_completion = []
    complete = worker._complete

    def delay_delivery(job) -> None:
        retained_completion.append(job)
        completion_pending.set()

    monkeypatch.setattr(worker, "_complete", delay_delivery)
    caller = asyncio.create_task(worker.submit(lambda: "completed without delivery"))
    await wait_for_event(completion_pending)
    assert not worker.busy
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert len(retained_completion) == 1
    complete(retained_completion[0])
    monkeypatch.setattr(worker, "_complete", complete)
    assert await asyncio.wait_for(worker.submit(lambda: "new result"), 5) == "new result"


def test_owner_tracks_running_work_after_its_loop_closes() -> None:
    owner = LegacyWorker("legacy-closed-loop-test")
    released = Event()
    finished = Event()

    def operation() -> None:
        loop.call_soon_threadsafe(started.set)
        if not released.wait(10):
            raise AssertionError("the test did not release the worker")

    async def start_and_abandon() -> None:
        owner.start()
        caller = asyncio.create_task(owner.submit(operation))
        await wait_for_event(started)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert owner._active is not None
        owner._active.result.add_done_callback(lambda _: finished.set())

    loop = asyncio.new_event_loop()
    started = asyncio.Event()
    try:
        loop.run_until_complete(start_and_abandon())
        loop.close()
        assert owner.busy and owner.abandoned
        assert not owner.can_replace
        owner.shutdown()
        released.set()
        assert finished.wait(5)
        assert not owner.busy and not owner.abandoned
        assert owner.can_replace
    finally:
        released.set()
        loop.close()
        owner.shutdown()
