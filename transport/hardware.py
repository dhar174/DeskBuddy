"""Hardware execution management and serialized worker tracking for DeskBuddy."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import threading
from enum import Enum
from typing import Any, Callable

from transport.protocol import (
    ExecutionOutcomeUnknown,
    TransportError,
    TransportErrorCode,
)

logger = logging.getLogger(__name__)


class OperationState(str, Enum):
    """Lifecycle states for hardware operations dispatched to the executor."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HardwareOperation:
    """Represents a discrete physical hardware task dispatched to the worker thread."""

    def __init__(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.state = OperationState.QUEUED
        self.result: Any = None
        self.exception: BaseException | None = None
        self.unresolved: bool = False
        self.lock = threading.Lock()


class HardwareExecutor:
    """Serialized hardware executor preventing concurrent physical servo/audio access.

    Guarantees:
    1. Single-worker serialized access to the Picoh hardware driver.
    2. Cancellation distinction:
       - Queued tasks cancelled before thread pickup are marked CANCELLED and skipped.
       - Running tasks cancelled due to coroutine timeouts are tracked as UNRESOLVED.
    3. Conflicting requests dispatched while an operation is UNRESOLVED are rejected,
       preventing hazardous concurrent or unexpected hardware motions.
    4. Upon completion of the blocked physical task, state is cleaned up and subsequent
       requests are accepted.
    """

    def __init__(self, thread_name_prefix: str = "picoh-hw") -> None:
        self.thread_name_prefix = thread_name_prefix
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name_prefix,
        )
        self._async_lock = asyncio.Lock()
        self._active_operation: HardwareOperation | None = None
        self._unresolved_operation: HardwareOperation | None = None

    @property
    def has_unresolved_operation(self) -> bool:
        """Return True if a synchronous operation was timed out/cancelled while still running."""
        return (
            self._unresolved_operation is not None
            and self._unresolved_operation.state in (OperationState.RUNNING, OperationState.QUEUED)
        )

    @property
    def active_operation(self) -> HardwareOperation | None:
        return self._active_operation

    @property
    def unresolved_operation(self) -> HardwareOperation | None:
        return self._unresolved_operation

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a synchronous hardware driver call serialized in the worker thread."""
        # 1. Reject if an unresolved operation is still occupying the physical driver
        if self.has_unresolved_operation:
            logger.warning(
                "Rejecting hardware operation: prior operation timed out and is still running"
            )
            raise TransportError(
                TransportErrorCode.EXECUTION_OUTCOME_UNKNOWN,
                "Hardware device is busy executing an unresolved prior operation",
            )

        if inspect.iscoroutinefunction(func):
            async with self._async_lock:
                if self.has_unresolved_operation:
                    raise TransportError(
                        TransportErrorCode.EXECUTION_OUTCOME_UNKNOWN,
                        "Hardware device is busy executing an unresolved prior operation",
                    )
                return await func(*args, **kwargs)

        loop = asyncio.get_running_loop()
        op = HardwareOperation(func, args, kwargs)

        async with self._async_lock:
            # Re-check unresolved state under lock
            if self.has_unresolved_operation:
                raise TransportError(
                    TransportErrorCode.EXECUTION_OUTCOME_UNKNOWN,
                    "Hardware device is busy executing an unresolved prior operation",
                )

            done_event = asyncio.Event()

            def _worker() -> None:
                with op.lock:
                    # If cancelled while waiting in queue, do not touch physical hardware
                    if op.state == OperationState.CANCELLED:
                        logger.debug("Skipping cancelled queued hardware operation")
                        return

                    op.state = OperationState.RUNNING
                    self._active_operation = op

                try:
                    res = op.func(*op.args, **op.kwargs)
                    with op.lock:
                        op.result = res
                        op.state = OperationState.COMPLETED
                except BaseException as exc:
                    with op.lock:
                        op.exception = exc
                        op.state = OperationState.FAILED
                finally:
                    with op.lock:
                        self._active_operation = None
                        if self._unresolved_operation is op:
                            logger.info("Unresolved hardware operation completed and cleared")
                            self._unresolved_operation = None
                            op.unresolved = False
                    if not done_event.is_set():
                        try:
                            loop.call_soon_threadsafe(done_event.set)
                        except RuntimeError:
                            # Loop may be closed during tear-down
                            pass

            future = self._thread_pool.submit(_worker)

            try:
                await done_event.wait()
            except asyncio.CancelledError:
                with op.lock:
                    if op.state == OperationState.COMPLETED:
                        # Operation completed on worker thread before or during cancellation
                        return op.result
                    elif op.state == OperationState.FAILED:
                        # Operation failed on worker thread before or during cancellation
                        if op.exception is not None:
                            raise op.exception
                        raise
                    elif op.state == OperationState.RUNNING:
                        # Operation started and cannot be forcibly cancelled on worker thread
                        op.unresolved = True
                        self._unresolved_operation = op
                        logger.warning(
                            "Hardware operation timed out/cancelled while still running on hardware thread"
                        )
                        raise ExecutionOutcomeUnknown(
                            request_id="hardware-operation",
                            command="hardware",
                            message="Hardware operation timed out while running on physical device",
                            code=TransportErrorCode.EXECUTION_OUTCOME_UNKNOWN,
                        )
                    elif op.state == OperationState.QUEUED:
                        cancelled = future.cancel()
                        if cancelled:
                            op.state = OperationState.CANCELLED
                            logger.debug("Successfully cancelled queued hardware operation before start")
                            raise
                        else:
                            # Worker thread already took work item; cannot stop execution
                            op.unresolved = True
                            self._unresolved_operation = op
                            logger.warning(
                                "Hardware operation cancellation failed; worker may be executing on physical device"
                            )
                            raise ExecutionOutcomeUnknown(
                                request_id="hardware-operation",
                                command="hardware",
                                message="Hardware operation cancellation failed; operation may be executing on physical device",
                                code=TransportErrorCode.EXECUTION_OUTCOME_UNKNOWN,
                            )
                raise

            if op.exception is not None:
                raise op.exception
            return op.result

    async def shutdown(self) -> None:
        """Shutdown the internal thread pool executor."""
        self._thread_pool.shutdown(wait=False, cancel_futures=True)


_default_executor: HardwareExecutor | None = None


def get_default_hardware_executor() -> HardwareExecutor:
    """Return the global default HardwareExecutor singleton."""
    global _default_executor
    if _default_executor is None:
        _default_executor = HardwareExecutor()
    return _default_executor


def reset_default_hardware_executor() -> None:
    """Reset the global default HardwareExecutor singleton (primarily for test isolation)."""
    global _default_executor
    if _default_executor is not None:
        try:
            _default_executor._thread_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _default_executor = None


async def run_hardware_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute blocking hardware driver calls off the event loop thread in a serialized worker."""
    executor = get_default_hardware_executor()
    return await executor.execute(func, *args, **kwargs)


async def run_hardware_op(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute hardware operation safely off the event loop thread."""
    executor = get_default_hardware_executor()
    return await executor.execute(func, *args, **kwargs)
