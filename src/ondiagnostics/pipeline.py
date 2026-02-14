from __future__ import annotations

import asyncio
import structlog
from rich.progress import Progress, TaskID


TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import AsyncIterator, Awaitable, Callable, Protocol, TypeVar

    T = TypeVar("T")
    R = TypeVar("R")

    class ProgressCallback(Protocol):
        """Protocol for progress callbacks."""

        def __call__(self, value: T, result: R | None, success: bool) -> None:
            """Called after a worker completes processing a value."""
            ...


async def producer(
    generator: AsyncIterator[T],
    queue: asyncio.Queue[T | None],
    on_complete: ProgressCallback | None = None,
) -> None:
    """
    Generic producer that consumes an async generator and puts items in a queue.

    Args:
        generator: Async generator that yields items
        queue: Queue to put items into
    """
    try:
        async for item in generator:
            await queue.put(item)
            if on_complete:
                on_complete(item, None, True)
    finally:
        await queue.put(None)


async def consumer(
    input_queue: asyncio.Queue[T | None],
    output_queue: asyncio.Queue[R | None] | None,
    worker: Callable[[T], Awaitable[R | None]],
    semaphore: asyncio.Semaphore,
    on_complete: ProgressCallback | None = None,
) -> None:
    """
    Generic consumer that processes datasets with a worker function.

    Args:
        input_queue: Queue to consume datasets from
        output_queue: Queue to route successful results to (None for terminal consumers)
        worker: Async function that processes a dataset (raises on failure)
        semaphore: Semaphore to limit concurrent workers
    """

    async def process_single_input(value: T) -> None:
        try:
            result, success = None, False
            try:
                res = await worker(value)
                success = res is not None
                if success and output_queue:
                    await output_queue.put(res)
            except Exception as e:
                logger = structlog.get_logger()
                logger.error("Worker failed", value=value, exc_info=e)
            finally:
                if on_complete:
                    on_complete(value, result, success)
        finally:
            semaphore.release()

    async with asyncio.TaskGroup() as tg:
        while (value := await input_queue.get()) is not None:
            await semaphore.acquire()
            tg.create_task(process_single_input(value))

    # Signal downstream consumer after all tasks complete
    if output_queue:
        await output_queue.put(None)


class ProgressQueue[T](asyncio.Queue):
    """Queue that updates a progress bar as items are added and removed."""

    def __init__(
        self,
        progress: Progress,
        put_task_id: TaskID | None = None,
        get_task_id: TaskID | None = None,
        maxsize: int = 0,
    ):
        super().__init__(maxsize=maxsize)
        self.progress = progress
        self.put_task_id = put_task_id
        self.get_task_id = get_task_id

    async def put(self, item: T) -> None:
        """Put an item and update the put progress bar."""
        await super().put(item)
        if self.put_task_id is not None and item is not None:
            self.progress.update(self.put_task_id, advance=1)
        if self.get_task_id is not None and item is not None:
            total = self.progress.tasks[self.get_task_id].total or 0
            self.progress.update(self.get_task_id, total=total + 1)
