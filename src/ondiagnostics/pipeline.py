from __future__ import annotations

import asyncio
from rich.progress import Progress, TaskID

from . import logger


TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import AsyncIterable
    from typing import Awaitable, Callable, TypeVar

    T = TypeVar("T")
    R = TypeVar("R")


async def producer(
    generator: AsyncIterable[T],
    queue: asyncio.Queue[T | None],
    on_complete: Callable[[T], None] | None = None,
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
                on_complete(item)
    finally:
        await queue.put(None)


async def consumer(
    input_queue: asyncio.Queue[T | None],
    output_queue: asyncio.Queue[R | None] | None,
    worker: Callable[[T], Awaitable[R | None]],
    semaphore: asyncio.Semaphore,
    on_complete: Callable[[T, R | None, bool], None] | None = None,
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


class ProgressQueue[T](asyncio.Queue[T]):
    """Queue that updates a progress bar as items are added and removed."""

    progress: Progress
    put_task_id: TaskID | None
    get_task_id: TaskID | None

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
