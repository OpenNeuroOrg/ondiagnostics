import asyncio
from enum import Enum
from typing import Annotated

import structlog
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn
from typer import Typer, Option

from .graphql import Dataset, create_client, get_dataset_count, datasets_generator
from .pipeline import producer, consumer, ProgressQueue
from .tasks import check_remote

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Awaitable, Callable, TypeVar

    T = TypeVar("T")
    R = TypeVar("R")

app = Typer()
logger = structlog.get_logger()


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


type DatasetQueue = ProgressQueue[Dataset | None]


def add_producer(
    name: str,
    generator: AsyncIterator[Dataset],
    progress: Progress,
    total: int,
    maxsize: int = 200,
) -> DatasetQueue:
    """Create a producer that feeds datasets from an async generator into a ProgressQueue."""
    task_id = progress.add_task(name, total=total, dataset="...")
    out_queue: DatasetQueue = ProgressQueue(
        progress=progress, put_task_id=task_id, maxsize=maxsize
    )

    def on_complete(dataset: Dataset, result: Dataset | None, success: bool) -> None:
        progress.update(task_id, dataset=dataset.id)

    asyncio.create_task(producer(generator, out_queue, on_complete=on_complete))
    return out_queue


def add_consumer(
    name: str,
    func: Callable[[Dataset], Awaitable[Dataset | None]],
    input_queue: DatasetQueue,
    max_concurrent: int,
) -> DatasetQueue:
    """Create a consumer that processes datasets from an input queue and routes successful results to an output queue."""
    progress = input_queue.progress
    task_id = progress.add_task(name, total=0, dataset="...")
    input_queue.get_task_id = task_id

    out_queue: DatasetQueue = ProgressQueue(
        progress=progress,
        maxsize=max_concurrent * 10,
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    def on_complete(dataset: Dataset, result: Dataset | None, success: bool) -> None:
        progress.update(task_id, advance=1, dataset=dataset.id)

    asyncio.create_task(
        consumer(
            input_queue=input_queue,
            output_queue=out_queue,
            worker=func,
            semaphore=semaphore,
            on_complete=on_complete,
        )
    )
    return out_queue


async def run_pipeline() -> int:
    """
    Run the OpenNeuro dataset validation pipeline.

    Args:
        aws_config: AWS configuration for S3 access
        cache_dir: Directory to cache cloned repositories
        dry_run: If True, don't actually delete S3 files

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Create GraphQL client
    client = create_client()

    # Get total dataset count for progress tracking
    total = await get_dataset_count(client)
    logger.debug("Total datasets", count=total)

    with Progress(
        TextColumn(
            "[progress.description]{task.description} {task.fields[dataset]:8s}"
        ),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        queue = add_producer(
            "Fetching", datasets_generator(client), progress, total=total
        )
        queue = add_consumer("Checking", check_remote, queue, 30)

        while await queue.get() is not None:
            pass

    return 0


@app.command()
def check_sync(
    dry_run: Annotated[bool, Option(help="Run without making changes")] = False,
    log_level: Annotated[LogLevel, Option(help="Set logging level")] = LogLevel.INFO,
):
    """
    Check synchronization status.
    """
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))
    logger.debug("Starting check-sync", dry_run=dry_run, log_level=log_level.value)
    # Run the async pipeline
    try:
        return asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Pipeline failed", exc_info=e)
        return 1


if __name__ == "__main__":
    app()
