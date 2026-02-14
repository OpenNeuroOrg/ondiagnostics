import asyncio
from enum import Enum
from typing import Annotated

import structlog
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn
from typer import Typer, Option

from .graphql import Dataset, create_client, get_dataset_count, datasets_generator
from .pipeline import producer, consumer, ProgressQueue
from .tasks import check_remote_worker

app = Typer()
logger = structlog.get_logger()


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


type DatasetQueue = asyncio.Queue[Dataset | None]


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
    logger.info("Total datasets", count=total)

    check_semaphore = asyncio.Semaphore(20)

    with Progress(
        TextColumn(
            "[progress.description]{task.description} {task.fields[dataset]:8s}"
        ),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        # Create progress tasks
        fetch_task = progress.add_task("Fetching", total=total, dataset="...")
        check_task = progress.add_task("Checking", total=0, dataset="...")

        # Create queues with progress tracking
        fetch_queue: DatasetQueue = ProgressQueue(
            progress=progress,
            put_task_id=fetch_task,
            get_task_id=check_task,
            maxsize=200,
        )

        # Start producer
        producer_task = asyncio.create_task(
            producer(
                generator=datasets_generator(client),
                queue=fetch_queue,
                on_complete=lambda dataset, result, success: progress.update(
                    fetch_task, dataset=dataset.id
                ),
            )
        )

        # Start check consumer
        check_consumer_task = asyncio.create_task(
            consumer(
                input_queue=fetch_queue,
                output_queue=None,
                # output_queue=clone_queue,
                worker=check_remote_worker,
                semaphore=check_semaphore,
                on_complete=lambda dataset, result, success: progress.update(
                    check_task, advance=1, dataset=dataset.id
                ),
            )
        )

        await producer_task
        await check_consumer_task

    logger.info("Pipeline complete")
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
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Pipeline failed", exc_info=e)
        return 1


if __name__ == "__main__":
    app()
