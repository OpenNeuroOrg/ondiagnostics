from __future__ import annotations
import asyncio
from enum import Enum
from pathlib import Path
from typing import Annotated

import structlog
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn
from typer import Typer, Argument, Option

from . import logger
from .awsconfig import AWSConfig
from .graphql import (
    Dataset,
    create_client,
    get_dataset_count,
    datasets_generator,
    datasets_by_ids_generator,
)
from .pipeline import producer, consumer, ProgressQueue
from .tasks.git import check_remote, clone_dataset
from .tasks.s3 import plan_cleanup, execute_cleanup

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import AsyncIterator, Awaitable, Callable, Protocol, TypeVar

    class HasId(Protocol):
        """Protocol for objects with an ID."""

        @property
        def id(self) -> str: ...

    T = TypeVar("T", bound=HasId)
    R = TypeVar("R", bound=HasId)


app = Typer()


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def add_producer(
    name: str,
    generator: AsyncIterator[Dataset],
    progress: Progress,
    total: int,
    maxsize: int = 200,
) -> ProgressQueue[Dataset | None]:
    """Create a producer that feeds datasets from an async generator into a ProgressQueue."""
    task_id = progress.add_task(name, total=total, dataset="...")
    out_queue: ProgressQueue[Dataset | None] = ProgressQueue(
        progress=progress, put_task_id=task_id, maxsize=maxsize
    )

    asyncio.create_task(
        producer(
            generator,
            out_queue,
            on_complete=lambda d: progress.update(task_id, dataset=d.id),
        )
    )
    return out_queue


def add_consumer(
    name: str,
    func: Callable[[T], Awaitable[R | None]],
    input_queue: ProgressQueue[T | None],
    max_concurrent: int,
) -> ProgressQueue[R | None]:
    """Create a consumer that processes datasets from an input queue and routes successful results to an output queue."""
    progress = input_queue.progress
    task_id = progress.add_task(name, total=0, dataset="...")
    input_queue.get_task_id = task_id

    out_queue: ProgressQueue[R | None] = ProgressQueue(
        progress=progress,
        maxsize=max_concurrent * 10,
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    asyncio.create_task(
        consumer(
            input_queue=input_queue,
            output_queue=out_queue,
            worker=func,
            semaphore=semaphore,
            on_complete=lambda d, r, s: progress.update(
                task_id, advance=1, dataset=d.id
            ),
        )
    )
    return out_queue


async def run_pipeline(
    cache_dir: Path | None = None,
    aws_config: AWSConfig | None = None,
    dry_run: bool = False,
    dataset_ids: list[str] | None = None,
) -> int:
    """
    Run the OpenNeuro dataset validation pipeline.

    Args:
        cache_dir: Directory to cache cloned repositories
        aws_config: AWS configuration for S3 access
        dry_run: If True, don't actually delete S3 files

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Create GraphQL client
    client = create_client()

    if dataset_ids:
        total = len(dataset_ids)
        dataset_source = datasets_by_ids_generator(client, dataset_ids)
        logger.debug("Using dataset IDs from command line", count=total)
    else:
        total = await get_dataset_count(client)
        dataset_source = datasets_generator(client)
        logger.debug("Processing all datasets", count=total)

    with Progress(
        TextColumn(
            "[progress.description]{task.description} {task.fields[dataset]:8s}"
        ),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        queue = add_producer("Fetching", dataset_source, progress, total=total)
        queue = add_consumer("Checking", check_remote, queue, 20)

        if aws_config and cache_dir:
            # Create aioboto3 session for S3 operations
            session = aws_config.create_session()
            bucket_name = aws_config.AWS_S3_BUCKET_NAME

            queue = add_consumer(
                "Cloning", lambda d: clone_dataset(d, cache_dir), queue, 10
            )
            plan_queue = add_consumer(
                "Checking S3",
                lambda d: plan_cleanup(d, cache_dir, session, bucket_name),
                queue,
                10,
            )
            queue = add_consumer(
                "Cleaning S3",
                lambda plan: execute_cleanup(plan, session, bucket_name, dry_run),
                plan_queue,
                5,
            )

        while await queue.get() is not None:
            pass

    return 0


@app.command()
def check_sync(
    dry_run: Annotated[bool, Option(help="Run without making changes")] = False,
    log_level: Annotated[LogLevel, Option(help="Set logging level")] = LogLevel.INFO,
    dataset_ids: Annotated[
        list[str] | None, Argument(help="Optional list of dataset IDs")
    ] = None,
) -> int:
    """
    Check synchronization status.
    """
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))
    logger.debug("Starting check-sync", dry_run=dry_run, log_level=log_level.value)
    # Run the async pipeline
    try:
        return asyncio.run(run_pipeline(dataset_ids=dataset_ids))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Pipeline failed", exc_info=e)
        return 1


@app.command()
def clean_s3(
    config: Annotated[Path | None, Option(help="Path to configuration file")] = None,
    dry_run: Annotated[bool, Option(help="Run without making changes")] = False,
    log_level: Annotated[LogLevel, Option(help="Set logging level")] = LogLevel.INFO,
    dataset_ids: Annotated[
        list[str] | None, Argument(help="Optional list of dataset IDs")
    ] = None,
) -> int:
    """
    Clean up S3 files that don't match the latest git tag.
    """
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))
    logger.debug("Starting clean-s3", dry_run=dry_run, log_level=log_level.value)

    aws_config = AWSConfig.from_file(config) if config else AWSConfig.from_env()

    try:
        return asyncio.run(
            run_pipeline(
                cache_dir=Path.home() / ".cache" / "ondiagnostics",
                aws_config=aws_config,
                dry_run=dry_run,
                dataset_ids=dataset_ids,
            )
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Pipeline failed", exc_info=e)
        return 1


if __name__ == "__main__":
    app()
