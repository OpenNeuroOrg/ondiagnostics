import asyncio
from enum import Enum
from pathlib import Path
from typing import Annotated

import boto3
import structlog
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn
from typer import Typer, Option

from .awsconfig import AWSConfig
from .graphql import Dataset, create_client, get_dataset_count, datasets_generator
from .pipeline import producer, consumer, ProgressQueue
from .tasks import check_remote, clone_dataset, s3_cleanup

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import AsyncIterator, Awaitable, Callable, TypeVar

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


async def run_pipeline(
    cache_dir: Path | None = None,
    bucket: boto3.resources.base.ServiceResource | None = None,
    dry_run: bool = False,
) -> int:
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

        if bucket and cache_dir:
            queue = add_consumer(
                "Cloning", lambda d: clone_dataset(d, cache_dir), queue, 10
            )
            queue = add_consumer(
                "Cleaning S3",
                lambda d: s3_cleanup(d, cache_dir, bucket, dry_run),
                queue,
                5,
            )

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


@app.command()
def clean_s3(
    config: Annotated[Path | None, Option(help="Path to configuration file")] = None,
    dry_run: Annotated[bool, Option(help="Run without making changes")] = False,
    log_level: Annotated[LogLevel, Option(help="Set logging level")] = LogLevel.INFO,
    ## Would be good to add, but we will need a new GraphQL query to restrict ourselves
    ## to specific datasets
    # dataset_ids: Annotated[
    #     list[str] | None, Argument(help="Optional list of dataset IDs")
    # ] = None,
):
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))
    logger.debug("Starting clean-s3", dry_run=dry_run, log_level=log_level.value)

    aws_config = AWSConfig.from_file(config) if config else AWSConfig.from_env()

    s3 = boto3.resource(
        "s3",
        region_name=aws_config.AWS_REGION,
        aws_access_key_id=aws_config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=aws_config.AWS_SECRET_ACCESS_KEY,
    )
    s3_bucket = s3.Bucket(aws_config.AWS_S3_BUCKET_NAME)
    try:
        return asyncio.run(
            run_pipeline(
                cache_dir=Path.home() / ".cache" / "ondiagnostics",
                bucket=s3_bucket,
                dry_run=dry_run,
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
