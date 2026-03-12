"""S3 tasks for dataset diagnostics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import aioboto3
import pygit2

from .. import logger
from ..graphql import Dataset

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import AsyncIterator, TypedDict

    class S3ObjectKey(TypedDict):
        Key: str

    class S3DeleteRequest(TypedDict):
        Objects: list[S3ObjectKey]

    class DeletionResult(TypedDict):
        Deleted: list[S3ObjectKey]


@dataclass
class S3CleanupPlan:
    """Plan for S3 cleanup - carries both dataset and cleanup info."""

    dataset: Dataset
    files_to_delete: list[str]

    @property
    def id(self) -> str:
        return self.dataset.id


async def list_s3_objects_pages(
    session: aioboto3.Session,
    bucket_name: str,
    prefix: str,
) -> AsyncIterator[list[str]]:
    """
    Yield S3 keys one page at a time using pagination.

    Args:
        session: aioboto3 Session
        bucket_name: S3 bucket name
        prefix: Prefix to filter objects

    Yields:
        Lists of S3 object keys, one page at a time
    """
    async with session.client("s3") as s3_client:
        paginator = s3_client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" in page:
                yield [obj["Key"] for obj in page["Contents"]]


async def plan_cleanup(
    dataset: Dataset,
    cache_dir: Path,
    session: aioboto3.Session,
    bucket_name: str,
) -> S3CleanupPlan | None:
    """
    Plan which S3 files should be deleted (worker for pipeline).

    Args:
        dataset: Dataset to clean up
        cache_dir: Directory containing cloned repositories
        session: aioboto3 Session
        bucket_name: S3 bucket name

    Returns:
        Cleanup plan, or None on failure
    """
    log = logger.bind(dataset=dataset.id, tag=dataset.tag)
    dataset_path = cache_dir / f"{dataset.id}.git"
    prefix = f"{dataset.id}/"

    # Get the tree from the git repository (blocking I/O, run in thread)
    def get_git_tree() -> pygit2.Tree:
        repo = pygit2.Repository(dataset_path)
        tag_ref = repo.references.get(f"refs/tags/{dataset.tag}")
        if not tag_ref:
            raise ValueError(f"Tag not found: {dataset.tag}")
        return tag_ref.peel().tree

    try:
        tree = await asyncio.to_thread(get_git_tree)
    except ValueError:
        log.error("Tag not found in dataset", tag=dataset.tag)
        return None
    except Exception as e:
        log.error("Failed to read repository", exc_info=e)
        return None

    # Helper to check which keys in a page should be deleted (blocking tree lookups)
    def check_page(keys: list[str]) -> list[str]:
        to_delete = []
        for key in keys:
            if not key.startswith(prefix):
                continue
            fname = key[len(prefix) :]
            if fname and fname not in tree:
                to_delete.append(key)
                log.debug("Will delete", filename=fname)
        return to_delete

    # Process S3 objects page by page
    tasks = []
    async with asyncio.TaskGroup() as tg:
        async for page in list_s3_objects_pages(session, bucket_name, prefix):
            tasks.append(tg.create_task(asyncio.to_thread(check_page, page)))

    files_to_delete = [key for task in tasks for key in task.result()]

    if not files_to_delete:
        log.debug("No files to delete")
        return None

    return S3CleanupPlan(
        dataset=dataset,
        files_to_delete=files_to_delete,
    )


async def execute_cleanup(
    plan: S3CleanupPlan,
    session: aioboto3.Session,
    bucket_name: str,
    dry_run: bool = False,
) -> Dataset | None:
    """
    Execute an S3 cleanup plan (worker for pipeline).

    Args:
        plan: Cleanup plan to execute
        session: aioboto3 Session
        bucket_name: S3 bucket name
        dry_run: If True, don't actually delete

    Returns:
        The dataset if successful, None on failure
    """
    log = logger.bind(dataset=plan.dataset.id, tag=plan.dataset.tag, dry_run=dry_run)

    # Batch delete using aioboto3
    async def delete_batch(keys: list[str]) -> int:
        objects: list[S3ObjectKey] = [{"Key": key} for key in keys]

        if not dry_run:
            async with session.client("s3") as s3_client:
                response = await s3_client.delete_objects(
                    Bucket=bucket_name, Delete={"Objects": objects}
                )
                deleted = response.get("Deleted", [])
        else:
            deleted = objects

        for obj in deleted:
            log.debug("Deleted", key=obj["Key"])
        return len(deleted)

    # Process in batches of 1000 (S3 limit)
    batch_size = 1000
    tasks = []
    async with asyncio.TaskGroup() as tg:
        for i in range(0, len(plan.files_to_delete), batch_size):
            batch = plan.files_to_delete[i : i + batch_size]
            tasks.append(tg.create_task(delete_batch(batch)))

    deleted_count = sum(task.result() for task in tasks)

    log.info("S3 cleanup complete", deleted=deleted_count)

    return plan.dataset
