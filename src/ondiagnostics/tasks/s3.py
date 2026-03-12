"""S3 tasks for dataset diagnostics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pygit2

from .. import logger
from ..graphql import Dataset

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Protocol, TypedDict

    class S3ObjectKey(TypedDict):
        Key: str

    class S3DeleteRequest(TypedDict):
        Objects: list[S3ObjectKey]

    class DeletionResult(TypedDict):
        Deleted: list[S3ObjectKey]

    class S3Object(Protocol):
        key: str

    class S3Objects(Protocol):
        def filter(self, Prefix: str) -> Iterable[S3Object]: ...

    class Bucket(Protocol):
        name: str
        objects: S3Objects

        def delete_objects(self, Delete: S3DeleteRequest) -> DeletionResult: ...


@dataclass
class S3CleanupPlan:
    """Plan for S3 cleanup - carries both dataset and cleanup info."""

    dataset: Dataset
    files_to_delete: list[str]

    @property
    def id(self) -> str:
        return self.dataset.id


async def plan_cleanup(
    dataset: Dataset,
    cache_dir: Path,
    s3_bucket: Bucket,
) -> S3CleanupPlan | None:
    """
    Plan which S3 files should be deleted (worker for pipeline).

    Args:
        dataset: Dataset to clean up
        cache_dir: Directory containing cloned repositories
        s3_bucket: S3 bucket to check

    Returns:
        Cleanup plan, or None on failure
    """
    log = logger.bind(dataset=dataset.id, tag=dataset.tag)
    dataset_path = cache_dir / f"{dataset.id}.git"

    # Get the tree from the git repository
    try:
        repo = pygit2.Repository(dataset_path)
        tag_ref = repo.references.get(f"refs/tags/{dataset.tag}")
        if not tag_ref:
            log.error("Tag not found in dataset", tag=dataset.tag)
            return None

        tree = tag_ref.peel().tree
    except Exception as e:
        log.error("Failed to read repository", exc_info=e)
        return None

    prefix = f"{dataset.id}/"

    # Get list of S3 objects
    def list_objects() -> list[str]:
        return [obj.key for obj in s3_bucket.objects.filter(Prefix=prefix)]

    s3_keys = await asyncio.to_thread(list_objects)

    # Determine what to delete
    files_to_delete = []

    for key in s3_keys:
        if not key.startswith(prefix):
            continue

        fname = key[len(prefix) :]
        if fname and fname not in tree:
            files_to_delete.append(key)
            log.debug("Will delete", filename=fname)

    if not files_to_delete:
        log.debug("No files to delete")
        return None

    return S3CleanupPlan(
        dataset=dataset,
        files_to_delete=files_to_delete,
    )


async def execute_cleanup(
    plan: S3CleanupPlan,
    s3_bucket: Bucket,
    dry_run: bool = False,
) -> Dataset | None:
    """
    Execute an S3 cleanup plan (worker for pipeline).

    Args:
        plan: Cleanup plan to execute
        s3_bucket: S3 bucket to delete from
        dry_run: If True, don't actually delete

    Returns:
        The dataset if successful, None on failure
    """
    log = logger.bind(dataset=plan.dataset.id, tag=plan.dataset.tag, dry_run=dry_run)

    # Batch delete
    async def delete_batch(keys: list[str]) -> int:
        def do_delete() -> int:
            objects: list[S3ObjectKey] = [{"Key": key} for key in keys]
            if not dry_run:
                result = s3_bucket.delete_objects(Delete={"Objects": objects})
                deleted = result.get("Deleted", [])
            else:
                deleted = objects
            for obj in deleted:
                log.info("Deleted", key=obj["Key"])
            return len(deleted)

        return await asyncio.to_thread(do_delete)

    # Process in batches of 1000
    batch_size = 1000
    delete_tasks = []
    for i in range(0, len(plan.files_to_delete), batch_size):
        batch = plan.files_to_delete[i : i + batch_size]
        delete_tasks.append(delete_batch(batch))

    # Run all delete batches concurrently
    try:
        results = await asyncio.gather(*delete_tasks)
        deleted_count = sum(results)

        log.info(
            "S3 cleanup complete",
            deleted=deleted_count,
        )

        return plan.dataset
    except Exception as e:
        log.error("Failed to execute S3 cleanup", exc_info=e)
        return None
