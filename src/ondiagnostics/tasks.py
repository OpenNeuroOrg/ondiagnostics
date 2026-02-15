"""Asyncio tasks for dataset diagnostics."""

import asyncio

import pygit2
import structlog

from .graphql import Dataset

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path
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


logger = structlog.get_logger()


async def git(*args: str) -> tuple[int, bytes, bytes]:
    """Run a git command and return the exit code, stdout, and stderr."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    assert proc.returncode is not None
    return proc.returncode, stdout, stderr


async def check_remote(dataset: Dataset) -> Dataset | None:
    """
    Check if the git remote has the expected tag and commit hash.

    Returns:
        The dataset if valid, None if validation fails
    """
    log = logger.bind(dataset=dataset.id, tag=dataset.tag)
    repo = f"https://github.com/OpenNeuroDatasets/{dataset.id}.git"

    ret, stdout, stderr = await git("ls-remote", "--exit-code", repo, dataset.tag)

    if ret:
        if b"Repository not found" in stderr:
            log.error("Missing repository")
        else:
            log.error("Missing latest tag")
        return None

    if not stdout.strip():
        log.error("Empty response from git ls-remote")
        return None

    stdout_text = stdout.decode().strip()

    shasum, ref = stdout_text.split(maxsplit=1)

    if shasum != dataset.hexsha:
        log.warning(f"mismatch: {shasum[:7]}({ref[10:]}) != {dataset.hexsha[:7]}")
        return None

    if ref != f"refs/tags/{dataset.tag}":
        log.error("Tag mismatch", stdout=stdout_text)
        return None

    return dataset


async def clone_dataset(dataset: Dataset, cache_dir: Path) -> Dataset | None:
    """
    Clone or update a dataset to the cache directory.

    Args:
        dataset: Dataset to clone
        cache_dir: Directory to clone into

    Returns:
        The dataset if successful
    """
    log = logger.bind(dataset=dataset.id, tag=dataset.tag)
    repo_url = f"https://github.com/OpenNeuroDatasets/{dataset.id}.git"
    dataset_path = cache_dir / f"{dataset.id}.git"

    if dataset_path.exists():
        repo = pygit2.Repository(dataset_path)
        tag_ref = repo.references.get(f"refs/tags/{dataset.tag}")
        if tag_ref:
            log.debug("Existing dataset already has the tag, assume clean")
            return dataset
        log.debug("Updating existing dataset")
        ret, _, stderr = await git(
            "-C",
            str(dataset_path),
            "fetch",
            "origin",
            dataset.tag,
        )

        if ret != 0:
            error_msg = stderr.decode()
            log.error("Failed to fetch", stderr=error_msg)
            return None
    else:
        log.debug("Cloning dataset")
        cache_dir.mkdir(parents=True, exist_ok=True)
        ret, stdout, stderr = await git(
            "clone",
            "--bare",
            "--filter=blob:none",
            "--depth=1",
            "--branch",
            dataset.tag,
            repo_url,
            str(dataset_path),
        )

        if ret != 0:
            error_msg = stderr.decode()
            log.error("Failed to clone", stderr=error_msg)
            return None

    return dataset


async def s3_cleanup(
    dataset: Dataset, cache_dir: Path, s3_bucket: Bucket, dry_run: bool = False
) -> Dataset | None:
    """
    Clean up S3 files not in the repository tag.

    Args:
        dataset: Dataset to clean up
        cache_dir: Directory containing cloned repositories
        s3_bucket: boto3 Bucket resource
        dry_run: If True, don't actually delete files

    Returns:
        The dataset if successful, None on failure
    """
    log = logger.bind(dataset=dataset.id, tag=dataset.tag)
    dataset_path = cache_dir / f"{dataset.id}.git"

    # Get the tree from the git repository
    repo = pygit2.Repository(dataset_path)
    tag_ref = repo.references.get(f"refs/tags/{dataset.tag}")
    if not tag_ref:
        log.error("Tag not found in dataset", tag=dataset.tag)
        return None

    tree = tag_ref.peel().tree
    prefix = f"{dataset.id}/"

    # log.info("Checking S3 bucket", bucket=s3_bucket.name, prefix=prefix)

    # Collect objects to delete (don't delete in the iteration loop)
    objects_to_delete: list[S3ObjectKey] = []
    kept_count = 0

    # Run S3 operations in thread pool since boto3 is synchronous
    def list_and_classify() -> None:
        nonlocal kept_count
        for obj in s3_bucket.objects.filter(Prefix=prefix):
            fname = obj.key[len(prefix) :]
            if fname and fname not in tree:
                objects_to_delete.append({"Key": obj.key})
                log.debug("Will delete", filename=fname)
            else:
                kept_count += 1

    await asyncio.to_thread(list_and_classify)

    # Batch delete objects (S3 supports up to 1000 objects per request)
    deleted_count = 0
    if objects_to_delete and not dry_run:

        async def delete_batch(batch: list[S3ObjectKey]) -> DeletionResult:
            def do_delete() -> DeletionResult:
                return s3_bucket.delete_objects(Delete={"Objects": batch})

            return await asyncio.to_thread(do_delete)

        # Process in batches of 1000
        batch_size = 1000
        delete_tasks = []
        for i in range(0, len(objects_to_delete), batch_size):
            batch = objects_to_delete[i : i + batch_size]
            delete_tasks.append(delete_batch(batch))

        # Run all delete batches concurrently
        results = await asyncio.gather(*delete_tasks)

        # Count successful deletions
        for result in results:
            if "Deleted" in result:
                deleted_count += len(result["Deleted"])
                for deleted in result["Deleted"]:
                    log.info("Deleted", key=deleted["Key"])
    else:
        deleted_count = len(objects_to_delete)

    if deleted_count:
        log.info(
            "S3 cleanup complete",
            deleted=deleted_count,
            kept=kept_count,
            dry_run=dry_run,
        )
    else:
        log.debug("Clean dataset", kept=kept_count)
    return dataset
