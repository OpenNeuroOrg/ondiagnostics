"""Asyncio tasks for dataset diagnostics."""

import asyncio

import pygit2
import structlog

from .graphql import Dataset

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path

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
    dataset_path = cache_dir / dataset.id

    if dataset_path.exists():
        repo = pygit2.Repository(dataset_path)
        tag_ref = repo.references.get(f"refs/tags/{dataset.tag}")
        if tag_ref:
            log.debug("Existing dataset already has the tag, assume clean")
            return None
        log.debug("Updating existing dataset")
        ret, _, stderr = await git("-C", str(dataset_path), "fetch", "--tags")

        if ret != 0:
            error_msg = stderr.decode()
            log.error("Failed to fetch", stderr=error_msg)
            return None
    else:
        log.debug("Cloning dataset")
        cache_dir.mkdir(parents=True, exist_ok=True)
        ret, stdout, stderr = await git(
            "clone",
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
