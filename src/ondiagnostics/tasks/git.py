from dataclasses import dataclass, field
from hashlib import sha1

import pygit2

from .. import logger
from ..graphql import Dataset
from ..subprocs import git

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class GitRefs:
    head: str | None
    branches: dict[str, str] = field(default_factory=dict)  # branch name -> SHA
    tags: dict[str, str] = field(default_factory=dict)  # tag name -> SHA


def worker_from_id(dataset_id: str) -> int:
    """Generate a worker number from the dataset ID."""
    # Use SHA-1 hash to get a consistent integer from the dataset ID
    hash_bytes = sha1(dataset_id.encode()).digest()
    return hash_bytes[-1] % 4


async def check_remote(dataset: Dataset) -> Dataset | None:
    """
    Check if the git remote has the expected tag and commit hash.

    Returns:
        The dataset if valid, None if validation fails
    """
    log = logger.bind(
        dataset=dataset.id, tag=dataset.tag, worker=worker_from_id(dataset.id)
    )
    repo = f"https://github.com/OpenNeuroDatasets/{dataset.id}.git"

    result = await git("ls-remote", "--exit-code", repo, dataset.tag)

    if result.returncode:
        if b"Repository not found" in result.stderr:
            log.error("Missing repository")
        else:
            log.error("Missing latest tag")
        return None

    if not result.stdout.strip():
        log.error("Empty response from git ls-remote")
        return None

    stdout_text = result.stdout.decode().strip()

    shasum, ref = stdout_text.split(maxsplit=1)

    if shasum != dataset.hexsha:
        log.warning(f"mismatch: {shasum[:7]}({ref[10:]}) != {dataset.hexsha[:7]}")
        return None

    if ref != f"refs/tags/{dataset.tag}":
        log.error("Tag mismatch", stdout=stdout_text)
        return None

    return dataset


async def list_refs(repo_url: str) -> GitRefs | None:
    """Run git ls-remote --symref and return structured ref map.

    Returns None if the repository is not found or the command fails.
    """
    log = logger.bind(repo_url=repo_url)

    result = await git("ls-remote", "--symref", repo_url)

    if result.returncode:
        if b"Repository not found" in result.stderr:
            log.error("Missing repository")
        else:
            log.error("git ls-remote failed", returncode=result.returncode)
        return None

    if not result.stdout.strip():
        log.error("Empty response from git ls-remote")
        return None

    lines = result.stdout.decode().strip().split("\n")
    head_ref = None
    branches: dict[str, str] = {}
    tags: dict[str, str] = {}

    for line in lines:
        parts = line.split()
        if parts[0] == "ref:":
            # Format: "ref: refs/heads/master\tHEAD"
            ref_target = parts[1]
            head_ref = ref_target.split("/")[-1]
            continue
        if len(parts) != 2:
            continue
        sha, ref = parts
        if ref == "HEAD":
            continue
        elif ref.startswith("refs/heads/"):
            branch_name = ref.replace("refs/heads/", "")
            branches[branch_name] = sha
        elif ref.startswith("refs/tags/"):
            tag_name = ref.replace("refs/tags/", "")
            tags[tag_name] = sha

    return GitRefs(head=head_ref, branches=branches, tags=tags)


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
        result = await git(
            "-C",
            str(dataset_path),
            "fetch",
            "origin",
            dataset.tag,
        )

        if result.returncode != 0:
            log.error("Failed to fetch", stderr=result.stderr.decode())
            return None
    else:
        log.debug("Cloning dataset")
        cache_dir.mkdir(parents=True, exist_ok=True)
        result = await git(
            "clone",
            "--bare",
            "--filter=blob:none",
            "--depth=1",
            "--branch",
            dataset.tag,
            repo_url,
            str(dataset_path),
        )

        if result.returncode != 0:
            log.error("Failed to clone", stderr=result.stderr.decode())
            return None

    return dataset
