from collections.abc import Iterator, MutableMapping
from pathlib import Path

import structlog
import pygit2
import pytest

from ondiagnostics.graphql import Dataset


@pytest.fixture(autouse=True)
def log_events() -> Iterator[list[MutableMapping[str, object]]]:

    with structlog.testing.capture_logs() as captured:
        yield captured


@pytest.fixture(scope="session")
def git_repo_with_tag(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """Create a real git repository with a tag (session-scoped)."""
    repo_path = tmp_path_factory.mktemp("repos") / "test_repo.git"
    repo_path.mkdir()

    # Initialize bare repo
    repo = pygit2.init_repository(str(repo_path), bare=True)

    # Create a commit
    tree_id = repo.TreeBuilder().write()
    author = pygit2.Signature("Test User", "test@example.com")
    commit_id = repo.create_commit(
        "refs/heads/main", author, author, "Initial commit", tree_id, []
    )

    # Create a tag
    repo.create_reference("refs/tags/1.0.0", commit_id)

    return repo_path, str(repo.revparse_single("1.0.0").id)


@pytest.fixture(scope="session")
def sample_dataset() -> Dataset:
    """Create a sample dataset for testing."""
    return Dataset(id="ds000001", tag="1.0.0", hexsha="abc123def456")
