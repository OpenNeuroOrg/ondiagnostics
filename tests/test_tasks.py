"""Tests for task workers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ondiagnostics.graphql import Dataset
from ondiagnostics.tasks import check_remote, clone_dataset


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create a sample dataset for testing."""
    return Dataset(id="ds000001", tag="1.0.0", hexsha="abc123def456")


@pytest.fixture
def mock_s3_bucket() -> MagicMock:
    """Create a mock S3 bucket."""
    bucket = MagicMock()
    bucket.name = "test-bucket"
    bucket.objects = MagicMock()
    bucket.delete_objects = MagicMock(return_value={"Deleted": []})
    return bucket


# Tests for check_remote()


@pytest.mark.asyncio
async def test_check_remote_success(sample_dataset: Dataset) -> None:
    """Test successful remote check."""
    ls_remote_output = f"{sample_dataset.hexsha}\trefs/tags/{sample_dataset.tag}\n"

    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (0, ls_remote_output.encode(), b"")

        result = await check_remote(sample_dataset)

        assert result == sample_dataset
        mock_git.assert_called_once_with(
            "ls-remote",
            "--exit-code",
            f"https://github.com/OpenNeuroDatasets/{sample_dataset.id}.git",
            sample_dataset.tag,
        )


@pytest.mark.asyncio
async def test_check_remote_repository_not_found(sample_dataset: Dataset) -> None:
    """Test check_remote when repository doesn't exist."""
    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (1, b"", b"Repository not found")

        result = await check_remote(sample_dataset)

        assert result is None


@pytest.mark.asyncio
async def test_check_remote_tag_missing(sample_dataset: Dataset) -> None:
    """Test check_remote when tag doesn't exist."""
    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (1, b"", b"")

        result = await check_remote(sample_dataset)

        assert result is None


@pytest.mark.asyncio
async def test_check_remote_empty_response(sample_dataset: Dataset) -> None:
    """Test check_remote with empty ls-remote output."""
    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (0, b"  ", b"")

        result = await check_remote(sample_dataset)

        assert result is None


@pytest.mark.asyncio
async def test_check_remote_sha_mismatch(sample_dataset: Dataset) -> None:
    """Test check_remote when commit SHA doesn't match."""
    wrong_sha = "different123"
    ls_remote_output = f"{wrong_sha}\trefs/tags/{sample_dataset.tag}\n"

    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (0, ls_remote_output.encode(), b"")

        result = await check_remote(sample_dataset)

        assert result is None


@pytest.mark.asyncio
async def test_check_remote_ref_mismatch(sample_dataset: Dataset) -> None:
    """Test check_remote when ref doesn't match expected format."""
    ls_remote_output = f"{sample_dataset.hexsha}\trefs/heads/{sample_dataset.tag}\n"

    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (0, ls_remote_output.encode(), b"")

        result = await check_remote(sample_dataset)

        assert result is None


# Tests for clone_dataset()


@pytest.mark.asyncio
async def test_clone_dataset_new_clone(sample_dataset: Dataset, tmp_path: Path) -> None:
    """Test cloning a new dataset."""
    cache_dir = tmp_path / "cache"

    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (0, b"Cloning...", b"")

        result = await clone_dataset(sample_dataset, cache_dir)

        assert result == sample_dataset
        assert cache_dir.exists()

        # Verify git clone was called
        call_args = mock_git.call_args[0]
        assert call_args[0] == "clone"
        assert "--bare" in call_args
        assert "--filter=blob:none" in call_args
        assert sample_dataset.tag in call_args


@pytest.mark.asyncio
async def test_clone_dataset_already_has_tag(
    sample_dataset: Dataset, tmp_path: Path, git_repo_with_tag: tuple[Path, str]
) -> None:
    """Test when dataset already exists and has the tag."""
    repo_path, commit_sha = git_repo_with_tag
    cache_dir = tmp_path / "cache"
    dataset_path = cache_dir / f"{sample_dataset.id}.git"

    # Copy the test repo to the expected location
    import shutil

    shutil.copytree(repo_path, dataset_path)

    # Update dataset to match the actual tag
    sample_dataset.tag = "1.0.0"

    result = await clone_dataset(sample_dataset, cache_dir)

    assert result == sample_dataset


@pytest.mark.asyncio
async def test_clone_dataset_needs_update(
    sample_dataset: Dataset, tmp_path: Path, git_repo_with_tag: tuple[Path, str]
) -> None:
    """Test updating an existing dataset without the tag."""
    repo_path, _ = git_repo_with_tag
    cache_dir = tmp_path / "cache"
    dataset_path = cache_dir / f"{sample_dataset.id}.git"

    # Copy the test repo but use a different tag that doesn't exist
    import shutil

    shutil.copytree(repo_path, dataset_path)

    sample_dataset.tag = "2.0.0"  # Tag that doesn't exist in repo

    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (0, b"Fetching...", b"")

        result = await clone_dataset(sample_dataset, cache_dir)

        assert result == sample_dataset

        # Verify git fetch was called
        call_args = mock_git.call_args[0]
        assert "-C" in call_args
        assert "fetch" in call_args
        assert sample_dataset.tag in call_args


@pytest.mark.asyncio
async def test_clone_dataset_clone_failure(
    sample_dataset: Dataset, tmp_path: Path
) -> None:
    """Test when git clone fails."""
    cache_dir = tmp_path / "cache"

    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (1, b"", b"fatal: repository not found")

        result = await clone_dataset(sample_dataset, cache_dir)

        assert result is None


@pytest.mark.asyncio
async def test_clone_dataset_fetch_failure(
    sample_dataset: Dataset, tmp_path: Path, git_repo_with_tag: tuple[Path, str]
) -> None:
    """Test when git fetch fails."""
    repo_path, _ = git_repo_with_tag
    cache_dir = tmp_path / "cache"
    dataset_path = cache_dir / f"{sample_dataset.id}.git"

    import shutil

    shutil.copytree(repo_path, dataset_path)

    sample_dataset.tag = "nonexistent"

    with patch("ondiagnostics.tasks.git") as mock_git:
        mock_git.return_value = (1, b"", b"fatal: couldn't find remote ref")

        result = await clone_dataset(sample_dataset, cache_dir)

        assert result is None
