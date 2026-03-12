"""Tests for S3 cleanup tasks."""

import asyncio

import pytest
import aioboto3

from aiomoto import mock_aws
from ondiagnostics.graphql import Dataset
from ondiagnostics.tasks.s3 import plan_cleanup

pytestmark = pytest.mark.asyncio


@pytest.fixture
def git_repo_simple(tmp_path, sample_dataset):
    """Create a simple git repo with just a few files at root level."""
    import pygit2

    repo_path = tmp_path / f"{sample_dataset.id}.git"
    repo_path.mkdir()

    repo = pygit2.init_repository(str(repo_path), bare=True)
    builder = repo.TreeBuilder()

    # Simple flat structure
    for filename in ["file1.txt", "file2.txt", "file3.txt"]:
        blob_oid = repo.create_blob(b"content")
        builder.insert(filename, blob_oid, pygit2.GIT_FILEMODE_BLOB)

    tree_oid = builder.write()
    author = pygit2.Signature("Test", "test@example.com")
    commit_oid = repo.create_commit(None, author, author, "Test commit", tree_oid, [])
    repo.create_reference(f"refs/tags/{sample_dataset.tag}", commit_oid)

    return tmp_path


@pytest.fixture
async def mock_session():
    """Provide a mocked S3 session and bucket."""
    async with mock_aws():
        session = aioboto3.Session()
        bucket_name = "test-bucket"

        # Create bucket
        async with session.client("s3", region_name="us-east-1") as s3:
            await s3.create_bucket(Bucket=bucket_name)

        yield session, bucket_name


async def populate_bucket(
    session: aioboto3.Session,
    bucket_name: str,
    data: dict[str, bytes],
) -> None:
    async with session.client("s3", region_name="us-east-1") as s3:
        async with asyncio.TaskGroup() as tg:
            for key, content in data.items():
                tg.create_task(s3.put_object(Bucket=bucket_name, Key=key, Body=content))


# ============================================================================
# plan_cleanup() tests
# ============================================================================


async def test_plan_cleanup_identifies_orphaned_files(
    mock_session, sample_dataset, git_repo_simple
):
    """Files in S3 but not in git tree should be marked for deletion."""

    session, bucket_name = mock_session
    prefix = f"{sample_dataset.id}/"

    # Populate S3: files from git plus some orphans
    await populate_bucket(
        *mock_session,
        {
            # Files that exist in git
            f"{prefix}file1.txt": b"content",
            f"{prefix}file2.txt": b"content",
            # Orphaned files not in git
            f"{prefix}orphaned.txt": b"old",
            f"{prefix}old_version.nii": b"old",
        },
    )

    # Run plan_cleanup
    plan = await plan_cleanup(sample_dataset, git_repo_simple, session, bucket_name)

    assert plan is not None
    assert plan.dataset == sample_dataset
    assert len(plan.files_to_delete) == 2
    assert f"{prefix}orphaned.txt" in plan.files_to_delete
    assert f"{prefix}old_version.nii" in plan.files_to_delete
    assert f"{prefix}file1.txt" not in plan.files_to_delete
    assert f"{prefix}file2.txt" not in plan.files_to_delete


async def test_plan_cleanup_returns_none_when_no_orphans(
    mock_session, sample_dataset, git_repo_simple
):
    """When S3 matches git tree exactly, should return None."""
    session, bucket_name = mock_session
    prefix = f"{sample_dataset.id}/"

    # S3 only has files that exist in git
    await populate_bucket(
        *mock_session,
        {
            f"{prefix}file1.txt": b"content",
            f"{prefix}file2.txt": b"content",
        },
    )

    plan = await plan_cleanup(sample_dataset, git_repo_simple, session, bucket_name)

    assert plan is None


async def test_plan_cleanup_returns_none_when_s3_empty(
    mock_session, sample_dataset, git_repo_simple
):
    """When S3 has no objects with the prefix, should return None."""
    session, bucket_name = mock_session

    plan = await plan_cleanup(sample_dataset, git_repo_simple, session, bucket_name)

    assert plan is None


async def test_plan_cleanup_handles_missing_tag(
    mock_session, sample_dataset, git_repo_simple
):
    """Should return None and log error when git tag doesn't exist."""
    session, bucket_name = mock_session

    # Use a dataset with a tag that doesn't exist
    bad_dataset = Dataset(id="ds000001", tag="99.99.99", hexsha="fake")

    plan = await plan_cleanup(bad_dataset, git_repo_simple, session, bucket_name)

    assert plan is None


async def test_plan_cleanup_handles_missing_repo(
    mock_session, sample_dataset, tmp_path
):
    """Should return None when git repository doesn't exist."""
    session, bucket_name = mock_session

    # tmp_path exists but has no repo
    plan = await plan_cleanup(sample_dataset, tmp_path, session, bucket_name)

    assert plan is None


async def test_plan_cleanup_processes_multiple_pages(
    mock_session, sample_dataset, git_repo_simple
):
    """Should process all pages when S3 listing is paginated."""
    session, bucket_name = mock_session
    prefix = f"{sample_dataset.id}/"

    # Add many orphaned files to trigger pagination
    await populate_bucket(
        *mock_session,
        {f"{prefix}orphaned_{i:03d}.txt": b"old" for i in range(150)},
    )

    plan = await plan_cleanup(sample_dataset, git_repo_simple, session, bucket_name)

    assert plan is not None
    assert len(plan.files_to_delete) == 150


async def test_plan_cleanup_ignores_wrong_prefix(
    mock_session, sample_dataset, git_repo_simple
):
    """Should only consider files with the correct dataset prefix."""
    session, bucket_name = mock_session
    prefix = f"{sample_dataset.id}/"

    await populate_bucket(
        *mock_session,
        {
            # Files from our dataset
            f"{prefix}orphaned.txt": b"old",
            # Files from other datasets (should be ignored)
            "ds000002/file.txt": b"other",
            "ds999999/file.txt": b"other",
        },
    )

    plan = await plan_cleanup(test_dataset, git_repo_simple, session, bucket_name)

    assert plan is not None
    assert len(plan.files_to_delete) == 1
    assert f"{prefix}orphaned.txt" in plan.files_to_delete
