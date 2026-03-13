"""Tests for main pipeline orchestration."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ondiagnostics.__main__ import (
    add_producer,
    add_consumer,
    run_pipeline,
)
from ondiagnostics.graphql import Dataset


# ============================================================================
# add_producer tests
# ============================================================================


@pytest.mark.asyncio
async def test_add_producer_creates_task(mock_progress: Mock) -> None:
    """Test add_producer creates a task and queue."""

    async def mock_generator() -> AsyncIterator[Dataset]:
        yield Dataset("ds000001", "1.0.0")
        yield Dataset("ds000002", "2.0.0")

    queue = add_producer(
        "Test Producer", mock_generator(), mock_progress, total=2, maxsize=10
    )

    # Queue should be created
    assert queue is not None
    assert queue.maxsize == 10

    # Should add task to progress
    mock_progress.add_task.assert_called_once()

    # Let the producer run
    await asyncio.sleep(0.1)

    # Should be able to get items from queue
    item1 = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert item1 is not None
    assert item1.id == "ds000001"


# ============================================================================
# add_consumer tests
# ============================================================================


@pytest.mark.asyncio
async def test_add_consumer_processes_items(mock_progress: Mock) -> None:
    """Test add_consumer processes items from input queue."""
    from ondiagnostics.pipeline import ProgressQueue

    # Create input queue with test data
    input_queue: ProgressQueue[Dataset | None] = ProgressQueue(
        mock_progress, maxsize=10
    )

    # Mock worker function
    processed = []

    async def mock_worker(dataset: Dataset) -> Dataset:
        processed.append(dataset.id)
        return dataset

    output_queue = add_consumer(
        "Test Consumer", mock_worker, input_queue, max_concurrent=2
    )

    # Add items to input queue
    await input_queue.put(Dataset("ds000001", "1.0.0"))
    await input_queue.put(Dataset("ds000002", "2.0.0"))
    await input_queue.put(None)  # Sentinel

    # Let consumer process
    while await output_queue.get() is not None:
        pass

    # Should have processed items
    assert len(processed) == 2
    assert "ds000001" in processed
    assert "ds000002" in processed


# ============================================================================
# run_pipeline tests
# ============================================================================


@pytest.mark.asyncio
async def test_run_pipeline_all_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_pipeline processes all datasets when no IDs specified."""
    mock_client = Mock()
    mock_count = 3

    async def mock_generator(client: Mock) -> AsyncIterator[Dataset]:
        yield Dataset("ds000001", "1.0.0")
        yield Dataset("ds000002", "2.0.0")
        yield Dataset("ds000003", "3.0.0")

    # Mock the GraphQL functions
    monkeypatch.setattr("ondiagnostics.__main__.create_client", lambda: mock_client)
    monkeypatch.setattr(
        "ondiagnostics.__main__.get_dataset_count", AsyncMock(return_value=mock_count)
    )
    monkeypatch.setattr("ondiagnostics.__main__.datasets_generator", mock_generator)

    # Mock check_remote to pass everything through
    monkeypatch.setattr(
        "ondiagnostics.__main__.check_remote", AsyncMock(side_effect=lambda d: d)
    )

    result = await run_pipeline(dataset_ids=None)

    assert result == 0


@pytest.mark.asyncio
async def test_run_pipeline_specific_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_pipeline processes only specific dataset IDs."""
    mock_client = Mock()

    async def mock_generator(client: Mock, ids: list[str]) -> AsyncIterator[Dataset]:
        for id in ids:
            yield Dataset(id, "1.0.0")

    monkeypatch.setattr("ondiagnostics.__main__.create_client", lambda: mock_client)
    monkeypatch.setattr(
        "ondiagnostics.__main__.datasets_by_ids_generator", mock_generator
    )
    monkeypatch.setattr(
        "ondiagnostics.__main__.check_remote", AsyncMock(side_effect=lambda d: d)
    )

    result = await run_pipeline(dataset_ids=["ds000001", "ds000002"])

    assert result == 0


# ============================================================================
# CLI command tests (using typer's CliRunner)
# ============================================================================


def test_check_sync_command() -> None:
    """Test check_sync CLI command."""
    from typer.testing import CliRunner
    from ondiagnostics.__main__ import app

    runner = CliRunner()

    # Mock run_pipeline to avoid actual execution
    with patch("ondiagnostics.__main__.asyncio.run", return_value=0):
        result = runner.invoke(app, ["check-sync", "--help"])
        assert result.exit_code == 0
        assert "Check synchronization status" in result.output


def test_clean_s3_command() -> None:
    """Test clean_s3 CLI command."""
    from typer.testing import CliRunner
    from ondiagnostics.__main__ import app

    runner = CliRunner()

    with patch("ondiagnostics.__main__.asyncio.run", return_value=0):
        result = runner.invoke(app, ["clean-s3", "--help"])
        assert result.exit_code == 0
        assert "Clean up S3" in result.output


def test_check_sync() -> None:
    """Test clean_s3 accepts dataset IDs and passes them to run_pipeline."""
    from typer.testing import CliRunner
    from ondiagnostics.__main__ import app

    runner = CliRunner()

    with patch(
        "ondiagnostics.__main__.run_pipeline", new_callable=AsyncMock
    ) as mock_pipeline:
        mock_pipeline.return_value = 0
        with patch("ondiagnostics.__main__.AWSConfig.from_env"):
            result = runner.invoke(app, ["check-sync"])

            assert result.exit_code == 0

            # Verify run_pipeline was called with the dataset IDs
            mock_pipeline.assert_called_once()


def test_clean_s3_with_dataset_ids() -> None:
    """Test clean_s3 accepts dataset IDs and passes them to run_pipeline."""
    from typer.testing import CliRunner
    from ondiagnostics.__main__ import app

    runner = CliRunner()

    with patch(
        "ondiagnostics.__main__.run_pipeline", new_callable=AsyncMock
    ) as mock_pipeline:
        mock_pipeline.return_value = 0
        with patch("ondiagnostics.__main__.AWSConfig.from_env"):
            result = runner.invoke(app, ["clean-s3", "ds000001", "ds000002"])

            assert result.exit_code == 0

            # Verify run_pipeline was called with the dataset IDs
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args.kwargs
            assert call_kwargs["dataset_ids"] == ["ds000001", "ds000002"]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_progress() -> Mock:
    """Mock Rich Progress object."""
    progress = Mock()
    progress.tasks = [(task := Mock())]
    task.total = 0
    progress.add_task = Mock(return_value=0)
    progress.update = Mock()
    return progress
