"""Tests for pipeline components."""

import asyncio
from collections.abc import AsyncIterable, Iterable
from typing import TypeVar

import pytest

from ondiagnostics.pipeline import producer, consumer, ProgressQueue

T = TypeVar("T")


async def as_async(iterable: Iterable[T]) -> AsyncIterable[T]:
    for i in iterable:
        yield i


@pytest.mark.asyncio
async def test_producer_base() -> None:
    """Test producer puts all items in queue."""
    queue: asyncio.Queue[int | None] = asyncio.Queue()
    await producer(as_async(range(5)), queue)

    value = 0
    while (item := await queue.get()) is not None:
        assert item == value
        value += 1

    assert item is None


@pytest.mark.asyncio
async def test_producer_with_callback() -> None:
    """Test producer calls callback for each item."""
    values = []

    def callback(item: int) -> None:
        values.append(item)

    queue: asyncio.Queue[int | None] = asyncio.Queue()
    await producer(as_async(range(5)), queue, on_complete=callback)
    while (item := await queue.get()) is not None:
        pass

    assert values == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_producer_exception_still_sends_sentinel() -> None:
    """Test producer sends None sentinel even if generator raises."""

    async def failing_generator() -> AsyncIterable[int]:
        yield 1
        yield 2
        raise ValueError("Test error")
        yield 3  # Never reached

    queue: asyncio.Queue[int | None] = asyncio.Queue()

    with pytest.raises(ValueError):
        await producer(failing_generator(), queue)

    items = []
    while (item := await queue.get()) is not None:
        items.append(item)

    assert items == [1, 2]
    assert item is None


@pytest.mark.asyncio
async def test_consumer_success() -> None:
    """Test consumer processes items and puts results in output queue."""
    input_queue: asyncio.Queue[int | None] = asyncio.Queue()
    output_queue: asyncio.Queue[float | None] = asyncio.Queue()

    await producer(as_async(range(5)), input_queue)

    async def double(x: int) -> float:
        return x * 2.0

    semaphore = asyncio.Semaphore(10)
    await consumer(input_queue, output_queue, double, semaphore)

    results = []
    while (item := await output_queue.get()) is not None:
        results.append(item)

    assert sorted(results) == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert item is None


@pytest.mark.asyncio
async def test_consumer_filters_none() -> None:
    """Test consumer doesn't pass None results downstream."""
    input_queue: asyncio.Queue[int | None] = asyncio.Queue()
    output_queue: asyncio.Queue[int | None] = asyncio.Queue()

    await producer(as_async(range(5)), input_queue)

    async def filter_even(x: int) -> int | None:
        return x if x % 2 == 0 else None

    semaphore = asyncio.Semaphore(10)
    await consumer(input_queue, output_queue, filter_even, semaphore)

    results = []
    while (item := await output_queue.get()) is not None:
        results.append(item)

    assert results == [0, 2, 4]


@pytest.mark.asyncio
async def test_consumer_callback() -> None:
    """Test consumer calls completion callback."""
    input_queue: asyncio.Queue[int | None] = asyncio.Queue()

    await producer(as_async(range(5)), input_queue)

    completed = []

    def on_complete(value: int, result: float | None, success: bool) -> None:
        completed.append((value, result, success))

    async def identity(x: int) -> float | None:
        # Exception on zero, None on 3
        return 1 / x if x != 3 else None

    semaphore = asyncio.Semaphore(10)
    await consumer(input_queue, None, identity, semaphore, on_complete=on_complete)

    assert len(completed) == 5
    assert completed[0] == (0, None, False)  # Exception
    assert completed[1] == (1, 1.0, True)
    assert completed[2] == (2, 0.5, True)
    assert completed[3] == (3, None, False)  # Filtered out
    assert completed[4] == (4, 0.25, True)
