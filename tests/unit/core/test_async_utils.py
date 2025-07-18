import asyncio
import re
from collections import deque
from unittest.mock import Mock, call, patch

import pytest

from oumi.core.async_utils import AsyncioPoliteSemaphore, safe_asyncio_run


@pytest.fixture
def mock_time():
    with patch("oumi.core.async_utils.time") as time_mock:
        yield time_mock


@pytest.fixture
def mock_asyncio_sleep():
    with patch("oumi.core.async_utils.asyncio.sleep") as sleep_mock:
        yield sleep_mock


@pytest.mark.asyncio
async def test_asyncio_polite_semaphore(mock_time, mock_asyncio_sleep):
    semaphore = AsyncioPoliteSemaphore(capacity=1, politeness_policy=10)
    mock_time.time.return_value = 1
    await semaphore.acquire()
    assert len(semaphore._queue) == 0
    await semaphore.release()
    assert semaphore._queue == deque([11])


@pytest.mark.asyncio
async def test_asyncio_polite_semaphore_subsequent_acquires_use_queue(
    mock_time, mock_asyncio_sleep
):
    """Test that subsequent acquires use the queue values for waiting."""
    semaphore = AsyncioPoliteSemaphore(capacity=2, politeness_policy=2.0)

    # First acquire - no wait needed since queue is empty
    mock_time.time.return_value = 10.0
    await semaphore.acquire()
    assert semaphore._queue == deque([-1])

    # Release - adds current time to queue
    await semaphore.release()
    assert semaphore._queue == deque([-1, 12.0])

    # Second acquire - should wait based on queue value
    mock_time.time.return_value = 11.0  # 1 second later
    await semaphore.acquire()

    assert semaphore._queue == deque([12.0])


@pytest.mark.asyncio
async def test_asyncio_polite_semaphore_multiple_releases_before_acquire(
    mock_time, mock_asyncio_sleep
):
    """Test that multiple releases create a queue that subsequent acquires respect."""
    semaphore = AsyncioPoliteSemaphore(capacity=2, politeness_policy=20)

    mock_time.time.side_effect = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 36.0, 37.0]

    async def acquire_and_release(semaphore: AsyncioPoliteSemaphore):
        await semaphore.acquire()
        await semaphore.release()

    tasks = [acquire_and_release(semaphore) for _ in range(4)]

    # Tasks 1 finishes at 11.0.
    # Task 2 finishes at 13.0.
    # Task 3 starts at time 14.0 and then sleeps for 17 seconds.
    # Task 4 starts at time 366.0 and does not sleep.

    await asyncio.gather(*tasks)
    mock_time.time.assert_has_calls(
        [call(), call(), call(), call(), call(), call(), call(), call()]
    )
    mock_asyncio_sleep.assert_has_calls([call(17.0)])


@pytest.mark.asyncio
async def test_asyncio_polite_semaphore_increments_queue(mock_time, mock_asyncio_sleep):
    """Test that queue is updated with each release and acquire."""
    semaphore = AsyncioPoliteSemaphore(capacity=2, politeness_policy=1.0)

    # First acquire and release
    mock_time.time.return_value = 10.0
    await semaphore.acquire()
    await semaphore.release()
    assert semaphore._queue == deque([-1, 11.0])

    # Second acquire and release
    mock_time.time.return_value = 11.0
    await semaphore.acquire()
    await semaphore.release()
    assert semaphore._queue == deque([11.0, 12.0])

    mock_time.time.return_value = 12.0
    await semaphore.acquire()
    await semaphore.release()
    assert semaphore._queue == deque([12.0, 13.0])

    mock_time.time.return_value = 13.0
    await semaphore.acquire()

    mock_asyncio_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_asyncio_polite_semaphore_wait_time_calculation(
    mock_time, mock_asyncio_sleep
):
    """Test specific wait time calculations."""
    semaphore = AsyncioPoliteSemaphore(capacity=1, politeness_policy=3.0)

    # First acquire and release
    mock_time.time.return_value = 100.0
    await semaphore.acquire()
    await semaphore.release()
    assert semaphore._queue == deque([103.0])

    # Second acquire - should wait exactly 3 seconds
    mock_time.time.return_value = 101.0  # 1 second later
    await semaphore.acquire()

    # Should wait: (100.0 + 3.0) - 101.0 = 2.0 seconds
    mock_asyncio_sleep.assert_called_once_with(2.0)


async def test_asyncio_polite_semaphore_no_wait_when_enough_time_passed(
    mock_time, mock_asyncio_sleep
):
    """Test that no wait occurs when enough time has passed since last release."""
    semaphore = AsyncioPoliteSemaphore(capacity=1, politeness_policy=2.0)

    # First acquire and release
    mock_time.time.return_value = 10.0
    await semaphore.acquire()
    await semaphore.release()
    assert semaphore._queue == deque([10.0])

    # Second acquire - wait more than politeness policy
    mock_time.time.return_value = 13.0  # 3 seconds later (more than 2.0 politeness)
    await semaphore.acquire()

    # Should not wait since enough time has passed
    mock_asyncio_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_asyncio_polite_semaphore_concurrent_acquires(
    mock_time, mock_asyncio_sleep
):
    """Test that concurrent acquires respect the politeness policy."""
    semaphore = AsyncioPoliteSemaphore(capacity=2, politeness_policy=1.0)

    # First two acquires should happen immediately
    mock_time.time.return_value = 10.0
    await semaphore.acquire()
    await semaphore.acquire()

    # Release first one
    await semaphore.release()
    assert semaphore._queue == deque([11.0])

    # Release second one
    mock_time.time.return_value = 11.0
    await semaphore.release()
    assert semaphore._queue == deque([11.0, 12.0])

    # Next acquire should wait based on oldest queue value
    mock_time.time.return_value = 12.0
    await semaphore.acquire()

    # Should wait: (10.0 + 1.0) - 12.0 = -1.0 (no wait needed)
    mock_asyncio_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_asyncio_polite_semaphore_empty_queue_no_wait(
    mock_time, mock_asyncio_sleep
):
    """Test that acquire with empty queue doesn't wait."""
    semaphore = AsyncioPoliteSemaphore(capacity=1, politeness_policy=5.0)

    # First acquire - no wait needed
    mock_time.time.return_value = 100.0
    await semaphore.acquire()

    # Should not sleep since queue is empty
    mock_asyncio_sleep.assert_not_called()


def test_safe_asyncio_run_nested_safe():
    async def nested():
        return 1

    def method_using_asyncio():
        return asyncio.run(nested())

    def method_using_safe_asyncio_run():
        return safe_asyncio_run(nested())

    with pytest.raises(
        RuntimeError,
        match=re.escape("asyncio.run() cannot be called from a running event loop"),
    ):

        async def main_async():
            return method_using_asyncio()

        # This will raise a RuntimeError because we are trying to run an async function
        # inside a running event loop.
        asyncio.run(main_async())

    async def safe_main():
        return method_using_safe_asyncio_run()

    # Verify using safe_asyncio_run within another safe_asyncio_run context.
    result = safe_asyncio_run(safe_main())
    assert result == 1


def test_safe_asyncio_run_nested_unsafe():
    async def nested():
        return 1

    def method_using_safe_asyncio():
        return safe_asyncio_run(nested())

    async def main():
        return method_using_safe_asyncio()

    # Here we run asyncio.run() at the top level, where the sub-loop is using
    # safe_asyncio_run.
    result = asyncio.run(main())
    assert result == 1


def test_safe_asyncio_run_nested_fails():
    def method_using_asyncio():
        coro = Mock()
        return asyncio.run(coro)

    async def main():
        return method_using_asyncio()

    # Here we run safe_asyncio_run at the top level, where the sub-loop is using
    # asyncio.run(). This will throw an exception as the new loop from safe_run_asyncio
    # is running in the same context as the asyncio.run() call.
    with pytest.raises(
        RuntimeError,
        match=re.escape("asyncio.run() cannot be called from a running event loop"),
    ):
        _ = safe_asyncio_run(main())
