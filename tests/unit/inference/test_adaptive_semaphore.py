# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections import deque
from unittest.mock import call, patch

import pytest

from oumi.inference.adaptive_semaphore import AdaptiveSemaphore, PoliteAdaptiveSemaphore


@pytest.fixture
def mock_time():
    with patch("oumi.inference.adaptive_semaphore.time") as time_mock:
        yield time_mock


@pytest.fixture
def mock_asyncio_sleep():
    with patch("oumi.inference.adaptive_semaphore.asyncio.sleep") as sleep_mock:
        yield sleep_mock


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore(mock_time, mock_asyncio_sleep):
    semaphore = PoliteAdaptiveSemaphore(capacity=1, politeness_policy=10)
    mock_time.time.return_value = 1
    await semaphore.acquire()
    assert len(semaphore._queue) == 0
    semaphore.release()
    # 11 = 1.0 + politeness_policy  (10.0)
    assert semaphore._queue == deque([11])


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_adjust_capacity(mock_time, mock_asyncio_sleep):
    semaphore = PoliteAdaptiveSemaphore(capacity=1, politeness_policy=10)
    mock_time.time.return_value = 0.0
    # Initially the semaphore has capacity of 1 and is unused, so the queue is [-1].
    assert semaphore._queue == deque([-1])
    await semaphore.adjust_capacity(2)
    # We expanded the capacity to 2, so the queue is padded with -1s.
    assert semaphore._queue == deque([-1, -1])
    mock_time.time.return_value = 1.0
    await semaphore.acquire()
    semaphore.release()
    mock_time.time.return_value = 2.0
    # We acquired the first permit (consumed -1) then released (appended the time +
    # politeness policy=10.0). The queue is now [-1, 11].
    assert semaphore._queue == deque([-1, 11])
    await semaphore.acquire()
    semaphore.release()
    mock_time.time.return_value = 3.0
    # We acquired the second permit (consumed -1) then released (appended the time +
    # politeness policy=10.0). The queue is now [11, 12].
    assert semaphore._queue == deque([11, 12])
    await semaphore.adjust_capacity(1)
    # We reduced the capacity to 1, so the queue is now [12].
    # 11 was removed from the queue because it was the oldest entry.
    assert semaphore._queue == deque([12])
    await semaphore.acquire()
    # We acquired the third permit (consumed 12). The queue is now empty.
    assert semaphore._queue == deque([])
    await semaphore.adjust_capacity(10)
    # We expanded the capacity to 10, so the queue is padded with -1s.
    assert semaphore._queue == deque([-1] * 10)


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_subsequent_acquires_use_queue(
    mock_time, mock_asyncio_sleep
):
    """Test that subsequent acquires use the queue values for waiting."""
    semaphore = PoliteAdaptiveSemaphore(capacity=2, politeness_policy=2.0)

    # First acquire - no wait needed since queue is empty
    mock_time.time.return_value = 10.0
    await semaphore.acquire()
    assert semaphore._queue == deque([-1])

    # Release - adds current time to queue
    semaphore.release()
    assert semaphore._queue == deque([-1, 12.0])

    # Second acquire - should wait based on queue value
    mock_time.time.return_value = 11.0  # 1 second later
    await semaphore.acquire()

    assert semaphore._queue == deque([12])


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_multiple_releases_before_acquire(
    mock_time, mock_asyncio_sleep
):
    """Test that multiple releases create a queue that subsequent acquires respect."""
    semaphore = PoliteAdaptiveSemaphore(capacity=2, politeness_policy=20)

    mock_time.time.side_effect = [
        10.0,
        11.0,
        12.0,
        36.0,  # release time for Task 3
        37.0,
        38.0,
    ]

    async def acquire_and_release(semaphore: PoliteAdaptiveSemaphore):
        await semaphore.acquire()
        semaphore.release()

    tasks = [acquire_and_release(semaphore) for _ in range(4)]

    # Task 1: Acquires at first available slot and finishes at 10.0. Sets slot to next
    # allowed start at 10.0 + 20 = 30.0.
    # Task 2: Acquires the other slot, finishes at 11.0. Sets slot to next allowed start
    # at 11.0 + 20 = 31.0
    # Task 3: Starts at 12.0. Both queue slots are now [30.0, 31.0]. Needs to wait for
    # queue.pop() - current time = 30.0 - 12.0 = 18.0 seconds.
    # Task 4: Starts at 37.0. The queue slots are now [31.0, 56.0]
    # (after Task 3’s release). 37.0 is after 31.0, so no sleep needed. Task 4 releases
    # at 38.0, setting queue to [56.0, 58.0].

    await asyncio.gather(*tasks)
    mock_time.time.assert_has_calls([call(), call(), call(), call(), call(), call()])
    mock_asyncio_sleep.assert_has_calls([call(18.0)])
    assert semaphore._queue == deque([56.0, 58.0])


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_increments_queue(
    mock_time, mock_asyncio_sleep
):
    """Test that queue is updated with each release and acquire."""
    semaphore = PoliteAdaptiveSemaphore(capacity=2, politeness_policy=1.0)

    # First acquire and release
    mock_time.time.return_value = 10.0
    await semaphore.acquire()
    semaphore.release()
    assert semaphore._queue == deque([-1, 11.0])

    # Second acquire and release
    mock_time.time.return_value = 11.0
    await semaphore.acquire()
    semaphore.release()
    assert semaphore._queue == deque([11.0, 12.0])

    mock_time.time.return_value = 12.0
    await semaphore.acquire()
    semaphore.release()
    assert semaphore._queue == deque([12.0, 13.0])

    mock_time.time.return_value = 13.0
    await semaphore.acquire()

    mock_asyncio_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_wait_time_calculation(
    mock_time, mock_asyncio_sleep
):
    """Test specific wait time calculations."""
    semaphore = PoliteAdaptiveSemaphore(capacity=1, politeness_policy=3.0)

    # First acquire and release
    mock_time.time.return_value = 100.0
    await semaphore.acquire()
    semaphore.release()
    assert semaphore._queue == deque([103.0])

    # Second acquire - should wait exactly 3 seconds
    mock_time.time.return_value = 101.0  # 1 second later
    await semaphore.acquire()

    # Should wait: (100.0 + 3.0) - 101.0 = 2.0 seconds
    mock_asyncio_sleep.assert_called_once_with(2.0)


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_no_wait_when_enough_time_passed(
    mock_time, mock_asyncio_sleep
):
    """Test that no wait occurs when enough time has passed since last release."""
    semaphore = PoliteAdaptiveSemaphore(capacity=1, politeness_policy=2.0)

    # First acquire and release
    mock_time.time.return_value = 10.0
    await semaphore.acquire()
    semaphore.release()
    assert semaphore._queue == deque([12.0])

    # Second acquire - wait more than politeness policy
    mock_time.time.return_value = 13.0  # 3 seconds later (more than 2.0 politeness)
    await semaphore.acquire()

    # Should not wait since enough time has passed
    mock_asyncio_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_concurrent_acquires(
    mock_time, mock_asyncio_sleep
):
    """Test that concurrent acquires respect the politeness policy."""
    semaphore = PoliteAdaptiveSemaphore(capacity=2, politeness_policy=1.0)

    # First two acquires should happen immediately
    mock_time.time.return_value = 10.0
    await semaphore.acquire()
    await semaphore.acquire()

    # Release first one
    semaphore.release()
    assert semaphore._queue == deque([11.0])

    # Release second one
    mock_time.time.return_value = 11.0
    semaphore.release()
    assert semaphore._queue == deque([11.0, 12.0])

    # Next acquire should wait based on oldest queue value
    mock_time.time.return_value = 12.0
    await semaphore.acquire()

    # Should wait: (10.0 + 1.0) - 12.0 = -1.0 (no wait needed)
    mock_asyncio_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_empty_queue_no_wait(
    mock_time, mock_asyncio_sleep
):
    """Test that acquire with empty queue doesn't wait."""
    semaphore = PoliteAdaptiveSemaphore(capacity=1, politeness_policy=5.0)

    # First acquire - no wait needed
    mock_time.time.return_value = 100.0
    await semaphore.acquire()

    # Should not sleep since queue is empty
    mock_asyncio_sleep.assert_not_called()


class TestAdaptiveSemaphore:
    """Test suite for AdaptiveSemaphore class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test semaphore initialization with different capacities."""
        # Test basic initialization
        semaphore = AdaptiveSemaphore(5)
        assert semaphore._max_capacity == 5
        assert semaphore._current_capacity == 5
        assert len(semaphore._waiters) == 0

        # Test with capacity of 1
        semaphore = AdaptiveSemaphore(1)
        assert semaphore._max_capacity == 1
        assert semaphore._current_capacity == 1

    @pytest.mark.asyncio
    async def test_basic_acquire_release(self):
        """Test basic acquire and release functionality."""
        semaphore = AdaptiveSemaphore(2)

        # First acquire should succeed immediately
        await semaphore.acquire()
        assert semaphore._current_capacity == 1

        # Second acquire should succeed immediately
        await semaphore.acquire()
        assert semaphore._current_capacity == 0

        # Release one permit
        semaphore.release()
        assert semaphore._current_capacity == 1

        # Release another permit
        semaphore.release()
        assert semaphore._current_capacity == 2

    @pytest.mark.asyncio
    async def test_acquire_blocking_when_capacity_reached(self):
        """Test that acquire blocks when capacity is reached."""
        semaphore = AdaptiveSemaphore(1)

        # First acquire should succeed immediately
        await semaphore.acquire()
        assert semaphore._current_capacity == 0

        # Second acquire should block
        acquire_task = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0.01)  # Give task a chance to run
        assert not acquire_task.done()
        assert len(semaphore._waiters) == 1

        # Release should unblock the waiting acquire
        semaphore.release()
        await acquire_task  # This should complete now
        assert semaphore._current_capacity == 0

    @pytest.mark.asyncio
    async def test_multiple_waiters(self):
        """Test handling of multiple waiting tasks."""
        semaphore = AdaptiveSemaphore(1)

        # Acquire the only permit
        await semaphore.acquire()
        assert semaphore._current_capacity == 0

        # Create multiple waiting tasks
        task1 = asyncio.create_task(semaphore.acquire())
        task2 = asyncio.create_task(semaphore.acquire())
        task3 = asyncio.create_task(semaphore.acquire())

        await asyncio.sleep(0.01)  # Let tasks start waiting
        assert len(semaphore._waiters) == 3
        assert not task1.done()
        assert not task2.done()
        assert not task3.done()

        # Release should wake up first waiter
        semaphore.release()
        await task1  # Should complete
        assert not task2.done()
        assert not task3.done()
        assert len(semaphore._waiters) == 2

        # Release again should wake up second waiter
        semaphore.release()
        await task2  # Should complete
        assert not task3.done()
        assert len(semaphore._waiters) == 1

        # Release again should wake up third waiter
        semaphore.release()
        await task3  # Should complete
        assert len(semaphore._waiters) == 0

    @pytest.mark.asyncio
    async def test_cancelled_waiters_handling(self):
        """Test that cancelled waiters are properly removed."""
        semaphore = AdaptiveSemaphore(1)

        # Acquire the only permit
        await semaphore.acquire()

        # Create waiting tasks
        task1 = asyncio.create_task(semaphore.acquire())
        task2 = asyncio.create_task(semaphore.acquire())

        await asyncio.sleep(0.01)
        assert len(semaphore._waiters) == 2

        # Cancel first task
        task1.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass

        # Release should wake up the non-cancelled waiter
        semaphore.release()
        await task2  # Should complete
        assert len(semaphore._waiters) == 0

    @pytest.mark.asyncio
    async def test_adjust_capacity_increase(self):
        """Test increasing semaphore capacity."""
        semaphore = AdaptiveSemaphore(2)

        # Acquire both permits
        await semaphore.acquire()
        await semaphore.acquire()
        assert semaphore._current_capacity == 0

        # Create waiting tasks
        task1 = asyncio.create_task(semaphore.acquire())
        task2 = asyncio.create_task(semaphore.acquire())

        await asyncio.sleep(0.01)
        assert len(semaphore._waiters) == 2

        # Increase capacity should wake up waiters
        await semaphore.adjust_capacity(4)
        assert semaphore._max_capacity == 4
        await asyncio.sleep(0.01)

        await task1  # Should complete
        await task2  # Should complete
        assert len(semaphore._waiters) == 0
        assert semaphore._current_capacity == 0  # 4 capacity - 4 acquired

    @pytest.mark.asyncio
    async def test_adjust_capacity_decrease(self):
        """Test decreasing semaphore capacity."""
        semaphore = AdaptiveSemaphore(5)

        # Acquire 2 permits
        await semaphore.acquire()
        await semaphore.acquire()
        assert semaphore._current_capacity == 3

        # Decrease capacity
        await semaphore.adjust_capacity(3)
        assert semaphore._max_capacity == 3
        assert semaphore._current_capacity == 1  # 3 - 2 acquired

        # Should be able to acquire one more
        await semaphore.acquire()
        assert semaphore._current_capacity == 0

        # Next acquire should block
        task = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0.01)
        assert not task.done()
        assert len(semaphore._waiters) == 1

        # Clean up
        semaphore.release()
        await task

    @pytest.mark.asyncio
    async def test_adjust_capacity_with_existing_waiters(self):
        """Test capacity adjustment when there are existing waiters."""
        semaphore = AdaptiveSemaphore(1)

        # Acquire the permit
        await semaphore.acquire()

        # Create multiple waiters
        tasks = [asyncio.create_task(semaphore.acquire()) for _ in range(3)]
        await asyncio.sleep(0.01)
        assert len(semaphore._waiters) == 3

        # Increase capacity to 3 should wake up 2 waiters
        await semaphore.adjust_capacity(3)
        await asyncio.sleep(0.01)

        # First two tasks should complete, third should still wait
        await tasks[0]
        await tasks[1]
        assert not tasks[2].done()
        assert len(semaphore._waiters) == 1

        # Release one more to complete the last task
        semaphore.release()
        await tasks[2]

    @pytest.mark.asyncio
    async def test_adjust_capacity_decrease_below_current_usage(self):
        """Test decreasing capacity below current usage doesn't revoke permits."""
        semaphore = AdaptiveSemaphore(5)

        # Acquire 4 permits
        for _ in range(4):
            await semaphore.acquire()
        assert semaphore._current_capacity == 1

        # Decrease capacity below current usage
        await semaphore.adjust_capacity(2)
        assert semaphore._max_capacity == 2
        # Current count should never go below 0
        assert semaphore._current_capacity == 0

        # No new acquires should succeed until enough permits are released
        task = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0.01)
        assert not task.done()

        # Release 3 permits to get back to positive count
        for _ in range(3):
            semaphore.release()

        # Now the waiting task should complete
        await task

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent acquire/release/adjust operations."""
        semaphore = AdaptiveSemaphore(3)

        async def worker(worker_id: int, results: list):
            try:
                await semaphore.acquire()
                await asyncio.sleep(0.01)  # Simulate work
                results.append(f"worker_{worker_id}_done")
                semaphore.release()
            except Exception as e:
                results.append(f"worker_{worker_id}_error_{e}")

        # Start multiple workers
        results = []
        tasks = [asyncio.create_task(worker(i, results)) for i in range(5)]

        # Adjust capacity while workers are running
        await asyncio.sleep(0.005)
        await semaphore.adjust_capacity(2)
        await asyncio.sleep(0.005)
        await semaphore.adjust_capacity(4)

        # Wait for all workers to complete
        await asyncio.gather(*tasks)

        # All workers should complete successfully
        assert len(results) == 5
        assert all("_done" in result for result in results)

    @pytest.mark.asyncio
    async def test_zero_capacity(self):
        """Test semaphore behavior with zero capacity."""
        semaphore = AdaptiveSemaphore(1)

        # Reduce to zero capacity
        with pytest.raises(ValueError):
            await semaphore.adjust_capacity(0)

        with pytest.raises(ValueError):
            await semaphore.adjust_capacity(-1)

        with pytest.raises(ValueError):
            semaphore = AdaptiveSemaphore(0)

    @pytest.mark.asyncio
    async def test_edge_case_empty_waiters_on_release(self):
        """Test release when there are no waiters."""
        semaphore = AdaptiveSemaphore(2)

        # Acquire one permit
        await semaphore.acquire()
        assert semaphore._current_capacity == 1

        # Release without any waiters
        semaphore.release()
        assert semaphore._current_capacity == 2
        assert len(semaphore._waiters) == 0

    @pytest.mark.asyncio
    async def test_waiter_cancellation_during_release(self):
        """Test handling of waiter cancellation during release."""
        semaphore = AdaptiveSemaphore(1)

        # Acquire the permit
        await semaphore.acquire()

        # Create waiters
        task1 = asyncio.create_task(semaphore.acquire())
        task2 = asyncio.create_task(semaphore.acquire())

        await asyncio.sleep(0.01)
        assert len(semaphore._waiters) == 2

        # Cancel first waiter
        task1.cancel()

        # Release should skip cancelled waiter and wake up second one
        semaphore.release()

        try:
            await task1
        except asyncio.CancelledError:
            pass

        await task2  # Should complete
        assert len(semaphore._waiters) == 0

    @pytest.mark.asyncio
    async def test_context_manager_basic(self):
        """Test basic context manager functionality."""
        semaphore = AdaptiveSemaphore(2)

        async with semaphore:
            assert semaphore._current_capacity == 1

        assert semaphore._current_capacity == 2

    @pytest.mark.asyncio
    async def test_context_manager_exception_handling(self):
        """Test context manager releases permit even when exception occurs."""
        semaphore = AdaptiveSemaphore(2)

        try:
            async with semaphore:
                assert semaphore._current_capacity == 1
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert semaphore._current_capacity == 2

    @pytest.mark.asyncio
    async def test_over_release_behavior(self):
        """Test behavior when releasing more permits than acquired."""
        semaphore = AdaptiveSemaphore(2)

        # Release without acquiring
        semaphore.release()

        # Current capacity should not exceed max capacity, no error should be raised
        assert semaphore._current_capacity == 2

    def test_string_representation(self):
        """Test string representation of semaphore."""
        semaphore = AdaptiveSemaphore(5)
        repr_str = repr(semaphore)
        assert "AdaptiveSemaphore" in repr_str
        assert "capacity=5" in repr_str
        assert "current_count=5" in repr_str

    @pytest.mark.asyncio
    async def test_locked_method(self):
        """Test locked() method returns correct state."""
        semaphore = AdaptiveSemaphore(2)

        assert not semaphore.locked()

        await semaphore.acquire()
        assert not semaphore.locked()

        await semaphore.acquire()
        assert semaphore.locked()

        semaphore.release()
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_concurrent_capacity_adjustments(self):
        """Test concurrent capacity adjustments don't cause race conditions."""
        semaphore = AdaptiveSemaphore(2)

        async def adjust_up():
            await semaphore.adjust_capacity(5)

        async def adjust_down():
            await semaphore.adjust_capacity(1)

        # Run concurrent adjustments
        await asyncio.gather(adjust_up(), adjust_down())

        # Final state should be consistent
        assert semaphore._max_capacity in [1, 5]
        assert semaphore._current_capacity <= semaphore._max_capacity

    @pytest.mark.asyncio
    async def test_many_waiters_performance(self):
        """Test handling of large numbers of waiters."""
        semaphore = AdaptiveSemaphore(1)

        # Acquire the permit
        await semaphore.acquire()

        # Create many waiters
        num_waiters = 1000
        tasks = [asyncio.create_task(semaphore.acquire()) for _ in range(num_waiters)]

        await asyncio.sleep(0.01)
        assert len(semaphore._waiters) == num_waiters

        # Increase capacity significantly to wake them all
        await semaphore.adjust_capacity(num_waiters + 1)

        # All should complete
        await asyncio.gather(*tasks)
        assert len(semaphore._waiters) == 0

    @pytest.mark.asyncio
    async def test_waiter_cleanup_on_exception(self):
        """Test that waiters are properly cleaned up when tasks fail."""
        semaphore = AdaptiveSemaphore(1)

        await semaphore.acquire()

        async def failing_acquire():
            await semaphore.acquire()
            raise RuntimeError("Task failed after acquire")

        task = asyncio.create_task(failing_acquire())

        # Release to let the task acquire
        semaphore.release()

        with pytest.raises(RuntimeError):
            await task

        # Semaphore should be in clean state
        assert len(semaphore._waiters) == 0
        assert semaphore._current_capacity == 0  # Still acquired by failed task
