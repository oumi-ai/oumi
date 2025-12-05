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
from unittest.mock import patch

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
async def test_polite_adaptive_semaphore_rate_limiting(mock_time, mock_asyncio_sleep):
    """Test that rate limiting enforces minimum interval between grants."""
    semaphore = PoliteAdaptiveSemaphore(capacity=10, politeness_policy=60.0)

    # Min interval should be 60 / 10 = 6 seconds per permit
    assert semaphore._calculate_min_interval() == 6.0

    # First acquire - no wait needed (last_grant_time is 0)
    mock_time.time.return_value = 10.0
    await semaphore.acquire()
    assert semaphore._last_grant_time == 10.0
    assert semaphore._active_workers == 1

    # Second acquire - should wait 6 seconds
    mock_time.time.return_value = 12.0  # Only 2 seconds passed
    await semaphore.acquire()
    # Should have waited: 6 - 2 = 4 seconds
    mock_asyncio_sleep.assert_called_once_with(4.0)

    semaphore.release()
    semaphore.release()


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_no_wait_when_enough_time_passed(
    mock_time, mock_asyncio_sleep
):
    """Test that no wait occurs when enough time has passed."""
    semaphore = PoliteAdaptiveSemaphore(capacity=5, politeness_policy=10.0)

    # Min interval = 10 / 5 = 2 seconds per permit

    # First acquire
    mock_time.time.return_value = 10.0
    await semaphore.acquire()

    # Second acquire - enough time passed
    mock_time.time.return_value = 13.0  # 3 seconds passed (> 2 second min interval)
    await semaphore.acquire()

    # Should not have waited
    mock_asyncio_sleep.assert_not_called()

    semaphore.release()
    semaphore.release()


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_capacity_adjustment():
    """Test that capacity adjustment updates rate calculation."""
    semaphore = PoliteAdaptiveSemaphore(capacity=10, politeness_policy=60.0)

    # Initial: 60 / 10 = 6 seconds per permit
    assert semaphore._calculate_min_interval() == 6.0

    # Increase capacity
    await semaphore.adjust_capacity(20)
    # New: 60 / 20 = 3 seconds per permit
    assert semaphore._calculate_min_interval() == 3.0

    # Decrease capacity
    await semaphore.adjust_capacity(5)
    # New: 60 / 5 = 12 seconds per permit
    assert semaphore._calculate_min_interval() == 12.0


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_active_worker_tracking():
    """Test that active workers are properly tracked."""
    semaphore = PoliteAdaptiveSemaphore(capacity=3, politeness_policy=1.0)

    assert semaphore._active_workers == 0

    # Acquire permits
    await semaphore.acquire()
    assert semaphore._active_workers == 1

    await semaphore.acquire()
    assert semaphore._active_workers == 2

    # Release permits
    semaphore.release()
    assert semaphore._active_workers == 1

    semaphore.release()
    assert semaphore._active_workers == 0


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_concurrent_grants(
    mock_time, mock_asyncio_sleep
):
    """Test that concurrent grants are serialized and rate-limited."""
    semaphore = PoliteAdaptiveSemaphore(capacity=2, politeness_policy=10.0)
    # Min interval = 10 / 2 = 5 seconds per permit

    # Simulate concurrent acquires
    # First acquire: time.time() returns 10.0 twice (before sleep check, after)
    # Second acquire: time.time() returns 12.0 (before), then 17.0 (after sleep)
    times = [10.0, 10.0, 12.0, 17.0]
    time_idx = [0]

    def mock_time_fn():
        result = times[time_idx[0]]
        time_idx[0] += 1
        return result

    mock_time.time.side_effect = mock_time_fn

    # First acquire at t=10
    # _last_grant_time = 0, so no wait needed
    await semaphore.acquire()
    # _last_grant_time is now 10.0

    # Second acquire at t=12 (only 2 seconds after first)
    # time_since_last = 12.0 - 10.0 = 2.0
    # min_interval = 5.0
    # Should wait: 5.0 - 2.0 = 3.0 seconds
    await semaphore.acquire()
    mock_asyncio_sleep.assert_called_once_with(3.0)

    semaphore.release()
    semaphore.release()


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_respects_capacity_limit():
    """Test that capacity limiting still works with rate limiting."""
    semaphore = PoliteAdaptiveSemaphore(capacity=2, politeness_policy=1.0)

    # Acquire both permits
    await semaphore.acquire()
    await semaphore.acquire()
    assert semaphore._current_capacity == 0

    # Third acquire should block
    task = asyncio.create_task(semaphore.acquire())
    await asyncio.sleep(0.01)
    assert not task.done()
    assert len(semaphore._waiters) == 1

    # Release one permit
    semaphore.release()
    await task

    # Clean up
    semaphore.release()
    semaphore.release()


@pytest.mark.asyncio
async def test_polite_adaptive_semaphore_initial_state():
    """Test initial state of rate-limited semaphore."""
    semaphore = PoliteAdaptiveSemaphore(capacity=5, politeness_policy=30.0)

    assert semaphore._politeness_policy == 30.0
    assert semaphore._max_capacity == 5
    assert semaphore._active_workers == 0
    assert semaphore._last_grant_time == 0.0


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
        # Current capacity can go negative to track excess active workers
        # Started with capacity=1, decreased by 3 (from 5 to 2), so: 1 - 3 = -2
        assert semaphore._current_capacity == -2

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
