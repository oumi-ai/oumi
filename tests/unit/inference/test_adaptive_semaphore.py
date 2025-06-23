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

import pytest

from oumi.inference.adaptive_semaphore import AdaptiveSemaphore


class TestAdaptiveSemaphore:
    """Test suite for AdaptiveSemaphore class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test semaphore initialization with different capacities."""
        # Test basic initialization
        semaphore = AdaptiveSemaphore(5)
        assert semaphore._capacity == 5
        assert semaphore._current_count == 5
        assert len(semaphore._waiters) == 0

        # Test with capacity of 1
        semaphore = AdaptiveSemaphore(1)
        assert semaphore._capacity == 1
        assert semaphore._current_count == 1

    @pytest.mark.asyncio
    async def test_basic_acquire_release(self):
        """Test basic acquire and release functionality."""
        semaphore = AdaptiveSemaphore(2)

        # First acquire should succeed immediately
        await semaphore.acquire()
        assert semaphore._current_count == 1

        # Second acquire should succeed immediately
        await semaphore.acquire()
        assert semaphore._current_count == 0

        # Release one permit
        semaphore.release()
        # Wait for async release to complete
        await asyncio.sleep(0.01)
        assert semaphore._current_count == 1

        # Release another permit
        semaphore.release()
        await asyncio.sleep(0.01)
        assert semaphore._current_count == 2

    @pytest.mark.asyncio
    async def test_acquire_blocking_when_capacity_reached(self):
        """Test that acquire blocks when capacity is reached."""
        semaphore = AdaptiveSemaphore(1)

        # First acquire should succeed immediately
        await semaphore.acquire()
        assert semaphore._current_count == 0

        # Second acquire should block
        acquire_task = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0.01)  # Give task a chance to run
        assert not acquire_task.done()
        assert len(semaphore._waiters) == 1

        # Release should unblock the waiting acquire
        semaphore.release()
        await asyncio.sleep(0.01)  # Give release task a chance to complete
        await acquire_task  # This should complete now
        assert semaphore._current_count == 0

    @pytest.mark.asyncio
    async def test_multiple_waiters(self):
        """Test handling of multiple waiting tasks."""
        semaphore = AdaptiveSemaphore(1)

        # Acquire the only permit
        await semaphore.acquire()
        assert semaphore._current_count == 0

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
        await asyncio.sleep(0.01)
        await task1  # Should complete
        assert not task2.done()
        assert not task3.done()
        assert len(semaphore._waiters) == 2

        # Release again should wake up second waiter
        semaphore.release()
        await asyncio.sleep(0.01)
        await task2  # Should complete
        assert not task3.done()
        assert len(semaphore._waiters) == 1

        # Release again should wake up third waiter
        semaphore.release()
        await asyncio.sleep(0.01)
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
        await asyncio.sleep(0.01)
        await task2  # Should complete
        assert len(semaphore._waiters) == 0

    @pytest.mark.asyncio
    async def test_adjust_capacity_increase(self):
        """Test increasing semaphore capacity."""
        semaphore = AdaptiveSemaphore(2)

        # Acquire both permits
        await semaphore.acquire()
        await semaphore.acquire()
        assert semaphore._current_count == 0

        # Create waiting tasks
        task1 = asyncio.create_task(semaphore.acquire())
        task2 = asyncio.create_task(semaphore.acquire())

        await asyncio.sleep(0.01)
        assert len(semaphore._waiters) == 2

        # Increase capacity should wake up waiters
        await semaphore.adjust_capacity(4)
        assert semaphore._capacity == 4
        await asyncio.sleep(0.01)

        await task1  # Should complete
        await task2  # Should complete
        assert len(semaphore._waiters) == 0
        assert semaphore._current_count == 0  # 4 capacity - 4 acquired

    @pytest.mark.asyncio
    async def test_adjust_capacity_decrease(self):
        """Test decreasing semaphore capacity."""
        semaphore = AdaptiveSemaphore(5)

        # Acquire 2 permits
        await semaphore.acquire()
        await semaphore.acquire()
        assert semaphore._current_count == 3

        # Decrease capacity
        await semaphore.adjust_capacity(3)
        assert semaphore._capacity == 3
        assert semaphore._current_count == 1  # 3 - 2 acquired

        # Should be able to acquire one more
        await semaphore.acquire()
        assert semaphore._current_count == 0

        # Next acquire should block
        task = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0.01)
        assert not task.done()
        assert len(semaphore._waiters) == 1

        # Clean up
        semaphore.release()
        await asyncio.sleep(0.01)
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
        await asyncio.sleep(0.01)
        await tasks[2]

    @pytest.mark.asyncio
    async def test_adjust_capacity_decrease_below_current_usage(self):
        """Test decreasing capacity below current usage doesn't revoke permits."""
        semaphore = AdaptiveSemaphore(5)

        # Acquire 4 permits
        for _ in range(4):
            await semaphore.acquire()
        assert semaphore._current_count == 1

        # Decrease capacity below current usage
        await semaphore.adjust_capacity(2)
        assert semaphore._capacity == 2
        # Current count becomes negative (2 - 4 = -2)
        assert semaphore._current_count == -2

        # No new acquires should succeed until enough permits are released
        task = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0.01)
        assert not task.done()

        # Release 3 permits to get back to positive count
        for _ in range(3):
            semaphore.release()
        await asyncio.sleep(0.01)

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
        await semaphore.adjust_capacity(0)
        assert semaphore._capacity == 0
        assert semaphore._current_count == 0

        # New acquires should block
        task = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0.01)
        assert not task.done()

        # Increase capacity should allow acquire
        await semaphore.adjust_capacity(1)
        await asyncio.sleep(0.01)
        await task  # Should complete now

    @pytest.mark.asyncio
    async def test_edge_case_empty_waiters_on_release(self):
        """Test release when there are no waiters."""
        semaphore = AdaptiveSemaphore(2)

        # Acquire one permit
        await semaphore.acquire()
        assert semaphore._current_count == 1

        # Release without any waiters
        semaphore.release()
        await asyncio.sleep(0.01)
        assert semaphore._current_count == 2
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
        await asyncio.sleep(0.01)

        try:
            await task1
        except asyncio.CancelledError:
            pass

        await task2  # Should complete
        assert len(semaphore._waiters) == 0
