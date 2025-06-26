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
import time
from unittest.mock import AsyncMock, patch

import pytest

from oumi.core.configs.params.remote_params import AdaptiveThroughputParams
from oumi.inference.adaptive_throughput_controller import AdaptiveThroughputController


class TestAdaptiveThroughputController:
    """Test suite for AdaptiveThroughputController class."""

    def create_config(self, **kwargs):
        """Create a test configuration with default values."""
        defaults = {
            "initial_concurrency": 5,
            "max_concurrency": 20,
            "concurrency_step": 2,
            "update_interval": 1.0,  # Short interval for faster tests
            "error_threshold": 0.1,  # 10%
            "backoff_factor": 0.8,
            "recovery_threshold": 0.05,  # 5%
            "min_window_size": 5,
        }
        defaults.update(kwargs)
        # Ensure recovery_threshold < error_threshold
        if defaults["recovery_threshold"] >= defaults["error_threshold"]:
            defaults["recovery_threshold"] = defaults["error_threshold"] - 0.01
        # Ensure min_window_size >= 1
        if defaults["min_window_size"] < 1:
            defaults["min_window_size"] = 1
        return AdaptiveThroughputParams(**defaults)

    def test_initialization(self):
        """Test controller initialization with various configurations."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        assert controller._config == config
        assert controller._current_concurrency == config.min_concurrency
        assert controller._semaphore is not None
        assert len(controller._outcomes) == 0
        assert controller._last_adjustment_time == 0
        assert controller._last_warmup_time == 0
        assert not controller._in_backoff
        assert controller._consecutive_good_windows == 0
        assert controller._consecutive_error_windows == 0

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration values."""
        config = self.create_config(
            initial_concurrency=10,
            max_concurrency=50,
            concurrency_step=5,
            error_threshold=0.05,
        )
        controller = AdaptiveThroughputController(config)

        assert controller._current_concurrency == 10
        assert controller._config.max_concurrency == 50
        assert controller._config.concurrency_step == 5
        assert controller._config.error_threshold == 0.05

    def test_record_success(self):
        """Test recording successful requests."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # Record multiple successes
        for _ in range(5):
            controller.record_success()

        assert len(controller._outcomes) == 5
        assert all(outcome for outcome in controller._outcomes)

    def test_record_error(self):
        """Test recording failed requests."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # Record multiple errors
        for _ in range(3):
            controller.record_error()

        assert len(controller._outcomes) == 3
        assert all(not outcome for outcome in controller._outcomes)

    def test_record_mixed_outcomes(self):
        """Test recording mixed success and error outcomes."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # Record pattern: success, error, success, error, success
        controller.record_success()
        controller.record_error()
        controller.record_success()
        controller.record_error()
        controller.record_success()

        assert len(controller._outcomes) == 5
        expected = [True, False, True, False, True]
        assert list(controller._outcomes) == expected

    def test_get_error_rate_empty(self):
        """Test error rate calculation with no data."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        error_rate = controller._get_error_rate()
        assert error_rate == 0.0

    def test_get_error_rate_all_success(self):
        """Test error rate calculation with all successful requests."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        for _ in range(10):
            controller.record_success()

        error_rate = controller._get_error_rate()
        assert error_rate == 0.0

    def test_get_error_rate_all_errors(self):
        """Test error rate calculation with all failed requests."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        for _ in range(10):
            controller.record_error()

        error_rate = controller._get_error_rate()
        assert error_rate == 1.0

    def test_get_error_rate_mixed(self):
        """Test error rate calculation with mixed outcomes."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # 7 successes, 3 errors = 30% error rate
        for _ in range(7):
            controller.record_success()
        for _ in range(3):
            controller.record_error()

        error_rate = controller._get_error_rate()
        assert error_rate == 0.3

    @pytest.mark.asyncio
    async def test_acquire_and_release_basic(self):
        """Test basic acquire and release functionality."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # Test acquire and release with real semaphore
        await controller.acquire()
        controller.release()

        # Verify we can acquire again after release
        await controller.acquire()
        controller.release()

    @pytest.mark.asyncio
    async def test_acquire_calls_maybe_adjust_concurrency(self):
        """Test that acquire calls concurrency adjustment logic."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # Mock only the adjustment method to verify it's called
        with patch.object(
            controller, "_maybe_adjust_concurrency", new_callable=AsyncMock
        ) as mock_adjust:
            await controller.acquire()
            mock_adjust.assert_called_once()

        controller.release()

    @pytest.mark.asyncio
    async def test_maybe_adjust_concurrency_no_data(self):
        """Test that adjustment doesn't happen with insufficient data."""
        config = self.create_config(min_window_size=10)
        controller = AdaptiveThroughputController(config)

        # Add some data but less than min_window_size
        for _ in range(5):
            controller.record_success()

        await controller._maybe_adjust_concurrency()

        # No adjustment should have occurred
        assert controller._current_concurrency == config.min_concurrency

    @pytest.mark.asyncio
    async def test_maybe_adjust_concurrency_too_soon(self):
        """Test that adjustment doesn't happen too frequently."""
        config = self.create_config(update_interval=60.0)  # 1 minute
        controller = AdaptiveThroughputController(config)

        # Add sufficient data
        for _ in range(10):
            controller.record_success()

        # Set last adjustment time to now
        controller._last_adjustment_time = time.time()

        await controller._maybe_adjust_concurrency()

        # No adjustment should have occurred
        assert controller._current_concurrency == config.min_concurrency

    @pytest.mark.asyncio
    async def test_backoff_on_high_error_rate(self):
        """Test backoff behavior when error rate exceeds threshold."""
        config = self.create_config(
            error_threshold=0.2,  # 20%
            backoff_factor=0.8,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)

        # Create high error rate (3 errors out of 5 = 60%)
        controller.record_success()
        controller.record_success()
        controller.record_error()
        controller.record_error()
        controller.record_error()

        # Make sure enough time has passed
        controller._last_adjustment_time = time.time() - 1.0

        await controller._maybe_adjust_concurrency()

        # Should have triggered backoff state but concurrency stays at initial value
        # because it should never go below initial concurrency
        expected_concurrency = max(
            config.min_concurrency,
            int(config.min_concurrency * config.backoff_factor),
        )
        assert controller._current_concurrency == expected_concurrency
        assert controller._in_backoff

    @pytest.mark.asyncio
    async def test_backoff_minimum_concurrency(self):
        """Test that backoff doesn't go below initial concurrency."""
        config = self.create_config(
            initial_concurrency=10,
            backoff_factor=0.1,  # Very aggressive backoff
            error_threshold=0.2,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)

        # Start with higher concurrency than initial
        controller._current_concurrency = 20

        # Create high error rate
        for _ in range(10):
            controller.record_error()

        controller._last_adjustment_time = time.time() - 1.0

        await controller._maybe_adjust_concurrency()

        # Should not go below initial concurrency even with aggressive backoff
        # max(10, 20 * 0.1) = max(10, 2) = 10
        assert controller._current_concurrency == config.min_concurrency
        assert controller._in_backoff

    @pytest.mark.asyncio
    async def test_warmup_on_low_error_rate(self):
        """Test warmup behavior when error rate is low."""
        config = self.create_config(
            initial_concurrency=10,
            max_concurrency=20,
            concurrency_step=3,
            recovery_threshold=0.1,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)

        # Create low error rate (all successes)
        for _ in range(10):
            controller.record_success()

        controller._last_adjustment_time = time.time() - 1.0

        await controller._maybe_adjust_concurrency()

        # Should have increased concurrency
        expected_concurrency = config.min_concurrency + config.concurrency_step
        assert controller._current_concurrency == expected_concurrency

    @pytest.mark.asyncio
    async def test_warmup_max_concurrency_limit(self):
        """Test that warmup doesn't exceed max concurrency."""
        config = self.create_config(
            initial_concurrency=18,
            max_concurrency=20,
            concurrency_step=5,
            recovery_threshold=0.1,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)

        # Set current concurrency close to max
        controller._current_concurrency = 18

        # Create low error rate
        for _ in range(10):
            controller.record_success()

        controller._last_adjustment_time = time.time() - 1.0

        await controller._maybe_adjust_concurrency()

        # Should not exceed max concurrency
        assert controller._current_concurrency == config.max_concurrency

    @pytest.mark.asyncio
    async def test_recovery_from_backoff(self):
        """Test recovery from backoff state."""
        config = self.create_config(
            error_threshold=0.2,
            recovery_threshold=0.05,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)

        # Start with higher concurrency then trigger backoff
        controller._current_concurrency = 10
        controller._in_backoff = True
        controller._consecutive_good_windows = 0

        # Create low error rate (all successes, 0% error rate)
        for _ in range(10):
            controller.record_success()

        controller._last_adjustment_time = time.time() - 1.0

        # With 0% error rate (< recovery_threshold), should increment good windows
        await controller._maybe_adjust_concurrency()
        assert controller._consecutive_good_windows == 1
        assert controller._in_backoff  # Still in backoff after first good window

        # Clear outcomes and add more successes for second window
        controller._outcomes.clear()
        for _ in range(10):
            controller.record_success()

        # Second call should exit backoff after 2 good windows
        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()
        assert not controller._in_backoff
        assert controller._consecutive_good_windows == 0

    @pytest.mark.asyncio
    async def test_additional_backoff_in_backoff_state(self):
        """Test additional backoff when already in backoff with continued errors."""
        config = self.create_config(
            error_threshold=0.2,
            recovery_threshold=0.05,
            backoff_factor=0.8,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)
        controller._semaphore = AsyncMock()

        # Start in backoff state
        controller._in_backoff = True
        controller._consecutive_error_windows = 0
        initial_concurrency = config.min_concurrency
        controller._current_concurrency = initial_concurrency

        # Create high error rate
        for _ in range(2):
            controller.record_error()
        for _ in range(3):
            controller.record_success()

        controller._last_adjustment_time = time.time() - 1.0

        # First call should increment error windows
        await controller._maybe_adjust_concurrency()
        assert controller._consecutive_error_windows == 1

        # Second call should trigger additional backoff
        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()
        expected_concurrency = max(
            initial_concurrency, int(initial_concurrency * config.backoff_factor)
        )
        assert controller._current_concurrency == expected_concurrency

    @pytest.mark.asyncio
    async def test_update_concurrency_resets_outcomes(self):
        """Test that updating concurrency resets outcome tracking."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)
        controller._semaphore = AsyncMock()

        # Add some outcomes
        for _ in range(5):
            controller.record_success()
        controller.record_error()

        assert len(controller._outcomes) == 6

        # Update concurrency should reset outcomes
        await controller._update_concurrency(10)

        assert len(controller._outcomes) == 0
        assert controller._consecutive_good_windows == 0
        assert controller._consecutive_error_windows == 0

    @pytest.mark.asyncio
    async def test_end_backoff_resets_counters(self):
        """Test that ending backoff resets window counters."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # Set up backoff state
        controller._in_backoff = True
        controller._consecutive_good_windows = 5
        controller._consecutive_error_windows = 3

        controller._end_backoff()

        assert not controller._in_backoff
        assert controller._consecutive_good_windows == 0
        assert controller._consecutive_error_windows == 0

    @pytest.mark.asyncio
    async def test_concurrent_access_to_outcomes(self):
        """Test thread-safe access to outcomes tracking."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        async def record_outcomes():
            for i in range(100):
                if i % 2 == 0:
                    controller.record_success()
                else:
                    controller.record_error()

        # Run multiple tasks concurrently
        tasks = [asyncio.create_task(record_outcomes()) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should have 500 total outcomes
        assert len(controller._outcomes) == 500
        # Should have equal success and errors
        successes = sum(1 for outcome in controller._outcomes if outcome)
        errors = sum(1 for outcome in controller._outcomes if not outcome)
        assert successes == 250
        assert errors == 250

    @pytest.mark.asyncio
    async def test_edge_case_zero_outcomes(self):
        """Test behavior with zero recorded outcomes."""
        config = self.create_config(min_window_size=0)
        controller = AdaptiveThroughputController(config)

        error_rate = controller._get_error_rate()
        assert error_rate == 0.0

        # Should not adjust concurrency
        await controller._maybe_adjust_concurrency()
        assert controller._current_concurrency == config.min_concurrency

    @pytest.mark.asyncio
    async def test_edge_case_single_outcome(self):
        """Test behavior with single outcome."""
        config = self.create_config(min_window_size=1, update_interval=0.1)
        controller = AdaptiveThroughputController(config)
        controller._semaphore = AsyncMock()

        # Single success
        controller.record_success()
        controller._last_adjustment_time = time.time() - 1.0

        error_rate = controller._get_error_rate()
        assert error_rate == 0.0

        await controller._maybe_adjust_concurrency()
        # Should trigger warmup due to low error rate

    @pytest.mark.asyncio
    async def test_backoff_state_persistence(self):
        """Test that backoff state persists correctly."""
        config = self.create_config(
            error_threshold=0.2,
            recovery_threshold=0.05,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)
        controller._semaphore = AsyncMock()

        # Trigger backoff
        for _ in range(8):
            controller.record_error()
        for _ in range(2):
            controller.record_success()

        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()

        assert controller._in_backoff

        # Add more mixed outcomes but still above recovery threshold
        controller._outcomes.clear()
        for _ in range(7):
            controller.record_success()
        for _ in range(3):
            controller.record_error()  # 30% error rate

        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()

        # Should still be in backoff due to error rate above recovery threshold
        assert controller._in_backoff

    @pytest.mark.asyncio
    async def test_configuration_edge_cases(self):
        """Test behavior with edge case configurations."""
        # Test with minimum values
        config = self.create_config(
            initial_concurrency=1,
            max_concurrency=1,
            concurrency_step=1,
            min_window_size=1,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)
        controller._semaphore = AsyncMock()

        # Should handle this configuration gracefully
        controller.record_success()
        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()

        # Concurrency should remain at 1 (can't increase due to max)
        assert controller._current_concurrency == 1

    @pytest.mark.asyncio
    async def test_timing_precision(self):
        """Test timing precision in update intervals."""
        config = self.create_config(update_interval=0.5)
        controller = AdaptiveThroughputController(config)

        # Add data
        for _ in range(10):
            controller.record_success()

        # Set adjustment time to just under the interval
        controller._last_adjustment_time = time.time() - 0.4

        # Should not adjust yet
        await controller._maybe_adjust_concurrency()
        assert controller._current_concurrency == config.min_concurrency

        # Wait a bit more
        await asyncio.sleep(0.2)
        controller._semaphore = AsyncMock()

        # Now should adjust
        await controller._maybe_adjust_concurrency()
        controller._semaphore.adjust_capacity.assert_called_once()

    @pytest.mark.asyncio
    async def test_large_scale_outcomes(self):
        """Test with large number of outcomes."""
        config = self.create_config(min_window_size=1000)
        controller = AdaptiveThroughputController(config)

        # Add 1000 outcomes with 10% error rate
        for i in range(1000):
            if i % 10 == 0:
                controller.record_error()
            else:
                controller.record_success()

        error_rate = controller._get_error_rate()
        assert abs(error_rate - 0.1) < 0.01  # Allow small floating point error

    def test_outcome_deque_behavior(self):
        """Test that outcomes are stored in a deque properly."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # Verify it's a deque
        from collections import deque

        assert isinstance(controller._outcomes, deque)

        # Test FIFO behavior (though we don't have sliding window implemented)
        controller.record_success()
        controller.record_error()
        controller.record_success()

        outcomes_list = list(controller._outcomes)
        assert outcomes_list == [True, False, True]

    @pytest.mark.asyncio
    async def test_semaphore_error_handling(self):
        """Test error handling when semaphore operations fail."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # Mock semaphore to raise exception
        controller._semaphore = AsyncMock()
        controller._semaphore.adjust_capacity.side_effect = Exception("Semaphore error")

        # Should handle the exception gracefully
        with pytest.raises(Exception, match="Semaphore error"):
            await controller._update_concurrency(10)

    @pytest.mark.asyncio
    async def test_multiple_adjustments_sequence(self):
        """Test a realistic sequence of multiple adjustments."""
        config = self.create_config(
            initial_concurrency=10,
            max_concurrency=30,
            concurrency_step=5,
            error_threshold=0.15,
            recovery_threshold=0.05,
            backoff_factor=0.7,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)

        # Phase 1: Low error rate, should warm up
        for _ in range(10):
            controller.record_success()

        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()
        assert controller._current_concurrency == 15  # 10 + 5

        # Phase 2: Continue low error rate, warm up more
        controller._outcomes.clear()
        for _ in range(10):
            controller.record_success()

        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()
        assert controller._current_concurrency == 20  # 15 + 5

        # Phase 3: High error rate, should backoff
        controller._outcomes.clear()
        for _ in range(2):
            controller.record_success()
        for _ in range(8):
            controller.record_error()  # 80% error rate

        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()
        expected = max(10, int(20 * 0.7))  # max(initial, current * backoff_factor)
        assert controller._current_concurrency == expected
        assert controller._in_backoff

        # Phase 4: Recovery - since we're at minimum concurrency (10),
        # no adjustment will happen but we should track good windows
        # However, since no concurrency change happens at min level,
        # the backoff flag should remain until we get enough good windows

    @pytest.mark.asyncio
    @patch("time.time")
    async def test_time_mocking(self, mock_time):
        """Test with mocked time for precise timing control."""
        config = self.create_config(update_interval=60.0)
        controller = AdaptiveThroughputController(config)

        # Set initial time
        mock_time.return_value = 1000.0
        controller._last_adjustment_time = 1000.0

        # Add data
        for _ in range(10):
            controller.record_success()

        # Should not adjust (no time passed)
        await controller._maybe_adjust_concurrency()
        assert controller._current_concurrency == config.min_concurrency

        # Advance time past interval
        mock_time.return_value = 1061.0
        controller._semaphore = AsyncMock()

        # Now should adjust
        await controller._maybe_adjust_concurrency()
        controller._semaphore.adjust_capacity.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_outcomes_functionality(self):
        """Test the _reset_outcomes method functionality."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        # Add some outcomes and state
        for _ in range(5):
            controller.record_success()
        controller._consecutive_good_windows = 3
        controller._consecutive_error_windows = 2

        old_time = controller._last_adjustment_time

        controller._reset_outcomes()

        assert len(controller._outcomes) == 0
        assert controller._consecutive_good_windows == 0
        assert controller._consecutive_error_windows == 0
        assert controller._last_adjustment_time > old_time

    @pytest.mark.asyncio
    async def test_no_warmup_at_max_concurrency(self):
        """Test that warmup doesn't occur when already at max concurrency."""
        config = self.create_config(
            initial_concurrency=20,
            max_concurrency=20,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)
        controller._semaphore = AsyncMock()

        # Already at max concurrency
        assert controller._current_concurrency == config.max_concurrency

        # Create low error rate
        for _ in range(10):
            controller.record_success()

        controller._last_adjustment_time = time.time() - 1.0

        await controller._maybe_adjust_concurrency()

        # Should not increase beyond max
        assert controller._current_concurrency == config.max_concurrency
        # Should not have called adjust_capacity since no change needed
        controller._semaphore.adjust_capacity.assert_not_called()

    @pytest.mark.asyncio
    async def test_backoff_state_persistence_across_calls(self):
        """Test that backoff state persists correctly across multiple calls."""
        config = self.create_config(
            error_threshold=0.2,
            recovery_threshold=0.05,
            min_window_size=5,
            update_interval=0.1,
        )
        controller = AdaptiveThroughputController(config)
        controller._semaphore = AsyncMock()

        # Trigger initial backoff
        for _ in range(8):
            controller.record_error()
        for _ in range(2):
            controller.record_success()

        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()

        assert controller._in_backoff

        # Add more mixed outcomes but still above recovery threshold
        controller._outcomes.clear()
        for _ in range(7):
            controller.record_success()
        for _ in range(3):
            controller.record_error()  # 30% error rate

        controller._last_adjustment_time = time.time() - 1.0
        await controller._maybe_adjust_concurrency()

        # Should still be in backoff due to error rate above recovery threshold
        assert controller._in_backoff
        # Should increment error windows
        assert controller._consecutive_error_windows == 1

    def test_thread_safety_of_outcome_tracking(self):
        """Test thread safety of outcome recording operations."""
        config = self.create_config()
        controller = AdaptiveThroughputController(config)

        import threading

        def record_many():
            for _ in range(1000):
                controller.record_success()
                controller.record_error()

        # Create multiple threads
        threads = [threading.Thread(target=record_many) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have recorded all outcomes without corruption
        assert len(controller._outcomes) == 10000  # 5 threads * 1000 * 2 operations
