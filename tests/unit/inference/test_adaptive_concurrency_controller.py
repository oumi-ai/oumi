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
import math
from unittest.mock import AsyncMock, patch

import pytest

from oumi.core.configs.params.remote_params import AdaptiveConcurrencyParams
from oumi.inference.adaptive_concurrency_controller import AdaptiveConcurrencyController


#
# Fixtures
#
@pytest.fixture
def mock_time():
    with patch("oumi.inference.adaptive_concurrency_controller.time") as time_mock:
        yield time_mock


def create_config(**kwargs):
    """Create a test configuration with default values."""
    defaults = {
        "min_concurrency": 5,
        "max_concurrency": 20,
        "concurrency_step": 2,
        "min_update_time": 1.0,  # Short interval for faster tests
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
    return AdaptiveConcurrencyParams(**defaults)


def test_initialization():
    """Test controller initialization with default configuration."""
    config = AdaptiveConcurrencyParams()
    controller = AdaptiveConcurrencyController(config)

    assert controller._config == config
    assert controller._current_concurrency == 52
    assert controller._semaphore is not None
    assert len(controller._outcomes) == 0
    assert controller._last_adjustment_time == 0
    assert controller._last_warmup_time == 0
    assert not controller._in_backoff
    assert controller._consecutive_good_windows_since_last_update == 0
    assert controller._consecutive_error_windows_since_last_update == 0


def test_initialization_with_custom_config():
    """Test initialization with custom configuration values."""
    config = create_config(
        min_concurrency=10,
        max_concurrency=50,
        concurrency_step=5,
        error_threshold=0.05,
    )
    controller = AdaptiveConcurrencyController(config)

    assert controller._current_concurrency == 30
    assert controller._config.max_concurrency == 50
    assert controller._config.concurrency_step == 5
    assert controller._config.error_threshold == 0.05


@pytest.mark.asyncio
async def test_record_success():
    """Test recording successful requests."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Record multiple successes
    for _ in range(5):
        await controller.record_success()

    assert len(controller._outcomes) == 5
    assert all(outcome for outcome in controller._outcomes)


@pytest.mark.asyncio
async def test_record_error():
    """Test recording failed requests."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Record multiple errors
    for _ in range(5):
        await controller.record_error()

    assert len(controller._outcomes) == 5
    assert all(not outcome for outcome in controller._outcomes)


@pytest.mark.asyncio
async def test_record_mixed_outcomes():
    """Test recording mixed success and error outcomes."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Record pattern: success, error, success, error, success
    await controller.record_success()
    await controller.record_error()
    await controller.record_success()
    await controller.record_error()
    await controller.record_success()

    assert len(controller._outcomes) == 5
    expected = [True, False, True, False, True]
    assert list(controller._outcomes) == expected


@pytest.mark.asyncio
async def test_get_error_rate_empty():
    """Test error rate calculation with no data."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    error_rate = await controller._get_error_rate()
    assert error_rate == 0.0


@pytest.mark.asyncio
async def test_get_error_rate_all_success():
    """Test error rate calculation with all successful requests."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    for _ in range(10):
        await controller.record_success()

    error_rate = await controller._get_error_rate()
    assert error_rate == 0.0


@pytest.mark.asyncio
async def test_get_error_rate_all_errors():
    """Test error rate calculation with all failed requests."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    for _ in range(10):
        await controller.record_error()

    error_rate = await controller._get_error_rate()
    assert error_rate == 1.0


@pytest.mark.asyncio
async def test_get_error_rate_mixed():
    """Test error rate calculation with mixed outcomes."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # 7 successes, 3 errors = 30% error rate
    for _ in range(7):
        await controller.record_success()
    for _ in range(3):
        await controller.record_error()

    error_rate = await controller._get_error_rate()
    assert error_rate == 0.3


@pytest.mark.asyncio
async def test_acquire_and_release_basic():
    """Test basic acquire and release functionality."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Test acquire and release with real semaphore
    await controller.acquire()
    controller.release()

    # Verify we can acquire again after release
    await controller.acquire()
    controller.release()


@pytest.mark.asyncio
async def test_acquire_calls_try_adjust_concurrency():
    """Test that acquire calls concurrency adjustment logic."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Mock only the adjustment method to verify it's called
    with patch.object(
        controller, "_try_adjust_concurrency", new_callable=AsyncMock
    ) as mock_adjust:
        await controller.acquire()
        mock_adjust.assert_called_once()
        mock_adjust.reset_mock()
        controller.release()
        mock_adjust.assert_not_called()


@pytest.mark.asyncio
async def test_try_adjust_concurrency_no_data():
    """Test that adjustment doesn't happen with insufficient data."""
    config = create_config(min_window_size=10)
    controller = AdaptiveConcurrencyController(config)
    initial_concurrency = controller._current_concurrency

    # Add some data but less than min_window_size
    for _ in range(5):
        await controller.record_success()

    await controller._try_adjust_concurrency()

    # No adjustment should have occurred
    assert controller._current_concurrency == initial_concurrency


@pytest.mark.asyncio
async def test_try_adjust_concurrency_too_soon(mock_time):
    """Test that adjustment doesn't happen too frequently."""
    config = create_config(min_update_time=60.0)  # 1 minute
    controller = AdaptiveConcurrencyController(config)
    initial_concurrency = controller._current_concurrency

    # Add sufficient data
    for _ in range(10):
        await controller.record_success()

    # Set last adjustment time to now
    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = config.min_update_time

    await controller._try_adjust_concurrency()

    # No adjustment should have occurred
    assert controller._current_concurrency == initial_concurrency


@pytest.mark.asyncio
async def test_backoff_on_high_error_rate(mock_time):
    """Test backoff behavior when error rate exceeds threshold."""
    config = create_config(
        error_threshold=0.2,  # 20%
        backoff_factor=0.8,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)

    await controller._update_concurrency(100)

    # Create high error rate (1 error out of 5 = 20%)
    await controller.record_success()
    await controller.record_success()
    await controller.record_success()
    await controller.record_success()
    await controller.record_error()

    # Make sure enough time has passed
    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()

    assert controller._current_concurrency == 80
    assert controller._in_backoff


@pytest.mark.asyncio
async def test_backoff_minimum_concurrency(mock_time):
    """Test that backoff doesn't go below initial concurrency."""
    config = create_config(
        min_concurrency=10,
        backoff_factor=0.1,  # Very aggressive backoff
        error_threshold=0.2,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)

    # Start with higher concurrency than initial
    await controller._update_concurrency(20)

    # Create high error rate
    for _ in range(10):
        await controller.record_error()

    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0

    await controller._try_adjust_concurrency()

    # Should not go below initial concurrency even with aggressive backoff
    assert controller._current_concurrency == config.min_concurrency
    assert controller._in_backoff


@pytest.mark.asyncio
async def test_warmup_on_low_error_rate(mock_time):
    """Test warmup behavior when error rate is low."""
    config = create_config(
        min_concurrency=10,
        max_concurrency=20,
        concurrency_step=3,
        recovery_threshold=0.1,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)
    initial_concurrency = controller._current_concurrency

    # Create low error rate (all successes)
    for _ in range(10):
        await controller.record_success()

    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0

    await controller._try_adjust_concurrency()

    # Should have increased concurrency
    assert controller._current_concurrency == (
        initial_concurrency + config.concurrency_step
    )


@pytest.mark.asyncio
async def test_warmup_max_concurrency_limit(mock_time):
    """Test that warmup doesn't exceed max concurrency."""
    config = create_config(
        min_concurrency=18,
        max_concurrency=20,
        concurrency_step=5,
        recovery_threshold=0.1,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)

    # Set current concurrency close to max
    await controller._update_concurrency(18)

    # Create low error rate
    for _ in range(10):
        await controller.record_success()

    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0

    await controller._try_adjust_concurrency()

    # Should not exceed max concurrency
    assert controller._current_concurrency == config.max_concurrency


@pytest.mark.asyncio
async def test_recovery_from_backoff(mock_time):
    """Test recovery from backoff state."""
    config = create_config(
        error_threshold=0.2,
        recovery_threshold=0.05,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)

    # Start with higher concurrency then trigger backoff
    initial_concurrency = 10
    await controller._update_concurrency(initial_concurrency)
    controller._in_backoff = True

    # Create low error rate (all successes, 0% error rate)
    for _ in range(10):
        await controller.record_success()

    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0

    # With 0% error rate (< recovery_threshold), should increment good windows
    await controller._try_adjust_concurrency()
    assert controller._consecutive_good_windows_since_last_update == 1
    assert controller._in_backoff  # Still in backoff after first good window

    # Clear outcomes and add more successes for second window
    controller._outcomes.clear()
    for _ in range(10):
        await controller.record_success()

    # Second call should exit backoff after 2 good windows
    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()
    assert not controller._in_backoff
    assert controller._consecutive_good_windows_since_last_update == 0

    # Validate that we warmup above previous concurrency
    assert (
        controller._current_concurrency == initial_concurrency + config.concurrency_step
    )


@pytest.mark.asyncio
async def test_additional_backoff_in_backoff_state(mock_time):
    """Test additional backoff when already in backoff with continued errors."""
    config = create_config(
        error_threshold=0.2,
        recovery_threshold=0.05,
        backoff_factor=0.8,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)
    controller._semaphore = AsyncMock()

    # Start in backoff state
    controller._in_backoff = True
    controller._consecutive_error_windows_since_last_update = 0
    initial_concurrency = 100
    await controller._update_concurrency(initial_concurrency)

    # Create high error rate
    for _ in range(2):
        await controller.record_error()
    for _ in range(3):
        await controller.record_success()

    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0

    # First call should increment error windows
    await controller._try_adjust_concurrency()
    assert controller._consecutive_error_windows_since_last_update == 1

    # Create high error rate
    for _ in range(2):
        await controller.record_error()
    for _ in range(3):
        await controller.record_success()

    # Second call should trigger additional backoff
    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()
    assert controller._current_concurrency == 80


@pytest.mark.asyncio
async def test_update_concurrency_resets_outcomes():
    """Test that updating concurrency resets outcome tracking."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)
    controller._semaphore = AsyncMock()

    # Add some outcomes
    for _ in range(5):
        await controller.record_success()
    await controller.record_error()

    assert len(controller._outcomes) == 6

    # Update concurrency should reset outcomes
    await controller._update_concurrency(10)

    assert len(controller._outcomes) == 0
    assert controller._consecutive_good_windows_since_last_update == 0
    assert controller._consecutive_error_windows_since_last_update == 0


@pytest.mark.asyncio
async def test_end_backoff_resets_counters():
    """Test that ending backoff resets window counters."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Set up backoff state
    controller._in_backoff = True
    controller._consecutive_good_windows_since_last_update = 5
    controller._consecutive_error_windows_since_last_update = 3

    controller._end_backoff()

    assert not controller._in_backoff
    assert controller._consecutive_good_windows_since_last_update == 0
    assert controller._consecutive_error_windows_since_last_update == 0


@pytest.mark.asyncio
async def test_concurrent_access_to_outcomes():
    """Test thread-safe access to outcomes tracking."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    async def record_outcomes():
        for i in range(100):
            if i % 2 == 0:
                await controller.record_success()
            else:
                await controller.record_error()

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
async def test_edge_case_zero_outcomes():
    """Test behavior with zero recorded outcomes."""
    config = create_config(min_window_size=0)
    controller = AdaptiveConcurrencyController(config)
    initial_concurrency = controller._current_concurrency

    error_rate = await controller._get_error_rate()
    assert error_rate == 0.0

    # Should not adjust concurrency
    await controller._try_adjust_concurrency()
    assert controller._current_concurrency == initial_concurrency


@pytest.mark.asyncio
async def test_backoff_state_persistence(mock_time):
    """Test that backoff state persists correctly."""
    config = create_config(
        error_threshold=0.2,
        recovery_threshold=0.05,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)
    controller._semaphore = AsyncMock()

    # Trigger backoff
    for _ in range(8):
        await controller.record_error()
    for _ in range(2):
        await controller.record_success()

    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()

    assert controller._in_backoff

    # Add more mixed outcomes but still above recovery threshold
    controller._outcomes.clear()
    for _ in range(7):
        await controller.record_success()
    for _ in range(3):
        await controller.record_error()  # 30% error rate

    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()

    # Should still be in backoff due to error rate above recovery threshold
    assert controller._in_backoff


@pytest.mark.asyncio
async def test_configuration_edge_cases(mock_time):
    """Test behavior with edge case configurations."""
    # Test with minimum values
    config = create_config(
        min_concurrency=1,
        max_concurrency=1,
        concurrency_step=1,
        min_window_size=1,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)
    controller._semaphore = AsyncMock()

    # Should handle this configuration gracefully
    await controller.record_success()
    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()

    # Concurrency should remain at 1 (can't increase due to max)
    assert controller._current_concurrency == 1


@pytest.mark.asyncio
async def test_large_scale_outcomes():
    """Test with large number of outcomes."""
    config = create_config(min_window_size=1000)
    controller = AdaptiveConcurrencyController(config)

    # Add 1000 outcomes with 10% error rate
    for i in range(1000):
        if i % 10 == 0:
            await controller.record_error()
        else:
            await controller.record_success()

    error_rate = await controller._get_error_rate()
    assert abs(error_rate - 0.1) < 0.01  # Allow small floating point error


@pytest.mark.asyncio
async def test_outcome_deque_behavior():
    """Test that outcomes are stored in a deque properly."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Verify it's a deque
    from collections import deque

    assert isinstance(controller._outcomes, deque)

    # Test FIFO behavior
    await controller.record_success()
    await controller.record_error()
    await controller.record_success()

    outcomes_list = list(controller._outcomes)
    assert outcomes_list == [True, False, True]


@pytest.mark.asyncio
async def test_semaphore_error_handling():
    """Test error handling when semaphore operations fail."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Mock semaphore to raise exception
    controller._semaphore = AsyncMock()
    controller._semaphore.adjust_capacity.side_effect = Exception("Semaphore error")

    # Should raise an exception
    with pytest.raises(Exception, match="Semaphore error"):
        await controller._update_concurrency(10)


@pytest.mark.asyncio
async def test_multiple_adjustments_sequence(mock_time):
    """Test a realistic sequence of multiple adjustments."""
    config = create_config(
        min_concurrency=10,
        max_concurrency=30,
        concurrency_step=5,
        error_threshold=0.15,
        recovery_threshold=0.05,
        backoff_factor=0.7,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)
    initial_concurrency = controller._current_concurrency

    # Phase 1: Low error rate, should warm up
    phase_1_concurrency = initial_concurrency + config.concurrency_step
    for _ in range(10):
        await controller.record_success()

    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()
    assert controller._current_concurrency == phase_1_concurrency

    # Phase 2: Continue low error rate, warm up more
    phase_2_concurrency = initial_concurrency + 2 * config.concurrency_step
    for _ in range(10):
        await controller.record_success()

    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()
    assert controller._current_concurrency == phase_2_concurrency

    # Phase 3: High error rate, should backoff
    phase_3_concurrency = math.floor(
        (initial_concurrency + 2 * config.concurrency_step) * config.backoff_factor
    )
    for _ in range(2):
        await controller.record_success()
    for _ in range(8):
        await controller.record_error()  # 80% error rate

    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()
    assert controller._current_concurrency == phase_3_concurrency
    assert controller._in_backoff

    # Phase 4: Recovery - first good window should not exit backoff
    phase_4_concurrency = phase_3_concurrency
    for _ in range(10):
        await controller.record_success()
    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()
    assert controller._current_concurrency == phase_4_concurrency
    assert controller._in_backoff

    # Phase 5: Recovery - second good window should exit backoff
    phase_5_concurrency = (
        math.floor(
            (initial_concurrency + 2 * config.concurrency_step) * config.backoff_factor
        )
        + config.concurrency_step
    )
    for _ in range(10):
        await controller.record_success()
    controller._last_adjustment_time = 0
    await controller._try_adjust_concurrency()
    assert controller._current_concurrency == phase_5_concurrency
    assert not controller._in_backoff


@pytest.mark.asyncio
async def test_reset_outcomes_functionality():
    """Test the _reset_outcomes method functionality."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Add some outcomes and state
    for _ in range(5):
        await controller.record_success()
    controller._consecutive_good_windows_since_last_update = 3
    controller._consecutive_error_windows_since_last_update = 2

    old_time = controller._last_adjustment_time

    await controller._reset_outcomes()

    assert len(controller._outcomes) == 0
    assert controller._consecutive_good_windows_since_last_update == 3
    assert controller._consecutive_error_windows_since_last_update == 2
    assert controller._last_adjustment_time == old_time


@pytest.mark.asyncio
async def test_clear_adjustment_state_functionality():
    """Test the _clear_adjustment_state method functionality."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Add some outcomes and state
    for _ in range(5):
        await controller.record_success()
    controller._consecutive_good_windows_since_last_update = 3
    controller._consecutive_error_windows_since_last_update = 2

    old_time = controller._last_adjustment_time

    await controller._clear_adjustment_state()

    assert len(controller._outcomes) == 5
    assert controller._consecutive_good_windows_since_last_update == 0
    assert controller._consecutive_error_windows_since_last_update == 0
    assert controller._last_adjustment_time > old_time


@pytest.mark.asyncio
async def test_thread_safety_of_outcome_tracking():
    """Test thread safety of outcome recording operations."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    import threading

    async def record_many(successes: int, errors: int):
        for _ in range(successes):
            await controller.record_success()
        for _ in range(errors):
            await controller.record_error()

    # Create multiple threads
    threads = [
        threading.Thread(target=asyncio.run, args=(record_many(10, 10),))
        for _ in range(5)
    ]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Should have recorded all outcomes without corruption
    assert len(controller._outcomes) == 100


@pytest.mark.asyncio
async def test_context_manager_basic():
    """Test basic context manager functionality."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)
    initial_concurrency = controller._current_concurrency

    # Check initial semaphore state
    initial_capacity = controller._semaphore._current_capacity
    assert initial_capacity == initial_concurrency

    async with controller:
        # Should have acquired the semaphore (capacity decreases by 1)
        assert controller._semaphore._current_capacity == initial_capacity - 1

    # Should have released the semaphore (capacity back to original)
    assert controller._semaphore._current_capacity == initial_capacity


@pytest.mark.asyncio
async def test_context_manager_with_exception():
    """Test context manager properly releases on exception."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)
    initial_concurrency = controller._current_concurrency

    # Check initial semaphore state
    initial_capacity = controller._semaphore._current_capacity
    assert initial_capacity == initial_concurrency

    with pytest.raises(ValueError):
        async with controller:
            # Verify semaphore was acquired (capacity should be decreased)
            assert controller._semaphore._current_capacity == initial_capacity - 1
            raise ValueError("Test exception")

    # Should still have released the semaphore even after exception
    assert controller._semaphore._current_capacity == initial_capacity


@pytest.mark.asyncio
async def test_outcomes_deque_max_size():
    """Test that outcomes deque respects maximum size limit."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    # Add more than max size
    for i in range(1200):
        await controller.record_success()

    # Should only keep the last 1000
    assert len(controller._outcomes) == 1000


@pytest.mark.asyncio
async def test_realistic_request_pattern():
    """Test with realistic request patterns and timing."""
    config = create_config(
        min_concurrency=5,
        max_concurrency=20,
        concurrency_step=3,
        error_threshold=0.15,
        recovery_threshold=0.05,
        min_window_size=10,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)

    # Simulate realistic usage pattern
    async def simulate_request(success_rate: float):
        async with controller:
            # Simulate request processing time
            await asyncio.sleep(0.09)
            if asyncio.get_event_loop().time() % 1.0 < success_rate:
                await controller.record_success()
            else:
                await controller.record_error()

    # Phase 1: High success rate (99%)
    tasks = [simulate_request(0.99) for _ in range(50)]
    await asyncio.gather(*tasks)

    # Should have warmed up
    assert controller._current_concurrency > config.min_concurrency
    warmup_concurrency = controller._current_concurrency

    # Phase 2: Lower success rate (1%) - should trigger backoff
    tasks = [simulate_request(0.01) for _ in range(50)]
    await asyncio.gather(*tasks)

    # Should have backed off
    assert controller._in_backoff
    assert controller._current_concurrency < warmup_concurrency


@pytest.mark.asyncio
async def test_backoff_warning_at_minimum_concurrency(mock_time):
    """Test warning is logged when backoff can't reduce concurrency further."""
    config = create_config(
        min_concurrency=5,
        max_concurrency=10,
        backoff_factor=0.5,
        error_threshold=0.2,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)

    await controller._update_concurrency(config.min_concurrency + 1)

    # Create high error rate
    for _ in range(4):
        await controller.record_success()
    await controller.record_error()  # 20% error rate

    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0

    # Mock logger to capture warnings
    with patch("oumi.inference.adaptive_concurrency_controller.logger") as mock_logger:
        await controller._try_adjust_concurrency()

        # Should be in backoff but no warning yet since concurrency changed
        assert controller._in_backoff
        mock_logger.warning.assert_not_called()

        # Submit more requests above error rate
        for _ in range(4):
            await controller.record_success()
        await controller.record_error()  # 20% error rate

        controller._last_adjustment_time = 0
        await controller._try_adjust_concurrency()

        # Should be in backoff but no warning yet we need multiple error windows
        assert controller._in_backoff
        mock_logger.warning.assert_not_called()

        # Trigger another backoff when already at minimum
        for _ in range(4):
            await controller.record_success()
        await controller.record_error()  # 20% error rate

        controller._last_adjustment_time = 0
        await controller._try_adjust_concurrency()

        # Now should log warning since we can't reduce further
        mock_logger.warning.assert_called_once()
        assert "already at minimum" in mock_logger.warning.call_args[0][0]


@pytest.mark.asyncio
async def test_warmup_warning_at_maximum_concurrency(mock_time):
    """Test warning is logged when warmup can't increase concurrency further."""
    config = create_config(
        min_concurrency=5,
        max_concurrency=5,  # Same as min, so can't increase
        concurrency_step=2,
        recovery_threshold=0.05,
        min_window_size=5,
        min_update_time=0.1,
    )
    controller = AdaptiveConcurrencyController(config)
    controller._semaphore = AsyncMock()

    # Create low error rate to trigger warmup
    for _ in range(10):
        await controller.record_success()

    mock_time.time.return_value = config.min_update_time
    controller._last_adjustment_time = 0

    # Mock logger to capture warnings
    with patch("oumi.inference.adaptive_concurrency_controller.logger") as mock_logger:
        await controller._try_adjust_concurrency()

        # Should log warning since we can't increase beyond maximum
        mock_logger.warning.assert_called_once()
        assert "already at maximum" in mock_logger.warning.call_args[0][0]


def test_consecutive_windows_constants():
    """Test that consecutive window requirements are as expected."""
    config = create_config()
    controller = AdaptiveConcurrencyController(config)

    assert controller._consecutive_good_windows_required_for_recovery == 2
    assert controller._consecutive_error_windows_for_additional_backoff == 2
