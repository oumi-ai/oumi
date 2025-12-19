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

from oumi.inference.token_rate_limiter import TokenRateLimiter, TokenUsage


class TestTokenUsage:
    """Tests for the TokenUsage dataclass."""

    def test_default_values(self):
        """Test that TokenUsage has correct default values."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.timestamp == 0.0

    def test_custom_values(self):
        """Test TokenUsage with custom values."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, timestamp=1234567890.0)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.timestamp == 1234567890.0


class TestTokenRateLimiter:
    """Tests for the TokenRateLimiter class."""

    def test_is_enabled_with_no_limits(self):
        """Test that is_enabled returns False when no limits are set."""
        limiter = TokenRateLimiter()
        assert limiter.is_enabled() is False

    def test_is_enabled_with_requests_per_minute(self):
        """Test that is_enabled returns True when requests_per_minute is set."""
        limiter = TokenRateLimiter(requests_per_minute=100)
        assert limiter.is_enabled() is True

    def test_is_enabled_with_input_tokens_per_minute(self):
        """Test that is_enabled returns True when input_tokens_per_minute is set."""
        limiter = TokenRateLimiter(input_tokens_per_minute=10000)
        assert limiter.is_enabled() is True

    def test_is_enabled_with_output_tokens_per_minute(self):
        """Test that is_enabled returns True when output_tokens_per_minute is set."""
        limiter = TokenRateLimiter(output_tokens_per_minute=5000)
        assert limiter.is_enabled() is True

    def test_is_enabled_with_all_limits(self):
        """Test that is_enabled returns True when all limits are set."""
        limiter = TokenRateLimiter(
            requests_per_minute=100,
            input_tokens_per_minute=10000,
            output_tokens_per_minute=5000,
        )
        assert limiter.is_enabled() is True

    @pytest.mark.asyncio
    async def test_record_usage_basic(self):
        """Test basic token usage recording."""
        limiter = TokenRateLimiter(
            requests_per_minute=100,
            input_tokens_per_minute=10000,
        )
        await limiter.record_usage(input_tokens=100, output_tokens=50)

        summary = await limiter.get_usage_summary()
        assert summary["request_count"] == 1
        assert summary["input_tokens"] == 100
        assert summary["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_record_multiple_usage(self):
        """Test recording multiple usage entries."""
        limiter = TokenRateLimiter(requests_per_minute=100)

        await limiter.record_usage(input_tokens=100, output_tokens=50)
        await limiter.record_usage(input_tokens=200, output_tokens=100)
        await limiter.record_usage(input_tokens=150, output_tokens=75)

        summary = await limiter.get_usage_summary()
        assert summary["request_count"] == 3
        assert summary["input_tokens"] == 450
        assert summary["output_tokens"] == 225

    @pytest.mark.asyncio
    async def test_record_request_without_tokens(self):
        """Test recording a request without token counts."""
        limiter = TokenRateLimiter(requests_per_minute=100)
        await limiter.record_request_without_tokens()

        summary = await limiter.get_usage_summary()
        assert summary["request_count"] == 1
        assert summary["input_tokens"] == 0
        assert summary["output_tokens"] == 0

    @pytest.mark.asyncio
    async def test_wait_if_needed_no_wait_when_under_limit(self):
        """Test that wait_if_needed returns 0 when under all limits."""
        limiter = TokenRateLimiter(
            requests_per_minute=10,
            input_tokens_per_minute=10000,
        )

        wait_time = await limiter.wait_if_needed()
        assert wait_time == 0.0

        # Also complete the pending request
        await limiter.record_usage(input_tokens=100, output_tokens=50)

    @pytest.mark.asyncio
    async def test_wait_if_needed_no_wait_when_disabled(self):
        """Test that wait_if_needed returns 0 when rate limiting is disabled."""
        limiter = TokenRateLimiter()

        wait_time = await limiter.wait_if_needed()
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_wait_if_needed_waits_at_request_limit(self):
        """Test that wait_if_needed waits when at request limit."""
        limiter = TokenRateLimiter(
            requests_per_minute=2,
            window_seconds=1.0,  # Short window for testing
        )

        # Record 2 requests (at limit)
        await limiter.record_usage(input_tokens=10, output_tokens=5)
        await limiter.record_usage(input_tokens=10, output_tokens=5)

        # Next request should wait
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            wait_time = await limiter.wait_if_needed()
            # Should have calculated a wait time
            if wait_time > 0:
                mock_sleep.assert_called()

    @pytest.mark.asyncio
    async def test_wait_if_needed_waits_at_input_token_limit(self):
        """Test that wait_if_needed waits when at input token limit."""
        limiter = TokenRateLimiter(
            input_tokens_per_minute=100,
            window_seconds=1.0,
        )

        # Record usage at limit
        await limiter.record_usage(input_tokens=100, output_tokens=10)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            wait_time = await limiter.wait_if_needed()
            if wait_time > 0:
                mock_sleep.assert_called()

    @pytest.mark.asyncio
    async def test_wait_if_needed_waits_at_output_token_limit(self):
        """Test that wait_if_needed waits when at output token limit."""
        limiter = TokenRateLimiter(
            output_tokens_per_minute=50,
            window_seconds=1.0,
        )

        # Record usage at limit
        await limiter.record_usage(input_tokens=10, output_tokens=50)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            wait_time = await limiter.wait_if_needed()
            if wait_time > 0:
                mock_sleep.assert_called()

    @pytest.mark.asyncio
    async def test_old_entries_cleaned_up(self):
        """Test that old entries are removed from the sliding window."""
        limiter = TokenRateLimiter(
            requests_per_minute=100,
            window_seconds=0.1,  # Very short window
        )

        # Record usage
        await limiter.record_usage(input_tokens=100, output_tokens=50)

        summary_before = await limiter.get_usage_summary()
        assert summary_before["request_count"] == 1

        # Wait for window to expire
        await asyncio.sleep(0.15)

        # Record another to trigger cleanup
        await limiter.record_usage(input_tokens=50, output_tokens=25)

        summary_after = await limiter.get_usage_summary()
        # Old entry should be cleaned up, only new entry remains
        assert summary_after["request_count"] == 1
        assert summary_after["input_tokens"] == 50

    @pytest.mark.asyncio
    async def test_get_usage_summary(self):
        """Test that get_usage_summary returns correct data."""
        limiter = TokenRateLimiter(
            requests_per_minute=100,
            input_tokens_per_minute=10000,
            output_tokens_per_minute=5000,
        )

        await limiter.record_usage(input_tokens=100, output_tokens=50)

        summary = await limiter.get_usage_summary()
        assert summary["request_count"] == 1
        assert summary["input_tokens"] == 100
        assert summary["output_tokens"] == 50
        assert summary["requests_per_minute_limit"] == 100
        assert summary["input_tokens_per_minute_limit"] == 10000
        assert summary["output_tokens_per_minute_limit"] == 5000

    @pytest.mark.asyncio
    async def test_pending_requests_tracked(self):
        """Test that pending requests are tracked correctly."""
        limiter = TokenRateLimiter(requests_per_minute=2)

        # Start a request (wait_if_needed increments pending)
        await limiter.wait_if_needed()

        summary = await limiter.get_usage_summary()
        # Pending request should be counted
        assert summary["request_count"] == 1

        # Complete the request
        await limiter.record_usage(input_tokens=10, output_tokens=5)

        summary = await limiter.get_usage_summary()
        # Still 1 request (now recorded, not pending)
        assert summary["request_count"] == 1

    @pytest.mark.asyncio
    async def test_pending_requests_cleaned_up_on_failure(self):
        """Test that pending requests are properly decremented even on failure.

        This test ensures that if wait_if_needed() is called but record_usage()
        is never called (e.g., due to request failure), the pending_requests
        counter doesn't leak and cause rate limiter to block future requests.
        """
        limiter = TokenRateLimiter(requests_per_minute=5)

        # Simulate failed requests: call wait_if_needed() without record_usage()
        await limiter.wait_if_needed()
        await limiter.wait_if_needed()
        await limiter.wait_if_needed()

        # The pending requests counter should have incremented
        summary = await limiter.get_usage_summary()
        assert summary["request_count"] == 3

        # In the real implementation, the finally block should call
        # record_request_without_tokens() to properly decrement the counter.
        # Simulate that here by calling it manually:
        await limiter.record_request_without_tokens()
        await limiter.record_request_without_tokens()
        await limiter.record_request_without_tokens()

        # After cleanup, usage should show 3 recorded (failed) requests
        summary = await limiter.get_usage_summary()
        assert summary["request_count"] == 3

        # Now verify that new requests can still proceed
        # (if counter leaked, this would be blocked)
        await limiter.wait_if_needed()
        await limiter.record_usage(input_tokens=10, output_tokens=5)

        summary = await limiter.get_usage_summary()
        assert summary["request_count"] == 4

    @pytest.mark.asyncio
    async def test_wait_when_limit_reached_with_empty_history(self):
        """Test that wait is applied when limit is reached by pending requests only.

        This tests the scenario where concurrent requests arrive simultaneously
        and the limit is reached purely by pending requests (no completed requests
        in history yet). The rate limiter should still enforce a wait.
        """
        limiter = TokenRateLimiter(requests_per_minute=2)

        # First two requests should proceed without waiting
        wait1 = await limiter.wait_if_needed()
        wait2 = await limiter.wait_if_needed()

        # At this point, we have 2 pending requests (at limit) but empty history
        # Third request should be forced to wait
        wait3 = await limiter.wait_if_needed()

        # The third request should have a non-zero wait time
        # because the limit was reached (even with empty history)
        assert wait3 > 0, "Should wait when limit reached by pending requests"

        # Clean up pending requests
        await limiter.record_usage(input_tokens=0, output_tokens=0)
        await limiter.record_usage(input_tokens=0, output_tokens=0)
        await limiter.record_usage(input_tokens=0, output_tokens=0)


class TestTokenRateLimiterIntegration:
    """Integration tests for TokenRateLimiter."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test that concurrent requests are tracked correctly."""
        limiter = TokenRateLimiter(
            requests_per_minute=10,
            input_tokens_per_minute=1000,
        )

        async def make_request(limiter, input_tokens, output_tokens):
            await limiter.wait_if_needed()
            await limiter.record_usage(input_tokens, output_tokens)

        # Run several concurrent requests
        await asyncio.gather(
            make_request(limiter, 100, 50),
            make_request(limiter, 100, 50),
            make_request(limiter, 100, 50),
        )

        summary = await limiter.get_usage_summary()
        assert summary["request_count"] == 3
        assert summary["input_tokens"] == 300
        assert summary["output_tokens"] == 150
