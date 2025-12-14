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
from unittest.mock import patch

import pytest

from oumi.inference.rate_limiter import RateLimiter, TokenUsage


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_token_usage_creation(self):
        """Test creating a TokenUsage object."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_defaults(self):
        """Test TokenUsage with default values."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_no_limits(self):
        """Test rate limiter with no limits configured."""
        limiter = RateLimiter()
        assert not limiter.has_limits()

        # Should not wait when no limits are set
        start_time = time.time()
        await limiter.acquire()
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_requests_per_minute_limit(self):
        """Test request rate limiting."""
        # Allow 10 requests per minute
        limiter = RateLimiter(requests_per_minute=10)
        assert limiter.has_limits()

        # Make 10 requests - should all go through without waiting
        for _ in range(10):
            await limiter.acquire()
            await limiter.record_usage(input_tokens=0, output_tokens=0)

        # 11th request should wait
        start_time = time.time()

        # Mock time to simulate waiting
        with patch('time.time') as mock_time:
            # Set initial time
            base_time = 1000.0
            call_count = [0]

            def time_side_effect():
                call_count[0] += 1
                # For the wait calculation, return time after 6 seconds
                if call_count[0] > 20:
                    return base_time + 6.1
                return base_time

            mock_time.side_effect = time_side_effect

            # This should calculate a wait time
            limiter_with_history = RateLimiter(requests_per_minute=10)
            for _ in range(10):
                await limiter_with_history.record_usage(input_tokens=0, output_tokens=0)

            # Check current usage
            requests, _, _ = await limiter_with_history.get_current_usage()
            assert requests == 10

    @pytest.mark.asyncio
    async def test_input_tokens_per_minute_limit(self):
        """Test input token rate limiting."""
        # Allow 1000 input tokens per minute
        limiter = RateLimiter(input_tokens_per_minute=1000)
        assert limiter.has_limits()

        # Use 900 tokens
        await limiter.record_usage(input_tokens=900, output_tokens=0)

        # Requesting 50 tokens should go through
        await limiter.acquire(estimated_input_tokens=50)
        await limiter.record_usage(input_tokens=50, output_tokens=0)

        # Check usage
        requests, input_tokens, output_tokens = await limiter.get_current_usage()
        assert requests == 2
        assert input_tokens == 950
        assert output_tokens == 0

    @pytest.mark.asyncio
    async def test_output_tokens_per_minute_limit(self):
        """Test output token rate limiting."""
        # Allow 500 output tokens per minute
        limiter = RateLimiter(output_tokens_per_minute=500)
        assert limiter.has_limits()

        # Use 400 tokens
        await limiter.record_usage(input_tokens=0, output_tokens=400)

        # Check usage
        requests, input_tokens, output_tokens = await limiter.get_current_usage()
        assert requests == 1
        assert input_tokens == 0
        assert output_tokens == 400

    @pytest.mark.asyncio
    async def test_combined_limits(self):
        """Test rate limiter with multiple limits configured."""
        limiter = RateLimiter(
            requests_per_minute=10,
            input_tokens_per_minute=1000,
            output_tokens_per_minute=500,
        )
        assert limiter.has_limits()

        # Make a few requests
        for _ in range(5):
            await limiter.acquire(estimated_input_tokens=100)
            await limiter.record_usage(input_tokens=100, output_tokens=50)

        # Check usage
        requests, input_tokens, output_tokens = await limiter.get_current_usage()
        assert requests == 5
        assert input_tokens == 500
        assert output_tokens == 250

    @pytest.mark.asyncio
    async def test_sliding_window(self):
        """Test that old records are removed from the sliding window."""
        limiter = RateLimiter(requests_per_minute=10)

        # Mock time to simulate passage of time
        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time

            # Record usage at base_time
            await limiter.record_usage(input_tokens=100, output_tokens=50)

            # Check usage immediately
            requests, input_tokens, _ = await limiter.get_current_usage()
            assert requests == 1
            assert input_tokens == 100

            # Move time forward by 61 seconds (past the 60-second window)
            mock_time.return_value = base_time + 61

            # Check usage again - should be 0 as the record expired
            requests, input_tokens, _ = await limiter.get_current_usage()
            assert requests == 0
            assert input_tokens == 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test rate limiter with concurrent requests."""
        limiter = RateLimiter(requests_per_minute=10)

        # Create multiple concurrent tasks
        async def make_request():
            await limiter.acquire()
            await limiter.record_usage(input_tokens=10, output_tokens=5)

        # Run 5 concurrent requests
        await asyncio.gather(*[make_request() for _ in range(5)])

        # Check usage
        requests, input_tokens, output_tokens = await limiter.get_current_usage()
        assert requests == 5
        assert input_tokens == 50
        assert output_tokens == 25

    @pytest.mark.asyncio
    async def test_record_usage_without_acquire(self):
        """Test recording usage without acquiring (for error cases)."""
        limiter = RateLimiter(requests_per_minute=10)

        # Record usage directly
        await limiter.record_usage(input_tokens=100, output_tokens=50)

        # Check usage
        requests, input_tokens, output_tokens = await limiter.get_current_usage()
        assert requests == 1
        assert input_tokens == 100
        assert output_tokens == 50
