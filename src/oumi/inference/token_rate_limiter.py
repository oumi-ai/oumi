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

"""Token-based rate limiter for API requests.

This module provides a sliding window rate limiter that tracks token usage
(input/output) and request counts to enforce rate limits imposed by API providers.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from oumi.utils.logging import logger


@dataclass
class TokenUsage:
    """Token usage from a single API request."""

    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: float = 0.0


class TokenRateLimiter:
    """A sliding window rate limiter for token-based and request-based limits.

    This limiter tracks token usage and request counts within a sliding time window
    (typically 60 seconds for "per minute" limits) and can calculate wait times
    when approaching limits.

    The limiter supports three types of limits:
    - Requests per minute (RPM)
    - Input tokens per minute (input TPM)
    - Output tokens per minute (output TPM)

    Example:
        ```python
        limiter = TokenRateLimiter(
            requests_per_minute=100,
            input_tokens_per_minute=100000,
            output_tokens_per_minute=50000,
        )

        # Before making a request, wait if needed
        await limiter.wait_if_needed()

        # After receiving response, record the usage
        await limiter.record_usage(input_tokens=150, output_tokens=50)
        ```
    """

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        input_tokens_per_minute: Optional[int] = None,
        output_tokens_per_minute: Optional[int] = None,
        window_seconds: float = 60.0,
    ):
        """Initialize the token rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute. None = unlimited.
            input_tokens_per_minute: Max input tokens per minute. None = unlimited.
            output_tokens_per_minute: Max output tokens per minute. None = unlimited.
            window_seconds: Time window in seconds (default 60 for per-minute limits).
        """
        self._requests_per_minute = requests_per_minute
        self._input_tokens_per_minute = input_tokens_per_minute
        self._output_tokens_per_minute = output_tokens_per_minute
        self._window_seconds = window_seconds

        # Sliding window of token usage records
        self._usage_history: deque[TokenUsage] = deque()
        self._lock = asyncio.Lock()

        # Track pending requests (requests that have been sent but not yet completed)
        self._pending_requests = 0

    def is_enabled(self) -> bool:
        """Check if any rate limiting is enabled."""
        return (
            self._requests_per_minute is not None
            or self._input_tokens_per_minute is not None
            or self._output_tokens_per_minute is not None
        )

    async def _cleanup_old_entries(self) -> None:
        """Remove entries outside the sliding window."""
        cutoff_time = time.time() - self._window_seconds
        while self._usage_history and self._usage_history[0].timestamp < cutoff_time:
            self._usage_history.popleft()

    async def _get_current_usage(self) -> tuple[int, int, int]:
        """Get current usage within the window.

        Returns:
            Tuple of (request_count, input_tokens, output_tokens).
        """
        await self._cleanup_old_entries()

        request_count = len(self._usage_history) + self._pending_requests
        input_tokens = sum(u.input_tokens for u in self._usage_history)
        output_tokens = sum(u.output_tokens for u in self._usage_history)

        return request_count, input_tokens, output_tokens

    async def _calculate_wait_time(self) -> float:
        """Calculate how long to wait before the next request.

        Returns:
            Wait time in seconds. 0 if no wait needed.
        """
        if not self.is_enabled():
            return 0.0

        request_count, input_tokens, output_tokens = await self._get_current_usage()
        current_time = time.time()
        wait_time = 0.0

        # Check request limit
        if (
            self._requests_per_minute is not None
            and request_count >= self._requests_per_minute
        ):
            # Find when the oldest request will expire from the window
            if self._usage_history:
                oldest_time = self._usage_history[0].timestamp
                wait_time = max(
                    wait_time, oldest_time + self._window_seconds - current_time
                )
            else:
                # All requests are pending (no completed history yet)
                # Wait for a small interval to allow pending requests to complete
                wait_time = max(wait_time, 0.1)

        # Check input token limit
        if (
            self._input_tokens_per_minute is not None
            and input_tokens >= self._input_tokens_per_minute
        ):
            if self._usage_history:
                oldest_time = self._usage_history[0].timestamp
                wait_time = max(
                    wait_time, oldest_time + self._window_seconds - current_time
                )
            else:
                # All requests are pending, wait briefly
                wait_time = max(wait_time, 0.1)

        # Check output token limit
        if (
            self._output_tokens_per_minute is not None
            and output_tokens >= self._output_tokens_per_minute
        ):
            if self._usage_history:
                oldest_time = self._usage_history[0].timestamp
                wait_time = max(
                    wait_time, oldest_time + self._window_seconds - current_time
                )
            else:
                # All requests are pending, wait briefly
                wait_time = max(wait_time, 0.1)

        return max(0.0, wait_time)

    async def wait_if_needed(self) -> float:
        """Wait if rate limits are being approached.

        Returns:
            The actual time waited in seconds.
        """
        async with self._lock:
            wait_time = await self._calculate_wait_time()
            if wait_time > 0:
                logger.debug(
                    f"Rate limit approaching, waiting {wait_time:.2f}s before request"
                )
            self._pending_requests += 1

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        return wait_time

    async def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record token usage from a completed request.

        Args:
            input_tokens: Number of input/prompt tokens used.
            output_tokens: Number of output/completion tokens used.
        """
        async with self._lock:
            self._pending_requests = max(0, self._pending_requests - 1)
            self._usage_history.append(
                TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    timestamp=time.time(),
                )
            )
            await self._cleanup_old_entries()

    async def record_request_without_tokens(self) -> None:
        """Record a request when token counts are not available.

        This is useful when the API doesn't return token counts but we still
        want to track request counts.
        """
        await self.record_usage(input_tokens=0, output_tokens=0)

    async def get_usage_summary(self) -> dict:
        """Get a summary of current usage within the window.

        Returns:
            Dictionary with request_count, input_tokens, output_tokens, and limits.
        """
        async with self._lock:
            request_count, input_tokens, output_tokens = await self._get_current_usage()
            return {
                "request_count": request_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "requests_per_minute_limit": self._requests_per_minute,
                "input_tokens_per_minute_limit": self._input_tokens_per_minute,
                "output_tokens_per_minute_limit": self._output_tokens_per_minute,
            }
