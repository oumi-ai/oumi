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
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenUsage:
    """Token usage information from an API response."""

    input_tokens: int = 0
    """Number of input tokens used."""

    output_tokens: int = 0
    """Number of output tokens used."""

    total_tokens: int = 0
    """Total number of tokens used (input + output)."""


@dataclass
class UsageRecord:
    """Record of a single request's usage."""

    timestamp: float
    """Time when the request was made."""

    input_tokens: int
    """Number of input tokens used."""

    output_tokens: int
    """Number of output tokens used."""


class RateLimiter:
    """Rate limiter for requests and tokens.

    This class implements a sliding window rate limiter that tracks:
    - Requests per minute (RPM)
    - Input tokens per minute (TPM)
    - Output tokens per minute (TPM)

    It calculates the wait time needed to stay within all configured limits.
    """

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        input_tokens_per_minute: Optional[int] = None,
        output_tokens_per_minute: Optional[int] = None,
    ):
        """Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute (None = no limit).
            input_tokens_per_minute: Maximum input tokens per minute (None = no limit).
            output_tokens_per_minute: Maximum output tokens per minute (None = no limit).
        """
        self.requests_per_minute = requests_per_minute
        self.input_tokens_per_minute = input_tokens_per_minute
        self.output_tokens_per_minute = output_tokens_per_minute

        # Use deque for efficient O(1) append/pop operations
        self._usage_history: deque[UsageRecord] = deque()
        self._lock = asyncio.Lock()

        # Window size in seconds (1 minute)
        self._window_size = 60.0

    def _remove_expired_records(self, current_time: float) -> None:
        """Remove records outside the sliding window.

        Args:
            current_time: Current timestamp.
        """
        cutoff_time = current_time - self._window_size
        while self._usage_history and self._usage_history[0].timestamp < cutoff_time:
            self._usage_history.popleft()

    def _get_current_usage(self, current_time: float) -> tuple[int, int, int]:
        """Get current usage within the sliding window.

        Args:
            current_time: Current timestamp.

        Returns:
            Tuple of (requests, input_tokens, output_tokens) within the window.
        """
        self._remove_expired_records(current_time)

        if not self._usage_history:
            return 0, 0, 0

        requests = len(self._usage_history)
        input_tokens = sum(record.input_tokens for record in self._usage_history)
        output_tokens = sum(record.output_tokens for record in self._usage_history)

        return requests, input_tokens, output_tokens

    def _calculate_wait_time(
        self,
        current_time: float,
        estimated_input_tokens: int = 0,
    ) -> float:
        """Calculate how long to wait before making a request.

        Args:
            current_time: Current timestamp.
            estimated_input_tokens: Estimated input tokens for the next request.

        Returns:
            Time in seconds to wait (0 if no wait needed).
        """
        requests, input_tokens, output_tokens = self._get_current_usage(current_time)

        max_wait_time = 0.0

        # Checking request limit
        if self.requests_per_minute is not None:
            # If we are at or over the limit, find when the oldest request expires
            if requests >= self.requests_per_minute and self._usage_history:
                oldest_request_time = self._usage_history[0].timestamp
                wait_until = oldest_request_time + self._window_size
                max_wait_time = max(max_wait_time, wait_until - current_time)

        # Checking input token limit
        if (
            self.input_tokens_per_minute is not None
            and estimated_input_tokens > 0
        ):
            projected_input_tokens = input_tokens + estimated_input_tokens
            if projected_input_tokens > self.input_tokens_per_minute:
                # Finding when enough tokens will expire to make room
                needed_tokens = projected_input_tokens - self.input_tokens_per_minute
                accumulated_tokens = 0
                for record in self._usage_history:
                    accumulated_tokens += record.input_tokens
                    if accumulated_tokens >= needed_tokens:
                        wait_until = record.timestamp + self._window_size
                        max_wait_time = max(max_wait_time, wait_until - current_time)
                        break

        # Checking output token limit
        if self.output_tokens_per_minute is not None:

            # For output tokens, we can only check if we're already over the limit
            # since we don't know the output size in advance
            if output_tokens >= self.output_tokens_per_minute and self._usage_history:
                # Find when enough tokens will expire
                accumulated_tokens = 0
                for record in self._usage_history:
                    accumulated_tokens += record.output_tokens
                    # Waiting for at least 10% of the limit to free up
                    if accumulated_tokens >= self.output_tokens_per_minute * 0.1:
                        wait_until = record.timestamp + self._window_size
                        max_wait_time = max(max_wait_time, wait_until - current_time)
                        break

        return max(0.0, max_wait_time)

    async def acquire(self, estimated_input_tokens: int = 0) -> None:
        """Acquire permission to make a request.

        This will wait if necessary to stay within rate limits.

        Args:
            estimated_input_tokens: Estimated number of input tokens for the request.
        """
        async with self._lock:
            current_time = time.time()
            wait_time = self._calculate_wait_time(current_time, estimated_input_tokens)

            if wait_time > 0:
                # Releasing the lock while sleeping
                pass

        # Sleeping outside the lock to allow other tasks to proceed
        if wait_time > 0:
            await asyncio.sleep(wait_time)

    async def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record usage after a request completes.

        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens used.
        """
        async with self._lock:
            current_time = time.time()
            self._remove_expired_records(current_time)

            record = UsageRecord(
                timestamp=current_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            self._usage_history.append(record)

    async def get_current_usage(self) -> tuple[int, int, int]:
        """Get current usage statistics.

        Returns:
            Tuple of (requests, input_tokens, output_tokens) within the window.
        """
        async with self._lock:
            current_time = time.time()
            return self._get_current_usage(current_time)

    def has_limits(self) -> bool:
        """Check if any rate limits are configured.

        Returns:
            True if any rate limits are set, False otherwise.
        """
        return (
            self.requests_per_minute is not None
            or self.input_tokens_per_minute is not None
            or self.output_tokens_per_minute is not None
        )
