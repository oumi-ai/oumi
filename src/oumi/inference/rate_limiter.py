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


@dataclass
class RequestRecord:
    """A single request's token usage recorded in the sliding window."""

    timestamp: float
    """Unix timestamp when this request was recorded."""

    input_tokens: int
    """Number of input (prompt) tokens consumed by this request."""

    output_tokens: int
    """Number of output (completion) tokens consumed by this request."""


@dataclass
class UsageStats:
    """Aggregated usage statistics over the current sliding window."""

    requests: int
    """Number of requests made within the sliding window."""

    input_tokens: int
    """Total input tokens consumed within the sliding window."""

    output_tokens: int
    """Total output tokens consumed within the sliding window."""


class UsageTracker:
    """Tracks API usage within a sliding time window.

    Maintains a deque of RequestRecord entries and purges records that have expired
    from the current window. Methods are thread safe using asyncio lock.
    """

    def __init__(self, window_size_seconds: float = 60.0):
        """Initialize the usage tracker.

        Args:
            window_size_seconds: Width of the sliding window in seconds.
            (Default is 60s per OpenAI and Anthropic)
        """
        self._window_size_seconds = window_size_seconds
        self._history: deque[RequestRecord] = deque()
        self._lock = asyncio.Lock()

    async def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record usage for a completed request.

        Args:
            input_tokens: Number of input (prompt) tokens used.
            output_tokens: Number of output (completion) tokens used.
        """
        async with self._lock:
            now = time.time()
            self._purge_expired(now)
            self._history.append(
                RequestRecord(
                    timestamp=now,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )

    # Unused but kept for tracking/debugging
    async def get_stats(self) -> UsageStats:
        """Return aggregated usage stats for the current sliding window.

        Returns:
            UsageStats with the number of requests, total input tokens, and
            total output tokens
        """
        async with self._lock:
            now = time.time()
            self._purge_expired(now)
            return UsageStats(
                requests=len(self._history),
                input_tokens=sum(r.input_tokens for r in self._history),
                output_tokens=sum(r.output_tokens for r in self._history),
            )

    def _purge_expired(self, now: float) -> None:
        """Remove records that have aged out of the sliding window.

        To be called while holding self._lock to prevent race conditions.

        Args:
            now: Current timestamp from time.
        """
        cutoff = now - self._window_size_seconds
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()


class RateLimiter:
    """Sliding window rate limiter for requests per minute and tokens per minute.

    Enforces up to three independent limits simultaneously:
    - Requests per minute (RPM)
    - Input tokens per minute (input TPM)
    - Output tokens per minute (output TPM)

    Before each request, call wait_if_needed() to sleep until all active
    limits have bandwidth.

    Token counts are consumed from the normalized output of
    RemoteInferenceEngine._extract_usage_from_response(), which maps to this
    format: prompt_tokens / completion_tokens:

    """

    def __init__(
        self,
        tracker: UsageTracker,
        requests_per_minute: int | None = None,
        input_tokens_per_minute: int | None = None,
        output_tokens_per_minute: int | None = None,
    ):
        """Initialize the rate limiter.

        Args:
            tracker: The UsageTracker instance that creates the sliding window.
            requests_per_minute: Maximum API calls allowed per minute.
            input_tokens_per_minute: Maximum input (prompt) tokens per minute.
            output_tokens_per_minute: Maximum output (completion) tokens per minute.
        """
        self._tracker = tracker
        self._requests_per_minute = requests_per_minute
        self._input_tokens_per_minute = input_tokens_per_minute
        self._output_tokens_per_minute = output_tokens_per_minute
        # Queue waiters to prevent synchronized wakeups in case of concurrent requests
        self._wait_queue_lock = asyncio.Lock()

    def has_limits(self) -> bool:
        """Return True if any of the rate limits are configured."""
        return any(
            [
                self._requests_per_minute is not None,
                self._input_tokens_per_minute is not None,
                self._output_tokens_per_minute is not None,
            ]
        )

    async def wait_if_needed(self) -> None:
        """Sleep until all configured rate limits have bandwidth.

        Computes the minimum wait time required to bring all active limits
        below their thresholds, then sleep for that duration. Waiters are
        serialized to prevent synchronized wakeups when many coroutines
        spawn on the limiter simultaneously. No action when no limits are
        configured or when bandwidth is available.
        """
        if not self.has_limits():
            return

        async with self._wait_queue_lock:  # Queue waiters
            while True:
                async with (
                    self._tracker._lock
                ):  # Hold the lock to purge expired request
                    now = time.time()
                    self._tracker._purge_expired(now)
                    wait_seconds = self._compute_wait_seconds(
                        self._tracker._history, now
                    )

                if wait_seconds <= 0:  # No need to wait, all limits have bandwidth
                    return
                await asyncio.sleep(wait_seconds)

    async def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from a completed API request.

        Should be called immediately after the API response is received and
        its usage field is extracted, to count tokens even if later conversion
        subsequently fails.

        Args:
            input_tokens: Prompt tokens consumed (prompt_tokens from the
                normalized usage dict).
            output_tokens: Completion tokens generated (completion_tokens from the
                from the normalized usage dict).
        """
        await self._tracker.record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _wait_seconds_for_rpm(
        self, history: deque[RequestRecord], now: float, window: float
    ) -> float:
        """Return the seconds to wait due to the RPM limit, or 0.0 if None."""
        if self._requests_per_minute is None:
            return 0.0
        if len(history) >= self._requests_per_minute and history:
            # wait till the oldest request expires (history[0] + window)
            return window - (now - history[0].timestamp)
        return 0.0

    def _wait_seconds_for_tpm(
        self,
        history: deque[RequestRecord],
        now: float,
        window: float,
        limit: int | None,
        token_field: str,
    ) -> float:
        """Return the seconds to wait due to a token per minute limit, or 0.0 if None.

        Traverses the history from oldest to newest, accumulating tokens, and
        returns the time until the record that would free enough bandwidth exits
        the sliding window.

        Args:
            history: Already-purged deque of RequestRecord entries.
            now: Current timestamp.
            window: Window size in seconds.
            limit: The token per minute limit. Returns 0.0 if None.
            token_field: Attribute name on RequestRecord to read ("input_tokens" or
            "output_tokens").
        """
        if limit is None or not history:
            return 0.0
        total = sum(getattr(r, token_field) for r in history)
        if total < limit:  # No need to wait
            return 0.0
        accumulated = 0
        for record in history:
            accumulated += getattr(record, token_field)
            if (
                total - accumulated
            ) < limit:  # Accumulated more than the limit, so wait
                return window - (now - record.timestamp)
        return 0.0

    def _compute_wait_seconds(self, history: deque[RequestRecord], now: float) -> float:
        """Compute the number of seconds to wait before the next request.

        Checks each configured limit independently and returns the maximum
        required sleep across all of them.

        Args:
            history: The current (already purged) deque of RequestRecord entries.
            now: Current timestamp from time.time().

        Returns:
            Seconds to sleep. Always >= 0.0.
            (Returns 0 if no sleep is required.)
        """
        window = self._tracker._window_size_seconds
        wait_rpm = self._wait_seconds_for_rpm(history, now, window)
        wait_itpm = self._wait_seconds_for_tpm(
            history, now, window, self._input_tokens_per_minute, "input_tokens"
        )
        wait_otpm = self._wait_seconds_for_tpm(
            history, now, window, self._output_tokens_per_minute, "output_tokens"
        )
        # Return the maximum of all the three wait times
        return max(0.0, wait_rpm, wait_itpm, wait_otpm)
