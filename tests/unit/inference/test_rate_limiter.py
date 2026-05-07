from collections import deque
from unittest.mock import patch

import pytest

from oumi.inference.rate_limiter import RateLimiter, RequestRecord, UsageTracker

_WINDOW = 60.0


@pytest.fixture
def mock_time():
    with patch("oumi.inference.rate_limiter.time") as time_mock:
        yield time_mock


@pytest.fixture
def mock_asyncio_sleep():
    with patch("oumi.inference.rate_limiter.asyncio.sleep") as sleep_mock:
        yield sleep_mock


# Helpers
def _record(
    timestamp: float, input_tokens: int = 0, output_tokens: int = 0
) -> RequestRecord:
    return RequestRecord(
        timestamp=timestamp, input_tokens=input_tokens, output_tokens=output_tokens
    )


def _history(*records: RequestRecord) -> deque[RequestRecord]:
    return deque(records)


# UsageTracker
@pytest.mark.asyncio
async def test_usage_tracker_record_and_purge(mock_time):
    mock_time.time.return_value = 1000.0
    tracker = UsageTracker(window_size_seconds=_WINDOW)

    await tracker.record(input_tokens=50, output_tokens=20)
    assert len(tracker._history) == 1
    assert tracker._history[0].input_tokens == 50
    assert tracker._history[0].output_tokens == 20
    assert tracker._history[0].timestamp == pytest.approx(1000.0)

    # Inject an expired entry
    tracker._history.appendleft(
        _record(timestamp=930.0, input_tokens=99, output_tokens=99)
    )
    assert len(tracker._history) == 2

    await tracker.record(input_tokens=10, output_tokens=5)
    # Expired entry purged, two new entries at ts=1000.0 remain
    assert len(tracker._history) == 2
    assert all(r.input_tokens != 99 for r in tracker._history)


# RateLimiter.has_limits
def test_has_limits():
    tracker = UsageTracker()
    assert not RateLimiter(tracker).has_limits()
    assert RateLimiter(tracker, requests_per_minute=10).has_limits()
    assert RateLimiter(tracker, input_tokens_per_minute=1000).has_limits()
    assert RateLimiter(tracker, output_tokens_per_minute=500).has_limits()


# RateLimiter.wait_if_needed: no limits
@pytest.mark.asyncio
async def test_wait_if_needed_returns_immediately_when_no_limits(mock_asyncio_sleep):
    limiter = RateLimiter(UsageTracker())
    await limiter.wait_if_needed()
    mock_asyncio_sleep.assert_not_called()


# RateLimiter.wait_if_needed: below RPM limit
@pytest.mark.asyncio
async def test_wait_if_needed_returns_immediately_below_rpm(
    mock_time, mock_asyncio_sleep
):
    now = 1000.0
    mock_time.time.return_value = now
    tracker = UsageTracker(window_size_seconds=_WINDOW)
    tracker._history = _history(
        _record(now - 20),
        _record(now - 10),
        _record(now - 5),
    )
    limiter = RateLimiter(tracker, requests_per_minute=5)
    await limiter.wait_if_needed()
    mock_asyncio_sleep.assert_not_called()


# RateLimiter._wait_seconds_for_rpm: pivot test
def test_wait_seconds_for_rpm_pivot():
    """Verify the correct pivot entry is used when history exceeds the RPM cap.

    At-limit: pivot is history[0], so only one entry must expire.
    Over-limit: pivot is history[len-RPM], not history[0], so the correct pivot is used.
    """
    tracker = UsageTracker(window_size_seconds=_WINDOW)
    limiter = RateLimiter(tracker, requests_per_minute=5)
    now = 1000.0

    # At-limit (len = RPM=5): pivot = history[0]
    at_limit = _history(
        _record(now - 30),
        _record(now - 20),
        _record(now - 15),
        _record(now - 10),
        _record(now - 5),
    )
    assert limiter._wait_seconds_for_rpm(at_limit, now, _WINDOW) == pytest.approx(30.0)

    # Over-limit (len=10, RPM=5): pivot = history[5]
    over_limit = _history(
        _record(now - 58),
        _record(now - 55),
        _record(now - 50),
        _record(now - 45),
        _record(now - 40),
        _record(now - 35),
        _record(now - 30),
        _record(now - 25),
        _record(now - 20),
        _record(now - 15),
    )
    assert limiter._wait_seconds_for_rpm(over_limit, now, _WINDOW) == pytest.approx(
        25.0
    )


# RateLimiter._wait_seconds_for_tpm
def test_wait_seconds_for_tpm():
    tracker = UsageTracker(window_size_seconds=_WINDOW)
    limiter = RateLimiter(tracker, input_tokens_per_minute=100)
    now = 1000.0

    # Below limit: returns 0.0
    below = _history(
        _record(now - 30, input_tokens=40),
        _record(now - 10, input_tokens=30),
    )
    wait = limiter._wait_seconds_for_tpm(below, now, _WINDOW, 100, "input_tokens")
    assert wait == pytest.approx(0.0, abs=1e-9)

    # Over limit: waits 30s
    over = _history(
        _record(now - 30, input_tokens=40),
        _record(now - 10, input_tokens=80),
    )
    assert limiter._wait_seconds_for_tpm(
        over, now, _WINDOW, 100, "input_tokens"
    ) == pytest.approx(30.0)


# RateLimiter._compute_wait_seconds: max of all limits
def test_compute_wait_seconds_returns_max():
    now = 1000.0
    tracker = UsageTracker(window_size_seconds=_WINDOW)
    limiter = RateLimiter(
        tracker,
        requests_per_minute=2,
        input_tokens_per_minute=50,
        output_tokens_per_minute=30,
    )
    history = _history(
        _record(now - 50, input_tokens=5, output_tokens=5),
        _record(now - 40, input_tokens=5, output_tokens=5),
        _record(now - 10, input_tokens=60, output_tokens=40),
    )
    tracker._history = history
    assert limiter._compute_wait_seconds(history, now) == pytest.approx(50.0)


# RateLimiter.wait_if_needed: sleeps for the correct duration
@pytest.mark.asyncio
async def test_wait_if_needed_sleeps_correct_duration(mock_time, mock_asyncio_sleep):
    t_start = 1000.0
    mock_time.time.side_effect = [t_start, t_start + 61]

    tracker = UsageTracker(window_size_seconds=_WINDOW)
    tracker._history = _history(_record(t_start - 30))
    limiter = RateLimiter(tracker, requests_per_minute=1)

    await limiter.wait_if_needed()
    mock_asyncio_sleep.assert_called_once_with(pytest.approx(30.0))
