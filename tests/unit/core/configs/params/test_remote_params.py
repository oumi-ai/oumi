import pytest

from oumi.core.configs.params.remote_params import RemoteParams


def test_remote_params_allows_empty():
    params = RemoteParams()
    params.finalize_and_validate()
    # No exception should be raised


def test_remote_params_validates_backoff_base():
    """Test that retry_backoff_base must be positive."""
    with pytest.raises(ValueError, match="Retry backoff base must be greater than 0"):
        params = RemoteParams(retry_backoff_base=0)
        params.finalize_and_validate()

    with pytest.raises(ValueError, match="Retry backoff base must be greater than 0"):
        params = RemoteParams(retry_backoff_base=-1)
        params.finalize_and_validate()


def test_remote_params_validates_backoff_max():
    """Test that retry_backoff_max is be greater than or equal to retry_backoff_base."""
    with pytest.raises(
        ValueError,
        match="Retry backoff max must be greater than or equal to retry backoff base",
    ):
        params = RemoteParams(retry_backoff_base=2, retry_backoff_max=1)
        params.finalize_and_validate()


def test_remote_params_accepts_valid_backoff():
    """Test that valid backoff parameters are accepted."""
    params = RemoteParams(retry_backoff_base=1, retry_backoff_max=30)
    params.finalize_and_validate()
    # No exception should be raised

    params = RemoteParams(retry_backoff_base=0.5, retry_backoff_max=0.5)
    params.finalize_and_validate()
    # No exception should be raised - equal values are allowed


def test_remote_params_validates_requests_per_minute():
    """Test that requests_per_minute must be at least 1 if set."""
    with pytest.raises(
        ValueError, match="Requests per minute must be greater than or equal to 1"
    ):
        params = RemoteParams(requests_per_minute=0)
        params.finalize_and_validate()

    with pytest.raises(
        ValueError, match="Requests per minute must be greater than or equal to 1"
    ):
        params = RemoteParams(requests_per_minute=-1)
        params.finalize_and_validate()


def test_remote_params_validates_input_tokens_per_minute():
    """Test that input_tokens_per_minute must be at least 1 if set."""
    with pytest.raises(
        ValueError, match="Input tokens per minute must be greater than or equal to 1"
    ):
        params = RemoteParams(input_tokens_per_minute=0)
        params.finalize_and_validate()

    with pytest.raises(
        ValueError, match="Input tokens per minute must be greater than or equal to 1"
    ):
        params = RemoteParams(input_tokens_per_minute=-1)
        params.finalize_and_validate()


def test_remote_params_validates_output_tokens_per_minute():
    """Test that output_tokens_per_minute must be at least 1 if set."""
    with pytest.raises(
        ValueError, match="Output tokens per minute must be greater than or equal to 1"
    ):
        params = RemoteParams(output_tokens_per_minute=0)
        params.finalize_and_validate()

    with pytest.raises(
        ValueError, match="Output tokens per minute must be greater than or equal to 1"
    ):
        params = RemoteParams(output_tokens_per_minute=-1)
        params.finalize_and_validate()


def test_remote_params_accepts_valid_rate_limits():
    """Test that valid rate limit parameters are accepted."""
    params = RemoteParams(
        requests_per_minute=100,
        input_tokens_per_minute=100000,
        output_tokens_per_minute=50000,
    )
    params.finalize_and_validate()
    # No exception should be raised


def test_remote_params_accepts_none_rate_limits():
    """Test that None rate limit parameters (unlimited) are accepted."""
    params = RemoteParams(
        requests_per_minute=None,
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
    )
    params.finalize_and_validate()
    # No exception should be raised
