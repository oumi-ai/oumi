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
