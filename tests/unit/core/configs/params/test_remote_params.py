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


def test_remote_params_validates_num_workers():
    """Test that num_workers must be non-negative."""
    with pytest.raises(
        ValueError, match="Number of num_workers must be greater than or equal to 0"
    ):
        params = RemoteParams(num_workers=-1)
        params.finalize_and_validate()


def test_remote_params_allows_num_workers_zero_with_adaptive():
    """Test that num_workers=0 is allowed when adaptive concurrency is enabled."""
    params = RemoteParams(num_workers=0, use_adaptive_concurrency=True)
    params.finalize_and_validate()
    # No exception should be raised


def test_remote_params_rejects_num_workers_zero_without_adaptive():
    """Test that num_workers=0 is rejected when adaptive concurrency is disabled."""
    with pytest.raises(
        ValueError,
        match="num_workers=0 is only allowed when use_adaptive_concurrency=True",
    ):
        params = RemoteParams(num_workers=0, use_adaptive_concurrency=False)
        params.finalize_and_validate()


def test_adaptive_concurrency_params_exponential_scaling():
    """Test exponential scaling parameters in AdaptiveConcurrencyParams."""
    from oumi.core.configs.params.remote_params import AdaptiveConcurrencyParams

    # Test valid exponential scaling configuration
    params = AdaptiveConcurrencyParams(
        use_exponential_scaling=True,
        exponential_scaling_factor=1.5,
    )
    # Should not raise an exception
    params.finalize_and_validate()

    # Test invalid exponential scaling factor
    with pytest.raises(
        ValueError, match="Exponential scaling factor must be greater than 1.0"
    ):
        params = AdaptiveConcurrencyParams(
            use_exponential_scaling=True,
            exponential_scaling_factor=1.0,
        )
        params.finalize_and_validate()

    # Test invalid exponential scaling factor (negative)
    with pytest.raises(
        ValueError, match="Exponential scaling factor must be greater than 1.0"
    ):
        params = AdaptiveConcurrencyParams(
            use_exponential_scaling=True,
            exponential_scaling_factor=0.5,
        )
        params.finalize_and_validate()


def test_adaptive_concurrency_params_defaults():
    """Test default values for adaptive concurrency parameters."""
    from oumi.core.configs.params.remote_params import AdaptiveConcurrencyParams

    params = AdaptiveConcurrencyParams()
    params.finalize_and_validate()

    # Check defaults for new parameters
    assert params.use_exponential_scaling is False
    assert params.exponential_scaling_factor == 2.0
