import pytest

from oumi.core.configs.params.remote_params import RemoteParams
from oumi.exceptions import OumiConfigError


def test_remote_params_allows_empty():
    params = RemoteParams()
    params.finalize_and_validate()
    # No exception should be raised


def test_remote_params_validates_backoff_base():
    """Test that retry_backoff_base must be positive."""
    with pytest.raises(
        OumiConfigError,
        match="Retry backoff base must be greater than 0",
    ):
        params = RemoteParams(retry_backoff_base=0)
        params.finalize_and_validate()

    with pytest.raises(
        OumiConfigError,
        match="Retry backoff base must be greater than 0",
    ):
        params = RemoteParams(retry_backoff_base=-1)
        params.finalize_and_validate()


def test_remote_params_validates_backoff_max():
    """Test that retry_backoff_max is be greater than or equal to retry_backoff_base."""
    with pytest.raises(
        OumiConfigError,
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


def test_remote_params_default_max_retries_is_five():
    from oumi.core.configs.params.remote_params import RemoteParams

    assert RemoteParams().max_retries == 5


def test_remote_params_accepts_extra_body():
    params = RemoteParams(extra_body={"provider": {"zdr": True}})
    params.finalize_and_validate()


@pytest.mark.parametrize("reserved_key", ["model", "messages"])
def test_remote_params_rejects_reserved_extra_body_keys(reserved_key):
    with pytest.raises(OumiConfigError, match="reserved request fields"):
        RemoteParams(extra_body={reserved_key: "x"}).finalize_and_validate()


def test_remote_params_extra_body_round_trips_through_yaml(tmp_path):
    from oumi.core.configs import InferenceConfig, ModelParams

    config = InferenceConfig(
        model=ModelParams(model_name="some/model"),
        remote_params=RemoteParams(
            extra_body={"provider": {"zdr": True, "ignore": ["a", "b"]}}
        ),
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)

    loaded = InferenceConfig.from_yaml(path)

    assert loaded.remote_params is not None
    assert loaded.remote_params.extra_body == {
        "provider": {"zdr": True, "ignore": ["a", "b"]}
    }
