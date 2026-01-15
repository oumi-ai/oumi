import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.fireworks_inference_engine import FireworksInferenceEngine


@pytest.fixture
def fireworks_engine():
    return FireworksInferenceEngine(
        model_params=ModelParams(model_name="fireworks-model"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_fireworks_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="fireworks-model")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = FireworksInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "fireworks-model"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_fireworks_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="fireworks-model")
    engine = FireworksInferenceEngine(model_params)
    assert engine._model_params.model_name == "fireworks-model"
    assert (
        engine._remote_params.api_url
        == "https://api.fireworks.ai/inference/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "FIREWORKS_API_KEY"
