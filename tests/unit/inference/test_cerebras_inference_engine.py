import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.cerebras_inference_engine import CerebrasInferenceEngine


@pytest.fixture
def cerebras_engine():
    return CerebrasInferenceEngine(
        model_params=ModelParams(model_name="cerebras-model"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_cerebras_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="cerebras-model")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = CerebrasInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "cerebras-model"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_cerebras_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="cerebras-model")
    engine = CerebrasInferenceEngine(model_params=model_params)
    assert engine._model_params.model_name == "cerebras-model"
    assert (
        engine._remote_params.api_url
        == "https://api.cerebras.ai/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "CEREBRAS_API_KEY"
