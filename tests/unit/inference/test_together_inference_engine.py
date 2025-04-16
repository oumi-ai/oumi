import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.together_inference_engine import TogetherInferenceEngine


@pytest.fixture
def together_engine():
    return TogetherInferenceEngine(
        model_params=ModelParams(model_name="together-model"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_together_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="together-model")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = TogetherInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "together-model"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_together_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="together-model")
    engine = TogetherInferenceEngine(model_params)
    assert engine._model_params.model_name == "together-model"
    assert (
        engine._remote_params.api_url == "https://api.together.xyz/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "TOGETHER_API_KEY"
