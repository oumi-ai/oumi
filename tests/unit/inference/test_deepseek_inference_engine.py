import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.deepseek_inference_engine import DeepSeekInferenceEngine


@pytest.fixture
def deepseek_engine():
    return DeepSeekInferenceEngine(
        model_params=ModelParams(model_name="deepseek-model"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_deepseek_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="deepseek-model")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = DeepSeekInferenceEngine(model_params, remote_params=remote_params)
    assert engine._model_params.model_name == "deepseek-model"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_deepseek_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="deepseek-model")
    engine = DeepSeekInferenceEngine(model_params)
    assert engine._model_params.model_name == "deepseek-model"
    assert (
        engine._remote_params.api_url == "https://api.deepseek.com/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "DEEPSEEK_API_KEY"
