import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.openai_inference_engine import OpenAIInferenceEngine


@pytest.fixture
def openai_engine():
    return OpenAIInferenceEngine(
        model_params=ModelParams(model_name="gpt-4"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_openai_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="gpt-4")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = OpenAIInferenceEngine(
        model_params=model_params, remote_params=remote_params
    )
    assert engine._model_params.model_name == "gpt-4"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_openai_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="gpt-4")
    engine = OpenAIInferenceEngine(model_params)
    assert engine._model_params.model_name == "gpt-4"
    assert engine._remote_params.api_url == "https://api.openai.com/v1/chat/completions"
    assert engine._remote_params.api_key_env_varname == "OPENAI_API_KEY"
