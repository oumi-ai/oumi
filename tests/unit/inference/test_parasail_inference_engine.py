import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.parasail_inference_engine import ParasailInferenceEngine


@pytest.fixture
def parasail_engine():
    return ParasailInferenceEngine(
        model_params=ModelParams(model_name="parasail-model"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_parasail_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="parasail-model")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = ParasailInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "parasail-model"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_parasail_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="parasail-model")
    engine = ParasailInferenceEngine(model_params=model_params)
    assert engine._model_params.model_name == "parasail-model"
    assert (
        engine._remote_params.api_url == "https://api.parasail.io/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "PARASAIL_API_KEY"
