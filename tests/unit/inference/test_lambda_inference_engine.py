import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.lambda_inference_engine import LambdaInferenceEngine


@pytest.fixture
def lambda_engine():
    return LambdaInferenceEngine(
        model_params=ModelParams(model_name="lambda-model"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_lambda_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="lambda-model")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = LambdaInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "lambda-model"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_lambda_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="lambda-model")
    engine = LambdaInferenceEngine(model_params)
    assert engine._model_params.model_name == "lambda-model"
    assert engine._remote_params.api_url == "https://api.lambda.ai/v1/chat/completions"
    assert engine._remote_params.api_key_env_varname == "LAMBDA_API_KEY"
