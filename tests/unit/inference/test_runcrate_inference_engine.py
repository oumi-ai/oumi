import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.runcrate_inference_engine import RuncrateInferenceEngine


@pytest.fixture
def runcrate_engine():
    return RuncrateInferenceEngine(
        model_params=ModelParams(model_name="Qwen/Qwen3-235B-A22B-Instruct-2507"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_runcrate_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="Qwen/Qwen3-235B-A22B-Instruct-2507")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = RuncrateInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "Qwen/Qwen3-235B-A22B-Instruct-2507"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_runcrate_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="Qwen/Qwen3-235B-A22B-Instruct-2507")
    engine = RuncrateInferenceEngine(model_params=model_params)
    assert engine._model_params.model_name == "Qwen/Qwen3-235B-A22B-Instruct-2507"
    assert (
        engine._remote_params.api_url == "https://api.runcrate.ai/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "RUNCRATE_API_KEY"


def test_runcrate_base_url_and_env_var(runcrate_engine):
    assert runcrate_engine.base_url == "https://api.runcrate.ai/v1/chat/completions"
    assert runcrate_engine.api_key_env_varname == "RUNCRATE_API_KEY"
