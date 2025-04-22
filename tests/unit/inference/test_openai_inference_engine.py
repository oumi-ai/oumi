import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
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


@pytest.mark.parametrize(
    ("model_name,logit_bias,temperature,expected_logit_bias,expected_temperature,"),
    [
        ("some_model", {"token": 0.0}, 0.0, {"token": 0.0}, 0.0),
        ("o1-preview", {"token": 0.0}, 0.0, {}, 1.0),
    ],
    ids=[
        "test_default_params",
        "test_default_params_o1_preview",
    ],
)
def test_default_params(
    model_name, logit_bias, temperature, expected_logit_bias, expected_temperature
):
    openai_engine = OpenAIInferenceEngine(
        model_params=ModelParams(model_name=model_name),
        generation_params=GenerationParams(
            temperature=temperature,
            logit_bias=logit_bias,
        ),
    )
    assert openai_engine._remote_params.num_workers == 50
    assert openai_engine._remote_params.politeness_policy == 60.0

    conversation = Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
        ]
    )

    api_input = openai_engine._convert_conversation_to_api_input(
        conversation, openai_engine._generation_params, openai_engine._model_params
    )

    assert api_input["model"] == model_name
    assert api_input["temperature"] == expected_temperature
    if expected_logit_bias:
        assert api_input["logit_bias"] == expected_logit_bias
    else:
        assert "logit_bias" not in api_input
