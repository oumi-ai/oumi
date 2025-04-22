import json
from unittest.mock import AsyncMock, patch

import PIL.Image
import pytest

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.inference.gcp_inference_engine import GoogleVertexInferenceEngine
from oumi.utils.image_utils import (
    create_png_bytes_from_image,
)


def create_test_remote_params():
    return RemoteParams(
        api_url="https://example.com/api",
        api_key="path/to/service_account.json",
        num_workers=1,
        max_retries=3,
        connection_timeout=30,
        politeness_policy=0.1,
    )


@pytest.fixture
def gcp_engine():
    model_params = ModelParams(model_name="gcp-model")
    return GoogleVertexInferenceEngine(
        model_params, remote_params=create_test_remote_params()
    )


@pytest.fixture
def remote_params():
    return create_test_remote_params()


@pytest.fixture
def generation_params():
    return GenerationParams(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
    )


@pytest.fixture
def inference_config(generation_params, remote_params):
    return InferenceConfig(
        generation=generation_params,
        remote_params=remote_params,
    )


def create_test_text_only_conversation():
    return Conversation(
        messages=[
            Message(content="Hello", role=Role.USER),
            Message(content="Hi there!", role=Role.ASSISTANT),
            Message(content="How are you?", role=Role.USER),
        ]
    )


def create_test_multimodal_text_image_conversation():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)
    return Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(binary=png_bytes, type=Type.IMAGE_BINARY),
                    ContentItem(content="Hello", type=Type.TEXT),
                ],
            ),
            Message(content="Hi there!", role=Role.ASSISTANT),
            Message(content="How are you?", role=Role.USER),
        ]
    )


def _generate_test_convesations() -> list[Conversation]:
    return [
        create_test_text_only_conversation(),
        create_test_multimodal_text_image_conversation(),
    ]


def test_get_api_key(gcp_engine, remote_params):
    with patch("google.oauth2.service_account.Credentials") as mock_credentials:
        mock_credentials.from_service_account_file.return_value.token = "fake_token"
        token = gcp_engine._get_api_key(remote_params)
        assert token == "fake_token"
        mock_credentials.from_service_account_file.assert_called_once_with(
            filename="path/to/service_account.json",
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )


def test_get_request_headers(gcp_engine, remote_params):
    with patch.object(gcp_engine, "_get_api_key", return_value="fake_token"):
        headers = gcp_engine._get_request_headers(remote_params)
        assert headers == {
            "Authorization": "Bearer fake_token",
            "Content-Type": "application/json",
        }


def test_convert_conversation_to_api_input_text(gcp_engine, inference_config):
    conversation = create_test_text_only_conversation()
    api_input = gcp_engine._convert_conversation_to_api_input(
        conversation, inference_config.generation, gcp_engine._model_params
    )
    assert api_input["model"] == "gcp-model"
    assert len(conversation.messages) == 3
    assert len(api_input["messages"]) == 3
    assert all([isinstance(m["content"], str) for m in api_input["messages"]]), (
        api_input["messages"]
    )
    assert api_input["max_completion_tokens"] == 100
    assert api_input["temperature"] == 0.7
    assert api_input["top_p"] == 0.9


def test_convert_conversation_to_api_input_multimodal(gcp_engine, inference_config):
    conversation = create_test_multimodal_text_image_conversation()
    api_input = gcp_engine._convert_conversation_to_api_input(
        conversation, inference_config.generation, gcp_engine._model_params
    )
    assert api_input["model"] == "gcp-model"
    assert len(conversation.messages) == 3
    assert len(api_input["messages"]) == 3
    assert isinstance(api_input["messages"][0]["content"], list)
    assert len(api_input["messages"][0]["content"]) == 2
    assert isinstance(api_input["messages"][1]["content"], str)
    assert isinstance(api_input["messages"][2]["content"], str)
    assert api_input["max_completion_tokens"] == 100
    assert api_input["temperature"] == 0.7
    assert api_input["top_p"] == 0.9


@pytest.mark.parametrize(
    "conversation",
    _generate_test_convesations(),
)
def test_infer_online_text(gcp_engine, conversation, inference_config):
    with patch.object(gcp_engine, "_infer", new_callable=AsyncMock) as mock_infer:
        mock_infer.return_value = [conversation]
        results = gcp_engine.infer_online([conversation], inference_config)

    assert len(results) == 1
    assert results[0] == conversation


@pytest.mark.parametrize(
    "conversation",
    _generate_test_convesations(),
)
def test_infer_from_file(gcp_engine, conversation, inference_config, tmp_path):
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        json.dump(conversation.to_dict(), f)
        f.write("\n")

    with patch.object(gcp_engine, "_infer", new_callable=AsyncMock) as mock_infer:
        mock_infer.return_value = [conversation]
        results = gcp_engine.infer_from_file(str(input_file), inference_config)

    assert len(results) == 1
    assert results[0] == conversation


def test_remote_params_defaults():
    gcp_engine = GoogleVertexInferenceEngine(
        model_params=ModelParams(model_name="some_model"),
    )
    assert gcp_engine._remote_params.num_workers == 10
    assert gcp_engine._remote_params.politeness_policy == 60.0


def test_setting_api_url_via_constructor_region_and_project_id():
    gcp_engine = GoogleVertexInferenceEngine(
        model_params=ModelParams(model_name="some_model"),
        project_id="test_project_id",
        region="test_region",
    )

    # The method `_set_required_fields_for_inference` is called right before querying
    # the remote API (with `_query_api`) to validate/update the remote params.
    remote_params = RemoteParams()
    gcp_engine._set_required_fields_for_inference(remote_params)

    expected_api_url = (
        "https://test_region-aiplatform.googleapis.com/v1beta1/projects/"
        "test_project_id/locations/test_region/endpoints/openapi/chat/completions"
    )
    assert remote_params.api_url == expected_api_url


@patch("os.getenv")
def test_setting_api_url_via_env_region_and_project_id(mock_getenv):
    def mock_getenv_fn(key):
        return {"PROJECT_ID": "test_project_id", "REGION": "test_region"}.get(key)

    mock_getenv.side_effect = mock_getenv_fn

    gcp_engine = GoogleVertexInferenceEngine(
        model_params=ModelParams(model_name="some_model"),
    )

    # The method `_set_required_fields_for_inference` is called right before querying
    # the remote API (with `_query_api`) to validate/update the remote params.
    remote_params = RemoteParams()
    gcp_engine._set_required_fields_for_inference(remote_params)

    expected_api_url = (
        "https://test_region-aiplatform.googleapis.com/v1beta1/projects/"
        "test_project_id/locations/test_region/endpoints/openapi/chat/completions"
    )
    assert remote_params.api_url == expected_api_url


@patch("os.getenv")
def test_setting_api_url_via_env_region_and_project_id_custom_keys(mock_getenv):
    def mock_getenv_fn(key):
        return {
            "CUSTOM_PROJECT_ID_KEY": "test_project_id",
            "CUSTOM_REGION_KEY": "test_region",
        }.get(key)

    mock_getenv.side_effect = mock_getenv_fn

    gcp_engine = GoogleVertexInferenceEngine(
        model_params=ModelParams(model_name="some_model"),
        project_id_env_key="CUSTOM_PROJECT_ID_KEY",
        region_env_key="CUSTOM_REGION_KEY",
    )

    # The method `_set_required_fields_for_inference` is called right before querying
    # the remote API (with `_query_api`) to validate/update the remote params.
    remote_params = RemoteParams()
    gcp_engine._set_required_fields_for_inference(remote_params)

    expected_api_url = (
        "https://test_region-aiplatform.googleapis.com/v1beta1/projects/"
        "test_project_id/locations/test_region/endpoints/openapi/chat/completions"
    )
    assert remote_params.api_url == expected_api_url


def test_not_setting_api_url_failure():
    gcp_engine = GoogleVertexInferenceEngine(
        model_params=ModelParams(model_name="some_model"),
    )

    with pytest.raises(ValueError) as exception_info:
        gcp_engine._set_required_fields_for_inference(RemoteParams())
    assert str(exception_info.value) == (
        "This inference engine requires that either `api_url` is set in `RemoteParams` "
        "or that both `project_id` and `region` are set. You can set the `project_id` "
        "and `region` when constructing a GoogleVertexInferenceEngine, or as "
        "environment variables: `PROJECT_ID` and `REGION`."
    )
