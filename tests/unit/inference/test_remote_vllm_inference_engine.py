import pytest

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.remote_vllm_inference_engine import RemoteVLLMInferenceEngine

_CONVERSATION = Conversation(messages=[Message(role=Role.USER, content="hi")])


def _engine(**model_kwargs) -> RemoteVLLMInferenceEngine:
    return RemoteVLLMInferenceEngine(
        ModelParams(model_name="served-model", **model_kwargs),
        remote_params=RemoteParams(api_url="http://localhost:8000", api_key="k"),
    )


def _api_input(
    engine: RemoteVLLMInferenceEngine, generation_params: GenerationParams
) -> dict:
    return engine._convert_conversation_to_api_input(
        _CONVERSATION, generation_params, engine._model_params
    )


def test_sends_chat_template_kwargs():
    engine = _engine(chat_template_kwargs={"enable_thinking": False})

    api_input = _api_input(engine, GenerationParams())

    assert api_input["chat_template_kwargs"] == {"enable_thinking": False}


def test_omits_chat_template_kwargs_when_unset():
    api_input = _api_input(_engine(), GenerationParams())

    assert "chat_template_kwargs" not in api_input


def test_sends_min_p_and_skip_special_tokens():
    api_input = _api_input(
        _engine(), GenerationParams(min_p=0.05, skip_special_tokens=False)
    )

    assert api_input["min_p"] == 0.05
    assert api_input["skip_special_tokens"] is False


@pytest.mark.parametrize(
    ("guided_decoding", "kind", "value"),
    [
        (GuidedDecodingParams(json={"type": "object"}), "json", {"type": "object"}),
        (GuidedDecodingParams(regex="[0-9]+"), "regex", "[0-9]+"),
        (GuidedDecodingParams(choice=["yes", "no"]), "choice", ["yes", "no"]),
    ],
)
def test_sends_constraints_as_structured_outputs_and_guided_keys(
    guided_decoding, kind, value
):
    api_input = _api_input(_engine(), GenerationParams(guided_decoding=guided_decoding))

    assert api_input["structured_outputs"] == {kind: value}
    assert api_input[f"guided_{kind}"] == value


def test_an_unconstrained_request_sends_no_structured_outputs():
    api_input = _api_input(_engine(), GenerationParams())

    assert "structured_outputs" not in api_input
    assert not [key for key in api_input if key.startswith("guided_")]


def test_min_p_and_skip_special_tokens_are_supported():
    assert {"min_p", "skip_special_tokens"} <= _engine().get_supported_params()
