import pytest
from aioresponses import aioresponses
from yarl import URL

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import ToolCall, ToolDefinition
from oumi.inference.remote_vllm_inference_engine import RemoteVLLMInferenceEngine
from oumi.inference.vllm_inference_engine import (
    _TOOL_PARSER_CAPABILITIES,
    VLLMInferenceEngine,
)

_CONVERSATION = Conversation(messages=[Message(role=Role.USER, content="hi")])
_CHAT_URL = "http://localhost:8000/v1/chat/completions"
_TOKENIZE_URL = "http://localhost:8000/tokenize"
_ANSWER = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "length",
        }
    ]
}


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


def _tool_call(call_id: str, name: str) -> ToolCall:
    return ToolCall.model_validate(
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": '{"city": "Tokyo", "days": 3}'},
        }
    )


_TOOL_CONVERSATION = Conversation(
    tools=[
        ToolDefinition.model_validate(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                        },
                        "required": ["city"],
                    },
                },
            }
        )
        for name in ("get_weather", "get_time")
    ],
    messages=[
        Message(role=Role.SYSTEM, content="Use the tools."),
        Message(role=Role.USER, content="Weather and time in Tokyo?"),
        Message(
            role=Role.ASSISTANT,
            content="Let me check both.",
            tool_calls=[_tool_call("a", "get_weather"), _tool_call("b", "get_time")],
        ),
        Message(role=Role.TOOL, content="22C", tool_call_id="a"),
        Message(role=Role.TOOL, content="9:30", tool_call_id="b"),
        Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[_tool_call("c", "get_weather")],
        ),
        Message(role=Role.TOOL, content="23C", tool_call_id="c"),
        Message(role=Role.ASSISTANT, content="22C and 9:30."),
        Message(role=Role.USER, content="And tomorrow?"),
    ],
)


@pytest.mark.parametrize("parser", [None, "hermes", *_TOOL_PARSER_CAPABILITIES])
def test_sends_the_history_and_tools_the_vllm_engine_renders(parser):
    """`vllm serve` renders the chat template VLLMInferenceEngine does."""
    offline = object.__new__(VLLMInferenceEngine)
    offline._tool_parser_name = parser
    engine = _engine(tool_call_parser=parser)

    api_input = engine._convert_conversation_to_api_input(
        _TOOL_CONVERSATION, GenerationParams(), engine._model_params
    )

    assert api_input["messages"] == offline._convert_conversation_to_vllm_input(
        _TOOL_CONVERSATION
    )
    [(_, offline_tools)] = VLLMInferenceEngine._group_by_tools([_TOOL_CONVERSATION])
    assert api_input["tools"] == offline_tools


def _chat_engine() -> RemoteVLLMInferenceEngine:
    return RemoteVLLMInferenceEngine(
        ModelParams(model_name="served-model"),
        generation_params=GenerationParams(max_new_tokens=512),
        remote_params=RemoteParams(api_url=_CHAT_URL, api_key="k", max_retries=0),
    )


def _sent_max_tokens(mocked: aioresponses) -> list[int]:
    return [
        call.kwargs["json"]["max_tokens"]
        for call in mocked.requests[("POST", URL(_CHAT_URL))]
    ]


def test_a_request_past_the_context_is_resent_capped_to_the_context_left():
    with aioresponses() as mocked:
        mocked.post(_CHAT_URL, status=400, body="maximum context length exceeded")
        mocked.post(_TOKENIZE_URL, payload={"count": 8000, "max_model_len": 8192})
        mocked.post(_CHAT_URL, payload=_ANSWER)

        answered = _chat_engine().infer([_CONVERSATION])

        assert _sent_max_tokens(mocked) == [512, 192]
    assert answered[0].messages[-1].content == "ok"


def test_a_rejected_request_within_the_context_keeps_its_error():
    with aioresponses() as mocked:
        mocked.post(_CHAT_URL, status=400, body="bad request")
        mocked.post(_TOKENIZE_URL, payload={"count": 10, "max_model_len": 8192})

        with pytest.raises(RuntimeError, match="HTTP 400"):
            _chat_engine().infer([_CONVERSATION])

        assert _sent_max_tokens(mocked) == [512]
