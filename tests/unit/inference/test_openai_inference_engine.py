from typing import Any

import pytest
from aioresponses import aioresponses

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import (
    FunctionCall,
    FunctionDefinition,
    JSONSchema,
    ToolCall,
    ToolDefinition,
)
from oumi.inference.openai_inference_engine import (
    OpenAIInferenceEngine,
    _is_reasoning_model,
)


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
        ("gpt-4", {"token": 0.5}, 0.7, {"token": 0.5}, 0.7),
        # Reasoning models should have temperature forced to 1.0 and logit_bias cleared
        ("o1-preview", {"token": 0.0}, 0.0, {}, 1.0),
        ("o1", {"token": 0.5}, 0.8, {}, 1.0),
        ("o1-2024-12-17", {"token": 0.5}, 0.5, {}, 1.0),
        ("o1-mini", {"token": 0.5}, 0.5, {}, 1.0),
        ("o3-mini", {"token": 0.5}, 0.5, {}, 1.0),
        ("o4-mini", {"token": 0.5}, 0.5, {}, 1.0),
        ("gpt-5", {"token": 0.5}, 0.8, {}, 1.0),
        ("gpt-5-2025-08-07", {"token": 0.5}, 0.8, {}, 1.0),
        ("gpt-5-mini", {"token": 0.5}, 0.8, {}, 1.0),
        ("gpt-5-mini-2025-08-07", {"token": 0.5}, 0.8, {}, 1.0),
        ("gpt-5-nano", {"token": 0.5}, 0.8, {}, 1.0),
        ("gpt-5-nano-2025-08-07", {"token": 0.5}, 0.8, {}, 1.0),
    ],
    ids=[
        "standard_model",
        "gpt4_standard",
        "o1_preview",
        "o1",
        "o1_dated",
        "o1_mini",
        "o3_mini",
        "o4_mini",
        "gpt5",
        "gpt5_dated",
        "gpt5_mini",
        "gpt5_mini_dated",
        "gpt5_nano",
        "gpt5_nano_dated",
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


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        # Reasoning models (should return True)
        ("o1", True),
        ("o1-preview", True),
        ("o1-mini", True),
        ("o1-2024-12-17", True),
        ("o3-mini", True),
        ("o4-mini", True),
        ("gpt-5", True),
        ("gpt-5-mini", True),
        ("gpt-5-nano", True),
        ("gpt-5-2025-08-07", True),
        ("gpt-5-mini-2025-08-07", True),
        ("gpt-5-nano-2025-08-07", True),
        # Non-reasoning models (should return False)
        ("gpt-4", False),
        ("gpt-4-turbo", False),
        ("gpt-4.1", False),
        ("gpt-4.1-mini", False),
        ("gpt-3.5-turbo", False),
        ("claude-3-opus", False),
        ("some-random-model", False),
        ("", False),
    ],
)
def test_is_reasoning_model(model_name: str, expected: bool):
    """Test _is_reasoning_model correctly identifies reasoning models."""
    assert _is_reasoning_model(model_name) == expected


_OPENAI_TEST_URL = "http://fakeopenai/v1/chat/completions"


def _weather_tool() -> ToolDefinition:
    return ToolDefinition(
        function=FunctionDefinition(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters=JSONSchema(
                type="object",
                properties={"city": JSONSchema(type="string")},
                required=["city"],
            ),
        ),
    )


def test_openai_native_tool_calling_round_trip():
    """End-to-end: tools forward in request, tool_calls parse from response."""
    captured: dict[str, Any] = {}

    def callback(url: str, **kwargs: Any):
        from aioresponses import CallbackResult

        captured["body"] = kwargs.get("json")
        return CallbackResult(
            status=200,
            payload={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_xyz",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    with aioresponses() as m:
        m.post(_OPENAI_TEST_URL, callback=callback)

        engine = OpenAIInferenceEngine(
            model_params=ModelParams(model_name="gpt-4o"),
            remote_params=RemoteParams(api_url=_OPENAI_TEST_URL, api_key="key"),
        )
        tool = _weather_tool()
        conversation = Conversation(
            messages=[Message(content="Weather in Paris?", role=Role.USER)],
            tools=[tool],
        )
        result = engine.infer(
            [conversation],
            InferenceConfig(generation=GenerationParams(max_new_tokens=5)),
        )

    # Request side: tools forwarded in OpenAI shape.
    assert "tools" in captured["body"]
    assert captured["body"]["tools"] == [
        tool.model_dump(mode="json", exclude_none=True)
    ]

    # Response side: tool_calls parsed into structured Message field.
    assert len(result) == 1
    assistant = result[0].messages[-1]
    assert assistant.role == Role.ASSISTANT
    assert assistant.content is None
    assert assistant.tool_calls is not None
    assert len(assistant.tool_calls) == 1
    tc = assistant.tool_calls[0]
    assert tc.id == "call_xyz"
    assert tc.function.name == "get_weather"
    assert tc.function.arguments == '{"city": "Paris"}'
    assert result[0].metadata.get("finish_reason") == "tool_calls"


def test_openai_multiturn_tool_dialogue_replays_tool_messages():
    """Assistant tool_calls + Role.TOOL response round-trip into the request body."""
    captured: dict[str, Any] = {}

    def callback(url: str, **kwargs: Any):
        from aioresponses import CallbackResult

        captured["body"] = kwargs.get("json")
        return CallbackResult(
            status=200,
            payload={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "It's 18C in Paris.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with aioresponses() as m:
        m.post(_OPENAI_TEST_URL, callback=callback)

        engine = OpenAIInferenceEngine(
            model_params=ModelParams(model_name="gpt-4o"),
            remote_params=RemoteParams(api_url=_OPENAI_TEST_URL, api_key="key"),
        )
        tool_call = ToolCall(
            id="call_abc",
            function=FunctionCall(name="get_weather", arguments='{"city": "Paris"}'),
        )
        conversation = Conversation(
            messages=[
                Message(content="Weather in Paris?", role=Role.USER),
                Message(role=Role.ASSISTANT, tool_calls=[tool_call]),
                Message(
                    role=Role.TOOL,
                    tool_call_id="call_abc",
                    content='{"temp_c": 18}',
                ),
            ],
            tools=[_weather_tool()],
        )
        engine.infer(
            [conversation],
            InferenceConfig(generation=GenerationParams(max_new_tokens=5)),
        )

    sent_messages = captured["body"]["messages"]
    assert len(sent_messages) == 3
    assert sent_messages[1]["role"] == "assistant"
    assert sent_messages[1]["content"] is None
    assert sent_messages[1]["tool_calls"] == [
        tool_call.model_dump(mode="json", exclude_none=True)
    ]
    assert sent_messages[2]["role"] == "tool"
    assert sent_messages[2]["tool_call_id"] == "call_abc"
    assert sent_messages[2]["content"] == '{"temp_c": 18}'
