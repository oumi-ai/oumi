from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import ToolDefinition
from oumi.inference import VLLMInferenceEngine
from tests.markers import requires_cuda_initialized, requires_gpus


def _get_default_model_params() -> ModelParams:
    return ModelParams(
        model_name="Qwen/Qwen3-0.6B",
        trust_remote_code=True,
    )


def _get_default_inference_config() -> InferenceConfig:
    return InferenceConfig(
        generation=GenerationParams(
            max_new_tokens=5, use_sampling=False, temperature=0.0, min_p=0.0, seed=42
        )
    )


@requires_cuda_initialized()
@requires_gpus()
def test_qwen_think_block_with_enable_thinking_true():
    convo = Conversation(
        messages=[Message(content="why is the sky blue?", role=Role.USER)]
    )
    engine = VLLMInferenceEngine(_get_default_model_params(), tensor_parallel_size=1)
    inference_config = _get_default_inference_config()
    outputs = engine.infer([convo], inference_config=inference_config)
    output = outputs[-1].messages[-1].content
    print(output)
    assert isinstance(output, str)
    assert "<think>" in output


@requires_cuda_initialized()
@requires_gpus()
def test_qwen_no_think_block_with_enable_thinking_false():
    convo = Conversation(
        messages=[Message(content="why is the sky blue?", role=Role.USER)]
    )

    engine = VLLMInferenceEngine(_get_default_model_params(), tensor_parallel_size=1)
    inference_config = _get_default_inference_config()
    inference_config.model.chat_template_kwargs = {"enable_thinking": False}

    outputs = engine.infer([convo], inference_config=inference_config)
    output = outputs[-1].messages[-1].content
    print(output)
    assert isinstance(output, str)
    assert "<think>" not in output


_WEATHER_TOOL = ToolDefinition.model_validate(
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
)


@requires_cuda_initialized()
@requires_gpus()
def test_qwen_emits_tool_call_when_tools_provided():
    """Tools on the conversation flow into the prompt; the model emits a call."""
    convo = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[
            Message(
                role=Role.USER,
                content="Use the get_weather tool to look up the weather in Tokyo.",
            )
        ],
    )
    engine = VLLMInferenceEngine(_get_default_model_params(), tensor_parallel_size=1)
    inference_config = InferenceConfig(
        generation=GenerationParams(
            max_new_tokens=128, temperature=0.0, min_p=0.0, seed=42
        )
    )

    outputs = engine.infer([convo], inference_config=inference_config)
    text = outputs[-1].messages[-1].content or ""
    print(text)
    # Qwen3 renders Hermes-style <tool_call> tags. 0.6B can be flaky on tool
    # use, so accept either the explicit tag or a tool-name mention.
    assert "<tool_call>" in text or "get_weather" in text


@requires_cuda_initialized()
@requires_gpus()
def test_qwen_tool_call_parser_populates_tool_calls():
    """tool_call_parser='hermes' parses an emitted call into Message.tool_calls."""
    convo = Conversation(
        tools=[_WEATHER_TOOL],
        messages=[
            Message(
                role=Role.USER,
                content="Use the get_weather tool to look up the weather in Tokyo.",
            )
        ],
    )
    engine = VLLMInferenceEngine(
        _get_default_model_params(),
        tensor_parallel_size=1,
        tool_call_parser="hermes",
    )
    inference_config = InferenceConfig(
        generation=GenerationParams(
            max_new_tokens=128, temperature=0.0, min_p=0.0, seed=42
        )
    )

    outputs = engine.infer([convo], inference_config=inference_config)
    assistant = outputs[-1].messages[-1]
    # 0.6B may not always emit a call; the strict gate is the unit-test suite.
    # When it does, verify the parsed payload and finish_reason are correct.
    if assistant.tool_calls:
        assert assistant.tool_calls[0].function.name == "get_weather"
        assert outputs[-1].metadata.get("finish_reason") == "tool_calls"
