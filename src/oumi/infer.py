# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.utils.logging import logger
from oumi.utils.provider_detection import detect_provider, is_yaml_path

if TYPE_CHECKING:
    pass

# Engine cache for reusing initialized engines across chat() calls
_ENGINE_CACHE: dict[str, BaseInferenceEngine] = {}


def get_engine(config: InferenceConfig) -> BaseInferenceEngine:
    """Returns the inference engine based on the provided config."""
    if config.engine is None:
        logger.warning(
            "No inference engine specified. Using the default 'native' engine."
        )
    return build_inference_engine(
        engine_type=config.engine or InferenceEngineType.NATIVE,
        model_params=config.model,
        remote_params=config.remote_params,
    )


def infer_interactive(
    config: InferenceConfig,
    *,
    input_image_bytes: list[bytes] | None = None,
    system_prompt: str | None = None,
) -> None:
    """Interactively provide the model response for a user-provided input."""
    # Create engine up front to avoid reinitializing it for each input.
    inference_engine = get_engine(config)
    while True:
        try:
            input_text = input("Enter your input prompt: ")
        except (EOFError, KeyboardInterrupt):  # Triggered by Ctrl+D/Ctrl+C
            print("\nExiting...")
            return
        model_response = infer(
            config=config,
            inputs=[
                input_text,
            ],
            system_prompt=system_prompt,
            input_image_bytes=input_image_bytes,
            inference_engine=inference_engine,
        )
        for g in model_response:
            print("------------")
            print(repr(g))
            print("------------")
        print()


def infer(
    config: InferenceConfig,
    inputs: list[str] | None = None,
    inference_engine: BaseInferenceEngine | None = None,
    *,
    input_image_bytes: list[bytes] | None = None,
    system_prompt: str | None = None,
) -> list[Conversation]:
    """Runs batch inference for a model using the provided configuration.

    Args:
        config: The configuration to use for inference.
        inputs: A list of inputs for inference.
        inference_engine: The engine to use for inference. If unspecified, the engine
            will be inferred from `config`.
        input_image_bytes: A list of input PNG image bytes to be used with `image+text`
            VLMs. Only used in interactive mode.
        system_prompt: System prompt for task-specific instructions.

    Returns:
        object: A list of model responses.
    """
    if not inference_engine:
        inference_engine = get_engine(config)

    # Pass None if no conversations are provided.
    conversations = None
    if inputs is not None and len(inputs) > 0:
        system_messages = (
            [Message(role=Role.SYSTEM, content=system_prompt)] if system_prompt else []
        )
        if input_image_bytes is None or len(input_image_bytes) == 0:
            conversations = [
                Conversation(
                    messages=(
                        system_messages + [Message(role=Role.USER, content=content)]
                    )
                )
                for content in inputs
            ]
        else:
            conversations = [
                Conversation(
                    messages=(
                        system_messages
                        + [
                            Message(
                                role=Role.USER,
                                content=(
                                    [
                                        ContentItem(
                                            type=Type.IMAGE_BINARY, binary=image_bytes
                                        )
                                        for image_bytes in input_image_bytes
                                    ]
                                    + [ContentItem(type=Type.TEXT, content=content)]
                                ),
                            )
                        ]
                    )
                )
                for content in inputs
            ]

    generations = inference_engine.infer(
        input=conversations,
        inference_config=config,
    )
    return generations


def clear_engine_cache() -> None:
    """Clear the inference engine cache.

    Use this to free memory when engines are no longer needed,
    or when you want to force re-initialization of engines.
    """
    _ENGINE_CACHE.clear()


def _get_cached_engine(
    engine_type: InferenceEngineType,
    model_name: str,
) -> BaseInferenceEngine:
    """Get or create a cached inference engine.

    Args:
        engine_type: The type of inference engine to create.
        model_name: The model name for the engine.

    Returns:
        A cached or newly created inference engine.
    """
    cache_key = f"{engine_type.value}:{model_name}"

    if cache_key not in _ENGINE_CACHE:
        model_params = ModelParams(model_name=model_name)
        _ENGINE_CACHE[cache_key] = build_inference_engine(
            engine_type=engine_type,
            model_params=model_params,
        )

    return _ENGINE_CACHE[cache_key]


@overload
def chat(
    model: str,
    message: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    conversation: Conversation | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    return_conversation: Literal[False] = False,
    use_cache: bool = True,
) -> str: ...


@overload
def chat(
    model: str,
    message: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    conversation: Conversation | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    return_conversation: Literal[True],
    use_cache: bool = True,
) -> Conversation: ...


def chat(
    model: str,
    message: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    conversation: Conversation | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    return_conversation: bool = False,
    use_cache: bool = True,
) -> str | Conversation:
    """Simple one-line chat interface for model inference.

    This function provides a streamlined way to interact with various LLM providers
    with automatic provider detection based on model name.

    Args:
        model: Model name with optional provider prefix.
            - "gpt-4o" -> auto-detected as OpenAI
            - "claude-3-opus" -> auto-detected as Anthropic
            - "openai/gpt-4o" -> explicit OpenAI
            - "meta-llama/Llama-3.1-8B-Instruct" -> HuggingFace model via vLLM
            - "config.yaml" -> load full InferenceConfig from YAML
        message: The user message to send. Required unless using messages/conversation.
        messages: List of message dicts with 'role' and 'content' keys (OpenAI format).
            Alternative to using message + system_prompt for multi-turn conversations.
        system_prompt: Optional system prompt for the conversation.
        conversation: Optional existing Conversation object to continue.
        temperature: Sampling temperature (default: provider-specific).
        max_tokens: Maximum tokens to generate (default: 1024).
        top_p: Top-p sampling parameter.
        return_conversation: If True, return full Conversation object instead of string.
        use_cache: If True, cache and reuse inference engines for better performance.

    Returns:
        str: The assistant's response text (default).
        Conversation: Full conversation object if return_conversation=True.

    Raises:
        ValueError: If neither message, messages, nor conversation is provided.
        RuntimeError: If no response is received from the model.

    Examples:
        Simple single message:
            >>> response = chat("gpt-4o", "What is machine learning?")

        With parameters:
            >>> response = chat("claude-3-opus", "Explain AI", temperature=0.7)

        Multi-turn with dict messages (OpenAI format):
            >>> response = chat("gpt-4o", messages=[
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hello!"},
            ... ])

        Continue a conversation:
            >>> conv = chat("gpt-4o", "Hi!", return_conversation=True)
            >>> response = chat("gpt-4o", "Tell me more", conversation=conv)

        Using YAML config:
            >>> response = chat("my_config.yaml", "Hello")

        Using explicit provider prefix:
            >>> response = chat("anthropic/claude-3-opus", "Hello")
    """
    # Handle YAML config path
    if is_yaml_path(model):
        config = InferenceConfig.from_yaml(model)
        input_text = message
        if messages:
            # Extract last user message from messages list
            user_msgs = [m for m in messages if m.get("role") == "user"]
            input_text = user_msgs[-1]["content"] if user_msgs else message
        results = infer(config, inputs=[input_text] if input_text else None)
        if return_conversation:
            return results[0] if results else Conversation(messages=[])
        if results and results[0].messages:
            last_msg = results[0].messages[-1]
            if last_msg.role == Role.ASSISTANT:
                content = last_msg.content
                if isinstance(content, str):
                    return content
                return last_msg.compute_flattened_text_content()
        return ""

    # Build conversation from various input formats first (for early validation)
    conv_messages: list[Message] = []

    if conversation is not None:
        # Continue existing conversation
        conv_messages = list(conversation.messages)
    elif messages is not None:
        # Build from dict messages (OpenAI format)
        for msg in messages:
            role_str = msg.get("role", "user").lower()
            content = msg.get("content", "")
            if role_str == "system":
                conv_messages.append(Message(role=Role.SYSTEM, content=content))
            elif role_str == "assistant":
                conv_messages.append(Message(role=Role.ASSISTANT, content=content))
            else:  # default to user
                conv_messages.append(Message(role=Role.USER, content=content))
    else:
        # Simple message + optional system prompt
        if system_prompt:
            conv_messages.append(Message(role=Role.SYSTEM, content=system_prompt))

    # Add the new user message if provided
    if message:
        conv_messages.append(Message(role=Role.USER, content=message))

    if not conv_messages:
        raise ValueError(
            "At least one of 'message', 'messages', or 'conversation' must be provided."
        )

    # Detect provider and clean model name
    engine_type, clean_model_name = detect_provider(model)

    # Build generation params
    gen_params = GenerationParams(
        temperature=temperature,
        max_new_tokens=max_tokens if max_tokens is not None else 1024,
        top_p=top_p,
    )

    # Get or create engine
    if use_cache:
        engine = _get_cached_engine(engine_type, clean_model_name)
    else:
        model_params = ModelParams(model_name=clean_model_name)
        engine = build_inference_engine(
            engine_type=engine_type,
            model_params=model_params,
        )

    conv = Conversation(messages=conv_messages)

    # Build inference config
    config = InferenceConfig(
        model=ModelParams(model_name=clean_model_name),
        generation=gen_params,
        engine=engine_type,
    )

    # Run inference
    results = engine.infer(input=[conv], inference_config=config)

    if not results:
        raise RuntimeError("No response received from model")

    result_conv = results[0]

    if return_conversation:
        return result_conv

    # Extract assistant response
    for msg in reversed(result_conv.messages):
        if msg.role == Role.ASSISTANT:
            content = msg.content
            if isinstance(content, str):
                return content
            return msg.compute_flattened_text_content()

    raise RuntimeError("No assistant response in conversation")
