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

"""Format converters for transforming raw data dictionaries to Conversation objects.

This module provides converters for various dataset formats (Oumi, Alpaca, ShareGPT,
etc.) that convert raw data into the standard Conversation format used by Oumi.

Example usage:
    from oumi.core.converters.format_converters import convert_alpaca

    example = {"instruction": "Translate", "input": "Hello", "output": "Bonjour"}
    conversation = convert_alpaca(example)
"""

from typing import Any, Callable, Optional

from oumi.core.registry import register_converter
from oumi.core.types.conversation import Conversation, Message, Role


# Type alias for converter functions
ConverterFn = Callable[[dict], Conversation]


#
# Core format converters
#


@register_converter("oumi")
def convert_oumi(example: dict) -> Conversation:
    """Convert Oumi/OpenAI format data.

    Expected format:
        {"messages": [{"role": "user", "content": "..."}, ...]}

    Args:
        example: Dictionary with "messages" key containing list of message dicts.

    Returns:
        Conversation object.

    Raises:
        ValueError: If the example doesn't have the expected structure.
    """
    if "messages" not in example:
        raise ValueError(
            "Oumi format requires 'messages' key. "
            f"Got keys: {list(example.keys())}"
        )
    return Conversation.model_validate(example)


@register_converter("alpaca")
def convert_alpaca(example: dict) -> Conversation:
    """Convert Alpaca format data.

    Expected format:
        {"instruction": "...", "input": "...", "output": "..."}

    Args:
        example: Dictionary with instruction, input, and output keys.

    Returns:
        Conversation object with user and assistant messages.

    Raises:
        ValueError: If required keys are missing.
    """
    required_keys = ["instruction", "input", "output"]
    missing_keys = [k for k in required_keys if k not in example]
    if missing_keys:
        raise ValueError(
            f"Alpaca format requires keys: {required_keys}. "
            f"Missing: {missing_keys}"
        )

    # Combine instruction and input for user message
    instruction = example["instruction"]
    input_text = example.get("input", "")

    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    messages = [
        Message(role=Role.USER, content=user_content),
        Message(role=Role.ASSISTANT, content=example["output"]),
    ]
    return Conversation(messages=messages)


@register_converter("sharegpt")
def convert_sharegpt(example: dict) -> Conversation:
    """Convert ShareGPT format data.

    Expected format:
        {"conversations": [{"from": "human/gpt", "value": "..."}]}

    Role mapping:
        - "human" -> USER
        - "gpt" -> ASSISTANT
        - "system" -> SYSTEM

    Args:
        example: Dictionary with "conversations" key.

    Returns:
        Conversation object.

    Raises:
        ValueError: If format is invalid or roles are unknown.
    """
    if "conversations" not in example:
        raise ValueError(
            "ShareGPT format requires 'conversations' key. "
            f"Got keys: {list(example.keys())}"
        )

    role_map = {
        "human": Role.USER,
        "gpt": Role.ASSISTANT,
        "system": Role.SYSTEM,
    }

    messages = []
    for turn in example["conversations"]:
        from_role = turn.get("from", "").lower()
        if from_role not in role_map:
            raise ValueError(
                f"Unknown ShareGPT role: '{turn.get('from')}'. "
                f"Expected one of: {list(role_map.keys())}"
            )
        messages.append(Message(role=role_map[from_role], content=turn["value"]))

    return Conversation(messages=messages)


@register_converter("conversations")
def convert_conversations(example: dict) -> Conversation:
    """Convert nested conversations format.

    Expected format:
        {"conversation": {"messages": [{"role": "...", "content": "..."}]}}

    Args:
        example: Dictionary with nested "conversation" key.

    Returns:
        Conversation object.

    Raises:
        ValueError: If the format is invalid.
    """
    if "conversation" not in example:
        raise ValueError(
            "Conversations format requires 'conversation' key. "
            f"Got keys: {list(example.keys())}"
        )
    return Conversation.model_validate(example["conversation"])


#
# Observability format converters
#


@register_converter("langfuse")
def convert_langfuse(example: dict) -> Conversation:
    """Convert Langfuse export format.

    Langfuse exports traces with input/output or prompt/completion structures.

    Expected formats:
        {"input": "...", "output": "..."}
        or
        {"prompt": "...", "completion": "..."}

    Args:
        example: Dictionary from Langfuse export.

    Returns:
        Conversation object.

    Raises:
        ValueError: If neither input/output nor prompt/completion found.
    """
    messages = []

    # Try input/output format first
    if "input" in example:
        messages.append(Message(role=Role.USER, content=str(example["input"])))
        if "output" in example and example["output"]:
            messages.append(
                Message(role=Role.ASSISTANT, content=str(example["output"]))
            )
    # Try prompt/completion format
    elif "prompt" in example:
        messages.append(Message(role=Role.USER, content=str(example["prompt"])))
        if "completion" in example and example["completion"]:
            messages.append(
                Message(role=Role.ASSISTANT, content=str(example["completion"]))
            )
    else:
        raise ValueError(
            "Langfuse format requires 'input'/'output' or 'prompt'/'completion' keys. "
            f"Got keys: {list(example.keys())}"
        )

    return Conversation(messages=messages)


@register_converter("opentelemetry")
def convert_opentelemetry(example: dict) -> Conversation:
    """Convert OpenTelemetry LLM semantic conventions format.

    OpenTelemetry defines semantic conventions for LLM observability with
    attributes like gen_ai.prompt and gen_ai.completion.

    Expected format:
        {"gen_ai.prompt": "...", "gen_ai.completion": "..."}
        or with nested attributes:
        {"attributes": {"gen_ai.prompt": "...", "gen_ai.completion": "..."}}

    Args:
        example: Dictionary with OpenTelemetry LLM attributes.

    Returns:
        Conversation object.

    Raises:
        ValueError: If required attributes are missing.
    """
    # Handle nested attributes structure
    attrs = example.get("attributes", example)

    messages = []

    # Check for gen_ai semantic convention attributes
    prompt_key = "gen_ai.prompt"
    completion_key = "gen_ai.completion"

    if prompt_key in attrs:
        messages.append(Message(role=Role.USER, content=str(attrs[prompt_key])))
        if completion_key in attrs and attrs[completion_key]:
            messages.append(
                Message(role=Role.ASSISTANT, content=str(attrs[completion_key]))
            )
    else:
        raise ValueError(
            f"OpenTelemetry format requires '{prompt_key}' attribute. "
            f"Got keys: {list(attrs.keys())}"
        )

    return Conversation(messages=messages)


@register_converter("langchain")
def convert_langchain(example: dict) -> Conversation:
    """Convert LangChain runs/traces format.

    LangChain traces contain inputs and outputs from chain runs.

    Expected format:
        {"inputs": {"input": "..."}, "outputs": {"output": "..."}}
        or
        {"input": "...", "output": "..."}

    Args:
        example: Dictionary from LangChain trace.

    Returns:
        Conversation object.

    Raises:
        ValueError: If the format is invalid.
    """
    messages = []

    # Handle nested inputs/outputs format
    if "inputs" in example:
        inputs = example["inputs"]
        input_text = inputs.get("input") or inputs.get("question") or str(inputs)
        messages.append(Message(role=Role.USER, content=str(input_text)))

        if "outputs" in example and example["outputs"]:
            outputs = example["outputs"]
            output_text = (
                outputs.get("output") or outputs.get("answer") or str(outputs)
            )
            messages.append(Message(role=Role.ASSISTANT, content=str(output_text)))
    # Handle flat format
    elif "input" in example:
        messages.append(Message(role=Role.USER, content=str(example["input"])))
        if "output" in example and example["output"]:
            messages.append(
                Message(role=Role.ASSISTANT, content=str(example["output"]))
            )
    else:
        raise ValueError(
            "LangChain format requires 'inputs'/'outputs' or 'input'/'output' keys. "
            f"Got keys: {list(example.keys())}"
        )

    return Conversation(messages=messages)


#
# Converter factories for configurable converters
#


def create_alpaca_converter(
    include_system_prompt: bool = False,
    system_prompt_with_context: Optional[str] = None,
    system_prompt_without_context: Optional[str] = None,
) -> ConverterFn:
    """Create an Alpaca converter with configurable system prompt.

    This factory creates a converter that can include system prompts similar to
    the original Stanford Alpaca format.

    Args:
        include_system_prompt: Whether to include a system prompt.
        system_prompt_with_context: System prompt when input context is provided.
            Defaults to Stanford Alpaca prompt.
        system_prompt_without_context: System prompt when no input context.
            Defaults to simplified Stanford Alpaca prompt.

    Returns:
        A converter function that converts Alpaca format to Conversation.

    Example:
        converter = create_alpaca_converter(include_system_prompt=True)
        conversation = converter({"instruction": "...", "input": "...", "output": "..."})
    """
    default_with_context = (
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes "
        "the request."
    )
    default_without_context = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request."
    )

    sys_with_ctx = system_prompt_with_context or default_with_context
    sys_without_ctx = system_prompt_without_context or default_without_context

    def converter(example: dict) -> Conversation:
        required_keys = ["instruction", "input", "output"]
        missing_keys = [k for k in required_keys if k not in example]
        if missing_keys:
            raise ValueError(
                f"Alpaca format requires keys: {required_keys}. "
                f"Missing: {missing_keys}"
            )

        messages = []

        has_input = "input" in example and len(str(example["input"])) > 0

        if include_system_prompt:
            system_prompt = sys_with_ctx if has_input else sys_without_ctx
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))

        if has_input:
            user_content = (
                f"{example['instruction']}\n\n### Input:\n{example['input']}"
            )
        else:
            user_content = example["instruction"]

        messages.append(Message(role=Role.USER, content=user_content))
        messages.append(Message(role=Role.ASSISTANT, content=example["output"]))

        return Conversation(messages=messages)

    return converter


#
# Auto-detection
#


def auto_detect_converter(example: dict) -> str:
    """Auto-detect format based on example structure.

    Examines the keys and structure of the example to determine which
    format converter should be used.

    Args:
        example: A single data example to analyze.

    Returns:
        The name of the detected converter (e.g., "oumi", "alpaca", "sharegpt").

    Raises:
        ValueError: If the format cannot be detected.
    """
    # Check for Oumi/OpenAI format: {"messages": [...]}
    if "messages" in example:
        messages = example["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            if all(
                isinstance(m, dict) and "role" in m and "content" in m
                for m in messages
            ):
                return "oumi"

    # Check for nested conversations format: {"conversation": {"messages": [...]}}
    if "conversation" in example:
        conv = example["conversation"]
        if isinstance(conv, dict) and "messages" in conv:
            return "conversations"

    # Check for Alpaca format: {"instruction", "input", "output"}
    if all(key in example for key in ["instruction", "input", "output"]):
        return "alpaca"

    # Check for ShareGPT format: {"conversations": [{"from": ..., "value": ...}]}
    if "conversations" in example:
        convs = example["conversations"]
        if isinstance(convs, list) and len(convs) > 0:
            if all("from" in c and "value" in c for c in convs):
                return "sharegpt"

    # Check for Langfuse format
    if ("input" in example and "output" in example) or (
        "prompt" in example and "completion" in example
    ):
        return "langfuse"

    # Check for OpenTelemetry format
    attrs = example.get("attributes", example)
    if "gen_ai.prompt" in attrs:
        return "opentelemetry"

    # Check for LangChain format
    if "inputs" in example and "outputs" in example:
        return "langchain"

    raise ValueError(
        "Unable to auto-detect format. Please specify a converter explicitly "
        "using the 'converter' parameter.\n"
        f"Got keys: {list(example.keys())}\n"
        "Supported formats:\n"
        "  - oumi: {'messages': [{'role': ..., 'content': ...}]}\n"
        "  - alpaca: {'instruction': ..., 'input': ..., 'output': ...}\n"
        "  - sharegpt: {'conversations': [{'from': ..., 'value': ...}]}\n"
        "  - langfuse: {'input': ..., 'output': ...}\n"
        "  - opentelemetry: {'gen_ai.prompt': ..., 'gen_ai.completion': ...}\n"
        "  - langchain: {'inputs': {...}, 'outputs': {...}}"
    )


def get_converter(name: str, **kwargs: Any) -> ConverterFn:
    """Get a converter function by name.

    This is a convenience function that retrieves a converter from the registry
    and optionally applies factory arguments.

    Args:
        name: The converter name (e.g., "alpaca", "oumi", "sharegpt").
        **kwargs: Additional arguments for converter factories.

    Returns:
        A converter function.

    Raises:
        ValueError: If the converter is not found.

    Example:
        # Simple converter
        converter = get_converter("alpaca")

        # Converter with factory kwargs
        converter = get_converter("alpaca_custom", include_system_prompt=True)
    """
    from oumi.core.registry import REGISTRY, RegistryType

    converter = REGISTRY.get_converter(name)
    if converter is None:
        raise ValueError(
            f"Unknown converter: '{name}'. "
            "Available converters can be listed via "
            "REGISTRY.get_all(RegistryType.CONVERTER)"
        )

    # If kwargs provided and converter is a factory, call it
    if kwargs:
        # Check if it's a factory (returns a callable when called)
        try:
            return converter(**kwargs)
        except TypeError:
            # Not a factory, ignore kwargs
            pass

    return converter
