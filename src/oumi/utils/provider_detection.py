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

"""Provider detection utilities for auto-selecting inference engines.

This module provides utilities for automatically detecting the appropriate
inference engine based on model names, following a hybrid approach:

1. Explicit prefix (highest priority): "openai/gpt-4o" -> OpenAI
2. Known model patterns: "gpt-4o" -> OpenAI, "claude-3" -> Anthropic
3. Default: "org/model" format -> vLLM (local HuggingFace models)

Example:
    >>> from oumi.utils.provider_detection import detect_provider
    >>> engine_type, model_name = detect_provider("gpt-4o")
    >>> print(engine_type)  # InferenceEngineType.OPENAI
    >>> print(model_name)   # "gpt-4o"
"""

from oumi.core.configs import InferenceEngineType


# Explicit provider prefixes (highest priority)
# Format: "prefix/model" -> engine_type, with prefix removed from model name
PROVIDER_PREFIXES: dict[str, InferenceEngineType] = {
    "openai/": InferenceEngineType.OPENAI,
    "anthropic/": InferenceEngineType.ANTHROPIC,
    "google/": InferenceEngineType.GOOGLE_GEMINI,
    "gemini/": InferenceEngineType.GOOGLE_GEMINI,
    "vertex/": InferenceEngineType.GOOGLE_VERTEX,
    "deepseek/": InferenceEngineType.DEEPSEEK,
    "together/": InferenceEngineType.TOGETHER,
    "parasail/": InferenceEngineType.PARASAIL,
    "sambanova/": InferenceEngineType.SAMBANOVA,
    "bedrock/": InferenceEngineType.BEDROCK,
    "lambda/": InferenceEngineType.LAMBDA,
    "vllm/": InferenceEngineType.VLLM,
    "llamacpp/": InferenceEngineType.LLAMACPP,
    "sglang/": InferenceEngineType.SGLANG,
}

# Known model name patterns for auto-detection
# These are checked when no explicit prefix is provided
OPENAI_MODEL_PREFIXES: set[str] = {
    "gpt-4",
    "gpt-3.5",
    "gpt-3",
    "o1",
    "o3",
    "chatgpt",
    "text-davinci",
    "text-embedding",
    "dall-e",
    "whisper",
    "tts",
}

ANTHROPIC_MODEL_PREFIXES: set[str] = {
    "claude-3",
    "claude-2",
    "claude-instant",
    "claude-opus",
    "claude-sonnet",
    "claude-haiku",
}

GEMINI_MODEL_PREFIXES: set[str] = {
    "gemini-pro",
    "gemini-ultra",
    "gemini-1.5",
    "gemini-2",
    "gemini-flash",
}

DEEPSEEK_MODEL_PREFIXES: set[str] = {
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-v2",
    "deepseek-v3",
    "deepseek-r1",
}


def detect_provider(model: str) -> tuple[InferenceEngineType, str]:
    """Detect the inference provider and clean model name from a model string.

    This function uses a hybrid approach to determine the appropriate inference
    engine:
    1. Check for explicit provider prefix (e.g., "openai/gpt-4o")
    2. Match against known model name patterns (e.g., "gpt-4o" -> OpenAI)
    3. Default to vLLM for "org/model" format (HuggingFace models)
    4. Fallback to vLLM for unknown models

    Args:
        model: Model name, optionally with provider prefix.

    Returns:
        Tuple of (InferenceEngineType, cleaned_model_name).

    Examples:
        >>> detect_provider("openai/gpt-4o")
        (InferenceEngineType.OPENAI, "gpt-4o")

        >>> detect_provider("gpt-4o")
        (InferenceEngineType.OPENAI, "gpt-4o")

        >>> detect_provider("claude-3-opus")
        (InferenceEngineType.ANTHROPIC, "claude-3-opus")

        >>> detect_provider("meta-llama/Llama-3.1-8B-Instruct")
        (InferenceEngineType.VLLM, "meta-llama/Llama-3.1-8B-Instruct")

        >>> detect_provider("unknown-model")
        (InferenceEngineType.VLLM, "unknown-model")
    """
    model_lower = model.lower()

    # 1. Check explicit provider prefix
    for prefix, engine_type in PROVIDER_PREFIXES.items():
        if model_lower.startswith(prefix):
            # Remove prefix from model name
            clean_name = model[len(prefix) :]
            return engine_type, clean_name

    # 2. Check known model patterns (case-insensitive prefix matching)
    for prefix in OPENAI_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return InferenceEngineType.OPENAI, model

    for prefix in ANTHROPIC_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return InferenceEngineType.ANTHROPIC, model

    for prefix in GEMINI_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return InferenceEngineType.GOOGLE_GEMINI, model

    for prefix in DEEPSEEK_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return InferenceEngineType.DEEPSEEK, model

    # 3. Default: assume HuggingFace model for "org/model" format -> vLLM
    # This covers models like "meta-llama/Llama-3.1-8B-Instruct"
    if "/" in model and not model.startswith("http"):
        return InferenceEngineType.VLLM, model

    # 4. Fallback to vLLM for unknown models
    return InferenceEngineType.VLLM, model


def is_yaml_path(value: str) -> bool:
    """Check if a string looks like a YAML config path.

    Args:
        value: String to check.

    Returns:
        True if the string ends with .yaml or .yml extension.

    Examples:
        >>> is_yaml_path("config.yaml")
        True

        >>> is_yaml_path("config.yml")
        True

        >>> is_yaml_path("gpt-4o")
        False

        >>> is_yaml_path("configs/training/sft.yaml")
        True
    """
    return value.endswith((".yaml", ".yml"))


def get_provider_help_message(model: str) -> str:
    """Generate a helpful error message when provider detection fails.

    Args:
        model: The model name that couldn't be resolved.

    Returns:
        A helpful message suggesting how to fix the issue.
    """
    return (
        f"Could not determine provider for model '{model}'. "
        f"You can specify the provider explicitly using a prefix:\n"
        f"  - openai/gpt-4o (OpenAI)\n"
        f"  - anthropic/claude-3-opus (Anthropic)\n"
        f"  - google/gemini-1.5-pro (Google Gemini)\n"
        f"  - together/meta-llama/Llama-3.1-8B (Together AI)\n"
        f"  - vllm/meta-llama/Llama-3.1-8B (Local vLLM)\n"
        f"\nOr use a full config file: chat('config.yaml', 'message')"
    )
