"""Model information utilities.

This module provides utilities for querying model-specific information
like context lengths, capabilities, etc.
"""

from typing import Optional


def get_context_length_for_engine(config) -> Optional[int]:
    """Get the appropriate context length for the given engine configuration.

    This is a centralized function to determine context length based on the
    engine type and model name. It handles both local engines (NATIVE, VLLM,
    LLAMACPP) that expose model_max_length, and API engines where we use
    known context limits.

    Args:
        config: The inference configuration object with engine and model attributes.

    Returns:
        Context length in tokens, or None if it cannot be determined.

    Note:
        For API engines, this currently uses hardcoded context limits based on
        model name patterns. Ideally, we should query provider SDKs directly
        (anthropic, openai, etc.) to get accurate, up-to-date context limits.
    """
    engine_type = str(config.engine) if config.engine else "NATIVE"

    # For local engines, check model_max_length
    if (
        "NATIVE" in engine_type
        or "VLLM" in engine_type
        or "LLAMACPP" in engine_type
    ):
        max_length = getattr(config.model, "model_max_length", None)
        if max_length is not None and max_length > 0:
            return max_length

    # For API engines, use hardcoded context limits based on model patterns
    model_name = getattr(config.model, "model_name", "").lower()

    # Anthropic context limits
    if "ANTHROPIC" in engine_type or "claude" in model_name:
        # All Claude 3+ models support 200K context
        return 200000

    # OpenAI context limits
    if "OPENAI" in engine_type or "gpt" in model_name:
        if "gpt-3.5" in model_name:
            return 16385  # GPT-3.5-turbo
        else:
            return 128000  # GPT-4, GPT-4o, and newer models

    # Together AI context limits
    if "TOGETHER" in engine_type:
        if "llama" in model_name:
            return 128000  # Llama 3+ models
        else:
            return 32768  # Conservative default for other models

    # DeepSeek context limits
    if "DEEPSEEK" in engine_type or "deepseek" in model_name:
        return 32768

    # Google Gemini context limits
    if "GOOGLE" in engine_type or "GEMINI" in engine_type or "gemini" in model_name:
        return 128000

    # If we can't determine, return None to signal fallback behavior
    return None
