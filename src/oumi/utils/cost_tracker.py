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

"""Cost tracking and estimation for API providers.

Pricing information is based on 2025 rates and may change. Always verify current pricing
on provider websites.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_price_per_million: float
    """Price per million input tokens in USD."""

    output_price_per_million: float
    """Price per million output tokens in USD."""

    reasoning_price_per_million: Optional[float] = None
    """Price per million reasoning tokens in USD (OpenAI o-series)."""

    cache_write_price_per_million: Optional[float] = None
    """Price per million tokens for cache writes in USD (Anthropic)."""

    cache_read_price_per_million: Optional[float] = None
    """Price per million tokens for cache reads in USD (Anthropic)."""

    thinking_price_per_million: Optional[float] = None
    """Price per million thinking tokens in USD (Anthropic Claude 4.1+)."""


# OpenAI Pricing (2025)
# Source: https://openai.com/api/pricing/
OPENAI_PRICING = {
    # GPT-4o series
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00),
    "gpt-4o-2024-08-06": ModelPricing(2.50, 10.00),
    "gpt-4o-2024-05-13": ModelPricing(5.00, 15.00),
    "gpt-4o-mini-2024-07-18": ModelPricing(0.15, 0.60),
    # GPT-4 Turbo
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-2024-04-09": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-preview": ModelPricing(10.00, 30.00),
    # GPT-4
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-4-0613": ModelPricing(30.00, 60.00),
    "gpt-4-0125-preview": ModelPricing(10.00, 30.00),
    # GPT-3.5
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-0125": ModelPricing(0.50, 1.50),
    # Reasoning models (o-series)
    "o1-preview": ModelPricing(15.00, 60.00, reasoning_price_per_million=60.00),
    "o1-mini": ModelPricing(3.00, 12.00, reasoning_price_per_million=12.00),
    "o3-mini": ModelPricing(1.10, 4.40, reasoning_price_per_million=4.40),
    "o4-mini": ModelPricing(1.10, 4.40, reasoning_price_per_million=4.40),
    # GPT-5 series (estimated)
    "gpt-5": ModelPricing(2.00, 8.00, reasoning_price_per_million=8.00),
    "gpt-5-mini": ModelPricing(0.20, 0.80),
    "gpt-5-chat-latest": ModelPricing(2.00, 8.00),
}

# Anthropic Pricing (2025)
# Source: https://www.anthropic.com/pricing
ANTHROPIC_PRICING = {
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": ModelPricing(
        3.00,
        15.00,
        cache_write_price_per_million=3.75,  # 1.25x for 5-min cache
        cache_read_price_per_million=0.30,  # 0.1x
    ),
    "claude-3-5-sonnet-20240620": ModelPricing(
        3.00,
        15.00,
        cache_write_price_per_million=3.75,
        cache_read_price_per_million=0.30,
    ),
    # Claude 3.7 Sonnet
    "claude-3-7-sonnet": ModelPricing(
        3.00,
        15.00,
        cache_write_price_per_million=3.75,
        cache_read_price_per_million=0.30,
    ),
    # Claude Opus 4
    "claude-opus-4": ModelPricing(
        15.00,
        75.00,
        cache_write_price_per_million=18.75,
        cache_read_price_per_million=1.50,
    ),
    "claude-opus-4-1": ModelPricing(
        15.00,
        75.00,
        cache_write_price_per_million=18.75,
        cache_read_price_per_million=1.50,
        thinking_price_per_million=75.00,  # Same as output for Claude 4.1+
    ),
    # Claude 3 Opus
    "claude-3-opus-20240229": ModelPricing(
        15.00,
        75.00,
        cache_write_price_per_million=18.75,
        cache_read_price_per_million=1.50,
    ),
    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": ModelPricing(
        3.00,
        15.00,
        cache_write_price_per_million=3.75,
        cache_read_price_per_million=0.30,
    ),
    # Claude 3 Haiku
    "claude-3-haiku-20240307": ModelPricing(
        0.25,
        1.25,
        cache_write_price_per_million=0.30,
        cache_read_price_per_million=0.03,
    ),
}

# Together.ai Pricing (2025)
# Note: Together.ai has many models with varying prices
# These are common examples. Check https://www.together.ai/pricing for full list
TOGETHER_PRICING = {
    # Llama models
    "meta-llama/Llama-3.1-8B-Instruct": ModelPricing(0.18, 0.18),
    "meta-llama/Llama-3.1-70B-Instruct": ModelPricing(0.88, 0.88),
    "meta-llama/Llama-3.1-405B-Instruct": ModelPricing(3.50, 3.50),
    "meta-llama/Llama-3.2-3B-Instruct": ModelPricing(0.06, 0.06),
    "meta-llama/Llama-3.3-70B-Instruct": ModelPricing(0.88, 0.88),
    # Qwen models
    "Qwen/Qwen2.5-7B-Instruct": ModelPricing(0.18, 0.18),
    "Qwen/Qwen2.5-72B-Instruct": ModelPricing(1.20, 1.20),
    # DeepSeek
    "deepseek-ai/DeepSeek-R1": ModelPricing(0.55, 2.19),
    "deepseek-ai/DeepSeek-V3": ModelPricing(0.27, 1.10),
    # Lite endpoints (lower cost)
    "meta-llama/Llama-3-8B-Lite": ModelPricing(0.10, 0.10),
}


def calculate_cost(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    thinking_tokens: int = 0,
    model_name: Optional[str] = None,
    pricing: Optional[ModelPricing] = None,
) -> float:
    """Calculate the cost of an API call.

    Args:
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.
        reasoning_tokens: Number of reasoning tokens (OpenAI o-series).
        cache_creation_tokens: Number of tokens for cache creation (Anthropic).
        cache_read_tokens: Number of tokens read from cache (Anthropic).
        thinking_tokens: Number of thinking tokens (Anthropic Claude 4.1+).
        model_name: Name of the model (used to look up pricing).
        pricing: Custom pricing information (overrides model_name lookup).

    Returns:
        Total cost in USD.
    """
    if pricing is None and model_name:
        # Try to find pricing for the model
        pricing = (
            OPENAI_PRICING.get(model_name)
            or ANTHROPIC_PRICING.get(model_name)
            or TOGETHER_PRICING.get(model_name)
        )

    if pricing is None:
        return 0.0

    cost = 0.0

    # Input tokens
    cost += (prompt_tokens / 1_000_000) * pricing.input_price_per_million

    # Output tokens
    cost += (completion_tokens / 1_000_000) * pricing.output_price_per_million

    # Reasoning tokens (OpenAI)
    if reasoning_tokens and pricing.reasoning_price_per_million:
        cost += (reasoning_tokens / 1_000_000) * pricing.reasoning_price_per_million

    # Cache creation (Anthropic)
    if cache_creation_tokens and pricing.cache_write_price_per_million:
        cost += (
            cache_creation_tokens / 1_000_000
        ) * pricing.cache_write_price_per_million

    # Cache reads (Anthropic)
    if cache_read_tokens and pricing.cache_read_price_per_million:
        cost += (cache_read_tokens / 1_000_000) * pricing.cache_read_price_per_million

    # Thinking tokens (Anthropic)
    if thinking_tokens and pricing.thinking_price_per_million:
        cost += (thinking_tokens / 1_000_000) * pricing.thinking_price_per_million

    return cost


def get_model_pricing(model_name: str) -> Optional[ModelPricing]:
    """Get pricing information for a model.

    Args:
        model_name: Name of the model.

    Returns:
        ModelPricing object if found, None otherwise.
    """
    return (
        OPENAI_PRICING.get(model_name)
        or ANTHROPIC_PRICING.get(model_name)
        or TOGETHER_PRICING.get(model_name)
    )
