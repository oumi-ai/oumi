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

"""Data structures for tracking token usage and costs."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenUsage:
    """Token usage information from an API response."""

    prompt_tokens: int = 0
    """Number of tokens in the prompt."""

    completion_tokens: int = 0
    """Number of tokens in the generated completion."""

    total_tokens: int = 0
    """Total number of tokens used (prompt + completion)."""

    # OpenAI reasoning models
    reasoning_tokens: Optional[int] = None
    """Number of reasoning tokens used by the model (o1, o3, o4 models)."""

    # Anthropic prompt caching
    cache_creation_input_tokens: Optional[int] = None
    """Number of tokens used to create the cache (Anthropic)."""

    cache_read_input_tokens: Optional[int] = None
    """Number of tokens read from the cache (Anthropic)."""

    # Anthropic thinking tokens
    thinking_tokens: Optional[int] = None
    """Number of thinking tokens used by the model (Claude 4.1+)."""

    # Audio tokens (OpenAI multimodal)
    audio_tokens: Optional[int] = None
    """Number of audio tokens used."""

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=(
                (self.reasoning_tokens or 0) + (other.reasoning_tokens or 0)
                if self.reasoning_tokens or other.reasoning_tokens
                else None
            ),
            cache_creation_input_tokens=(
                (self.cache_creation_input_tokens or 0)
                + (other.cache_creation_input_tokens or 0)
                if self.cache_creation_input_tokens or other.cache_creation_input_tokens
                else None
            ),
            cache_read_input_tokens=(
                (self.cache_read_input_tokens or 0)
                + (other.cache_read_input_tokens or 0)
                if self.cache_read_input_tokens or other.cache_read_input_tokens
                else None
            ),
            thinking_tokens=(
                (self.thinking_tokens or 0) + (other.thinking_tokens or 0)
                if self.thinking_tokens or other.thinking_tokens
                else None
            ),
            audio_tokens=(
                (self.audio_tokens or 0) + (other.audio_tokens or 0)
                if self.audio_tokens or other.audio_tokens
                else None
            ),
        )


@dataclass
class CostEstimate:
    """Cost estimate for API usage."""

    input_cost: float = 0.0
    """Cost for input tokens (in USD)."""

    output_cost: float = 0.0
    """Cost for output tokens (in USD)."""

    reasoning_cost: float = 0.0
    """Cost for reasoning tokens (in USD)."""

    cache_write_cost: float = 0.0
    """Cost for writing to cache (in USD)."""

    cache_read_cost: float = 0.0
    """Cost for reading from cache (in USD)."""

    thinking_cost: float = 0.0
    """Cost for thinking tokens (in USD)."""

    total_cost: float = 0.0
    """Total cost (in USD)."""

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        """Add two CostEstimate objects together."""
        return CostEstimate(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            reasoning_cost=self.reasoning_cost + other.reasoning_cost,
            cache_write_cost=self.cache_write_cost + other.cache_write_cost,
            cache_read_cost=self.cache_read_cost + other.cache_read_cost,
            thinking_cost=self.thinking_cost + other.thinking_cost,
            total_cost=self.total_cost + other.total_cost,
        )


@dataclass
class UsageInfo:
    """Combined usage and cost information."""

    token_usage: TokenUsage
    """Token usage information."""

    cost_estimate: Optional[CostEstimate] = None
    """Cost estimate (if pricing information is available)."""

    model_name: Optional[str] = None
    """Name of the model used."""

    def __add__(self, other: "UsageInfo") -> "UsageInfo":
        """Add two UsageInfo objects together."""
        return UsageInfo(
            token_usage=self.token_usage + other.token_usage,
            cost_estimate=(
                self.cost_estimate + other.cost_estimate
                if self.cost_estimate and other.cost_estimate
                else self.cost_estimate or other.cost_estimate
            ),
            model_name=self.model_name or other.model_name,
        )
