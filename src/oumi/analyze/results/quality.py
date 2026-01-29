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

"""Data quality metrics result model."""

from pydantic import BaseModel, Field


class DataQualityMetrics(BaseModel):
    """Result model for data quality analysis of conversations.

    Contains various quality checks that can be run without an LLM,
    making them fast and cheap to compute on large datasets.

    Example:
        >>> result = DataQualityMetrics(
        ...     has_alternating_turns=True,
        ...     has_empty_turns=False,
        ...     has_invalid_values=False,
        ...     fits_4k_context=True,
        ...     appears_truncated=False,
        ...     has_policy_refusal=False,
        ...     has_unbalanced_tags=False,
        ... )
        >>> print(result.has_alternating_turns)
        True
    """

    # Turn pattern checks
    has_alternating_turns: bool = Field(
        description="Whether conversation has proper alternating user-assistant turns"
    )
    turn_sequence: str = Field(
        description="Sequence of roles (e.g., 'user,assistant,user,assistant')"
    )
    num_consecutive_same_role: int = Field(
        default=0,
        description="Number of consecutive messages from the same role (excluding system)"
    )

    # Empty content checks
    has_empty_turns: bool = Field(
        description="Whether any message has empty content"
    )
    empty_turn_count: int = Field(
        default=0,
        description="Number of messages with empty or whitespace-only content"
    )
    empty_turn_indices: list[int] = Field(
        default_factory=list,
        description="Indices of empty messages"
    )

    # Invalid value checks
    has_invalid_values: bool = Field(
        description="Whether content contains invalid serialized values like 'NaN', 'null', 'None'"
    )
    invalid_value_patterns: list[str] = Field(
        default_factory=list,
        description="List of invalid patterns found"
    )

    # Context length checks
    estimated_tokens: int = Field(
        description="Estimated token count (rough word-based estimate)"
    )
    fits_4k_context: bool = Field(
        description="Whether conversation likely fits in 4K token context"
    )
    fits_8k_context: bool = Field(
        description="Whether conversation likely fits in 8K token context"
    )

    # Truncation checks
    appears_truncated: bool = Field(
        description="Whether the last message appears to be abruptly truncated"
    )
    ends_mid_sentence: bool = Field(
        default=False,
        description="Whether the conversation ends without proper punctuation"
    )
    truncation_reason: str | None = Field(
        default=None,
        description="Reason for truncation detection (if applicable)"
    )

    # Policy refusal checks
    has_policy_refusal: bool = Field(
        description="Whether any assistant message appears to be a policy refusal"
    )
    refusal_count: int = Field(
        default=0,
        description="Number of messages containing refusal patterns"
    )
    refusal_phrases: list[str] = Field(
        default_factory=list,
        description="List of refusal phrases detected"
    )

    # Think token / tag checks
    has_think_tags: bool = Field(
        description="Whether any message contains thinking/reasoning tags"
    )
    has_unbalanced_tags: bool = Field(
        description="Whether there are unmatched opening/closing tags"
    )
    unmatched_tags: list[str] = Field(
        default_factory=list,
        description="List of unmatched tags found"
    )

    # Overall quality flag
    passes_basic_quality: bool = Field(
        description="Whether the conversation passes all basic quality checks"
    )
    quality_issues: list[str] = Field(
        default_factory=list,
        description="List of quality issues found"
    )
