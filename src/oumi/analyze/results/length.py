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

"""Length metrics result model."""

from pydantic import BaseModel, Field


class LengthMetrics(BaseModel):
    """Result model for length analysis of conversations.

    Contains token counts at both the conversation level and per-message breakdown.

    Example:
        >>> result = LengthMetrics(
        ...     total_tokens=25,
        ...     avg_tokens_per_message=12.5,
        ...     message_token_counts=[10, 15],
        ...     num_messages=2,
        ... )
        >>> print(result.total_tokens)
        25
    """

    # Conversation-level totals
    total_tokens: int = Field(
        description="Total number of tokens across all messages"
    )

    # Averages
    avg_tokens_per_message: float = Field(
        description="Average tokens per message"
    )

    # Per-message breakdowns
    message_token_counts: list[int] = Field(
        description="Token count for each message in order"
    )

    # Message count
    num_messages: int = Field(
        description="Number of messages in the conversation"
    )

    # Role-specific stats (optional)
    user_total_tokens: int | None = Field(
        default=None,
        description="Total tokens in user messages"
    )
    assistant_total_tokens: int | None = Field(
        default=None,
        description="Total tokens in assistant messages"
    )
    system_total_tokens: int | None = Field(
        default=None,
        description="Total tokens in system messages"
    )
