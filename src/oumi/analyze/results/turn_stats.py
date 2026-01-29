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

"""Turn statistics result model."""

from pydantic import BaseModel, Field


class TurnStatsMetrics(BaseModel):
    """Result model for turn statistics analysis of conversations.

    Contains metrics about conversation structure including turn counts,
    per-role statistics, and length ratios.

    Example:
        >>> result = TurnStatsMetrics(
        ...     num_turns=4,
        ...     num_user_turns=2,
        ...     num_assistant_turns=2,
        ...     has_system_message=False,
        ...     avg_user_chars=50.0,
        ...     avg_assistant_chars=150.0,
        ...     response_ratio=3.0,
        ...     first_turn_role="user",
        ...     last_turn_role="assistant",
        ... )
        >>> print(result.response_ratio)
        3.0
    """

    # Turn counts
    num_turns: int = Field(
        description="Total number of turns (messages) in the conversation"
    )
    num_user_turns: int = Field(
        description="Number of user turns in the conversation"
    )
    num_assistant_turns: int = Field(
        description="Number of assistant turns in the conversation"
    )
    has_system_message: bool = Field(
        description="Whether the conversation has a system message"
    )

    # Length statistics
    avg_user_chars: float = Field(
        description="Average character length of user messages"
    )
    avg_assistant_chars: float = Field(
        description="Average character length of assistant messages"
    )
    total_user_chars: int = Field(
        default=0,
        description="Total character length of all user messages"
    )
    total_assistant_chars: int = Field(
        default=0,
        description="Total character length of all assistant messages"
    )

    # Ratios
    response_ratio: float = Field(
        description="Ratio of average assistant to user message length (assistant/user)"
    )
    assistant_turn_ratio: float = Field(
        default=0.0,
        description="Ratio of assistant turns to total non-system turns"
    )

    # Turn order
    first_turn_role: str = Field(
        description="Role of the first message in the conversation"
    )
    last_turn_role: str = Field(
        description="Role of the last message in the conversation"
    )
