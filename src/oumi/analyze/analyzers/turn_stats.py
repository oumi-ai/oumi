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

"""Turn statistics analyzer implementation and result model."""

from pydantic import BaseModel, Field

from oumi.analyze.base import ConversationAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.core.types.conversation import Conversation, Role

__all__ = ["TurnStatsMetrics", "TurnStatsAnalyzer"]


class TurnStatsMetrics(BaseModel):
    """Result model for turn statistics analysis of conversations.

    Contains metrics about conversation structure including turn counts
    and per-role statistics.

    Example:
        >>> result = TurnStatsMetrics(
        ...     num_turns=4,
        ...     num_user_turns=2,
        ...     num_assistant_turns=2,
        ...     has_system_message=False,
        ...     first_turn_role="user",
        ...     last_turn_role="assistant",
        ... )
        >>> print(result.num_turns)
        4
    """

    # Turn counts
    num_turns: int = Field(
        description="Total number of turns (messages) in the conversation"
    )
    num_user_turns: int = Field(description="Number of user turns in the conversation")
    num_assistant_turns: int = Field(
        description="Number of assistant turns in the conversation"
    )
    num_tool_turns: int = Field(
        default=0, description="Number of tool turns in the conversation"
    )
    has_system_message: bool = Field(
        description="Whether the conversation has a system message"
    )

    # Turn order
    first_turn_role: str | None = Field(
        default=None,
        description="Role of the first message in the conversation, or None if empty",
    )
    last_turn_role: str | None = Field(
        default=None,
        description="Role of the last message in the conversation, or None if empty",
    )


@register_sample_analyzer("typed_turn_stats")
class TurnStatsAnalyzer(ConversationAnalyzer[TurnStatsMetrics]):
    """Analyzer for computing turn statistics of conversations.

    Computes turn counts and per-role statistics to help understand
    conversation structure and balance.

    Example:
        >>> from oumi.analyze.analyzers.turn_stats import TurnStatsAnalyzer
        >>> from oumi.core.types.conversation import Conversation, Message, Role
        >>>
        >>> analyzer = TurnStatsAnalyzer()
        >>> conversation = Conversation(messages=[
        ...     Message(role=Role.USER, content="What is Python?"),
        ...     Message(
        ...         role=Role.ASSISTANT,
        ...         content="Python is a programming language.",
        ...     ),
        ... ])
        >>> result = analyzer.analyze(conversation)
        >>> print(f"Turns: {result.num_turns}")
        Turns: 2
    """

    def analyze(self, conversation: Conversation) -> TurnStatsMetrics:
        """Analyze turn statistics for a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            TurnStatsMetrics containing turn counts and statistics.
        """
        num_user = 0
        num_assistant = 0
        num_tool = 0
        has_system = False
        first_role: str | None = None
        last_role: str | None = None

        for i, message in enumerate(conversation.messages):
            role_value = message.role.value

            # Track first and last roles
            if i == 0:
                first_role = role_value
            last_role = role_value

            if message.role == Role.USER:
                num_user += 1
            elif message.role == Role.ASSISTANT:
                num_assistant += 1
            elif message.role == Role.TOOL:
                num_tool += 1
            elif message.role == Role.SYSTEM:
                has_system = True

        return TurnStatsMetrics(
            num_turns=len(conversation.messages),
            num_user_turns=num_user,
            num_assistant_turns=num_assistant,
            num_tool_turns=num_tool,
            has_system_message=has_system,
            first_turn_role=first_role,
            last_turn_role=last_role,
        )
