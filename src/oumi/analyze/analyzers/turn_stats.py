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

"""Turn statistics analyzer implementation."""

from oumi.analyze.base import ConversationAnalyzer
from oumi.analyze.results.turn_stats import TurnStatsMetrics
from oumi.core.types.conversation import Conversation, Role


class TurnStatsAnalyzer(ConversationAnalyzer[TurnStatsMetrics]):
    """Analyzer for computing turn statistics of conversations.

    Computes turn counts, per-role statistics, and length ratios to help
    understand conversation structure and balance.

    Example:
        >>> from oumi.analyze.analyzers.turn_stats import TurnStatsAnalyzer
        >>> from oumi.core.types.conversation import Conversation, Message, Role
        >>>
        >>> analyzer = TurnStatsAnalyzer()
        >>> conversation = Conversation(messages=[
        ...     Message(role=Role.USER, content="What is Python?"),
        ...     Message(role=Role.ASSISTANT, content="Python is a programming language."),
        ... ])
        >>> result = analyzer.analyze(conversation)
        >>> print(f"Response ratio: {result.response_ratio:.2f}")
        Response ratio: 2.13

    Args:
        include_system_in_counts: Whether to include system messages in turn counts.
    """

    def __init__(
        self,
        include_system_in_counts: bool = False,
    ):
        """Initialize the turn statistics analyzer.

        Args:
            include_system_in_counts: Whether to include system messages in total
                turn counts. Default False (only count user/assistant turns).
        """
        self.include_system_in_counts = include_system_in_counts

    def analyze(self, conversation: Conversation) -> TurnStatsMetrics:
        """Analyze turn statistics for a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            TurnStatsMetrics containing turn counts and statistics.
        """
        user_lengths: list[int] = []
        assistant_lengths: list[int] = []
        has_system = False
        first_role = ""
        last_role = ""

        for i, message in enumerate(conversation.messages):
            text = self.get_text_content(message)
            role_value = (
                message.role.value
                if hasattr(message.role, "value")
                else str(message.role)
            )

            # Track first and last roles
            if i == 0:
                first_role = role_value
            last_role = role_value

            if message.role == Role.USER:
                user_lengths.append(len(text))
            elif message.role == Role.ASSISTANT:
                assistant_lengths.append(len(text))
            elif message.role == Role.SYSTEM:
                has_system = True

        # Calculate averages
        total_user_chars = sum(user_lengths)
        total_assistant_chars = sum(assistant_lengths)
        avg_user = total_user_chars / len(user_lengths) if user_lengths else 0.0
        avg_assistant = (
            total_assistant_chars / len(assistant_lengths) if assistant_lengths else 0.0
        )

        # Calculate response ratio (how much more verbose is the assistant)
        response_ratio = avg_assistant / avg_user if avg_user > 0 else 0.0

        # Calculate turn counts
        num_user = len(user_lengths)
        num_assistant = len(assistant_lengths)
        non_system_turns = num_user + num_assistant
        assistant_turn_ratio = (
            num_assistant / non_system_turns if non_system_turns > 0 else 0.0
        )

        # Total turns depends on configuration
        if self.include_system_in_counts:
            num_turns = len(conversation.messages)
        else:
            num_turns = non_system_turns

        return TurnStatsMetrics(
            num_turns=num_turns,
            num_user_turns=num_user,
            num_assistant_turns=num_assistant,
            has_system_message=has_system,
            avg_user_chars=avg_user,
            avg_assistant_chars=avg_assistant,
            total_user_chars=total_user_chars,
            total_assistant_chars=total_assistant_chars,
            response_ratio=response_ratio,
            assistant_turn_ratio=assistant_turn_ratio,
            first_turn_role=first_role,
            last_turn_role=last_role,
        )
