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

"""Length analyzer implementation and result model."""

from typing import Any

from pydantic import BaseModel, Field

from oumi.analyze.base import ConversationAnalyzer
from oumi.core.types.conversation import Conversation, Role

__all__ = ["LengthMetrics", "LengthAnalyzer"]


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
    total_tokens: int = Field(description="Total number of tokens across all messages")

    # Averages
    avg_tokens_per_message: float = Field(description="Average tokens per message")

    # Per-message breakdowns
    message_token_counts: list[int] = Field(
        description="Token count for each message in order"
    )

    # Message count
    num_messages: int = Field(description="Number of messages in the conversation")

    # Role-specific stats (optional)
    user_total_tokens: int | None = Field(
        default=None, description="Total tokens in user messages"
    )
    assistant_total_tokens: int | None = Field(
        default=None, description="Total tokens in assistant messages"
    )
    system_total_tokens: int | None = Field(
        default=None, description="Total tokens in system messages"
    )


class LengthAnalyzer(ConversationAnalyzer[LengthMetrics]):
    """Analyzer for computing token length metrics of conversations.

    Computes token counts for conversations using either a provided tokenizer
    or tiktoken. Provides both conversation-level totals and per-message breakdowns.

    Example:
        >>> from oumi.analyze import LengthAnalyzer
        >>> from oumi.core.types.conversation import Conversation, Message, Role
        >>>
        >>> analyzer = LengthAnalyzer()
        >>> conversation = Conversation(messages=[
        ...     Message(role=Role.USER, content="Hello, how are you?"),
        ...     Message(role=Role.ASSISTANT, content="I'm doing well, thanks!"),
        ... ])
        >>> result = analyzer.analyze(conversation)
        >>> print(f"Total tokens: {result.total_tokens}")
        Total tokens: 12

    Args:
        tokenizer: Tokenizer instance for token counting. If None,
            will use tiktoken with the specified encoding.
        tiktoken_encoding: Tiktoken encoding name to use if no tokenizer
            is provided. Defaults to "cl100k_base" (GPT-4 encoding).
        compute_role_stats: Whether to compute per-role token counts.
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        tiktoken_encoding: str = "cl100k_base",
        compute_role_stats: bool = True,
    ):
        """Initialize the length analyzer.

        Args:
            tokenizer: Optional tokenizer for token counting.
            tiktoken_encoding: Tiktoken encoding name if using tiktoken.
            compute_role_stats: Whether to compute per-role statistics.
        """
        self.tokenizer = tokenizer
        self.tiktoken_encoding = tiktoken_encoding
        self.compute_role_stats = compute_role_stats
        self._tiktoken_encoder = None

        # Initialize tiktoken if no tokenizer provided
        if tokenizer is None:
            self._tiktoken_encoder = self._load_tiktoken_encoder()

    def _load_tiktoken_encoder(self) -> Any | None:
        """Load tiktoken encoder lazily.

        Returns:
            Tiktoken encoder or None if not available.
        """
        try:
            import tiktoken

            return tiktoken.get_encoding(self.tiktoken_encoding)
        except ImportError:
            return None
        except Exception:
            return None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize.

        Returns:
            Token count (0 if tokenizer not available).
        """
        if self.tokenizer is not None:
            # Use provided tokenizer (HuggingFace style)
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception:
                return 0

        if self._tiktoken_encoder is not None:
            # Use tiktoken
            try:
                tokens = self._tiktoken_encoder.encode(text)
                return len(tokens)
            except Exception:
                return 0

        return 0

    def analyze(self, conversation: Conversation) -> LengthMetrics:
        """Analyze token length metrics for a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            LengthMetrics containing token counts.
        """
        message_token_counts: list[int] = []

        # Role-specific accumulators
        role_token_counts: dict[Role, int] = {
            Role.USER: 0,
            Role.ASSISTANT: 0,
            Role.SYSTEM: 0,
        }

        for message in conversation.messages:
            text = self.get_text_content(message)

            # Token count
            token_count = self._count_tokens(text)
            message_token_counts.append(token_count)

            # Role-specific counts
            if self.compute_role_stats and message.role in role_token_counts:
                role_token_counts[message.role] += token_count

        # Compute totals
        total_tokens = sum(message_token_counts)
        num_messages = len(conversation.messages)

        # Compute average
        avg_tokens = total_tokens / num_messages if num_messages > 0 else 0.0

        return LengthMetrics(
            total_tokens=total_tokens,
            avg_tokens_per_message=avg_tokens,
            message_token_counts=message_token_counts,
            num_messages=num_messages,
            user_total_tokens=role_token_counts[Role.USER]
            if self.compute_role_stats
            else None,
            assistant_total_tokens=role_token_counts[Role.ASSISTANT]
            if self.compute_role_stats
            else None,
            system_total_tokens=role_token_counts[Role.SYSTEM]
            if self.compute_role_stats
            else None,
        )

    def analyze_text(self, text: str) -> LengthMetrics:
        """Analyze token length metrics for a single text string.

        Convenience method for analyzing text without creating a Conversation.

        Args:
            text: The text to analyze.

        Returns:
            LengthMetrics for the text (treated as a single message).
        """
        token_count = self._count_tokens(text)

        return LengthMetrics(
            total_tokens=token_count,
            avg_tokens_per_message=float(token_count),
            message_token_counts=[token_count],
            num_messages=1,
            user_total_tokens=None,
            assistant_total_tokens=None,
            system_total_tokens=None,
        )
