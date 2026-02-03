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
from oumi.core.registry import register_sample_analyzer
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

    # Role-specific stats
    user_total_tokens: int = Field(
        default=0, description="Total tokens in user messages"
    )
    assistant_total_tokens: int = Field(
        default=0, description="Total tokens in assistant messages"
    )
    system_total_tokens: int = Field(
        default=0, description="Total tokens in system messages"
    )
    tool_total_tokens: int = Field(
        default=0, description="Total tokens in tool messages"
    )


@register_sample_analyzer("typed_length")
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
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        tiktoken_encoding: str = "cl100k_base",
    ):
        """Initialize the length analyzer.

        Args:
            tokenizer: Optional custom tokenizer (e.g., HuggingFace) for model-specific
                token counting. Must have an `encode(text) -> list` method.
            tiktoken_encoding: Tiktoken encoding name to use as fallback when no
                custom tokenizer is provided.
        """
        # User-provided tokenizer takes priority (for model-specific token counts)
        self.tokenizer = tokenizer
        self.tiktoken_encoding = tiktoken_encoding

        # Fallback: use tiktoken (fast, widely available) when no custom tokenizer
        self._tiktoken_encoder = None
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

        Priority:
        1. Custom tokenizer (if provided) - for model-specific token counts
        2. Tiktoken (fallback) - fast default using OpenAI's tokenizer

        Args:
            text: Text to tokenize.

        Returns:
            Token count. Returns 0 if encoding fails.

        Raises:
            RuntimeError: If no tokenizer is available (neither custom nor tiktoken).
        """
        # Priority 1: Custom tokenizer (e.g., HuggingFace model tokenizer)
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception:
                return 0

        # Priority 2: Tiktoken fallback (fast, widely available)
        if self._tiktoken_encoder is not None:
            try:
                tokens = self._tiktoken_encoder.encode(text)
                return len(tokens)
            except Exception:
                return 0

        # No tokenizer available - fail explicitly rather than silently returning 0
        raise RuntimeError(
            "No tokenizer available. Either provide a custom tokenizer or "
            "install tiktoken: pip install tiktoken"
        )

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
            Role.TOOL: 0,
        }

        for message in conversation.messages:
            text = self.get_text_content(message)

            # Token count
            token_count = self._count_tokens(text)
            message_token_counts.append(token_count)

            # Role-specific counts
            if message.role in role_token_counts:
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
            user_total_tokens=role_token_counts[Role.USER],
            assistant_total_tokens=role_token_counts[Role.ASSISTANT],
            system_total_tokens=role_token_counts[Role.SYSTEM],
            tool_total_tokens=role_token_counts[Role.TOOL],
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
        )
