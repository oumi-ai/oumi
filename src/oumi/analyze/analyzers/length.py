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

"""Length analyzer implementation."""

from typing import Any

from oumi.analyze.base import ConversationAnalyzer
from oumi.analyze.results.length import LengthMetrics
from oumi.core.types.conversation import Conversation, Role


class LengthAnalyzer(ConversationAnalyzer[LengthMetrics]):
    """Analyzer for computing length metrics of conversations.

    Computes character counts, word counts, and optionally token counts
    for conversations. Provides both conversation-level totals and
    per-message breakdowns.

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
        >>> print(f"Total words: {result.total_words}")
        Total words: 9

    Args:
        count_tokens: Whether to compute token counts (requires tokenizer).
        tokenizer: Tokenizer instance for token counting. If None and
            count_tokens is True, will attempt to use tiktoken.
        tiktoken_encoding: Tiktoken encoding name to use if no tokenizer
            is provided. Defaults to "cl100k_base" (GPT-4 encoding).
        compute_role_stats: Whether to compute per-role word counts.
    """

    def __init__(
        self,
        count_tokens: bool = False,
        tokenizer: Any | None = None,
        tiktoken_encoding: str = "cl100k_base",
        compute_role_stats: bool = True,
    ):
        """Initialize the length analyzer.

        Args:
            count_tokens: Whether to compute token counts.
            tokenizer: Optional tokenizer for token counting.
            tiktoken_encoding: Tiktoken encoding name if using tiktoken.
            compute_role_stats: Whether to compute per-role statistics.
        """
        self.count_tokens = count_tokens
        self.tokenizer = tokenizer
        self.tiktoken_encoding = tiktoken_encoding
        self.compute_role_stats = compute_role_stats
        self._tiktoken_encoder = None

        # Initialize tiktoken if needed
        if count_tokens and tokenizer is None:
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

    def _count_tokens(self, text: str) -> int | None:
        """Count tokens in text.

        Args:
            text: Text to tokenize.

        Returns:
            Token count or None if tokenizer not available.
        """
        if not self.count_tokens:
            return None

        if self.tokenizer is not None:
            # Use provided tokenizer (HuggingFace style)
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception:
                return None

        if self._tiktoken_encoder is not None:
            # Use tiktoken
            try:
                tokens = self._tiktoken_encoder.encode(text)
                return len(tokens)
            except Exception:
                return None

        return None

    def analyze(self, conversation: Conversation) -> LengthMetrics:
        """Analyze length metrics for a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            LengthMetrics containing character, word, and token counts.
        """
        message_char_counts: list[int] = []
        message_word_counts: list[int] = []
        message_token_counts: list[int] = [] if self.count_tokens else []

        # Role-specific accumulators
        role_word_counts: dict[Role, int] = {
            Role.USER: 0,
            Role.ASSISTANT: 0,
            Role.SYSTEM: 0,
        }

        for message in conversation.messages:
            text = self.get_text_content(message)

            # Character count
            char_count = len(text)
            message_char_counts.append(char_count)

            # Word count
            words = text.split()
            word_count = len(words)
            message_word_counts.append(word_count)

            # Token count (if enabled)
            if self.count_tokens:
                token_count = self._count_tokens(text)
                if token_count is not None:
                    message_token_counts.append(token_count)

            # Role-specific counts
            if self.compute_role_stats and message.role in role_word_counts:
                role_word_counts[message.role] += word_count

        # Compute totals
        total_chars = sum(message_char_counts)
        total_words = sum(message_word_counts)
        total_tokens = sum(message_token_counts) if message_token_counts else None

        num_messages = len(conversation.messages)

        # Compute averages
        avg_chars = total_chars / num_messages if num_messages > 0 else 0.0
        avg_words = total_words / num_messages if num_messages > 0 else 0.0
        avg_tokens = (
            total_tokens / num_messages
            if total_tokens is not None and num_messages > 0
            else None
        )

        return LengthMetrics(
            total_chars=total_chars,
            total_words=total_words,
            total_tokens=total_tokens,
            avg_chars_per_message=avg_chars,
            avg_words_per_message=avg_words,
            avg_tokens_per_message=avg_tokens,
            message_char_counts=message_char_counts,
            message_word_counts=message_word_counts,
            message_token_counts=message_token_counts if message_token_counts else None,
            num_messages=num_messages,
            user_total_words=role_word_counts[Role.USER]
            if self.compute_role_stats
            else None,
            assistant_total_words=role_word_counts[Role.ASSISTANT]
            if self.compute_role_stats
            else None,
            system_total_words=role_word_counts[Role.SYSTEM]
            if self.compute_role_stats
            else None,
        )

    def analyze_text(self, text: str) -> LengthMetrics:
        """Analyze length metrics for a single text string.

        Convenience method for analyzing text without creating a Conversation.

        Args:
            text: The text to analyze.

        Returns:
            LengthMetrics for the text (treated as a single message).
        """
        char_count = len(text)
        word_count = len(text.split())
        token_count = self._count_tokens(text) if self.count_tokens else None

        return LengthMetrics(
            total_chars=char_count,
            total_words=word_count,
            total_tokens=token_count,
            avg_chars_per_message=float(char_count),
            avg_words_per_message=float(word_count),
            avg_tokens_per_message=float(token_count) if token_count else None,
            message_char_counts=[char_count],
            message_word_counts=[word_count],
            message_token_counts=[token_count] if token_count else None,
            num_messages=1,
            user_total_words=None,
            assistant_total_words=None,
            system_total_words=None,
        )
