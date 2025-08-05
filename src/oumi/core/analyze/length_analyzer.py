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

"""Length analyzer for text content."""

import re
from typing import Any, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oumi.core.analyze.dataset_analyzer import (
    ConversationAnalysisResult,
    MessageAnalysisResult,
    SampleAnalysisResult,
)
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.core.types.conversation import Conversation


@register_sample_analyzer("length")
class LengthAnalyzer(SampleAnalyzer):
    """Analyzer that computes various length metrics for text content."""

    def __init__(
        self,
        *,
        char_count: bool = True,
        word_count: bool = True,
        sentence_count: bool = True,
        token_count: bool = False,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        include_special_tokens: bool = True,
    ):
        """Initialize the length analyzer.

        Args:
            char_count: Whether to compute character count
            word_count: Whether to compute word count
            sentence_count: Whether to compute sentence count
            token_count: Whether to compute token count
            tokenizer: Tokenizer to use for token counting
                (required if token_count=True)
            include_special_tokens: Whether to include special tokens in token count.
                Defaults to True to match training tokenization. Set to False for raw
                content analysis only.
        """
        self.char_count = char_count
        self.word_count = word_count
        self.sentence_count = sentence_count
        self.token_count = token_count
        self.tokenizer = tokenizer
        self.include_special_tokens = include_special_tokens
        # Validate tokenizer requirements
        if self.token_count and tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when token_count=True. "
                "Set token_count=False or provide a tokenizer."
            )

    def compute_metrics(
        self,
        conversation: Conversation,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    ) -> SampleAnalysisResult:
        """Compute both message-level and conversation-level length metrics.

        This implementation:
        1. Analyzes each message individually for message-level metrics
        2. Renders the entire conversation as text for conversation-level metrics
           (which may give different results than aggregating message metrics)

        Args:
            conversation: The conversation object to analyze
            tokenizer: Optional tokenizer to use for token counting

        Returns:
            SampleAnalysisResult: Complete analysis result containing both
            message-level and conversation-level metrics for the conversation.
        """
        # Step 1: Compute message-level metrics
        message_results = []
        for msg_idx, message in enumerate(conversation.messages):
            # Get text content for this message
            if isinstance(message.content, str):
                text_content = message.content
            else:
                # For multimodal content, extract text only
                text_content = message.compute_flattened_text_content()

            # Compute metrics for this message
            message_metrics = self._compute_message_metrics(text_content, tokenizer)

            # Create MessageAnalysisResult
            message_result = MessageAnalysisResult(
                message_index=msg_idx,
                role=message.role.value,
                message_id=message.id or f"msg_{msg_idx}",
                text_content=text_content,
                analyzer_metrics=message_metrics,
            )
            message_results.append(message_result)

        # Step 2: Compute conversation-level metrics
        # Render the entire conversation as text
        conversation_text = self._render_conversation_as_text(conversation)

        # Compute conversation-level metrics
        conversation_metrics = self._compute_message_metrics(
            conversation_text, tokenizer
        )

        # Create ConversationAnalysisResult
        conversation_result = ConversationAnalysisResult(
            analyzer_metrics=conversation_metrics,
        )

        # Create and return SampleAnalysisResult
        return SampleAnalysisResult(
            conversation_id=conversation.conversation_id or "unknown",
            conversation_index=0,  # Single conversation
            messages=message_results,
            conversation=conversation_result,
        )

    def _compute_message_metrics(
        self,
        text_content: str,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    ) -> dict[str, Any]:
        """Compute length metrics for a single text content.

        Args:
            text_content: The text content to analyze
            tokenizer: Optional tokenizer to use for token counting

        Returns:
            Dictionary containing requested length metrics
        """
        metrics = {}

        if self.char_count:
            metrics["char_count"] = len(text_content)

        if self.word_count:
            # Simple word count - split on whitespace
            metrics["word_count"] = len(text_content.split())

        if self.sentence_count:
            # Simple sentence count - split on common sentence endings
            sentences = re.split(r"[.!?]+", text_content)
            # Filter out empty strings
            sentences = [s.strip() for s in sentences if s.strip()]
            metrics["sentence_count"] = len(sentences)

        if self.token_count:
            # Use provided tokenizer or fall back to instance tokenizer
            tokenizer_to_use = tokenizer or self.tokenizer
            if tokenizer_to_use is not None:
                # Use tokenizer for accurate token count
                tokens = tokenizer_to_use.encode(
                    text_content, add_special_tokens=self.include_special_tokens
                )
                metrics["token_count"] = len(tokens)

        return metrics

    def _render_conversation_as_text(self, conversation: Conversation) -> str:
        """Render a conversation as a single text string.

        Args:
            conversation: The conversation to render

        Returns:
            The conversation rendered as text with role prefixes
        """
        rendered_parts = []

        for message in conversation.messages:
            # Get the text content of the message
            if isinstance(message.content, str):
                text_content = message.content
            else:
                # For multimodal content, extract text only
                text_content = message.compute_flattened_text_content()

            # Add role prefix and message content
            role_prefix = f"{message.role.value}: "
            rendered_parts.append(role_prefix + text_content)

        # Join all messages with newlines
        return "\n".join(rendered_parts)
