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
from typing import Any, Optional, Union, cast

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from oumi.core.analyze.dataset_analyzer import (
    ConversationAnalysisResult,
    MessageAnalysisResult,
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

    def analyze_sample(
        self,
        conversation: Conversation,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    ) -> tuple[list[MessageAnalysisResult], ConversationAnalysisResult]:
        """Analyze a conversation sample and return comprehensive length metrics.

        1. Analyzes each message individually for message-level metrics
        2. Computes conversation-level metrics by:
           - Aggregating message-level char, word, and sentence counts
           - Using dataset tokenization for conversation-level token count

        Args:
            conversation: The conversation object to analyze
            tokenizer: Optional tokenizer to use for token counting

        Returns:
            Tuple containing:
            - List of MessageAnalysisResult objects for each message
            - ConversationAnalysisResult for the conversation as a whole
        """
        # Step 1: Compute message-level metrics
        message_results = self.compute_message_metrics(conversation)

        # Step 2: Compute conversation-level metrics using message results
        conversation_result = self.compute_conversation_metrics(
            conversation, message_results
        )

        # Return individual components
        return message_results, conversation_result

    def compute_message_metrics(
        self, conversation: Conversation
    ) -> list[MessageAnalysisResult]:
        """Compute message-level length metrics for each message in the conversation.

        Args:
            conversation: The conversation object to analyze
            tokenizer: Optional tokenizer to use for token counting

        Returns:
            List of MessageAnalysisResult objects, one for each message
        """
        message_results = []
        for msg_idx, message in enumerate(conversation.messages):
            # Get text content for this message
            if isinstance(message.content, str):
                text_content = message.content
            else:
                # For multimodal content, extract text only
                text_content = message.compute_flattened_text_content()

            # Compute metrics for this message
            message_metrics = self.compute_length_metrics(text_content)

            # Create MessageAnalysisResult
            message_result = MessageAnalysisResult(
                message_index=msg_idx,
                role=message.role.value,
                message_id=message.id or f"msg_{msg_idx}",
                text_content=text_content,
                analyzer_metrics=message_metrics,
            )
            message_results.append(message_result)

        return message_results

    def compute_conversation_metrics(
        self,
        conversation: Conversation,
        message_results: Optional[list[MessageAnalysisResult]] = None,
    ) -> ConversationAnalysisResult:
        """Compute conversation-level length metrics for the entire conversation.

        Args:
            conversation: The conversation object to analyze
            tokenizer: Optional tokenizer to use for token counting
            message_results: Optional pre-computed message results for aggregation

        Returns:
            ConversationAnalysisResult containing conversation-level metrics
        """
        # For char, word, and sentence counts, aggregate message-level metrics
        conversation_metrics = {}

        # Aggregate message-level metrics for non-token metrics
        if self.char_count:
            if message_results is not None:
                # Use pre-computed message results
                total_chars = sum(
                    msg.analyzer_metrics.get("char_count", 0) for msg in message_results
                )
                conversation_metrics["char_count"] = total_chars

        if self.word_count:
            if message_results is not None:
                # Use pre-computed message results
                total_words = sum(
                    msg.analyzer_metrics.get("word_count", 0) for msg in message_results
                )
                conversation_metrics["word_count"] = total_words

        if self.sentence_count:
            if message_results is not None:
                # Use pre-computed message results
                total_sentences = sum(
                    msg.analyzer_metrics.get("sentence_count", 0)
                    for msg in message_results
                )
                conversation_metrics["sentence_count"] = total_sentences

        # For token count, tokenize the full rendered conversation using the tokenizer's
        # chat template for full control (independent of dataset internals).
        if self.token_count:
            tokenizer_to_use = self.tokenizer
            if tokenizer_to_use is not None:
                # First render the conversation to a prompt string via chat template
                prompt_text = tokenizer_to_use.apply_chat_template(
                    conversation,  # type: ignore
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prompt_text = cast(str, prompt_text)
                # Then encode with explicit controls
                conv_tokens = tokenizer_to_use.encode(
                    prompt_text,
                    add_special_tokens=self.include_special_tokens,
                )
                conversation_metrics["token_count"] = len(conv_tokens)

        # Create ConversationAnalysisResult
        return ConversationAnalysisResult(
            analyzer_metrics=conversation_metrics,
        )

    def compute_length_metrics(self, text_content: str) -> dict[str, Any]:
        """Compute length metrics for a single text content.

        This is a helper function that can be used by both message-level and
        conversation-level analysis.

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
            # Use instance tokenizer only
            tokenizer_to_use = self.tokenizer
            if tokenizer_to_use is not None:
                # Use tokenizer for accurate token count
                tokens = tokenizer_to_use.encode(
                    text_content, add_special_tokens=self.include_special_tokens
                )
                metrics["token_count"] = len(tokens)

        return metrics
