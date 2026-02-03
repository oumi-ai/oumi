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

"""Base analyzer classes for the typed analyzer framework.

This module defines the base classes for different analyzer scopes:
- BaseAnalyzer: Common base for all analyzers (metadata methods)
- MessageAnalyzer: Analyzes individual messages
- ConversationAnalyzer: Analyzes complete conversations
- DatasetAnalyzer: Analyzes entire datasets (cross-sample operations)
- PreferenceAnalyzer: Analyzes preference pairs (for DPO data)

Each analyzer returns strongly-typed Pydantic models as results.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from oumi.core.types.conversation import ContentItem, Conversation, Message

# Type variable for analyzer results - must be a Pydantic BaseModel
TResult = TypeVar("TResult", bound=BaseModel)


class BaseAnalyzer(ABC, Generic[TResult]):
    """Base class for all analyzer types.

    Provides common metadata methods for inspecting the result type and schema
    of an analyzer, enabling introspection of available metrics.

    All concrete analyzer types (MessageAnalyzer, ConversationAnalyzer, etc.)
    inherit from this class.

    Attributes:
        analyzer_id: Optional custom identifier for this analyzer instance.
            If not set, the class name is used as the identifier.
    """

    analyzer_id: str | None = None

    @classmethod
    def get_scope(cls) -> str:
        """Get the scope of this analyzer.

        Returns:
            Scope string ('message', 'conversation', 'dataset', or 'preference').
        """
        return "unknown"

    @classmethod
    def _require_result_type(cls) -> type[BaseModel]:
        """Get the result type, raising if not available.

        Returns:
            The result type class (a BaseModel subclass).

        Raises:
            TypeError: If the analyzer doesn't have a valid result type.
        """
        result_type = cls._get_result_type()
        if result_type is None:
            raise TypeError(
                f"{cls.__name__} does not have a valid result type. "
                f"Ensure the class specifies a Pydantic BaseModel as the "
                f"generic type parameter, e.g., "
                f"`class {cls.__name__}(ConversationAnalyzer[YourResultModel])`"
            )
        return result_type

    @classmethod
    def get_result_schema(cls) -> dict:
        """Get the JSON schema for this analyzer's result model.

        This allows users to discover what metrics the analyzer produces
        before running analysis. Useful for documentation, UI generation,
        and config validation.

        Returns:
            JSON schema dictionary for the result model.

        Raises:
            TypeError: If the analyzer doesn't have a valid result type.
        """
        return cls._require_result_type().model_json_schema()

    @classmethod
    def get_metric_names(cls) -> list[str]:
        """Get the list of metric field names this analyzer produces.

        Returns:
            List of metric field names.

        Raises:
            TypeError: If the analyzer doesn't have a valid result type.
        """
        return list(cls._require_result_type().model_fields.keys())

    @classmethod
    def get_metric_descriptions(cls) -> dict[str, str]:
        """Get descriptions for each metric field.

        Returns:
            Dictionary mapping field names to descriptions.

        Raises:
            TypeError: If the analyzer doesn't have a valid result type.
        """
        return {
            name: field_info.description or ""
            for name, field_info in cls._require_result_type().model_fields.items()
        }

    @classmethod
    def _get_result_type(cls) -> type[BaseModel] | None:
        """Get the result type from the generic parameter.

        Walks through the class's base classes to find the generic type
        argument (TResult) from the analyzer base class.

        Returns:
            The result type class (a BaseModel subclass), or None if not found.
        """
        # Walk through original bases to find generic type parameter
        # __orig_bases__ exists on classes inheriting from Generic (PEP 560)
        # Use getattr since it's a runtime attribute not in the static type system
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is not None:
                args = get_args(base)
                if (
                    args
                    and isinstance(args[0], type)
                    and issubclass(args[0], BaseModel)
                ):
                    return args[0]
        return None

    @staticmethod
    def get_text_content(message: Message) -> str:
        """Extract text content from a message.

        Handles both simple string content and multimodal content lists.

        Args:
            message: The message to extract text from.

        Returns:
            The text content as a string.
        """
        if isinstance(message.content, str):
            return message.content
        # For multimodal content, concatenate text items
        text_parts = []
        for item in message.content:
            if isinstance(item, ContentItem) and isinstance(item.content, str):
                text_parts.append(item.content)
        return " ".join(text_parts)


class MessageAnalyzer(BaseAnalyzer[TResult]):
    """Base class for analyzers that operate on individual messages.

    MessageAnalyzers process single messages and return typed results.
    Use this for metrics that are meaningful at the message level,
    such as length, format detection, or content analysis.

    Example:
        class FormatAnalyzer(MessageAnalyzer[FormatMetrics]):
            def analyze(self, message: Message) -> FormatMetrics:
                text = self.get_text_content(message)
                return FormatMetrics(
                    has_markdown=self._detect_markdown(text),
                    has_code_blocks=self._detect_code_blocks(text),
                )
    """

    @classmethod
    def get_scope(cls) -> str:
        """Get the scope of this analyzer.

        Returns:
            Scope string ('message').
        """
        return "message"

    @abstractmethod
    def analyze(self, message: Message) -> TResult:
        """Analyze a single message and return typed results.

        Args:
            message: The message to analyze.

        Returns:
            Typed result model containing analysis metrics.
        """
        ...

    def analyze_batch(self, messages: list[Message]) -> list[TResult]:
        """Analyze multiple messages and return results for each.

        Override this method to implement vectorized/batched processing
        for better performance with large datasets.

        Args:
            messages: List of messages to analyze.

        Returns:
            List of typed results, one per message.
        """
        return [self.analyze(m) for m in messages]

    def __call__(self, message: Message) -> TResult:
        """Allow analyzer to be called directly.

        Args:
            message: The message to analyze.

        Returns:
            Typed result model.
        """
        return self.analyze(message)


class ConversationAnalyzer(BaseAnalyzer[TResult]):
    """Base class for analyzers that operate on complete conversations.

    ConversationAnalyzers process entire conversations and return typed results.
    Use this for metrics that require context across messages, such as
    turn patterns, coherence analysis, or conversation-level quality scores.

    Example:
        class LengthAnalyzer(ConversationAnalyzer[LengthMetrics]):
            def analyze(self, conversation: Conversation) -> LengthMetrics:
                total_words = sum(
                    len(m.content.split())
                    for m in conversation.messages
                    if isinstance(m.content, str)
                )
                return LengthMetrics(total_words=total_words, ...)
    """

    @classmethod
    def get_scope(cls) -> str:
        """Get the scope of this analyzer.

        Returns:
            Scope string ('conversation').
        """
        return "conversation"

    @abstractmethod
    def analyze(self, conversation: Conversation) -> TResult:
        """Analyze a complete conversation and return typed results.

        Args:
            conversation: The conversation to analyze.

        Returns:
            Typed result model containing analysis metrics.
        """
        ...

    def analyze_batch(self, conversations: list[Conversation]) -> list[TResult]:
        """Analyze multiple conversations and return results for each.

        Override this method to implement batched processing for better
        performance, especially for analyzers that benefit from batching
        (e.g., those using ML models).

        Args:
            conversations: List of conversations to analyze.

        Returns:
            List of typed results, one per conversation.
        """
        return [self.analyze(c) for c in conversations]

    def __call__(self, conversation: Conversation) -> TResult:
        """Allow analyzer to be called directly.

        Args:
            conversation: The conversation to analyze.

        Returns:
            Typed result model.
        """
        return self.analyze(conversation)

    @staticmethod
    def get_conversation_text(
        conversation: Conversation,
        tokenizer: PreTrainedTokenizerBase,
    ) -> str:
        """Get the full text of a conversation using a tokenizer's chat template.

        Args:
            conversation: The conversation to extract text from.
            tokenizer: Tokenizer with a chat template for formatting.

        Returns:
            Full conversation text as a single string.

        Raises:
            ValueError: If the tokenizer doesn't have a chat template.
        """
        if tokenizer.chat_template is None:
            raise ValueError(
                f"Tokenizer {type(tokenizer).__name__} does not have a chat template. "
                "A chat template is required to format conversation text."
            )

        result = tokenizer.apply_chat_template(
            conversation=conversation,  # type: ignore
            tokenize=False,
            return_dict=False,
        )

        if not isinstance(result, str):
            raise TypeError(
                f"apply_chat_template returned {type(result).__name__}, expected str"
            )

        return result


class DatasetAnalyzer(BaseAnalyzer[TResult]):
    """Base class for analyzers that operate on entire datasets.

    DatasetAnalyzers have access to all conversations at once, enabling
    cross-sample operations like deduplication, clustering, or computing
    dataset-wide statistics.

    Example:
        class DeduplicationAnalyzer(DatasetAnalyzer[DeduplicationResult]):
            def analyze(self, conversations: list[Conversation]) -> DeduplicationResult:
                embeddings = self._compute_embeddings(conversations)
                duplicates = self._find_duplicates(embeddings)
                return DeduplicationResult(
                    duplicate_groups=duplicates,
                    total_duplicates=len(duplicates),
                )
    """

    @classmethod
    def get_scope(cls) -> str:
        """Get the scope of this analyzer.

        Returns:
            Scope string ('dataset').
        """
        return "dataset"

    @abstractmethod
    def analyze(self, conversations: list[Conversation]) -> TResult:
        """Analyze an entire dataset and return typed results.

        This method receives all conversations at once, enabling
        cross-sample operations that require global context.

        Args:
            conversations: All conversations in the dataset.

        Returns:
            Typed result model containing dataset-level analysis.
        """
        ...

    def __call__(self, conversations: list[Conversation]) -> TResult:
        """Allow analyzer to be called directly.

        Args:
            conversations: All conversations to analyze.

        Returns:
            Typed result model.
        """
        return self.analyze(conversations)


class PreferenceAnalyzer(BaseAnalyzer[TResult]):
    """Base class for analyzers that operate on preference pairs.

    PreferenceAnalyzers process chosen/rejected conversation pairs,
    which is the format used for DPO (Direct Preference Optimization)
    and similar preference-based training methods.

    Example:
        class PreferenceMarginAnalyzer(PreferenceAnalyzer[PreferenceMetrics]):
            def analyze(
                self, chosen: Conversation, rejected: Conversation
            ) -> PreferenceMetrics:
                chosen_score = self._compute_quality(chosen)
                rejected_score = self._compute_quality(rejected)
                return PreferenceMetrics(
                    margin=chosen_score - rejected_score,
                    chosen_score=chosen_score,
                    rejected_score=rejected_score,
                )
    """

    @classmethod
    def get_scope(cls) -> str:
        """Get the scope of this analyzer.

        Returns:
            Scope string ('preference').
        """
        return "preference"

    @abstractmethod
    def analyze(self, chosen: Conversation, rejected: Conversation) -> TResult:
        """Analyze a preference pair and return typed results.

        Args:
            chosen: The preferred/chosen conversation.
            rejected: The rejected/dispreferred conversation.

        Returns:
            Typed result model containing preference analysis.
        """
        ...

    def analyze_batch(
        self,
        pairs: list[tuple[Conversation, Conversation]],
    ) -> list[TResult]:
        """Analyze multiple preference pairs.

        Args:
            pairs: List of (chosen, rejected) conversation tuples.

        Returns:
            List of typed results, one per pair.
        """
        return [self.analyze(chosen, rejected) for chosen, rejected in pairs]

    def __call__(self, chosen: Conversation, rejected: Conversation) -> TResult:
        """Allow analyzer to be called directly.

        Args:
            chosen: The preferred conversation.
            rejected: The rejected conversation.

        Returns:
            Typed result model.
        """
        return self.analyze(chosen, rejected)
