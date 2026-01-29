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
- MessageAnalyzer: Analyzes individual messages
- ConversationAnalyzer: Analyzes complete conversations
- DatasetAnalyzer: Analyzes entire datasets (cross-sample operations)
- PreferenceAnalyzer: Analyzes preference pairs (for DPO data)

Each analyzer returns strongly-typed Pydantic models as results.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from oumi.core.types.conversation import Conversation, Message

# Type variable for analyzer results - must be a Pydantic BaseModel
TResult = TypeVar("TResult", bound=BaseModel)


class MessageAnalyzer(ABC, Generic[TResult]):
    """Base class for analyzers that operate on individual messages.

    MessageAnalyzers process single messages and return typed results.
    Use this for metrics that are meaningful at the message level,
    such as length, format detection, or content analysis.

    Example:
        class FormatAnalyzer(MessageAnalyzer[FormatMetrics]):
            def analyze(self, message: Message) -> FormatMetrics:
                text = self._get_text_content(message)
                return FormatMetrics(
                    has_markdown=self._detect_markdown(text),
                    has_code_blocks=self._detect_code_blocks(text),
                )
    """

    @classmethod
    def get_result_schema(cls) -> dict:
        """Get the JSON schema for this analyzer's result model.

        This allows users to discover what metrics the analyzer produces
        before running analysis.

        Returns:
            JSON schema dictionary for the result model.
        """
        # Get the result type from the generic parameter
        result_type = cls._get_result_type()
        if result_type and hasattr(result_type, "model_json_schema"):
            return result_type.model_json_schema()
        return {}

    @classmethod
    def get_metric_names(cls) -> list[str]:
        """Get the list of metric field names this analyzer produces.

        Returns:
            List of metric field names.
        """
        result_type = cls._get_result_type()
        if result_type and hasattr(result_type, "model_fields"):
            return list(result_type.model_fields.keys())
        return []

    @classmethod
    def get_metric_descriptions(cls) -> dict[str, str]:
        """Get descriptions for each metric field.

        Returns:
            Dictionary mapping field names to descriptions.
        """
        result_type = cls._get_result_type()
        if result_type and hasattr(result_type, "model_fields"):
            descriptions = {}
            for name, field_info in result_type.model_fields.items():
                desc = field_info.description or ""
                descriptions[name] = desc
            return descriptions
        return {}

    @classmethod
    def _get_result_type(cls) -> type | None:
        """Get the result type from the generic parameter."""
        import typing

        for base in getattr(cls, "__orig_bases__", []):
            if hasattr(base, "__origin__") and base.__origin__ is MessageAnalyzer:
                args = typing.get_args(base)
                if args:
                    return args[0]
        return None

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
            if hasattr(item, "content") and isinstance(item.content, str):
                text_parts.append(item.content)
        return " ".join(text_parts)


class ConversationAnalyzer(ABC, Generic[TResult]):
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
    def get_result_schema(cls) -> dict:
        """Get the JSON schema for this analyzer's result model."""
        result_type = cls._get_result_type()
        if result_type and hasattr(result_type, "model_json_schema"):
            return result_type.model_json_schema()
        return {}

    @classmethod
    def get_metric_names(cls) -> list[str]:
        """Get the list of metric field names this analyzer produces."""
        result_type = cls._get_result_type()
        if result_type and hasattr(result_type, "model_fields"):
            return list(result_type.model_fields.keys())
        return []

    @classmethod
    def get_metric_descriptions(cls) -> dict[str, str]:
        """Get descriptions for each metric field."""
        result_type = cls._get_result_type()
        if result_type and hasattr(result_type, "model_fields"):
            descriptions = {}
            for name, field_info in result_type.model_fields.items():
                desc = field_info.description or ""
                descriptions[name] = desc
            return descriptions
        return {}

    @classmethod
    def _get_result_type(cls) -> type | None:
        """Get the result type from the generic parameter."""
        import typing

        for base in getattr(cls, "__orig_bases__", []):
            if hasattr(base, "__origin__") and base.__origin__ is ConversationAnalyzer:
                args = typing.get_args(base)
                if args:
                    return args[0]
        return None

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
    def get_text_content(message: Message) -> str:
        """Extract text content from a message.

        Handles both simple string content and multimodal content lists.

        Args:
            message: The message to extract text from.

        Returns:
            The text content as a string.
        """
        return MessageAnalyzer.get_text_content(message)

    @staticmethod
    def get_conversation_text(conversation: Conversation) -> str:
        """Get the full text content of a conversation.

        Concatenates all message contents with role prefixes.

        Args:
            conversation: The conversation to extract text from.

        Returns:
            Full conversation text as a single string.
        """
        parts = []
        for message in conversation.messages:
            text = ConversationAnalyzer.get_text_content(message)
            parts.append(f"{message.role.value}: {text}")
        return "\n".join(parts)


class DatasetAnalyzer(ABC, Generic[TResult]):
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
    def get_result_schema(cls) -> dict:
        """Get the JSON schema for this analyzer's result model."""
        result_type = cls._get_result_type()
        if result_type and hasattr(result_type, "model_json_schema"):
            return result_type.model_json_schema()
        return {}

    @classmethod
    def get_metric_names(cls) -> list[str]:
        """Get the list of metric field names this analyzer produces."""
        result_type = cls._get_result_type()
        if result_type and hasattr(result_type, "model_fields"):
            return list(result_type.model_fields.keys())
        return []

    @classmethod
    def get_metric_descriptions(cls) -> dict[str, str]:
        """Get descriptions for each metric field."""
        result_type = cls._get_result_type()
        if result_type and hasattr(result_type, "model_fields"):
            descriptions = {}
            for name, field_info in result_type.model_fields.items():
                desc = field_info.description or ""
                descriptions[name] = desc
            return descriptions
        return {}

    @classmethod
    def _get_result_type(cls) -> type | None:
        """Get the result type from the generic parameter."""
        import typing

        for base in getattr(cls, "__orig_bases__", []):
            if hasattr(base, "__origin__") and base.__origin__ is DatasetAnalyzer:
                args = typing.get_args(base)
                if args:
                    return args[0]
        return None

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


class PreferenceAnalyzer(ABC, Generic[TResult]):
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
