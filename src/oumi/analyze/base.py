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

"""Base analyzer classes for the typed analyzer framework."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, get_args

from pydantic import BaseModel

from oumi.core.types.conversation import Conversation, Message

TResult = TypeVar("TResult", bound=BaseModel)


class _AnalyzerMetaMixin:
    """Mixin providing common metadata methods for all analyzer types.

    Attributes:
        analyzer_id: Optional custom identifier for this analyzer instance.
            If not set, the class name is used as the identifier.
    """

    analyzer_id: str | None = None

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
            return {
                name: field_info.description or ""
                for name, field_info in result_type.model_fields.items()
            }
        return {}

    @classmethod
    def _get_result_type(cls) -> type | None:
        """Get the result type from the generic parameter.

        Walks the MRO to find the generic type argument (TResult) from
        the analyzer base class.
        """
        analyzer_bases = (
            MessageAnalyzer,
            ConversationAnalyzer,
            DatasetAnalyzer,
            PreferenceAnalyzer,
        )
        for klass in cls.__mro__:
            for base in getattr(klass, "__orig_bases__", []):
                if hasattr(base, "__origin__") and base.__origin__ in analyzer_bases:
                    args = get_args(base)
                    if args:
                        return args[0]
        return None


class MessageAnalyzer(_AnalyzerMetaMixin, ABC, Generic[TResult]):
    """Base class for analyzers that operate on individual messages."""

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
        """Call analyze() directly."""
        return self.analyze(message)

    @staticmethod
    def get_text_content(message: Message) -> str:
        """Extract text content from a message.

        Handles both simple string content and multimodal content lists.
        """
        if isinstance(message.content, str):
            return message.content
        text_parts = []
        for item in message.content:
            if hasattr(item, "content") and isinstance(item.content, str):
                text_parts.append(item.content)
        return " ".join(text_parts)


class ConversationAnalyzer(_AnalyzerMetaMixin, ABC, Generic[TResult]):
    """Base class for analyzers that operate on complete conversations."""

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
        """Call analyze() directly."""
        return self.analyze(conversation)

    @staticmethod
    def get_text_content(message: Message) -> str:
        """Extract text content from a message.

        Handles both simple string content and multimodal content lists.
        """
        return MessageAnalyzer.get_text_content(message)

    @staticmethod
    def get_conversation_text(conversation: Conversation) -> str:
        """Get the full text content of a conversation.

        Concatenates all message contents with role prefixes.
        """
        parts = []
        for message in conversation.messages:
            text = ConversationAnalyzer.get_text_content(message)
            parts.append(f"{message.role.value}: {text}")
        return "\n".join(parts)


class DatasetAnalyzer(_AnalyzerMetaMixin, ABC, Generic[TResult]):
    """Base class for analyzers that operate on entire datasets."""

    @abstractmethod
    def analyze(self, conversations: list[Conversation]) -> TResult:
        """Analyze an entire dataset and return typed results.

        Args:
            conversations: All conversations in the dataset.

        Returns:
            Typed result model containing dataset-level analysis.
        """
        ...

    def __call__(self, conversations: list[Conversation]) -> TResult:
        """Call analyze() directly."""
        return self.analyze(conversations)


class PreferenceAnalyzer(_AnalyzerMetaMixin, ABC, Generic[TResult]):
    """Base class for analyzers that operate on preference pairs."""

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
        """Call analyze() directly."""
        return self.analyze(chosen, rejected)
