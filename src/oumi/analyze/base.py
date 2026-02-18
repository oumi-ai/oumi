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
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from oumi.core.types.conversation import ContentItem, Conversation, Message

TResult = TypeVar("TResult", bound=BaseModel)


class BaseAnalyzer(ABC, Generic[TResult]):
    """Base class for all analyzer types.

    Subclasses must implement metadata methods to describe their result schema.
    The generic type parameter TResult provides type safety for the analyze() method.

    All concrete analyzer types (MessageAnalyzer, ConversationAnalyzer, etc.)
    inherit from this class.

    Subclasses should set ``_result_model`` to their Pydantic result class to get
    automatic implementations of ``get_result_schema``, ``get_metric_names``, and
    ``get_metric_descriptions`` for free.

    Attributes:
        analyzer_id: Optional custom identifier for this analyzer instance.
            If not set, the class name is used as the identifier.
        _result_model: Pydantic model class for this analyzer's result type.
            Set this in subclasses to enable automatic schema/metric introspection.
    """

    analyzer_id: str | None = None
    _result_model: ClassVar[type[BaseModel] | None] = None

    @classmethod
    def get_scope(cls) -> str:
        """Get the scope of this analyzer.

        Returns:
            Scope string ('message', 'conversation', 'dataset', or 'preference').
        """
        return "unknown"

    @classmethod
    @abstractmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get JSON schema for this analyzer's configuration.

        Returns:
            JSON-schema-like dict with ``properties``, ``required``, etc.
        """
        ...

    @classmethod
    def get_result_schema(cls) -> dict:
        """Get the JSON schema for this analyzer's result model.

        Uses ``_result_model.model_json_schema()`` when ``_result_model`` is set.
        Subclasses that cannot set ``_result_model`` may override this method.

        Returns:
            JSON schema dictionary for the result model.
        """
        if cls._result_model is None:
            return {}
        return cls._result_model.model_json_schema()

    @classmethod
    def get_metric_names(cls) -> list[str]:
        """Get the list of metric field names this analyzer produces.

        Uses ``_result_model.model_fields`` when ``_result_model`` is set.
        Subclasses that cannot set ``_result_model`` may override this method.

        Returns:
            List of metric field names.
        """
        if cls._result_model is None:
            return []
        return list(cls._result_model.model_fields.keys())

    def get_available_metric_names(self) -> list[str]:
        """Get metric names this instance will actually produce.

        Subclasses can override to exclude metrics that depend on config
        (e.g., ``rendered_tokens`` requires a HuggingFace tokenizer).
        Default: returns all metric names from the result schema.
        """
        return self.get_metric_names()

    @classmethod
    def get_metric_descriptions(cls) -> dict[str, str]:
        """Get descriptions for each metric field.

        Uses ``_result_model.model_fields`` when ``_result_model`` is set.
        Subclasses that cannot set ``_result_model`` may override this method.

        Returns:
            Dictionary mapping field names to descriptions.
        """
        if cls._result_model is None:
            return {}
        return {
            name: field.description or ""
            for name, field in cls._result_model.model_fields.items()
        }

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
        text_parts = []
        for item in message.content:
            if isinstance(item, ContentItem) and isinstance(item.content, str):
                text_parts.append(item.content)
        return " ".join(text_parts)


class MessageAnalyzer(BaseAnalyzer[TResult]):
    """Base class for analyzers that operate on individual messages."""

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
    """Base class for analyzers that operate on complete conversations."""

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
    """Base class for analyzers that operate on entire datasets."""

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
    """Base class for analyzers that operate on preference pairs."""

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
