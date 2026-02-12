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

import inspect
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from oumi.core.types.conversation import ContentItem, Conversation, Message

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
    def get_config_schema(cls) -> dict[str, Any]:
        """Get JSON schema for this analyzer's constructor parameters.

        Introspects ``__init__`` to discover what configuration the analyzer
        accepts.  Analyzers with no parameters return an empty schema.

        Subclasses can override this to provide a custom schema (e.g. when
        the constructor takes a non-serialisable object like a tokenizer but
        the *user-facing* config is a simpler set of options).

        Returns:
            JSON-schema-like dict with ``properties``, ``required``, etc.
        """
        sig = inspect.signature(cls.__init__)
        properties: dict[str, Any] = {}
        required: list[str] = []

        _TYPE_MAP = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # Skip *args and **kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            prop: dict[str, Any] = {}

            # Resolve type annotation
            annotation = param.annotation
            if annotation is not inspect.Parameter.empty:
                prop["type"] = _TYPE_MAP.get(annotation, "string")
            else:
                prop["type"] = "string"

            # Default value
            if param.default is not inspect.Parameter.empty:
                # Skip non-serialisable defaults (None is fine)
                if param.default is None or isinstance(
                    param.default, (str, int, float, bool)
                ):
                    prop["default"] = param.default
            else:
                required.append(name)

            properties[name] = prop

        schema: dict[str, Any] = {"properties": properties}
        if required:
            schema["required"] = required
        return schema

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

        Returns:
            The result type class (a BaseModel subclass), or None if not found.
        """
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
