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

from typing import Any, ClassVar, Protocol, runtime_checkable

import tiktoken
from pydantic import BaseModel, Field

from oumi.analyze.base import ConversationAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.core.types.conversation import Conversation, Role

__all__ = [
    "LengthMetrics",
    "LengthAnalyzer",
    "Tokenizer",
    "default_tokenizer",
    "huggingface_tokenizer",
]


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers used by LengthAnalyzer."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...


def default_tokenizer(encoding: str = "cl100k_base") -> tiktoken.Encoding:
    """Get the default tiktoken tokenizer.

    Args:
        encoding: Tiktoken encoding name. Defaults to "cl100k_base" (GPT-4).

    Returns:
        Tiktoken encoder instance.
    """
    return tiktoken.get_encoding(encoding)


def huggingface_tokenizer(model_name: str, trust_remote_code: bool = False) -> Any:
    """Get a HuggingFace tokenizer with chat template support.

    This enables the `rendered_tokens` field which counts tokens after
    applying the model's chat template (e.g., ChatML, Llama format).

    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        trust_remote_code: Whether to trust remote code for custom tokenizers.

    Returns:
        HuggingFace tokenizer instance with chat_template support.

    Raises:
        ImportError: If transformers is not installed.
        OSError: If the model/tokenizer cannot be loaded.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "HuggingFace transformers is required for HuggingFace tokenizers. "
            "Install with: pip install transformers"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer


class LengthMetrics(BaseModel):
    """Result model for length analysis of conversations.

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

    total_tokens: int = Field(description="Total number of tokens across all messages")
    rendered_tokens: int | None = Field(
        default=None,
        description="Token count of the full conversation rendered with chat template. "
        "None if tokenizer doesn't support apply_chat_template.",
    )
    avg_tokens_per_message: float = Field(description="Average tokens per message")
    message_token_counts: list[int] = Field(
        description="Token count for each message in order"
    )
    num_messages: int = Field(description="Number of messages in the conversation")
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


@register_sample_analyzer("length")
class LengthAnalyzer(ConversationAnalyzer[LengthMetrics]):
    """Analyzer for computing token length metrics of conversations.

    Computes token counts for conversations using a provided tokenizer.
    Provides both conversation-level totals and per-message breakdowns.

    Example:
        >>> from oumi.analyze.analyzers.length import LengthAnalyzer, default_tokenizer
        >>> from oumi.core.types.conversation import Conversation, Message, Role
        >>>
        >>> analyzer = LengthAnalyzer(tokenizer=default_tokenizer())
        >>> conversation = Conversation(messages=[
        ...     Message(role=Role.USER, content="Hello, how are you?"),
        ...     Message(role=Role.ASSISTANT, content="I'm doing well, thanks!"),
        ... ])
        >>> result = analyzer.analyze(conversation)
        >>> print(f"Total tokens: {result.total_tokens}")
        Total tokens: 12

    Args:
        tokenizer: Tokenizer instance for token counting. Must have an
            `encode(text) -> list` method. Use `default_tokenizer()` for
            tiktoken, or pass a HuggingFace tokenizer for model-specific counts.
    """

    # Custom config schema: the actual __init__ takes a Tokenizer object,
    # but the user-facing config is tokenizer selection params that the
    # worker uses to construct the tokenizer.
    _CONFIG_SCHEMA: ClassVar[dict[str, Any]] = {
        "properties": {
            "tokenizer_type": {
                "type": "string",
                "enum": ["tiktoken", "huggingface"],
                "default": "tiktoken",
                "description": "Tokenizer backend to use",
            },
            "encoding": {
                "type": "string",
                "default": "cl100k_base",
                "description": "Tiktoken encoding name (only for tiktoken)",
                "enum": ["cl100k_base", "o200k_base", "p50k_base", "r50k_base"],
            },
            "model_name": {
                "type": "string",
                "description": (
                    "HuggingFace model ID (only for huggingface), "
                    "e.g. meta-llama/Llama-3.1-8B-Instruct"
                ),
            },
            "trust_remote_code": {
                "type": "boolean",
                "default": False,
                "description": "Trust remote code for HuggingFace tokenizers",
            },
        },
    }

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Return user-facing config schema for tokenizer selection."""
        return cls._CONFIG_SCHEMA

    def __init__(self, tokenizer: Tokenizer | None = None):
        """Initialize the analyzer."""
        self.tokenizer = tokenizer

    def get_available_metric_names(self) -> list[str]:
        """Return metrics this instance will produce.

        Excludes ``rendered_tokens`` when the tokenizer doesn't support
        ``apply_chat_template`` (i.e. tiktoken or no tokenizer).
        """
        names = self.get_metric_names()
        if not hasattr(self.tokenizer, "apply_chat_template"):
            names = [n for n in names if n != "rendered_tokens"]
        return names

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError(
                "No tokenizer configured. Either pass a tokenizer to __init__ "
                "or use default_tokenizer()."
            )

        try:
            # tiktoken: encode literal special tokens (e.g. <|endoftext|>) as normal text
            tokens = self.tokenizer.encode(text, disallowed_special=())
        except TypeError:
            # tokenizer doesn't accept disallowed_special (e.g. HuggingFace)
            tokens = self.tokenizer.encode(text)
        return len(tokens)

    def _count_rendered_tokens(self, conversation: Conversation) -> int | None:
        """Count tokens in the chat-template-rendered conversation.

        This gives the actual token count the model sees during training/inference,
        including special tokens added by the chat template.

        Args:
            conversation: The conversation to render and tokenize.

        Returns:
            Token count of rendered conversation, or None if tokenizer doesn't
            support chat templates.
        """
        if self.tokenizer is None:
            return None

        # Check if tokenizer has a chat template before proceeding
        if getattr(self.tokenizer, "chat_template", None) is None:
            return None

        if not conversation.messages:
            return 0

        try:
            # Use base class method to render conversation with chat template
            # Type ignore: we've verified tokenizer has chat_template attribute above
            rendered_text = self.get_conversation_text(conversation, self.tokenizer)  # type: ignore[arg-type]
            return self._count_tokens(rendered_text)
        except (ValueError, AttributeError):
            # Unexpected error during rendering
            return None

    def analyze(self, conversation: Conversation) -> LengthMetrics:
        """Analyze token length metrics for a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            LengthMetrics containing token counts.
        """
        message_token_counts: list[int] = []
        role_token_counts: dict[Role, int] = {role: 0 for role in Role}

        for message in conversation.messages:
            text = self.get_text_content(message)
            token_count = self._count_tokens(text)
            message_token_counts.append(token_count)

            if message.role in role_token_counts:
                role_token_counts[message.role] += token_count

        total_tokens = sum(message_token_counts)
        num_messages = len(conversation.messages)
        avg_tokens = total_tokens / num_messages if num_messages > 0 else 0.0
        rendered_tokens = self._count_rendered_tokens(conversation)

        return LengthMetrics(
            total_tokens=total_tokens,
            rendered_tokens=rendered_tokens,
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
