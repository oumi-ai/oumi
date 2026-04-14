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

from typing import Any, Protocol, runtime_checkable

import tiktoken
from pydantic import BaseModel, Field

from oumi.analyze.base import ConversationAnalyzer
from oumi.builders import build_tokenizer
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.registry import register_sample_analyzer
from oumi.core.types.conversation import Conversation, Role

__all__ = [
    "LengthAnalyzerConfig",
    "LengthMetrics",
    "LengthAnalyzer",
    "Tokenizer",
]


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers used by LengthAnalyzer."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...


def _default_tokenizer(encoding: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding)


class LengthAnalyzerConfig(BaseModel):
    """Configuration for LengthAnalyzer."""

    tokenizer_name: str = Field(
        default="cl100k_base",
        description=(
            "Tokenizer name. For tiktoken, use encoding names like "
            "'cl100k_base' (GPT-4), 'o200k_base' (GPT-4o), etc. "
            "For HuggingFace, use model IDs like "
            "'meta-llama/Llama-3.1-8B-Instruct'. "
            "Automatically detects backend based on name."
        ),
    )
    trust_remote_code: bool = Field(
        default=False,
        description=(
            "Trust remote code for HuggingFace tokenizers "
            "(only applicable when using HF models)"
        ),
    )


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
        >>> from oumi.analyze.analyzers.length import LengthAnalyzer
        >>> from oumi.core.types.conversation import Conversation, Message, Role
        >>>
        >>> analyzer = LengthAnalyzer.from_config({"tokenizer_name": "cl100k_base"})
        >>> conversation = Conversation(messages=[
        ...     Message(role=Role.USER, content="Hello, how are you?"),
        ...     Message(role=Role.ASSISTANT, content="I'm doing well, thanks!"),
        ... ])
        >>> result = analyzer.analyze(conversation)
        >>> print(f"Total tokens: {result.total_tokens}")
        Total tokens: 12

    Args:
        tokenizer: Tokenizer instance for token counting. Must have an
            `encode(text) -> list` method. Use `from_config()` to construct
            from a tokenizer name, or pass any compatible tokenizer directly.
    """

    _result_model = LengthMetrics

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get JSON schema for this analyzer's configuration."""
        return LengthAnalyzerConfig.model_json_schema()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LengthAnalyzer":
        """Create a LengthAnalyzer from a config dictionary.

        Args:
            config: See ``LengthAnalyzerConfig`` for supported keys.

        Returns:
            LengthAnalyzer instance with configured tokenizer.
        """
        cfg = LengthAnalyzerConfig(**config)

        # Known tiktoken encodings — auto-detect backend based on name.
        # Note: "gpt2" is intentionally excluded because it is also a valid
        # HuggingFace model ID; users who want tiktoken's gpt2 encoding should
        # use the tiktoken API directly.
        TIKTOKEN_ENCODINGS = {
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "r50k_base",
            "p50k_edit",
        }

        if cfg.tokenizer_name in TIKTOKEN_ENCODINGS:
            tokenizer = _default_tokenizer(cfg.tokenizer_name)
        else:
            # Use build_tokenizer so token counts align with training/inference
            # and oumi's internal model configs (padding side, etc.) are applied.
            tokenizer = build_tokenizer(
                ModelParams(
                    model_name=cfg.tokenizer_name,
                    trust_remote_code=cfg.trust_remote_code,
                )
            )

        return cls(tokenizer=tokenizer)

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
                "or use from_config({'tokenizer_name': 'cl100k_base'})."
            )

        if isinstance(self.tokenizer, tiktoken.Encoding):
            # Encode special tokens (e.g. <|endoftext|>) as literal text
            tokens = self.tokenizer.encode(text, disallowed_special=())
        else:
            tokens = self.tokenizer.encode(text)
        return len(tokens)

    def _count_rendered_tokens(self, conversation: Conversation) -> int | None:
        """Count tokens after applying the tokenizer's chat template.

        Returns None if the tokenizer doesn't support chat templates.
        """
        if self.tokenizer is None:
            return None

        try:
            rendered_text = self.get_conversation_text(conversation, self.tokenizer)  # type: ignore[arg-type]
            return self._count_tokens(rendered_text)
        except Exception:
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
