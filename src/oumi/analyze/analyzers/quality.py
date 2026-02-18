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

"""Data quality analyzer implementation."""

import re

from pydantic import BaseModel, Field

from oumi.analyze.base import ConversationAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.core.types.conversation import Conversation

__all__ = ["DataQualityMetrics", "DataQualityAnalyzer"]

# Invalid serialization patterns: (regex, display_name)
_INVALID_VALUE_PATTERNS = [
    (re.compile(r"\bNaN\b"), "NaN"),
    (re.compile(r"\bnan\b"), "nan"),
    (re.compile(r"^null$"), "null"),
    (re.compile(r"^None$"), "None"),
    (re.compile(r"^undefined$"), "undefined"),
]


class DataQualityMetrics(BaseModel):
    """Result model for data quality checks on a conversation.

    Example:
        >>> result = DataQualityMetrics(
        ...     has_alternating_turns=True,
        ...     has_empty_turns=False,
        ...     empty_turn_count=0,
        ...     has_invalid_values=False,
        ...     invalid_value_patterns=[],
        ... )
        >>> print(result.has_alternating_turns)
        True
    """

    has_alternating_turns: bool = Field(
        description=(
            "True if all non-system messages alternate between user and assistant roles"
        )
    )
    has_empty_turns: bool = Field(
        description="True if any message has empty or whitespace-only content"
    )
    empty_turn_count: int = Field(
        description="Number of messages with empty or whitespace-only content"
    )
    has_invalid_values: bool = Field(
        description=(
            "True if any message contains values serialized as strings "
            "(e.g. 'NaN', 'null', 'None', 'undefined')"
        )
    )
    invalid_value_patterns: list[str] = Field(
        description="List of invalid value patterns found across all messages"
    )


@register_sample_analyzer("quality")
class DataQualityAnalyzer(ConversationAnalyzer[DataQualityMetrics]):
    """Analyzer for basic data quality checks on conversations.

    Checks for three common data quality issues without requiring an LLM:
    - Non-alternating user/assistant message patterns
    - Empty or whitespace-only turns
    - Values serialized as strings (NaN, null, None, undefined)

    Example:
        >>> from oumi.analyze.analyzers.quality import DataQualityAnalyzer
        >>> from oumi.core.types.conversation import Conversation, Message, Role
        >>>
        >>> analyzer = DataQualityAnalyzer()
        >>> conversation = Conversation(messages=[
        ...     Message(role=Role.USER, content="Hello"),
        ...     Message(role=Role.ASSISTANT, content="Hi there!"),
        ... ])
        >>> result = analyzer.analyze(conversation)
        >>> print(result.has_alternating_turns)
        True
    """

    _result_model = DataQualityMetrics

    @classmethod
    def get_config_schema(cls) -> dict:
        """Get JSON schema for DataQualityAnalyzer configuration."""
        return {"properties": {}}

    def analyze(self, conversation: Conversation) -> DataQualityMetrics:
        """Analyze data quality for a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            DataQualityMetrics with the quality check results.
        """
        # 1. Alternating turns check (ignoring system messages)
        roles = [m.role.value for m in conversation.messages]
        non_system = [r for r in roles if r != "system"]
        has_alternating = True
        for i in range(1, len(non_system)):
            if non_system[i] == non_system[i - 1]:
                has_alternating = False
                break

        # 2. Empty turns check
        def _text(m) -> str:  # type: ignore[no-untyped-def]
            c = m.content
            return c if isinstance(c, str) else (str(c) if c else "")

        empty_count = sum(
            1 for m in conversation.messages if not _text(m).strip()
        )

        # 3. Invalid serialized values check
        patterns_found: set[str] = set()
        for message in conversation.messages:
            content = _text(message)
            for pattern, name in _INVALID_VALUE_PATTERNS:
                if pattern.search(content):
                    patterns_found.add(name)

        return DataQualityMetrics(
            has_alternating_turns=has_alternating,
            has_empty_turns=empty_count > 0,
            empty_turn_count=empty_count,
            has_invalid_values=len(patterns_found) > 0,
            invalid_value_patterns=sorted(patterns_found),
        )
