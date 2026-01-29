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

"""Data quality analyzer implementation.

This analyzer performs basic data quality checks on conversations without
requiring an LLM, making it fast and cheap to run on large datasets.
"""

import re
from typing import Any

from oumi.analyze.base import ConversationAnalyzer
from oumi.analyze.results.quality import DataQualityMetrics
from oumi.core.types.conversation import Conversation


# Common refusal phrases used by AI assistants
REFUSAL_PATTERNS = [
    r"i cannot",
    r"i can't",
    r"i am not able to",
    r"i'm not able to",
    r"i apologize,? but",
    r"i'm sorry,? but i cannot",
    r"as an ai",
    r"as a language model",
    r"i don't have the ability",
    r"it would be inappropriate",
    r"i must decline",
    r"i cannot assist with",
    r"against my guidelines",
    r"violates my guidelines",
    r"i'm unable to",
    r"i am unable to",
]

# Tags to check for balance
TAG_PAIRS = [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
    ("<answer>", "</answer>"),
    ("<response>", "</response>"),
]

# Invalid serialization patterns
INVALID_VALUE_PATTERNS = [
    (r"\bNaN\b", "NaN"),
    (r"\bnan\b", "nan"),
    (r"^null$", "null"),
    (r"^None$", "None"),
    (r"^undefined$", "undefined"),
]


class DataQualityAnalyzer(ConversationAnalyzer[DataQualityMetrics]):
    """Analyzer for basic data quality checks on conversations.

    Performs fast, non-LLM quality checks including:
    - Alternating turn pattern validation
    - Empty message detection
    - Invalid serialized value detection (NaN, null, None)
    - Context length estimation
    - Truncation detection
    - Policy refusal detection
    - Think tag balance checking

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
        >>> print(f"Passes quality: {result.passes_basic_quality}")
        Passes quality: True

    Args:
        check_turn_pattern: Whether to check for alternating turns.
        check_empty_content: Whether to check for empty messages.
        check_invalid_values: Whether to check for invalid serialized values.
        check_truncation: Whether to check for truncation.
        check_refusals: Whether to check for policy refusals.
        check_tags: Whether to check for unbalanced tags.
        context_4k_threshold: Token threshold for 4K context (default 4096).
        context_8k_threshold: Token threshold for 8K context (default 8192).
        tokens_per_word: Estimated tokens per word for rough counting.
    """

    def __init__(
        self,
        check_turn_pattern: bool = True,
        check_empty_content: bool = True,
        check_invalid_values: bool = True,
        check_truncation: bool = True,
        check_refusals: bool = True,
        check_tags: bool = True,
        context_4k_threshold: int = 4096,
        context_8k_threshold: int = 8192,
        tokens_per_word: float = 1.3,
    ):
        """Initialize the data quality analyzer."""
        self.check_turn_pattern = check_turn_pattern
        self.check_empty_content = check_empty_content
        self.check_invalid_values = check_invalid_values
        self.check_truncation = check_truncation
        self.check_refusals = check_refusals
        self.check_tags = check_tags
        self.context_4k_threshold = context_4k_threshold
        self.context_8k_threshold = context_8k_threshold
        self.tokens_per_word = tokens_per_word

        # Pre-compile refusal patterns
        self._refusal_patterns = [
            re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS
        ]

    def _get_content(self, message: Any) -> str:
        """Extract text content from a message."""
        content = message.content if hasattr(message, "content") else ""
        if not isinstance(content, str):
            content = str(content) if content else ""
        return content

    def _get_role(self, message: Any) -> str:
        """Extract role from a message."""
        role = message.role if hasattr(message, "role") else "unknown"
        if hasattr(role, "value"):
            return role.value
        return str(role)

    def _check_turn_pattern(
        self, conversation: Conversation
    ) -> tuple[bool, str, int]:
        """Check if conversation has proper alternating turns.

        Returns:
            Tuple of (has_alternating, turn_sequence, max_consecutive).
        """
        roles = [self._get_role(m) for m in conversation.messages]
        turn_sequence = ",".join(roles)

        # Filter out system messages for alternation check
        non_system = [r for r in roles if r != "system"]

        if len(non_system) <= 1:
            return True, turn_sequence, 0

        alternating = True
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(non_system)):
            if non_system[i] == non_system[i - 1]:
                alternating = False
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        return alternating, turn_sequence, max_consecutive if not alternating else 0

    def _check_empty_content(
        self, conversation: Conversation
    ) -> tuple[bool, int, list[int]]:
        """Check for empty message content.

        Returns:
            Tuple of (has_empty, count, indices).
        """
        empty_indices = []

        for i, message in enumerate(conversation.messages):
            content = self._get_content(message)
            if len(content.strip()) == 0:
                empty_indices.append(i)

        return len(empty_indices) > 0, len(empty_indices), empty_indices

    def _check_invalid_values(
        self, conversation: Conversation
    ) -> tuple[bool, list[str]]:
        """Check for invalid serialized values.

        Returns:
            Tuple of (has_invalid, patterns_found).
        """
        patterns_found = set()

        for message in conversation.messages:
            content = self._get_content(message)

            for pattern, name in INVALID_VALUE_PATTERNS:
                if re.search(pattern, content):
                    patterns_found.add(name)

        return len(patterns_found) > 0, list(patterns_found)

    def _estimate_tokens(self, conversation: Conversation) -> int:
        """Estimate token count based on word count."""
        total_words = 0
        for message in conversation.messages:
            content = self._get_content(message)
            total_words += len(content.split())

        return int(total_words * self.tokens_per_word)

    def _check_truncation(
        self, conversation: Conversation
    ) -> tuple[bool, bool, str | None]:
        """Check if conversation appears truncated.

        Returns:
            Tuple of (appears_truncated, ends_mid_sentence, reason).
        """
        if not conversation.messages:
            return False, False, None

        last_msg = conversation.messages[-1]
        content = self._get_content(last_msg).rstrip()

        if not content:
            return True, True, "Empty last message"

        last_char = content[-1]

        # Check for proper ending punctuation
        proper_endings = ".!?\"')]}>"
        ends_properly = last_char in proper_endings

        # Check for truncation signs
        truncation_reasons = []

        if content.endswith("...") and len(content) > 100:
            truncation_reasons.append("Ends with ellipsis")

        if content.endswith(","):
            truncation_reasons.append("Ends with comma")

        trailing_words = content.lower().split()[-2:] if len(content.split()) >= 2 else []
        if trailing_words:
            incomplete_phrases = ["and", "the", "a", "an", "to", "of", "in", "for"]
            if trailing_words[-1] in incomplete_phrases:
                truncation_reasons.append(f"Ends with incomplete word '{trailing_words[-1]}'")

        appears_truncated = not ends_properly or len(truncation_reasons) > 0
        reason = "; ".join(truncation_reasons) if truncation_reasons else (
            "No proper ending punctuation" if not ends_properly else None
        )

        return appears_truncated, not ends_properly, reason

    def _check_refusals(
        self, conversation: Conversation
    ) -> tuple[bool, int, list[str]]:
        """Check for policy refusal patterns.

        Returns:
            Tuple of (has_refusal, count, phrases).
        """
        refusal_count = 0
        all_phrases = []

        for message in conversation.messages:
            role = self._get_role(message)
            if role != "assistant":
                continue

            content = self._get_content(message)
            message_phrases = []

            for pattern in self._refusal_patterns:
                if pattern.search(content):
                    message_phrases.append(pattern.pattern)

            if message_phrases:
                refusal_count += 1
                all_phrases.extend(message_phrases)

        return refusal_count > 0, refusal_count, list(set(all_phrases))

    def _check_tags(
        self, conversation: Conversation
    ) -> tuple[bool, bool, list[str]]:
        """Check for think tags and tag balance.

        Returns:
            Tuple of (has_think_tags, has_unbalanced, unmatched_tags).
        """
        has_think_tags = False
        unmatched_tags = []

        for message in conversation.messages:
            content = self._get_content(message)

            for open_tag, close_tag in TAG_PAIRS:
                open_count = content.count(open_tag)
                close_count = content.count(close_tag)

                if open_count > 0 or close_count > 0:
                    has_think_tags = True

                if open_count > close_count:
                    unmatched_tags.extend([open_tag] * (open_count - close_count))
                elif close_count > open_count:
                    unmatched_tags.extend([close_tag] * (close_count - open_count))

            # Check code blocks
            code_block_count = content.count("```")
            if code_block_count % 2 != 0:
                unmatched_tags.append("```")

        return has_think_tags, len(unmatched_tags) > 0, unmatched_tags

    def analyze(self, conversation: Conversation) -> DataQualityMetrics:
        """Analyze data quality metrics for a conversation.

        Args:
            conversation: The conversation to analyze.

        Returns:
            DataQualityMetrics containing quality check results.
        """
        quality_issues = []

        # Turn pattern check
        if self.check_turn_pattern:
            has_alt, turn_seq, max_consec = self._check_turn_pattern(conversation)
            if not has_alt:
                quality_issues.append("Non-alternating turns detected")
        else:
            has_alt, turn_seq, max_consec = True, "", 0

        # Empty content check
        if self.check_empty_content:
            has_empty, empty_count, empty_indices = self._check_empty_content(
                conversation
            )
            if has_empty:
                quality_issues.append(f"{empty_count} empty message(s)")
        else:
            has_empty, empty_count, empty_indices = False, 0, []

        # Invalid values check
        if self.check_invalid_values:
            has_invalid, invalid_patterns = self._check_invalid_values(conversation)
            if has_invalid:
                quality_issues.append(f"Invalid values: {invalid_patterns}")
        else:
            has_invalid, invalid_patterns = False, []

        # Token estimation
        estimated_tokens = self._estimate_tokens(conversation)
        fits_4k = estimated_tokens <= self.context_4k_threshold
        fits_8k = estimated_tokens <= self.context_8k_threshold

        if not fits_4k:
            quality_issues.append(f"Exceeds 4K context ({estimated_tokens} est. tokens)")

        # Truncation check
        if self.check_truncation:
            appears_trunc, ends_mid, trunc_reason = self._check_truncation(conversation)
            if appears_trunc:
                quality_issues.append(f"Appears truncated: {trunc_reason}")
        else:
            appears_trunc, ends_mid, trunc_reason = False, False, None

        # Refusal check
        if self.check_refusals:
            has_refusal, refusal_count, refusal_phrases = self._check_refusals(
                conversation
            )
            if has_refusal:
                quality_issues.append(f"{refusal_count} policy refusal(s)")
        else:
            has_refusal, refusal_count, refusal_phrases = False, 0, []

        # Tag balance check
        if self.check_tags:
            has_tags, has_unbalanced, unmatched = self._check_tags(conversation)
            if has_unbalanced:
                quality_issues.append(f"Unbalanced tags: {unmatched}")
        else:
            has_tags, has_unbalanced, unmatched = False, False, []

        # Overall quality
        passes_quality = len(quality_issues) == 0

        return DataQualityMetrics(
            # Turn pattern
            has_alternating_turns=has_alt,
            turn_sequence=turn_seq,
            num_consecutive_same_role=max_consec,
            # Empty content
            has_empty_turns=has_empty,
            empty_turn_count=empty_count,
            empty_turn_indices=empty_indices,
            # Invalid values
            has_invalid_values=has_invalid,
            invalid_value_patterns=invalid_patterns,
            # Context length
            estimated_tokens=estimated_tokens,
            fits_4k_context=fits_4k,
            fits_8k_context=fits_8k,
            # Truncation
            appears_truncated=appears_trunc,
            ends_mid_sentence=ends_mid,
            truncation_reason=trunc_reason,
            # Refusals
            has_policy_refusal=has_refusal,
            refusal_count=refusal_count,
            refusal_phrases=refusal_phrases,
            # Tags
            has_think_tags=has_tags,
            has_unbalanced_tags=has_unbalanced,
            unmatched_tags=unmatched,
            # Overall
            passes_basic_quality=passes_quality,
            quality_issues=quality_issues,
        )
