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

"""Training quality analyzer for SFT instruction datasets.

This analyzer evaluates the quality of instruction-response pairs for
supervised fine-tuning, providing metrics that predict training effectiveness.
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("training_quality")
class TrainingQualityAnalyzer(SampleAnalyzer):
    """Analyzer for evaluating SFT training quality metrics.

    This analyzer computes metrics that predict how well instruction-response
    pairs will train a language model:

    Instruction Clarity Metrics:
        - instruction_clarity_score: Composite score (0-1) for instruction quality
        - has_clear_intent: Whether the instruction has a clear goal
        - has_specificity: Whether the instruction includes specific details
        - instruction_word_count: Word count for the instruction

    Response Completeness Metrics:
        - response_completeness_score: Composite score (0-1) for response quality
        - has_proper_ending: Whether the response ends properly (not truncated)
        - has_structure: Whether the response uses lists, code blocks, etc.
        - response_word_count: Word count for the response

    Turn Quality Metrics (for multi-turn conversations):
        - turn_quality_score: Composite score (0-1) for turn quality
        - references_context: Whether the message references previous context
        - role_appropriate: Whether the message is appropriate for its role
    """

    # Imperative verbs that indicate clear instructions
    _IMPERATIVE_VERBS = {
        "write",
        "create",
        "generate",
        "explain",
        "describe",
        "list",
        "summarize",
        "analyze",
        "compare",
        "translate",
        "convert",
        "implement",
        "design",
        "develop",
        "build",
        "fix",
        "debug",
        "optimize",
        "refactor",
        "review",
        "edit",
        "rewrite",
        "provide",
        "give",
        "show",
        "tell",
        "help",
        "find",
        "search",
        "calculate",
        "compute",
        "solve",
        "answer",
        "respond",
        "define",
        "clarify",
        "elaborate",
        "expand",
        "simplify",
        "outline",
        "draft",
        "compose",
        "format",
        "organize",
        "classify",
        "categorize",
        "identify",
        "extract",
        "parse",
        "validate",
        "verify",
        "check",
        "test",
        "evaluate",
        "assess",
        "rate",
        "rank",
        "recommend",
        "suggest",
        "propose",
    }

    # Vague terms that reduce instruction clarity
    _VAGUE_TERMS = {
        "something",
        "stuff",
        "things",
        "whatever",
        "anything",
        "somehow",
        "somewhat",
        "kind of",
        "sort of",
        "basically",
        "actually",
        "really",
        "very",
        "maybe",
        "perhaps",
        "probably",
        "etc",
        "and so on",
        "and stuff",
    }

    # Patterns for detecting specificity
    _NUMBER_PATTERN = re.compile(r"\b\d+\b")
    _QUOTED_PATTERN = re.compile(r'["\'][^"\']+["\']')
    _CODE_INDICATOR_PATTERN = re.compile(
        r"`[^`]+`|```|\b(?:function|class|def|var|const|let|import|from)\b"
    )

    # Patterns for proper endings
    _TRUNCATION_PATTERNS = [
        re.compile(r"\.\.\.$"),  # Trailing ellipsis
        re.compile(r"[^.!?)\]\"'`]$"),  # No punctuation at end
        re.compile(r"\b(?:and|or|but|the|a|an|to|of|in|for|with)\s*$", re.IGNORECASE),
    ]

    # Patterns for structured responses
    _STRUCTURE_PATTERNS = [
        re.compile(r"^\s*[-*]\s+", re.MULTILINE),  # Bullet points
        re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE),  # Numbered lists
        re.compile(r"```[\s\S]*?```"),  # Code blocks
        re.compile(r"^\s*#{1,6}\s+", re.MULTILINE),  # Headers
        re.compile(r"\*\*[^*]+\*\*"),  # Bold text
        re.compile(r"^\s*>\s+", re.MULTILINE),  # Block quotes
    ]

    # Context reference patterns
    _CONTEXT_REFERENCE_PATTERNS = [
        re.compile(
            r"\b(?:you mentioned|as you said|earlier|above|previously|before)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:your question|your request|you asked|you wanted)\b", re.IGNORECASE
        ),
        re.compile(r"\b(?:the code|the example|the function|the error)\b", re.IGNORECASE),
        re.compile(r"\b(?:this|that|these|those|it)\s+(?:is|are|was|were)\b"),
    ]

    def __init__(
        self,
        *,
        compute_instruction_clarity: bool = True,
        compute_response_completeness: bool = True,
        compute_turn_quality: bool = True,
        min_instruction_words: int = 3,
        max_instruction_words: int = 500,
        min_response_words: int = 5,
        clarity_vague_penalty: float = 0.15,
        clarity_no_verb_penalty: float = 0.2,
        clarity_too_short_penalty: float = 0.3,
        clarity_too_long_penalty: float = 0.1,
    ):
        """Initialize the TrainingQualityAnalyzer.

        Args:
            compute_instruction_clarity: Whether to compute instruction clarity metrics.
            compute_response_completeness: Whether to compute response completeness.
            compute_turn_quality: Whether to compute turn quality metrics.
            min_instruction_words: Minimum words for a good instruction.
            max_instruction_words: Maximum words before instruction is too long.
            min_response_words: Minimum words for a complete response.
            clarity_vague_penalty: Penalty for each vague term found.
            clarity_no_verb_penalty: Penalty if no imperative verb found.
            clarity_too_short_penalty: Penalty if instruction is too short.
            clarity_too_long_penalty: Penalty if instruction is too long.
        """
        self.compute_instruction_clarity = compute_instruction_clarity
        self.compute_response_completeness = compute_response_completeness
        self.compute_turn_quality = compute_turn_quality
        self.min_instruction_words = min_instruction_words
        self.max_instruction_words = max_instruction_words
        self.min_response_words = min_response_words
        self.clarity_vague_penalty = clarity_vague_penalty
        self.clarity_no_verb_penalty = clarity_no_verb_penalty
        self.clarity_too_short_penalty = clarity_too_short_penalty
        self.clarity_too_long_penalty = clarity_too_long_penalty

    def _compute_instruction_clarity(self, text: str) -> dict[str, Any]:
        """Compute instruction clarity metrics.

        Args:
            text: Instruction text to analyze.

        Returns:
            Dictionary with clarity metrics.
        """
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        # Start with perfect score
        score = 1.0
        has_clear_intent = False
        has_specificity = False

        # Check for imperative verbs (clear intent)
        for word in words[:10]:  # Check first 10 words
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word in self._IMPERATIVE_VERBS:
                has_clear_intent = True
                break

        # Check for question marks (also indicates clear intent)
        if "?" in text:
            has_clear_intent = True

        if not has_clear_intent:
            score -= self.clarity_no_verb_penalty

        # Check for vague terms
        vague_count = 0
        for term in self._VAGUE_TERMS:
            if term in text_lower:
                vague_count += 1

        score -= vague_count * self.clarity_vague_penalty

        # Check for specificity indicators
        has_numbers = bool(self._NUMBER_PATTERN.search(text))
        has_quotes = bool(self._QUOTED_PATTERN.search(text))
        has_code = bool(self._CODE_INDICATOR_PATTERN.search(text))

        if has_numbers or has_quotes or has_code:
            has_specificity = True
            score += 0.1  # Bonus for specificity

        # Check length appropriateness
        if word_count < self.min_instruction_words:
            score -= self.clarity_too_short_penalty
        elif word_count > self.max_instruction_words:
            score -= self.clarity_too_long_penalty

        # Clamp score
        score = max(0.0, min(1.0, score))

        return {
            "instruction_clarity_score": round(score, 3),
            "has_clear_intent": has_clear_intent,
            "has_specificity": has_specificity,
            "instruction_word_count": word_count,
        }

    def _compute_response_completeness(self, text: str) -> dict[str, Any]:
        """Compute response completeness metrics.

        Args:
            text: Response text to analyze.

        Returns:
            Dictionary with completeness metrics.
        """
        words = text.split()
        word_count = len(words)

        score = 1.0
        has_proper_ending = True
        has_structure = False

        # Check for truncation patterns
        text_stripped = text.strip()
        for pattern in self._TRUNCATION_PATTERNS:
            if pattern.search(text_stripped):
                has_proper_ending = False
                score -= 0.3
                break

        # Check for structure
        for pattern in self._STRUCTURE_PATTERNS:
            if pattern.search(text):
                has_structure = True
                score += 0.05  # Small bonus for structure
                break

        # Check minimum length
        if word_count < self.min_response_words:
            score -= 0.3

        # Empty or near-empty responses
        if word_count < 2:
            score = 0.0

        # Clamp score
        score = max(0.0, min(1.0, score))

        return {
            "response_completeness_score": round(score, 3),
            "has_proper_ending": has_proper_ending,
            "has_structure": has_structure,
            "response_word_count": word_count,
        }

    def _compute_turn_quality(self, text: str, role: str) -> dict[str, Any]:
        """Compute turn quality metrics for multi-turn conversations.

        Args:
            text: Message text to analyze.
            role: Role of the message (user, assistant, system).

        Returns:
            Dictionary with turn quality metrics.
        """
        score = 1.0
        references_context = False
        role_appropriate = True

        # Check for context references
        for pattern in self._CONTEXT_REFERENCE_PATTERNS:
            if pattern.search(text):
                references_context = True
                break

        # Role-appropriate checks
        role_lower = role.lower() if role else ""

        if role_lower == "assistant":
            # Assistant should not ask for clarification excessively
            clarification_patterns = re.compile(
                r"\b(?:could you|can you|please provide|what do you mean|"
                r"i need more|not sure what)\b",
                re.IGNORECASE,
            )
            clarification_count = len(clarification_patterns.findall(text))
            if clarification_count > 2:
                role_appropriate = False
                score -= 0.2

        elif role_lower == "user":
            # User instructions should have some substance
            if len(text.split()) < 3:
                role_appropriate = False
                score -= 0.2

        elif role_lower == "system":
            # System messages should be directive
            pass  # System messages are usually fine

        # Clamp score
        score = max(0.0, min(1.0, score))

        return {
            "turn_quality_score": round(score, 3),
            "references_context": references_context,
            "role_appropriate": role_appropriate,
        }

    def _analyze_message(self, text: str, role: str) -> dict[str, Any]:
        """Analyze a single message for training quality.

        Args:
            text: Message text to analyze.
            role: Role of the message.

        Returns:
            Dictionary of training quality metrics.
        """
        results = {}
        role_lower = role.lower() if role else ""

        # Compute instruction clarity for user messages
        if self.compute_instruction_clarity and role_lower == "user":
            clarity_results = self._compute_instruction_clarity(text)
            results.update(clarity_results)

        # Compute response completeness for assistant messages
        if self.compute_response_completeness and role_lower == "assistant":
            completeness_results = self._compute_response_completeness(text)
            results.update(completeness_results)

        # Compute turn quality for all messages
        if self.compute_turn_quality:
            turn_results = self._compute_turn_quality(text, role)
            results.update(turn_results)

        return results

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for training quality metrics.

        This analyzer is role-aware:
        - User messages get instruction clarity metrics
        - Assistant messages get response completeness metrics
        - All messages get turn quality metrics

        Args:
            df: Input DataFrame with text fields and role column.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added training quality analysis columns.
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for training quality "
                "analysis. Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df

        # Find the role column
        role_column = None
        for col, config in schema.items():
            if (
                config.get("content_type") == ContentType.CATEGORICAL
                and col in df.columns
                and "role" in col.lower()
            ):
                role_column = col
                break

        analyzer_id = getattr(self, "analyzer_id", "training_quality")

        for column in text_columns:
            # Analyze each row with its role
            if role_column is not None:
                analysis_results = df.apply(
                    lambda row: self._analyze_message(
                        str(row[column]), str(row.get(role_column, ""))
                    ),
                    axis=1,
                )
            else:
                # No role column - analyze as generic text
                analysis_results = df[column].astype(str).apply(
                    lambda text: self._analyze_message(text, "")
                )

            # Extract instruction clarity metrics (for user messages)
            if self.compute_instruction_clarity:
                result_df[
                    f"{column}_{analyzer_id}_instruction_clarity_score"
                ] = analysis_results.apply(
                    lambda r: r.get("instruction_clarity_score", None)
                )
                result_df[
                    f"{column}_{analyzer_id}_has_clear_intent"
                ] = analysis_results.apply(lambda r: r.get("has_clear_intent", None))
                result_df[
                    f"{column}_{analyzer_id}_has_specificity"
                ] = analysis_results.apply(lambda r: r.get("has_specificity", None))
                result_df[
                    f"{column}_{analyzer_id}_instruction_word_count"
                ] = analysis_results.apply(
                    lambda r: r.get("instruction_word_count", None)
                )

            # Extract response completeness metrics (for assistant messages)
            if self.compute_response_completeness:
                result_df[
                    f"{column}_{analyzer_id}_response_completeness_score"
                ] = analysis_results.apply(
                    lambda r: r.get("response_completeness_score", None)
                )
                result_df[
                    f"{column}_{analyzer_id}_has_proper_ending"
                ] = analysis_results.apply(lambda r: r.get("has_proper_ending", None))
                result_df[
                    f"{column}_{analyzer_id}_has_structure"
                ] = analysis_results.apply(lambda r: r.get("has_structure", None))
                result_df[
                    f"{column}_{analyzer_id}_response_word_count"
                ] = analysis_results.apply(
                    lambda r: r.get("response_word_count", None)
                )

            # Extract turn quality metrics (for all messages)
            if self.compute_turn_quality:
                result_df[
                    f"{column}_{analyzer_id}_turn_quality_score"
                ] = analysis_results.apply(lambda r: r.get("turn_quality_score", None))
                result_df[
                    f"{column}_{analyzer_id}_references_context"
                ] = analysis_results.apply(lambda r: r.get("references_context", None))
                result_df[
                    f"{column}_{analyzer_id}_role_appropriate"
                ] = analysis_results.apply(lambda r: r.get("role_appropriate", None))

        return result_df
