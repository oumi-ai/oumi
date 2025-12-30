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

"""Response completeness analyzer for detecting truncated/partial responses.

This analyzer focuses specifically on detecting incomplete, truncated,
or partial responses which is a common issue in synthetic data generation.
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.column_utils import make_analyzer_column_name
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("response_completeness")
class ResponseCompletenessAnalyzer(SampleAnalyzer):
    """Analyzer for detecting incomplete/truncated responses.

    This analyzer specifically targets the detection of:
        - Truncated responses (cut off mid-sentence)
        - Incomplete lists (started but not finished)
        - Partial code blocks (unclosed)
        - Missing conclusions
        - Abandoned explanations

    Metrics computed:
        - is_complete: Boolean flag for complete responses
        - completeness_score: Overall completeness score (0-1)
        - ends_naturally: Whether response has a natural ending
        - has_conclusion: Whether response has a concluding statement
        - truncation_type: Type of truncation detected (if any)
    """

    # Patterns indicating truncation (cut off mid-sentence)
    _TRUNCATION_PATTERNS = [
        # Ends with connector words
        re.compile(
            r"\b(?:and|or|but|the|a|an|to|of|in|for|with|"
            r"that|this|which|who|where|when|because|since|"
            r"although|while|if|as|so|however|therefore)\s*$",
            re.IGNORECASE,
        ),
        # Ends with incomplete phrases
        re.compile(
            r"\b(?:such\s+as|for\s+example|including|like|"
            r"e\.?g\.?|i\.?e\.?|etc)\s*$",
            re.IGNORECASE,
        ),
        # Ends with ellipsis suggesting more to come
        re.compile(r"\.\.\.\s*$"),
        # Ends with comma or colon (expecting more)
        re.compile(r"[,:]\s*$"),
        # Ends with opening bracket/parenthesis
        re.compile(r"[\(\[\{]\s*$"),
    ]

    # Patterns indicating incomplete lists
    _INCOMPLETE_LIST_PATTERNS = [
        # Numbered list that ends abruptly
        re.compile(
            r"^\s*(?:1\.|\(1\)|\*|-)\s+.+(?:\n\s*(?:2\.|\(2\)|\*|-).+)?$", re.MULTILINE
        ),
        # List that mentions "first" but no "second/finally"
        re.compile(
            r"\b(?:first(?:ly)?|1\))\b(?!.*\b(?:second(?:ly)?|finally|lastly|2\)|3\))\b)",
            re.IGNORECASE | re.DOTALL,
        ),
    ]

    # Patterns indicating incomplete code blocks
    _INCOMPLETE_CODE_PATTERNS = [
        # Unclosed code block
        re.compile(r"```\w*\n(?!.*```)", re.DOTALL),
        # Unclosed function/class
        re.compile(r"\b(?:def|function|class)\s+\w+.*:\s*$", re.MULTILINE),
        # Unclosed brackets in code
        re.compile(r"```[\s\S]*?(?:\{[^}]*|\[[^\]]*|\([^)]*)\s*```"),
    ]

    # Patterns indicating natural endings
    _NATURAL_ENDING_PATTERNS = [
        # Proper sentence endings
        re.compile(r"[.!?][\"\'\)]?\s*$"),
        # Code block endings
        re.compile(r"```\s*$"),
        # Closing brackets (for structured data)
        re.compile(r"[\}\]\)]\s*$"),
    ]

    # Patterns indicating conclusions
    _CONCLUSION_PATTERNS = [
        re.compile(
            r"\b(?:in\s+conclusion|to\s+conclude|in\s+summary|"
            r"to\s+summarize|overall|finally|lastly|"
            r"in\s+short|to\s+sum\s+up)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:hope\s+this\s+helps|let\s+me\s+know|"
            r"feel\s+free\s+to\s+ask|if\s+you\s+have\s+(?:any\s+)?"
            r"(?:more\s+)?questions)\b",
            re.IGNORECASE,
        ),
    ]

    # Minimum word count for meaningful response
    _MIN_MEANINGFUL_WORDS = 5

    def __init__(
        self,
        *,
        analyze_assistant_only: bool = True,
        strict_mode: bool = False,
        include_truncation_type: bool = True,
    ):
        """Initialize the ResponseCompletenessAnalyzer.

        Args:
            analyze_assistant_only: If True, only analyze assistant messages.
            strict_mode: If True, require natural endings for completeness.
            include_truncation_type: Include type of truncation detected.
        """
        self.analyze_assistant_only = analyze_assistant_only
        self.strict_mode = strict_mode
        self.include_truncation_type = include_truncation_type

    def _detect_truncation(self, text: str) -> Optional[str]:
        """Detect the type of truncation in the response.

        Args:
            text: Response text.

        Returns:
            Type of truncation or None if not truncated.
        """
        text_stripped = text.strip()

        # Check for truncation patterns
        for pattern in self._TRUNCATION_PATTERNS:
            if pattern.search(text_stripped):
                return "mid_sentence"

        # Check for incomplete lists
        for pattern in self._INCOMPLETE_LIST_PATTERNS:
            if pattern.search(text):
                return "incomplete_list"

        # Check for incomplete code blocks
        for pattern in self._INCOMPLETE_CODE_PATTERNS:
            if pattern.search(text):
                return "incomplete_code"

        return None

    def _ends_naturally(self, text: str) -> bool:
        """Check if the response ends naturally.

        Args:
            text: Response text.

        Returns:
            True if natural ending.
        """
        for pattern in self._NATURAL_ENDING_PATTERNS:
            if pattern.search(text.strip()):
                return True
        return False

    def _has_conclusion(self, text: str) -> bool:
        """Check if the response has a concluding statement.

        Args:
            text: Response text.

        Returns:
            True if has conclusion.
        """
        # Check last 20% of text for conclusion patterns
        text_end = text[int(len(text) * 0.8) :]
        for pattern in self._CONCLUSION_PATTERNS:
            if pattern.search(text_end):
                return True
        return False

    def _compute_completeness_score(
        self,
        text: str,
        truncation_type: Optional[str],
        ends_naturally: bool,
        has_conclusion: bool,
    ) -> float:
        """Compute overall completeness score.

        Args:
            text: Response text.
            truncation_type: Type of truncation detected.
            ends_naturally: Whether response ends naturally.
            has_conclusion: Whether response has conclusion.

        Returns:
            Completeness score (0-1).
        """
        score = 1.0  # Start with complete

        # Truncation is a major penalty
        if truncation_type:
            if truncation_type == "mid_sentence":
                score -= 0.5
            elif truncation_type == "incomplete_code":
                score -= 0.4
            elif truncation_type == "incomplete_list":
                score -= 0.3

        # Natural ending bonus
        if ends_naturally:
            score += 0.1
        else:
            score -= 0.2

        # Conclusion bonus for longer responses
        word_count = len(text.split())
        if word_count > 50 and has_conclusion:
            score += 0.1
        elif word_count > 100 and not has_conclusion:
            score -= 0.1

        # Very short responses are likely incomplete
        if word_count < self._MIN_MEANINGFUL_WORDS:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _analyze_response(self, text: str) -> dict[str, Any]:
        """Analyze a response for completeness.

        Args:
            text: Response text.

        Returns:
            Dictionary of completeness metrics.
        """
        if not text or not text.strip():
            return {
                "is_complete": False,
                "completeness_score": 0.0,
                "ends_naturally": False,
                "has_conclusion": False,
                "truncation_type": "empty",
            }

        truncation_type = self._detect_truncation(text)
        ends_naturally = self._ends_naturally(text)
        has_conclusion = self._has_conclusion(text)

        completeness_score = self._compute_completeness_score(
            text, truncation_type, ends_naturally, has_conclusion
        )

        # Determine if complete
        if self.strict_mode:
            is_complete = truncation_type is None and ends_naturally
        else:
            is_complete = completeness_score >= 0.7

        result = {
            "is_complete": is_complete,
            "completeness_score": round(completeness_score, 3),
            "ends_naturally": ends_naturally,
            "has_conclusion": has_conclusion,
        }

        if self.include_truncation_type:
            result["truncation_type"] = truncation_type or ""

        return result

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields for response completeness.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added completeness analysis columns.
            generated column schema dict).
        """
        result_df = df.copy()
        generated_schema = {}

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for response "
                "completeness analysis. Please provide a column schema dict "
                "that specifies which columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df, generated_schema

        # Find the role column if needed
        role_column = None
        if self.analyze_assistant_only:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and col in df.columns
                    and "role" in col.lower()
                ):
                    role_column = col
                    break

        analyzer_id = getattr(self, "analyzer_id", "response_completeness")

        for column in text_columns:
            if self.analyze_assistant_only and role_column is not None:
                # Only analyze assistant messages
                analysis_results = df.apply(
                    lambda row: (
                        self._analyze_response(str(row[column]))
                        if str(row.get(role_column, "")).lower() == "assistant"
                        else {
                            "is_complete": None,
                            "completeness_score": None,
                            "ends_naturally": None,
                            "has_conclusion": None,
                            "truncation_type": None,
                        }
                    ),
                    axis=1,
                )
            else:
                analysis_results = df[column].astype(str).apply(self._analyze_response)

            # Extract results to columns
            col_name = make_analyzer_column_name(column, analyzer_id, "is_complete")
            result_df[col_name] = analysis_results.apply(lambda r: r.get("is_complete"))
            generated_schema[col_name] = {
                "type": ColumnType.BOOL,
                "content_type": ContentType.BOOLEAN,
                "description": "Whether response appears complete",
            }

            col_name = make_analyzer_column_name(column, analyzer_id, "score")
            result_df[col_name] = analysis_results.apply(
                lambda r: r.get("completeness_score")
            )
            generated_schema[col_name] = {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": "Completeness score (0.0 = incomplete, 1.0 = complete)",
            }

            col_name = make_analyzer_column_name(column, analyzer_id, "ends_naturally")
            result_df[col_name] = analysis_results.apply(
                lambda r: r.get("ends_naturally")
            )
            generated_schema[col_name] = {
                "type": ColumnType.BOOL,
                "content_type": ContentType.BOOLEAN,
                "description": "Whether response ends naturally",
            }

            col_name = make_analyzer_column_name(column, analyzer_id, "has_conclusion")
            result_df[col_name] = analysis_results.apply(
                lambda r: r.get("has_conclusion")
            )
            generated_schema[col_name] = {
                "type": ColumnType.BOOL,
                "content_type": ContentType.BOOLEAN,
                "description": "Whether response has a conclusion",
            }

            if self.include_truncation_type:
                col_name = make_analyzer_column_name(column, analyzer_id, "truncation_type")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("truncation_type")
                )
                generated_schema[col_name] = {
                    "type": ColumnType.STRING,
                    "content_type": ContentType.CATEGORICAL,
                    "description": "Type of truncation detected (if any)",
                }

        return result_df, generated_schema
