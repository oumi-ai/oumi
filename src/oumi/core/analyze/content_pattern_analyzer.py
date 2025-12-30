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

"""Content pattern analyzer for detecting AI-specific quality issues in datasets.

This analyzer identifies quality issues commonly found in AI-generated training data:
- Placeholder text ([Name], [Product Name], etc.)
- AI hallucinated personal experiences
- Nooutput tags and AI refusals
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.column_utils import make_analyzer_column_name
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("content_pattern")
class ContentPatternAnalyzer(SampleAnalyzer):
    """Analyzer for detecting AI-specific content quality issues.

    This analyzer identifies common quality problems in AI-generated training data:
        - Placeholder text: [Name], [Product Name], [Your...], [Insert...], etc.
        - AI hallucinated experiences: Fabricated first-person stories
        - Nooutput markers: <nooutput>, N/A, and similar non-responses
        - AI refusals: "I cannot provide...", "I'm unable to..."

    Quality metrics computed:
        - has_placeholder: Boolean indicating placeholder text detected
        - placeholder_count: Number of placeholders found
        - placeholder_types: Types of placeholders detected
        - has_hallucinated_experience: Boolean for AI fabricated stories
        - has_nooutput: Boolean for nooutput/NA markers
        - has_refusal: Boolean for AI refusal patterns
    """

    # Bracketed placeholders: [Name], [Product Name], [Your Name], etc.
    _BRACKET_PLACEHOLDER_PATTERN = re.compile(
        r"\[(?:Name|Product\s*Name|Company\s*Name|Your\s*\w*|Insert\s*\w*|"
        r"Add\s*\w*|Fill\s*(?:in\s*)?\w*|Replace\s*(?:with\s*)?\w*|"
        r"Enter\s*\w*|Specify\s*\w*|X{2,}|Placeholder|TBD|TODO|FIXME|"
        r"Person|Place|Date|Time|Location|Address|City|Country|"
        r"Email|Phone|Number|Amount|Price|Value|Title|Subject|Topic)\]",
        re.IGNORECASE,
    )

    # Angle bracket placeholders: <name>, <your_company>, etc.
    _ANGLE_PLACEHOLDER_PATTERN = re.compile(
        r"<(?:name|your[_\s]?\w*|company[_\s]?\w*|product[_\s]?\w*|"
        r"insert[_\s]?\w*|fill[_\s]?\w*|placeholder|tbd|todo|"
        r"person|place|date|time|email|phone)\s*/?>",
        re.IGNORECASE,
    )

    # AI hallucinated experience patterns
    _HALLUCINATED_EXPERIENCE_PATTERNS = [
        # "I had to... when I was working as..."
        re.compile(
            r"\bI\s+(?:had\s+to|needed\s+to|was\s+asked\s+to|decided\s+to|"
            r"chose\s+to)\s+\w+.*?(?:when\s+I\s+(?:was|worked)|"
            r"during\s+my\s+(?:time|tenure|career))",
            re.IGNORECASE,
        ),
        # "When I was a [profession]..."
        re.compile(
            r"\bWhen\s+I\s+was\s+(?:a\s+|an\s+)?(?:project\s+manager|"
            r"software\s+engineer|developer|doctor|nurse|teacher|lawyer|"
            r"manager|director|CEO|founder|consultant|analyst|designer|"
            r"researcher|scientist|professor|engineer|accountant|"
            r"marketing\s+manager|sales\s+representative|HR\s+manager)\b",
            re.IGNORECASE,
        ),
        # "In my experience as a..."
        re.compile(
            r"\b(?:In\s+my|From\s+my|Based\s+on\s+my)\s+"
            r"(?:experience|time|years?|work|career)\s+"
            r"(?:as\s+|at\s+|in\s+|with\s+|working\s+)?(?:a\s+|an\s+)?",
            re.IGNORECASE,
        ),
        # "During my career/time at..."
        re.compile(
            r"\bDuring\s+my\s+(?:career|time|tenure|work|years?)\s+"
            r"(?:at|with|in|as)\s+",
            re.IGNORECASE,
        ),
        # "I recently visited/attended..."
        re.compile(
            r"\bI\s+recently\s+(?:visited|attended|went\s+to|completed|"
            r"finished|started|joined)\s+",
            re.IGNORECASE,
        ),
    ]

    # Nooutput and non-response patterns
    _NOOUTPUT_PATTERNS = [
        re.compile(r"<\s*nooutput\s*>", re.IGNORECASE),
        re.compile(r"<\s*no[_\s-]?output\s*>", re.IGNORECASE),
        re.compile(r"\[no[_\s-]?output\]", re.IGNORECASE),
        re.compile(r"^\s*N/?A\s*$"),
        re.compile(r"^\s*\[N/?A\]\s*$"),
        re.compile(r"^\s*None\s*$", re.IGNORECASE),
        re.compile(r"^\s*-\s*$"),
    ]

    # AI refusal patterns
    _REFUSAL_PATTERNS = [
        re.compile(
            r"\bI(?:'m|\s+am)?\s*(?:cannot|can't|unable\s+to|won't|will\s+not|"
            r"refuse\s+to|not\s+able\s+to)\s+"
            r"(?:provide|generate|create|write|produce|help|"
            r"assist|complete|fulfill)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:This|That|The)\s+(?:request|question|prompt|task)\s+"
            r"(?:is|seems|appears)\s+(?:inappropriate|harmful|unethical|"
            r"impossible|beyond\s+my\s+capabilities)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:This|That)\s+(?:type\s+of\s+)?(?:instruction|request)\s+"
            r"cannot\s+be\s+(?:fulfilled|completed|processed)",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        *,
        detect_placeholders: bool = True,
        detect_hallucinated_experiences: bool = True,
        detect_nooutput: bool = True,
        detect_refusals: bool = True,
        placeholder_whitelist: Optional[list[str]] = None,
        check_output_only: bool = False,
    ):
        """Initialize the ContentPatternAnalyzer.

        Args:
            detect_placeholders: Whether to detect placeholder text like [Name].
            detect_hallucinated_experiences: Whether to detect AI fabricated stories.
            detect_nooutput: Whether to detect nooutput/NA markers.
            detect_refusals: Whether to detect AI refusal patterns.
            placeholder_whitelist: List of placeholder patterns to ignore.
            check_output_only: If True, only analyze assistant/output messages.
        """
        self.detect_placeholders = detect_placeholders
        self.detect_hallucinated_experiences = detect_hallucinated_experiences
        self.detect_nooutput = detect_nooutput
        self.detect_refusals = detect_refusals
        self.placeholder_whitelist = set(placeholder_whitelist or [])
        self.check_output_only = check_output_only

    def _detect_placeholders(self, text: str) -> dict[str, Any]:
        """Detect placeholder text in content.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with placeholder detection results.
        """
        placeholder_types = []
        total_count = 0

        # Check bracketed placeholders
        bracket_matches = self._BRACKET_PLACEHOLDER_PATTERN.findall(text)
        for match in bracket_matches:
            if match not in self.placeholder_whitelist:
                placeholder_types.append("bracket")
                total_count += 1

        # Check angle bracket placeholders
        angle_matches = self._ANGLE_PLACEHOLDER_PATTERN.findall(text)
        for match in angle_matches:
            if match not in self.placeholder_whitelist:
                placeholder_types.append("angle")
                total_count += 1

        return {
            "has_placeholder": total_count > 0,
            "placeholder_count": total_count,
            "placeholder_types": ",".join(set(placeholder_types))
            if placeholder_types
            else "",
        }

    def _detect_hallucinated_experience(self, text: str) -> bool:
        """Detect AI hallucinated personal experiences.

        Args:
            text: Input text to analyze.

        Returns:
            True if hallucinated experience patterns are detected.
        """
        for pattern in self._HALLUCINATED_EXPERIENCE_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _detect_nooutput(self, text: str) -> bool:
        """Detect nooutput/NA markers.

        Args:
            text: Input text to analyze.

        Returns:
            True if nooutput patterns are detected.
        """
        text_stripped = text.strip()

        # Check if entire response is very short and matches nooutput
        if len(text_stripped) < 50:
            for pattern in self._NOOUTPUT_PATTERNS:
                if pattern.search(text_stripped):
                    return True

        # Also check for nooutput tags anywhere in text
        if re.search(r"<\s*nooutput\s*>", text, re.IGNORECASE):
            return True

        return False

    def _detect_refusal(self, text: str) -> bool:
        """Detect AI refusal patterns.

        Args:
            text: Input text to analyze.

        Returns:
            True if refusal patterns are detected.
        """
        for pattern in self._REFUSAL_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _analyze_text(self, text: str) -> dict[str, Any]:
        """Analyze a single text sample for content pattern issues.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary of content pattern metrics.
        """
        results = {}

        # Placeholder detection
        if self.detect_placeholders:
            placeholder_results = self._detect_placeholders(text)
            results.update(placeholder_results)

        # Hallucinated experience detection
        if self.detect_hallucinated_experiences:
            results["has_hallucinated_experience"] = (
                self._detect_hallucinated_experience(text)
            )

        # Nooutput detection
        if self.detect_nooutput:
            results["has_nooutput"] = self._detect_nooutput(text)

        # Refusal detection
        if self.detect_refusals:
            results["has_refusal"] = self._detect_refusal(text)

        return results

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields for content pattern issues.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added content pattern analysis columns.
            generated column schema dict).
        """
        result_df = df.copy()
        generated_schema = {}

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for content pattern "
                "analysis. Please provide a column schema dict that specifies "
                "which columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df, generated_schema

        # Find the role column if we need to filter by role
        role_column = None
        if self.check_output_only:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and col in df.columns
                    and "role" in col.lower()
                ):
                    role_column = col
                    break

        analyzer_id = getattr(self, "analyzer_id", "content_pattern")

        for column in text_columns:
            # Analyze all texts in the column
            if self.check_output_only and role_column is not None:
                # Only analyze assistant/output messages
                analysis_results = df.apply(
                    lambda row: (
                        self._analyze_text(str(row[column]))
                        if str(row.get(role_column, "")).lower() == "assistant"
                        else {}
                    ),
                    axis=1,
                )
            else:
                # Analyze all messages
                analysis_results = df[column].astype(str).apply(self._analyze_text)

            # Add columns for each metric
            if self.detect_placeholders:
                col_name = make_analyzer_column_name(column, analyzer_id, "has_placeholder")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("has_placeholder", None)
                )
                generated_schema[col_name] = {
                    "type": ColumnType.BOOL,
                    "content_type": ContentType.BOOLEAN,
                    "description": "Whether text contains placeholder patterns",
                }

                col_name = make_analyzer_column_name(column, analyzer_id, "placeholder_count")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("placeholder_count", None)
                )
                generated_schema[col_name] = {
                    "type": ColumnType.INT,
                    "content_type": ContentType.NUMERIC,
                    "description": "Number of placeholder patterns detected",
                }

                col_name = make_analyzer_column_name(column, analyzer_id, "placeholder_types")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("placeholder_types", None)
                )
                generated_schema[col_name] = {
                    "type": ColumnType.STRING,
                    "content_type": ContentType.LIST,
                    "description": "Comma-separated list of placeholder types detected",
                }

            if self.detect_hallucinated_experiences:
                col_name = make_analyzer_column_name(column, analyzer_id, "has_hallucinated_experience")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("has_hallucinated_experience", None)
                )
                generated_schema[col_name] = {
                    "type": ColumnType.BOOL,
                    "content_type": ContentType.BOOLEAN,
                    "description": "Whether text contains hallucinated personal experience",
                }

            if self.detect_nooutput:
                col_name = make_analyzer_column_name(column, analyzer_id, "has_nooutput")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("has_nooutput", None)
                )
                generated_schema[col_name] = {
                    "type": ColumnType.BOOL,
                    "content_type": ContentType.BOOLEAN,
                    "description": "Whether response indicates no output/empty response",
                }

            if self.detect_refusals:
                col_name = make_analyzer_column_name(column, analyzer_id, "has_refusal")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("has_refusal", None)
                )
                generated_schema[col_name] = {
                    "type": ColumnType.BOOL,
                    "content_type": ContentType.BOOLEAN,
                    "description": "Whether response contains a refusal pattern",
                }

        return result_df, generated_schema
