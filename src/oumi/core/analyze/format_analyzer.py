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

"""Format analyzer for detecting structural patterns in text content."""

import json
import re
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("format")
class FormatAnalyzer(SampleAnalyzer):
    """Analyzer that detects formatting patterns and structural elements in text.

    This analyzer identifies various format patterns commonly found in training
    data, such as markdown formatting, code blocks, JSON content, URLs, and
    email addresses. It's useful for understanding the structure of your dataset
    and filtering samples based on content type.

    Detection features:
        - has_markdown: Detect markdown formatting (headers, lists, bold, italic, links)
        - has_json_blocks: Detect JSON content (code blocks or inline)
        - has_code_blocks: Detect fenced code blocks (```)
        - code_block_count: Count the number of code blocks
        - code_block_languages: Extract programming languages from code blocks
        - has_urls: Detect URLs in text
        - has_emails: Detect email addresses
        - format_complexity_score: Composite score of formatting complexity
    """

    # Compiled regex patterns for efficiency
    _MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)
    _MARKDOWN_LIST_PATTERN = re.compile(
        r"^[\s]*[-*+]\s+.+|^\s*\d+\.\s+.+", re.MULTILINE
    )
    _MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
    _MARKDOWN_ITALIC_PATTERN = re.compile(r"\*[^*]+\*|_[^_]+_")
    _MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    _MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    _CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    _INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")

    _JSON_BLOCK_PATTERN = re.compile(
        r"```(?:json)?\s*\n(\{.*?\}|\[.*?\])\s*```", re.DOTALL
    )
    _INLINE_JSON_PATTERN = re.compile(
        r"(?<![a-zA-Z])(\{[^{}]*\}|\[[^\[\]]*\])(?![a-zA-Z])"
    )

    _URL_PATTERN = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
        r"(?:/[^\s\]\)\"\'>]*)?"
    )

    _EMAIL_PATTERN = re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    )

    def __init__(
        self,
        *,
        detect_markdown: bool = True,
        detect_json: bool = True,
        detect_code_blocks: bool = True,
        detect_urls: bool = True,
        detect_emails: bool = False,
        compute_complexity: bool = True,
    ):
        """Initialize the FormatAnalyzer.

        Args:
            detect_markdown: Whether to detect markdown formatting elements.
                Includes headers, lists, bold, italic, links, and images.
            detect_json: Whether to detect JSON content in the text.
                Includes both fenced JSON code blocks and inline JSON.
            detect_code_blocks: Whether to detect fenced code blocks.
                Also extracts the count and languages of code blocks.
            detect_urls: Whether to detect URLs in the text.
            detect_emails: Whether to detect email addresses in the text.
                Disabled by default as it's less commonly needed.
            compute_complexity: Whether to compute a format complexity score.
                This is a composite score based on the presence of various
                formatting elements.
        """
        self.detect_markdown = detect_markdown
        self.detect_json = detect_json
        self.detect_code_blocks = detect_code_blocks
        self.detect_urls = detect_urls
        self.detect_emails = detect_emails
        self.compute_complexity = compute_complexity

    def _has_markdown(self, text: str) -> bool:
        """Check if text contains markdown formatting.

        Args:
            text: Input text to check.

        Returns:
            True if markdown formatting is detected.
        """
        patterns = [
            self._MARKDOWN_HEADER_PATTERN,
            self._MARKDOWN_LIST_PATTERN,
            self._MARKDOWN_BOLD_PATTERN,
            self._MARKDOWN_ITALIC_PATTERN,
            self._MARKDOWN_LINK_PATTERN,
            self._MARKDOWN_IMAGE_PATTERN,
        ]
        return any(pattern.search(text) for pattern in patterns)

    def _has_json(self, text: str) -> bool:
        """Check if text contains JSON content.

        Args:
            text: Input text to check.

        Returns:
            True if JSON content is detected.
        """
        # Check for JSON code blocks
        if self._JSON_BLOCK_PATTERN.search(text):
            return True

        # Check for inline JSON and validate it
        for match in self._INLINE_JSON_PATTERN.finditer(text):
            try:
                json.loads(match.group(1))
                return True
            except (json.JSONDecodeError, ValueError):
                continue

        return False

    def _get_code_block_info(self, text: str) -> tuple[bool, int, list[str]]:
        """Extract code block information from text.

        Args:
            text: Input text to analyze.

        Returns:
            Tuple of (has_code_blocks, count, languages).
        """
        matches = self._CODE_BLOCK_PATTERN.findall(text)
        if not matches:
            return False, 0, []

        languages = [lang.strip().lower() for lang, _ in matches if lang.strip()]
        return True, len(matches), languages

    def _has_urls(self, text: str) -> bool:
        """Check if text contains URLs.

        Args:
            text: Input text to check.

        Returns:
            True if URLs are detected.
        """
        return bool(self._URL_PATTERN.search(text))

    def _has_emails(self, text: str) -> bool:
        """Check if text contains email addresses.

        Args:
            text: Input text to check.

        Returns:
            True if email addresses are detected.
        """
        return bool(self._EMAIL_PATTERN.search(text))

    def _compute_complexity_score(
        self,
        has_markdown: bool,
        has_json: bool,
        has_code_blocks: bool,
        code_block_count: int,
        has_urls: bool,
        has_emails: bool,
    ) -> float:
        """Compute a format complexity score.

        The score is a weighted sum of detected formatting features,
        normalized to a 0-1 scale.

        Args:
            has_markdown: Whether markdown was detected.
            has_json: Whether JSON was detected.
            has_code_blocks: Whether code blocks were detected.
            code_block_count: Number of code blocks.
            has_urls: Whether URLs were detected.
            has_emails: Whether emails were detected.

        Returns:
            Complexity score between 0 and 1.
        """
        # Weights for different features
        score = 0.0
        max_score = 0.0

        # Markdown (weight: 2)
        max_score += 2.0
        if has_markdown:
            score += 2.0

        # JSON (weight: 2)
        max_score += 2.0
        if has_json:
            score += 2.0

        # Code blocks (weight: 3, plus bonus for multiple blocks)
        max_score += 4.0  # Base + max bonus
        if has_code_blocks:
            score += 3.0
            # Bonus for multiple code blocks (max 1.0)
            score += min(1.0, (code_block_count - 1) * 0.5)

        # URLs (weight: 1)
        max_score += 1.0
        if has_urls:
            score += 1.0

        # Emails (weight: 1)
        max_score += 1.0
        if has_emails:
            score += 1.0

        return score / max_score if max_score > 0 else 0.0

    def _analyze_text(self, text: str) -> dict[str, any]:
        """Analyze a single text sample for format features.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary of detected features.
        """
        results = {}

        # Detect markdown
        has_markdown = self._has_markdown(text) if self.detect_markdown else False
        if self.detect_markdown:
            results["has_markdown"] = has_markdown

        # Detect JSON
        has_json = self._has_json(text) if self.detect_json else False
        if self.detect_json:
            results["has_json"] = has_json

        # Detect code blocks
        if self.detect_code_blocks:
            code_info = self._get_code_block_info(text)
            has_code_blocks, code_block_count, languages = code_info
            results["has_code_blocks"] = has_code_blocks
            results["code_block_count"] = code_block_count
            results["code_block_languages"] = ",".join(languages) if languages else ""
        else:
            has_code_blocks = False
            code_block_count = 0

        # Detect URLs
        has_urls = self._has_urls(text) if self.detect_urls else False
        if self.detect_urls:
            results["has_urls"] = has_urls

        # Detect emails
        has_emails = self._has_emails(text) if self.detect_emails else False
        if self.detect_emails:
            results["has_emails"] = has_emails

        # Compute complexity score
        if self.compute_complexity:
            results["format_complexity_score"] = self._compute_complexity_score(
                has_markdown=has_markdown,
                has_json=has_json,
                has_code_blocks=has_code_blocks,
                code_block_count=code_block_count,
                has_urls=has_urls,
                has_emails=has_emails,
            )

        return results

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields and return format detection results.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added format analysis columns.
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for format analysis. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            # No text columns to analyze in this DataFrame, return unchanged
            return result_df

        # Get analyzer ID for column naming
        analyzer_id = getattr(self, "analyzer_id", "format")

        for column in text_columns:
            # Analyze all texts in the column
            analysis_results = df[column].astype(str).apply(self._analyze_text)

            # Add columns for each detected feature
            if self.detect_markdown:
                col_name = f"{column}_{analyzer_id}_has_markdown"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r["has_markdown"]
                )

            if self.detect_json:
                col_name = f"{column}_{analyzer_id}_has_json"
                result_df[col_name] = analysis_results.apply(lambda r: r["has_json"])

            if self.detect_code_blocks:
                col_name = f"{column}_{analyzer_id}_has_code_blocks"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r["has_code_blocks"]
                )

                col_name = f"{column}_{analyzer_id}_code_block_count"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r["code_block_count"]
                )

                col_name = f"{column}_{analyzer_id}_code_block_languages"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r["code_block_languages"]
                )

            if self.detect_urls:
                col_name = f"{column}_{analyzer_id}_has_urls"
                result_df[col_name] = analysis_results.apply(lambda r: r["has_urls"])

            if self.detect_emails:
                col_name = f"{column}_{analyzer_id}_has_emails"
                result_df[col_name] = analysis_results.apply(lambda r: r["has_emails"])

            if self.compute_complexity:
                col_name = f"{column}_{analyzer_id}_format_complexity_score"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r["format_complexity_score"]
                )

        return result_df
