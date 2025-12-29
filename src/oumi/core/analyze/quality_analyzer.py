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

"""Quality and safety analyzer for detecting PII, toxic content, and data issues."""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("quality")
class QualityAnalyzer(SampleAnalyzer):
    """Analyzer for detecting quality and safety issues in text content.

    This analyzer identifies potential data quality and safety issues including:
        - PII detection (emails, phone numbers, SSNs, credit cards, API keys)
        - Language detection (requires optional langdetect package)
        - Encoding issues (invalid UTF-8, mojibake patterns)
        - Repetition detection (repeated phrases or patterns)

    Quality metrics computed:
        - has_pii: Boolean indicating if any PII was detected
        - pii_types: Comma-separated list of detected PII types
        - pii_count: Total count of PII instances detected
        - detected_language: ISO 639-1 language code (if language detection enabled)
        - language_confidence: Confidence score for language detection
        - has_encoding_issues: Boolean for potential encoding problems
        - repetition_ratio: Ratio of repeated n-grams to total n-grams
        - has_high_repetition: Boolean for high repetition detected
    """

    # PII detection patterns
    _EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    _PHONE_PATTERN = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    _SSN_PATTERN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
    _CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    _API_KEY_PATTERN = re.compile(
        r"(?:api[_-]?key|secret|token|password|auth)[\"']?\s*[:=]\s*[\"']?[\w-]{16,}",
        re.IGNORECASE,
    )

    # Common encoding issue patterns (mojibake)
    _MOJIBAKE_PATTERNS = [
        re.compile(r"[ÃÂ]{2,}"),  # UTF-8 decoded as Latin-1
        re.compile(r"â€[™˜œ]"),  # Common smart quote mojibake
        re.compile(r"Ã©|Ã¨|Ã |Ã¢"),  # French accents mojibake
        re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"),  # Control characters
        re.compile(r"\ufffd"),  # Unicode replacement character
    ]

    def __init__(
        self,
        *,
        detect_pii: bool = True,
        detect_emails: bool = True,
        detect_phones: bool = True,
        detect_ssn: bool = True,
        detect_credit_cards: bool = True,
        detect_api_keys: bool = True,
        detect_language: bool = False,
        detect_encoding_issues: bool = True,
        detect_repetition: bool = True,
        repetition_ngram_size: int = 3,
        repetition_threshold: float = 0.3,
    ):
        """Initialize the QualityAnalyzer.

        Args:
            detect_pii: Master switch for PII detection.
            detect_emails: Whether to detect email addresses.
            detect_phones: Whether to detect phone numbers.
            detect_ssn: Whether to detect Social Security Numbers.
            detect_credit_cards: Whether to detect credit card numbers.
            detect_api_keys: Whether to detect API keys and secrets.
            detect_language: Whether to detect language (requires langdetect).
            detect_encoding_issues: Whether to detect encoding problems.
            detect_repetition: Whether to detect repetitive content.
            repetition_ngram_size: Size of n-grams for repetition detection.
            repetition_threshold: Threshold above which repetition is flagged.
        """
        self.detect_pii = detect_pii
        self.detect_emails = detect_emails
        self.detect_phones = detect_phones
        self.detect_ssn = detect_ssn
        self.detect_credit_cards = detect_credit_cards
        self.detect_api_keys = detect_api_keys
        self.detect_language = detect_language
        self.detect_encoding_issues = detect_encoding_issues
        self.detect_repetition = detect_repetition
        self.repetition_ngram_size = repetition_ngram_size
        self.repetition_threshold = repetition_threshold

        # Check if langdetect is available
        self._langdetect_available = False
        if self.detect_language:
            try:
                import langdetect  # noqa: F401

                self._langdetect_available = True
            except ImportError:
                logger.warning(
                    "langdetect package not installed. Language detection disabled. "
                    "Install with: pip install langdetect"
                )
                self.detect_language = False

    def _detect_pii(self, text: str) -> dict[str, Any]:
        """Detect PII in text.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with PII detection results.
        """
        pii_types = []
        pii_count = 0

        if self.detect_emails:
            matches = self._EMAIL_PATTERN.findall(text)
            if matches:
                pii_types.append("email")
                pii_count += len(matches)

        if self.detect_phones:
            matches = self._PHONE_PATTERN.findall(text)
            if matches:
                pii_types.append("phone")
                pii_count += len(matches)

        if self.detect_ssn:
            matches = self._SSN_PATTERN.findall(text)
            if matches:
                pii_types.append("ssn")
                pii_count += len(matches)

        if self.detect_credit_cards:
            matches = self._CREDIT_CARD_PATTERN.findall(text)
            if matches:
                pii_types.append("credit_card")
                pii_count += len(matches)

        if self.detect_api_keys:
            matches = self._API_KEY_PATTERN.findall(text)
            if matches:
                pii_types.append("api_key")
                pii_count += len(matches)

        return {
            "has_pii": len(pii_types) > 0,
            "pii_types": ",".join(pii_types) if pii_types else "",
            "pii_count": pii_count,
        }

    def _detect_language(self, text: str) -> dict[str, Any]:
        """Detect the language of text.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary with language detection results.
        """
        if not self._langdetect_available or len(text.strip()) < 20:
            return {
                "detected_language": "",
                "language_confidence": 0.0,
            }

        try:
            from langdetect import detect_langs

            results = detect_langs(text)
            if results:
                top_result = results[0]
                return {
                    "detected_language": top_result.lang,
                    "language_confidence": round(top_result.prob, 3),
                }
        except Exception:
            pass

        return {
            "detected_language": "",
            "language_confidence": 0.0,
        }

    def _detect_encoding_issues(self, text: str) -> bool:
        """Detect potential encoding issues in text.

        Args:
            text: Input text to analyze.

        Returns:
            True if encoding issues are detected.
        """
        for pattern in self._MOJIBAKE_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _compute_repetition_ratio(self, text: str) -> float:
        """Compute the ratio of repeated n-grams in text.

        Args:
            text: Input text to analyze.

        Returns:
            Ratio of repeated n-grams (0-1).
        """
        words = text.lower().split()
        if len(words) < self.repetition_ngram_size * 2:
            return 0.0

        # Generate n-grams
        ngrams = []
        for i in range(len(words) - self.repetition_ngram_size + 1):
            ngram = " ".join(words[i : i + self.repetition_ngram_size])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        # Count unique and total n-grams
        unique_ngrams = set(ngrams)
        repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))

        return round(repetition_ratio, 3)

    def _analyze_text(self, text: str) -> dict[str, Any]:
        """Analyze a single text sample for quality issues.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary of quality metrics.
        """
        results = {}

        # PII detection
        if self.detect_pii:
            pii_results = self._detect_pii(text)
            results.update(pii_results)

        # Language detection
        if self.detect_language:
            lang_results = self._detect_language(text)
            results.update(lang_results)

        # Encoding issues
        if self.detect_encoding_issues:
            results["has_encoding_issues"] = self._detect_encoding_issues(text)

        # Repetition
        if self.detect_repetition:
            repetition_ratio = self._compute_repetition_ratio(text)
            results["repetition_ratio"] = repetition_ratio
            results["has_high_repetition"] = (
                repetition_ratio > self.repetition_threshold
            )

        return results

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields for quality and safety issues.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added quality analysis columns.
            generated column schema dict).
        """
        result_df = df.copy()
        generated_schema = {}

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for quality analysis. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df, generated_schema

        analyzer_id = getattr(self, "analyzer_id", "quality")

        for column in text_columns:
            # Analyze all texts in the column
            analysis_results = df[column].astype(str).apply(self._analyze_text)

            # Add columns for each metric
            if self.detect_pii:
                col_name = f"{column}_{analyzer_id}_has_pii"
                result_df[col_name] = analysis_results.apply(lambda r: r["has_pii"])
                generated_schema[col_name] = {
                    "type": ColumnType.BOOL,
                    "content_type": ContentType.BOOLEAN,
                    "description": f"Whether text contains PII",
                }

                col_name = f"{column}_{analyzer_id}_pii_types"
                result_df[col_name] = analysis_results.apply(lambda r: r["pii_types"])
                generated_schema[col_name] = {
                    "type": ColumnType.STRING,
                    "content_type": ContentType.LIST,
                    "description": (f"Comma-separated list of PII types detected."),
                }

                col_name = f"{column}_{analyzer_id}_pii_count"
                result_df[col_name] = analysis_results.apply(lambda r: r["pii_count"])
                generated_schema[col_name] = {
                    "type": ColumnType.INT,
                    "content_type": ContentType.NUMERIC,
                    "description": f"Total count of PII instances.",
                }

            if self.detect_language:
                col_name = f"{column}_{analyzer_id}_detected_language"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("detected_language", "")
                )
                generated_schema[col_name] = {
                    "type": ColumnType.STRING,
                    "content_type": ContentType.CATEGORICAL,
                    "description": f"Detected language.",
                }

                col_name = f"{column}_{analyzer_id}_language_confidence"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("language_confidence", 0.0)
                )
                generated_schema[col_name] = {
                    "type": ColumnType.FLOAT,
                    "content_type": ContentType.NUMERIC,
                    "description": f"Language detection confidence score.",
                }

            if self.detect_encoding_issues:
                col_name = f"{column}_{analyzer_id}_has_encoding_issues"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r["has_encoding_issues"]
                )
                generated_schema[col_name] = {
                    "type": ColumnType.BOOL,
                    "content_type": ContentType.BOOLEAN,
                    "description": f"Whether text has encoding issues",
                }

            if self.detect_repetition:
                col_name = f"{column}_{analyzer_id}_repetition_ratio"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r["repetition_ratio"]
                )
                generated_schema[col_name] = {
                    "type": ColumnType.FLOAT,
                    "content_type": ContentType.NUMERIC,
                    "description": (
                        "Ratio of repeated n-grams "
                        "(0.0 = no repetition, 1.0 = complete repetition)"
                    ),
                }

                col_name = f"{column}_{analyzer_id}_has_high_repetition"
                result_df[col_name] = analysis_results.apply(
                    lambda r: r["has_high_repetition"]
                )
                generated_schema[col_name] = {
                    "type": ColumnType.BOOL,
                    "content_type": ContentType.BOOLEAN,
                    "description": f"Whether text has high repetition",
                }

        return result_df, generated_schema
