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

"""Safety analyzer for detecting potentially unsafe content.

Based on safety assessment practices from "Fixing It in Post" paper
which used Llama-Guard for safety scoring. This analyzer provides
heuristic-based safety assessment when model-based scoring is not available.
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.column_utils import make_analyzer_column_name
from oumi.core.analyze.sample_analyzer import DEFAULT_TEXT_COLUMNS, SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("safety")
class SafetyAnalyzer(SampleAnalyzer):
    """Analyzer for detecting potentially unsafe content.

    This analyzer evaluates content for safety concerns across multiple
    categories based on common safety taxonomies (similar to Llama-Guard).

    Safety categories checked:
        - violence: Violence, threats, harm
        - hate: Hate speech, discrimination
        - sexual: Sexual content
        - self_harm: Self-harm, suicide references
        - illegal: Illegal activities
        - deception: Misinformation, fraud
        - dangerous: Dangerous activities, weapons
        - privacy: Privacy violations, doxxing

    Metrics computed:
        - safety_score: Overall safety score (0-1, higher is safer)
        - is_safe: Boolean flag for safe content
        - risk_level: Risk level (safe, low, medium, high)
        - safety_categories: Comma-separated list of flagged categories
    """

    # Violence patterns
    _VIOLENCE_PATTERNS = [
        re.compile(
            r"\b(?:kill|murder|assassinate|execute|slaughter|massacre|"
            r"butcher|eliminate|exterminate)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:attack|assault|beat|punch|kick|stab|shoot|strangle|"
            r"choke|torture|maim|injure|wound)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:bomb|explosive|weapon|gun|firearm|knife|blade|"
            r"poison|toxic|lethal)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:terrorist|terrorism|jihad|radicalize|extremist)\b",
            re.IGNORECASE,
        ),
    ]

    # Hate speech patterns
    _HATE_PATTERNS = [
        re.compile(
            r"\b(?:hate|despise|loathe)\s+(?:all\s+)?(?:the\s+)?(?:\w+\s+)?"
            r"(?:people|race|religion|group)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:inferior|superior)\s+(?:race|people|group)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:racial|ethnic|religious)\s+(?:slur|discrimination|"
            r"prejudice|hatred)\b",
            re.IGNORECASE,
        ),
    ]

    # Self-harm patterns
    _SELF_HARM_PATTERNS = [
        re.compile(
            r"\b(?:suicide|suicidal|kill\s+(?:my|your)?self|"
            r"end\s+(?:my|your)\s+life)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:self[- ]?harm|cut(?:ting)?\s+(?:my|your)?self|"
            r"hurt\s+(?:my|your)?self)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:overdose|OD|swallow\s+pills|hang\s+(?:my|your)?self)\b",
            re.IGNORECASE,
        ),
    ]

    # Illegal activity patterns
    _ILLEGAL_PATTERNS = [
        re.compile(
            r"\b(?:how\s+to\s+)?(?:hack|crack|exploit|breach|"
            r"bypass|circumvent)\s+(?:into|a|the)?\s*(?:\w+\s+)?"
            r"(?:system|account|password|security)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:steal|theft|rob|burglar|shoplift|embezzle|"
            r"counterfeit|forge|fraud)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:drug|cocaine|heroin|meth|marijuana|cannabis|"
            r"narcotics?)\s*(?:deal|sell|buy|make|cook|produce)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:launder|laundering|money\s+laundering|tax\s+evasion)\b",
            re.IGNORECASE,
        ),
    ]

    # Dangerous activities patterns
    _DANGEROUS_PATTERNS = [
        re.compile(
            r"\b(?:how\s+to\s+)?(?:make|build|create|construct)\s+"
            r"(?:a\s+)?(?:bomb|explosive|weapon|poison|toxin)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:synthesize|manufacture|produce)\s+(?:\w+\s+)?"
            r"(?:chemical|biological|radiological)\b",
            re.IGNORECASE,
        ),
    ]

    # Privacy violation patterns
    _PRIVACY_PATTERNS = [
        re.compile(
            r"\b(?:dox|doxx|reveal|expose)\s+(?:\w+\s+)?(?:personal|"
            r"private)\s+(?:information|data|address|phone)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:social\s+security|SSN|credit\s+card|bank\s+account)\s+"
            r"(?:number|info)\b",
            re.IGNORECASE,
        ),
    ]

    # Deception patterns
    _DECEPTION_PATTERNS = [
        re.compile(
            r"\b(?:how\s+to\s+)?(?:scam|deceive|defraud|trick|"
            r"manipulate|con)\s+(?:\w+\s+)?(?:people|someone|victim)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:phishing|social\s+engineering|catfish|impersonate)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:spread|create|generate)\s+(?:\w+\s+)?(?:misinformation|"
            r"disinformation|fake\s+news|propaganda)\b",
            re.IGNORECASE,
        ),
    ]

    # Category weights for overall score
    _CATEGORY_WEIGHTS = {
        "violence": 0.9,
        "hate": 0.85,
        "self_harm": 0.95,
        "illegal": 0.8,
        "dangerous": 0.9,
        "privacy": 0.7,
        "deception": 0.75,
    }

    def __init__(
        self,
        *,
        strict_mode: bool = False,
        include_categories: bool = True,
    ):
        """Initialize the SafetyAnalyzer.

        Args:
            strict_mode: If True, flag content with any pattern match.
            include_categories: Include detailed category flags.
        """
        self.strict_mode = strict_mode
        self.include_categories = include_categories

        # Compile pattern groups
        self._pattern_groups = {
            "violence": self._VIOLENCE_PATTERNS,
            "hate": self._HATE_PATTERNS,
            "self_harm": self._SELF_HARM_PATTERNS,
            "illegal": self._ILLEGAL_PATTERNS,
            "dangerous": self._DANGEROUS_PATTERNS,
            "privacy": self._PRIVACY_PATTERNS,
            "deception": self._DECEPTION_PATTERNS,
        }

    def _check_category(self, text: str, patterns: list[re.Pattern]) -> int:
        """Check how many patterns match in a category.

        Args:
            text: Text to check.
            patterns: List of patterns to match.

        Returns:
            Number of pattern matches.
        """
        count = 0
        for pattern in patterns:
            matches = pattern.findall(text)
            count += len(matches)
        return count

    def _analyze_text(self, text: str) -> dict[str, Any]:
        """Analyze text for safety concerns.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary of safety metrics.
        """
        if not text or not text.strip():
            return {
                "safety_score": 1.0,
                "is_safe": True,
                "risk_level": "safe",
                "safety_categories": "",
            }

        # Check each category
        category_scores = {}
        flagged_categories = []

        for category, patterns in self._pattern_groups.items():
            match_count = self._check_category(text, patterns)
            if match_count > 0:
                flagged_categories.append(category)
                # More matches = lower category score
                category_scores[category] = max(0.0, 1.0 - (match_count * 0.3))
            else:
                category_scores[category] = 1.0

        # Calculate overall safety score
        if not flagged_categories:
            safety_score = 1.0
        else:
            # Weighted average of category scores
            total_weight = 0
            weighted_sum = 0
            for category, score in category_scores.items():
                weight = self._CATEGORY_WEIGHTS.get(category, 0.5)
                weighted_sum += score * weight
                total_weight += weight

            safety_score = weighted_sum / total_weight if total_weight > 0 else 1.0

        # Determine if safe
        if self.strict_mode:
            is_safe = len(flagged_categories) == 0
        else:
            is_safe = safety_score >= 0.7

        # Determine risk level
        if safety_score >= 0.9:
            risk_level = "safe"
        elif safety_score >= 0.7:
            risk_level = "low"
        elif safety_score >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"

        result = {
            "safety_score": round(safety_score, 3),
            "is_safe": is_safe,
            "risk_level": risk_level,
            "safety_categories": ",".join(flagged_categories),
        }

        return result

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for safety concerns.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added safety analysis columns.
            generated column schema dict).
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for safety "
                "analysis. Please provide a column schema dict that specifies "
                "which columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df

        analyzer_id = getattr(self, "analyzer_id", "safety")

        for column in text_columns:
            analysis_results = df[column].astype(str).apply(self._analyze_text)

            # Extract results to columns
            col_name = make_analyzer_column_name(column, analyzer_id, "score")
            result_df[col_name] = analysis_results.apply(
                lambda r: r.get("safety_score")
            )

            col_name = make_analyzer_column_name(column, analyzer_id, "is_safe")
            result_df[col_name] = analysis_results.apply(lambda r: r.get("is_safe"))

            col_name = make_analyzer_column_name(column, analyzer_id, "risk_level")
            result_df[col_name] = analysis_results.apply(lambda r: r.get("risk_level"))

            if self.include_categories:
                col_name = make_analyzer_column_name(
                    column, analyzer_id, "categories_triggered"
                )
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("safety_categories")
                )

        return result_df
