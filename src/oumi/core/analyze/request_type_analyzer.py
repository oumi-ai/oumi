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

"""Request type pattern analyzer for classifying and analyzing request distributions."""

import math
import re
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer

# Default patterns for common request types
DEFAULT_PATTERNS: dict[str, list[str]] = {
    "explanation": [
        r"\bexplain\b",
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bhow does\b",
        r"\bwhy is\b",
        r"\bwhy do\b",
        r"\bdescribe\b",
        r"\bdefine\b",
    ],
    "code_generation": [
        r"\bwrite (?:a |some )?code\b",
        r"\bimplement\b",
        r"\bcreate (?:a )?function\b",
        r"\bcode (?:for|to)\b",
        r"\bprogram\b",
        r"\bscript\b",
    ],
    "summarization": [
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\btl;?dr\b",
        r"\bbrief\b",
        r"\bshorten\b",
        r"\bcondense\b",
    ],
    "debugging": [
        r"\bfix\b",
        r"\bdebug\b",
        r"\berror\b",
        r"\bbug\b",
        r"\bnot working\b",
        r"\bissue\b",
        r"\bproblem with\b",
    ],
    "translation": [
        r"\btranslate\b",
        r"\bconvert to\b",
        r"\bin (?:spanish|french|german|chinese|japanese)\b",
    ],
    "creative_writing": [
        r"\bwrite (?:a |an )?(?:story|poem|essay|article)\b",
        r"\bcreative\b",
        r"\bimagine\b",
        r"\bfiction\b",
    ],
    "analysis": [
        r"\banalyze\b",
        r"\breview\b",
        r"\bevaluate\b",
        r"\bassess\b",
        r"\bcritique\b",
    ],
    "comparison": [
        r"\bcompare\b",
        r"\bdifference between\b",
        r"\bvs\.?\b",
        r"\bversus\b",
    ],
    "how_to": [
        r"\bhow to\b",
        r"\bhow do i\b",
        r"\bhow can i\b",
        r"\bsteps to\b",
        r"\bguide\b",
    ],
    "list_generation": [
        r"\blist\b",
        r"\bgive me (?:\d+ )?\b",
        r"\bprovide (?:\d+ )?\b",
        r"\benumerate\b",
    ],
}


@register_sample_analyzer("request_type")
class RequestTypeAnalyzer(SampleAnalyzer):
    """Analyzer that classifies requests by type using pattern matching.

    Identifies request types and analyzes their distribution across the dataset.
    """

    def __init__(
        self,
        *,
        patterns: Optional[dict[str, list[str]]] = None,
        case_sensitive: bool = False,
        min_type_percentage: float = 0.01,
        apply_to_role: Optional[str] = "user",
        tokenizer=None,
    ):
        """Initialize the RequestTypeAnalyzer.

        Args:
            patterns: Dict mapping type names to lists of regex patterns.
                If None, uses DEFAULT_PATTERNS.
            case_sensitive: Whether pattern matching is case-sensitive.
            min_type_percentage: Flag types appearing in less than this fraction.
            apply_to_role: Only classify messages with this role (None for all).
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.patterns = patterns or DEFAULT_PATTERNS
        self.case_sensitive = case_sensitive
        self.min_type_percentage = min_type_percentage
        self.apply_to_role = apply_to_role

        # Compile patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        self._compiled_patterns: dict[str, list[re.Pattern]] = {
            type_name: [re.compile(p, flags) for p in pattern_list]
            for type_name, pattern_list in self.patterns.items()
        }

    def _classify_text(self, text: str) -> tuple[str, list[str]]:
        """Classify text into a request type.

        Returns:
            Tuple of (primary_type, all_matched_types)
        """
        matched_types = []
        for type_name, compiled_list in self._compiled_patterns.items():
            for pattern in compiled_list:
                if pattern.search(text):
                    matched_types.append(type_name)
                    break  # Only count each type once

        if not matched_types:
            return "unknown", []

        return matched_types[0], matched_types

    def _compute_entropy(self, distribution: dict[str, int], total: int) -> float:
        """Compute Shannon entropy of distribution."""
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for request type classification.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema to identify text fields.

        Returns:
            DataFrame with added columns:
            - request_type: Primary classified type
            - request_type_matches: All matched types
            - request_type_is_unknown: Whether no type matched
        """
        if not schema:
            raise ValueError("schema is required to identify text fields.")

        result_df = df.copy()

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            raise ValueError("No text fields found in DataFrame.")

        # Use first text column for classification
        text_column = text_columns[0]

        # Filter by role if specified
        if self.apply_to_role and "role" in df.columns:
            role_mask = df["role"].str.lower() == self.apply_to_role.lower()
        else:
            role_mask = pd.Series([True] * len(df), index=df.index)

        # Classify each row
        def classify_row(row_idx: int) -> tuple[str, list[str]]:
            if not role_mask.iloc[row_idx]:
                return "n/a", []
            text = str(df[text_column].iloc[row_idx])
            return self._classify_text(text)

        classifications = [classify_row(i) for i in range(len(df))]

        result_df["request_type"] = [c[0] for c in classifications]
        result_df["request_type_matches"] = [c[1] for c in classifications]
        result_df["request_type_is_unknown"] = result_df["request_type"] == "unknown"

        # Compute distribution statistics (excluding n/a)
        valid_types = result_df[result_df["request_type"] != "n/a"]["request_type"]
        type_counts = valid_types.value_counts().to_dict()
        total_valid = len(valid_types)

        # Identify underrepresented types
        underrepresented = []
        for type_name in self.patterns.keys():
            count = type_counts.get(type_name, 0)
            if total_valid > 0 and count / total_valid < self.min_type_percentage:
                underrepresented.append(type_name)

        # Add underrepresented flag
        result_df["request_type_is_underrepresented"] = result_df["request_type"].isin(
            underrepresented
        )

        return result_df, {}
