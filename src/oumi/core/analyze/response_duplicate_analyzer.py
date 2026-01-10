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

"""Response duplicate analyzer for detecting duplicate assistant messages."""

import hashlib
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("response_duplicate")
class ResponseDuplicateAnalyzer(SampleAnalyzer):
    """Analyzer for detecting duplicate responses (assistant messages).

    Response duplication should generally be low (<5%) as responses should be
    diverse and contextual. High duplication (>10%) indicates:
    - Generic/templated responses (e.g., "I don't know", "Yes", "No")
    - Lack of response diversity
    - Potential quality issues with generated data
    - Copy-paste errors in dataset creation

    Very short responses (1-2 words) are expected to have higher duplication.
    """

    def __init__(
        self,
        *,
        acceptable_duplication: float = 0.05,
        high_duplication_threshold: float = 0.10,
        short_response_length: int = 20,
        normalize_whitespace: bool = True,
        case_sensitive: bool = False,
        tokenizer=None,
    ):
        """Initialize the ResponseDuplicateAnalyzer.

        Args:
            acceptable_duplication: Acceptable duplication rate (0.05 = 5%).
                Below this is considered good diversity.
            high_duplication_threshold: Threshold above which duplication is
                concerning (0.10 = 10%). Above this flags quality issues.
            short_response_length: Character count below which a response is
                considered "short" (20 chars). Short responses naturally have
                higher duplication.
            normalize_whitespace: Collapse multiple whitespace to single space.
            case_sensitive: If False, convert to lowercase before hashing.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.acceptable_duplication = acceptable_duplication
        self.high_duplication_threshold = high_duplication_threshold
        self.short_response_length = short_response_length
        self.normalize_whitespace = normalize_whitespace
        self.case_sensitive = case_sensitive

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent hashing."""
        if not self.case_sensitive:
            text = text.lower()
        if self.normalize_whitespace:
            text = " ".join(text.split())
        return text

    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of normalized text."""
        normalized = self._normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze assistant messages (responses) for duplicates.

        Args:
            df: Input DataFrame with role and text content columns.
            schema: Column schema to identify text fields.

        Returns:
            Tuple of (DataFrame with response duplicate analysis columns, empty dict).

            New columns added (only for assistant role messages):
            - response_hash: Hash of response content
            - response_duplicate_count: Number of times this response appears
            - response_is_duplicate: Whether this response is duplicated (>1 occurrence)
            - response_is_short: Whether response is short (<= short_response_length)
            - response_duplication_level: Classification of duplication
                ("unique", "acceptable", "high")
            - response_is_generic: Whether response is a common generic response
                (very high duplication count)
        """
        if not schema:
            raise ValueError("schema is required to identify text fields.")

        result_df = df.copy()

        # Check for required columns
        if "role" not in df.columns:
            raise ValueError("DataFrame must have a 'role' column.")

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            raise ValueError("No text fields found in DataFrame.")

        # Use first text column (typically 'text_content')
        text_column = text_columns[0]

        # Initialize columns
        result_df["response_hash"] = None
        result_df["response_duplicate_count"] = None
        result_df["response_is_duplicate"] = False
        result_df["response_is_short"] = False
        result_df["response_duplication_level"] = None
        result_df["response_is_generic"] = False

        # Filter to assistant messages only
        assistant_mask = df["role"] == "assistant"
        assistant_df = df[assistant_mask]

        if len(assistant_df) == 0:
            return result_df, {}

        # Compute hashes for all assistant messages
        assistant_hashes = assistant_df[text_column].astype(str).apply(self._compute_hash)
        result_df.loc[assistant_mask, "response_hash"] = assistant_hashes

        # Count occurrences
        hash_counts = assistant_hashes.value_counts()
        result_df.loc[assistant_mask, "response_duplicate_count"] = assistant_hashes.map(
            hash_counts
        )

        # Mark duplicates
        result_df.loc[assistant_mask, "response_is_duplicate"] = (
            result_df.loc[assistant_mask, "response_duplicate_count"] > 1
        )

        # Mark short responses
        response_lengths = assistant_df[text_column].astype(str).str.len()
        result_df.loc[assistant_mask, "response_is_short"] = (
            response_lengths <= self.short_response_length
        )

        # Calculate duplication rate and classify
        total_responses = len(assistant_df)
        duplicate_responses = (result_df.loc[assistant_mask, "response_is_duplicate"]).sum()
        duplication_rate = duplicate_responses / total_responses if total_responses > 0 else 0

        # Identify generic responses (appear in >1% of all responses)
        generic_threshold = max(int(total_responses * 0.01), 10)

        # Classify duplication level for each response
        for idx in result_df[assistant_mask].index:
            count = result_df.at[idx, "response_duplicate_count"]
            is_short = result_df.at[idx, "response_is_short"]

            # Generic response detection
            if count >= generic_threshold:
                result_df.at[idx, "response_is_generic"] = True

            # Duplication level classification
            if count == 1:
                level = "unique"
            elif is_short:
                # Short responses can have higher duplication
                level = "acceptable_short"
            elif duplication_rate <= self.acceptable_duplication:
                level = "acceptable"
            elif duplication_rate <= self.high_duplication_threshold:
                level = "moderate"
            else:
                level = "high"
            result_df.at[idx, "response_duplication_level"] = level

        return result_df, {}
