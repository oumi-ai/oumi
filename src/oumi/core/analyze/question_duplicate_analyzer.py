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

"""Question duplicate analyzer for detecting duplicate user messages."""

import hashlib
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("question_duplicate")
class QuestionDuplicateAnalyzer(SampleAnalyzer):
    """Analyzer for detecting duplicate questions (user messages).

    Some question duplication (5-15%) is acceptable since multiple users may ask
    the same question. However, excessive duplication (>20%) indicates:
    - Insufficient question diversity
    - Copy-paste errors in dataset creation
    - Need for more varied training data
    """

    def __init__(
        self,
        *,
        acceptable_duplication: float = 0.15,
        high_duplication_threshold: float = 0.20,
        normalize_whitespace: bool = True,
        case_sensitive: bool = False,
        tokenizer=None,
    ):
        """Initialize the QuestionDuplicateAnalyzer.

        Args:
            acceptable_duplication: Acceptable duplication rate (0.15 = 15%).
                Below this is considered normal/good diversity.
            high_duplication_threshold: Threshold above which duplication is
                concerning (0.20 = 20%). Above this flags quality issues.
            normalize_whitespace: Collapse multiple whitespace to single space.
            case_sensitive: If False, convert to lowercase before hashing.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.acceptable_duplication = acceptable_duplication
        self.high_duplication_threshold = high_duplication_threshold
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
        """Analyze user messages (questions) for duplicates.

        Args:
            df: Input DataFrame with role and text content columns.
            schema: Column schema to identify text fields.

        Returns:
            Tuple of (DataFrame with question duplicate analysis columns, empty dict).

            New columns added (only for user role messages):
            - question_hash: Hash of question content
            - question_duplicate_count: Number of times this question appears
            - question_is_duplicate: Whether this question is duplicated (>1 occurrence)
            - question_duplication_level: Classification of duplication
                ("unique", "acceptable", "high")
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
        result_df["question_hash"] = None
        result_df["question_duplicate_count"] = None
        result_df["question_is_duplicate"] = False
        result_df["question_duplication_level"] = None

        # Filter to user messages only
        user_mask = df["role"] == "user"
        user_df = df[user_mask]

        if len(user_df) == 0:
            return result_df, {}

        # Compute hashes for all user messages
        user_hashes = user_df[text_column].astype(str).apply(self._compute_hash)
        result_df.loc[user_mask, "question_hash"] = user_hashes

        # Count occurrences
        hash_counts = user_hashes.value_counts()
        result_df.loc[user_mask, "question_duplicate_count"] = user_hashes.map(
            hash_counts
        )

        # Mark duplicates
        result_df.loc[user_mask, "question_is_duplicate"] = (
            result_df.loc[user_mask, "question_duplicate_count"] > 1
        )

        # Calculate duplication rate and classify
        total_questions = len(user_df)
        duplicate_questions = (result_df.loc[user_mask, "question_is_duplicate"]).sum()
        duplication_rate = duplicate_questions / total_questions if total_questions > 0 else 0

        # Classify duplication level for each question
        for idx in result_df[user_mask].index:
            count = result_df.at[idx, "question_duplicate_count"]
            if count == 1:
                level = "unique"
            elif duplication_rate <= self.acceptable_duplication:
                level = "acceptable"
            elif duplication_rate <= self.high_duplication_threshold:
                level = "moderate"
            else:
                level = "high"
            result_df.at[idx, "question_duplication_level"] = level

        return result_df, {}
