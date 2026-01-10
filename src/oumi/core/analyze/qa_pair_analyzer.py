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

"""Question-Answer pair analyzer for detecting duplicate QA pairs."""

import hashlib
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("qa_pair_duplicate")
class QuestionAnswerPairAnalyzer(SampleAnalyzer):
    """Analyzer for detecting duplicate question-answer pairs.

    QA pairs should generally be unique. High duplication (>5-10%) indicates:
    - Redundant training data
    - Copy-paste errors
    - Insufficient dataset diversity

    This analyzer pairs consecutive user-assistant messages and detects
    when the same (question, answer) combination appears multiple times.
    """

    def __init__(
        self,
        *,
        duplicate_threshold: float = 0.05,
        normalize_whitespace: bool = True,
        case_sensitive: bool = False,
        tokenizer=None,
    ):
        """Initialize the QuestionAnswerPairAnalyzer.

        Args:
            duplicate_threshold: Threshold above which duplication is concerning
                (0.05 = 5%). If >5% of QA pairs are duplicates, flag as issue.
            normalize_whitespace: Collapse multiple whitespace to single space.
            case_sensitive: If False, convert to lowercase before hashing.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.duplicate_threshold = duplicate_threshold
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
        """Analyze question-answer pairs for duplicates.

        Args:
            df: Input DataFrame with role and text content columns.
            schema: Column schema to identify text fields.

        Returns:
            Tuple of (DataFrame with QA pair analysis columns, empty dict).

            New columns added:
            - qa_pair_hash: Hash of (question, answer) pair
            - qa_pair_duplicate_count: Number of times this QA pair appears
            - qa_pair_is_duplicate: Whether this QA pair is duplicated
            - qa_pair_high_duplication: Whether duplication exceeds threshold
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
        result_df["qa_pair_hash"] = None
        result_df["qa_pair_duplicate_count"] = None
        result_df["qa_pair_is_duplicate"] = False
        result_df["qa_pair_high_duplication"] = False

        # Check if we have conversation_id and message_index for proper pairing
        has_conversation_id = "conversation_id" in df.columns
        has_message_index = "message_index" in df.columns

        if not has_conversation_id or not has_message_index:
            # Fallback: pair consecutive messages if they're user-assistant
            user_mask = df["role"] == "user"
            assistant_mask = df["role"] == "assistant"

            qa_pairs = []
            for i in range(len(df) - 1):
                if user_mask.iloc[i] and assistant_mask.iloc[i + 1]:
                    question = str(df.iloc[i][text_column])
                    answer = str(df.iloc[i + 1][text_column])
                    pair_text = f"Q:{question}|A:{answer}"
                    pair_hash = self._compute_hash(pair_text)
                    qa_pairs.append((i, i + 1, pair_hash))

            # Map hashes to rows
            if qa_pairs:
                pair_hashes = [p[2] for p in qa_pairs]
                hash_counts = pd.Series(pair_hashes).value_counts()

                for user_idx, assistant_idx, pair_hash in qa_pairs:
                    count = hash_counts[pair_hash]
                    result_df.at[user_idx, "qa_pair_hash"] = pair_hash
                    result_df.at[user_idx, "qa_pair_duplicate_count"] = count
                    result_df.at[user_idx, "qa_pair_is_duplicate"] = count > 1
                    result_df.at[assistant_idx, "qa_pair_hash"] = pair_hash
                    result_df.at[assistant_idx, "qa_pair_duplicate_count"] = count
                    result_df.at[assistant_idx, "qa_pair_is_duplicate"] = count > 1
        else:
            # Proper conversation-aware pairing
            for conv_id in df["conversation_id"].unique():
                conv_mask = df["conversation_id"] == conv_id
                conv_df = df[conv_mask].sort_values("message_index")

                # Find user-assistant pairs in this conversation
                for i in range(len(conv_df) - 1):
                    current_row = conv_df.iloc[i]
                    next_row = conv_df.iloc[i + 1]

                    if current_row["role"] == "user" and next_row["role"] == "assistant":
                        question = str(current_row[text_column])
                        answer = str(next_row[text_column])
                        pair_text = f"Q:{question}|A:{answer}"
                        pair_hash = self._compute_hash(pair_text)

                        # Store hash in both user and assistant rows
                        user_idx = current_row.name
                        assistant_idx = next_row.name
                        result_df.at[user_idx, "qa_pair_hash"] = pair_hash
                        result_df.at[assistant_idx, "qa_pair_hash"] = pair_hash

            # Compute duplicate counts
            qa_hashes = result_df["qa_pair_hash"].dropna()
            if len(qa_hashes) > 0:
                hash_counts = qa_hashes.value_counts()
                # Each QA pair has 2 rows (user + assistant), so divide count by 2
                # to get actual number of pair occurrences
                result_df.loc[result_df["qa_pair_hash"].notna(), "qa_pair_duplicate_count"] = (
                    result_df.loc[result_df["qa_pair_hash"].notna(), "qa_pair_hash"].map(hash_counts) // 2
                )
                result_df["qa_pair_is_duplicate"] = (
                    result_df["qa_pair_duplicate_count"] > 1
                )

        # Calculate if duplication is high (exceeds threshold)
        total_pairs = result_df["qa_pair_hash"].notna().sum() / 2  # Divide by 2 since each pair has 2 rows
        if total_pairs > 0:
            duplicate_pairs = result_df[result_df["qa_pair_is_duplicate"] == True]["qa_pair_hash"].nunique()
            duplication_rate = duplicate_pairs / total_pairs if total_pairs > 0 else 0
            result_df["qa_pair_high_duplication"] = duplication_rate > self.duplicate_threshold

        return result_df, {}
