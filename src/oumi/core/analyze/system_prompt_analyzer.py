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

"""System prompt analyzer for detecting system prompt patterns and issues."""

import hashlib
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("system_prompt")
class SystemPromptAnalyzer(SampleAnalyzer):
    """Analyzer for system prompt patterns, duplication, and issues.

    System prompts typically have high duplication (80-100%) which is expected.
    This analyzer detects:
    - Missing system prompts (conversations without system role)
    - System prompt templates and their usage
    - Abnormal system prompt diversity (too many unique prompts may indicate issues)
    """

    def __init__(
        self,
        *,
        expected_duplication_threshold: float = 0.80,
        max_unique_templates: int = 10,
        min_template_frequency: float = 0.05,
        normalize_whitespace: bool = True,
        case_sensitive: bool = False,
        tokenizer=None,
    ):
        """Initialize the SystemPromptAnalyzer.

        Args:
            expected_duplication_threshold: Expected % of conversations with same
                system prompt (0.80 = 80%). Above this is normal, below may indicate
                inconsistent formatting.
            max_unique_templates: Maximum expected unique system prompts.
                More than this may indicate quality issues.
            min_template_frequency: Minimum frequency for a template to be considered
                common (0.05 = 5%). Templates below this are flagged.
            normalize_whitespace: Collapse multiple whitespace to single space.
            case_sensitive: If False, convert to lowercase before hashing.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.expected_duplication_threshold = expected_duplication_threshold
        self.max_unique_templates = max_unique_templates
        self.min_template_frequency = min_template_frequency
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
        """Analyze system prompts for patterns and issues.

        Args:
            df: Input DataFrame with role and text content columns.
            schema: Column schema to identify text fields.

        Returns:
            Tuple of (DataFrame with system prompt analysis columns, empty dict).

            New columns added (conversation-level):
            - system_prompt_missing: Whether conversation lacks system prompt
            - system_prompt_hash: Hash of system prompt content
            - system_prompt_is_common_template: Whether system prompt matches
                a common template (>= min_template_frequency)
            - system_prompt_template_rank: Rank of template by frequency (1=most common)
            - system_prompt_is_unusual: Whether system prompt is rare/unique

            New columns added (message-level for system messages):
            - system_prompt_length: Character count of system prompt
            - system_prompt_duplicate_count: How many times this prompt appears
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
        result_df["system_prompt_missing"] = False
        result_df["system_prompt_hash"] = None
        result_df["system_prompt_is_common_template"] = False
        result_df["system_prompt_template_rank"] = None
        result_df["system_prompt_is_unusual"] = False
        result_df["system_prompt_length"] = None
        result_df["system_prompt_duplicate_count"] = None

        # Check if we have conversation_id for grouping
        has_conversation_id = "conversation_id" in df.columns

        if has_conversation_id:
            # Conversation-level analysis
            for conv_id in df["conversation_id"].unique():
                conv_mask = df["conversation_id"] == conv_id
                conv_df = df[conv_mask]

                # Check for system role in this conversation
                system_messages = conv_df[conv_df["role"] == "system"]

                if len(system_messages) == 0:
                    result_df.loc[conv_mask, "system_prompt_missing"] = True
                else:
                    # Get first system message as the system prompt
                    first_system = system_messages.iloc[0]
                    system_text = str(first_system[text_column])
                    system_hash = self._compute_hash(system_text)

                    result_df.loc[conv_mask, "system_prompt_hash"] = system_hash
        else:
            # No conversation grouping - analyze system messages directly
            system_mask = df["role"] == "system"
            system_messages = df[system_mask]

            if len(system_messages) > 0:
                # Compute hashes for all system messages
                system_hashes = system_messages[text_column].astype(str).apply(
                    self._compute_hash
                )
                result_df.loc[system_mask, "system_prompt_hash"] = system_hashes

        # Compute template statistics from all system prompts
        all_system_hashes = result_df["system_prompt_hash"].dropna()

        if len(all_system_hashes) > 0:
            hash_counts = all_system_hashes.value_counts()
            total_with_system = len(all_system_hashes)

            # Identify common templates (>= min_template_frequency)
            common_templates = hash_counts[
                hash_counts / total_with_system >= self.min_template_frequency
            ]

            # Assign template ranks
            template_ranks = {
                hash_val: rank + 1 for rank, hash_val in enumerate(hash_counts.index)
            }

            # Mark common templates and unusual prompts
            for idx, row in result_df.iterrows():
                if pd.notna(row["system_prompt_hash"]):
                    hash_val = row["system_prompt_hash"]
                    result_df.at[idx, "system_prompt_template_rank"] = template_ranks[
                        hash_val
                    ]
                    result_df.at[idx, "system_prompt_is_common_template"] = (
                        hash_val in common_templates.index
                    )
                    result_df.at[idx, "system_prompt_is_unusual"] = (
                        hash_val not in common_templates.index
                    )

        # Add message-level stats for system role messages
        system_mask = result_df["role"] == "system"
        if system_mask.any():
            # Length
            result_df.loc[system_mask, "system_prompt_length"] = (
                result_df.loc[system_mask, text_column].astype(str).str.len()
            )

            # Duplicate count
            system_hashes = result_df.loc[system_mask, text_column].astype(str).apply(
                self._compute_hash
            )
            hash_counts = system_hashes.value_counts()
            result_df.loc[system_mask, "system_prompt_duplicate_count"] = (
                system_hashes.map(hash_counts)
            )

        return result_df, {}
