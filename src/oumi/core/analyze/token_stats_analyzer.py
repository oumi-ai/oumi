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

"""Token stats analyzer for aggregating tokens by message role.

This analyzer aggregates message-level token counts by role to create
conversation-level metrics:
- System tokens: Sum of tokens in all system messages
- User tokens: Sum of tokens in all user messages
- Input tokens: Sum of tokens in system + user messages
- Output tokens: Sum of tokens in all assistant messages
"""

import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.column_utils import (
    make_analyzer_column_name,
    parse_analyzer_column_name,
)
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("token_stats")
class TokenStatsAnalyzer(SampleAnalyzer):
    """Analyzer that aggregates message-level token counts by role.

    This analyzer works on message-level DataFrames and aggregates token counts
    by role to create conversation-level metrics. It requires the length analyzer
    to have been run first to generate token_count columns.

    Metrics computed (conversation-level):
        - system_tokens: Sum of tokens in all system messages
        - user_tokens: Sum of tokens in all user messages
        - input_tokens: Sum of tokens in system + user messages
            (total input to model during training)
        - output_tokens: Sum of tokens in all assistant messages
    """

    def __init__(
        self,
        *,
        token_count_column: str | None = None,
    ):
        """Initialize the TokenStatsAnalyzer.

        Args:
            token_count_column: Optional name of the token count column to use.
                If None, will automatically detect columns matching
                'text_content__length__token_count' pattern.
        """
        self.token_count_column = token_count_column

    def _find_token_count_column(self, df: pd.DataFrame) -> str | None:
        """Find the token count column in the DataFrame.

        Looks for columns matching the pattern from the length analyzer:
        'text_content__length__token_count' or similar patterns.

        Args:
            df: DataFrame to search for token count column.

        Returns:
            Column name if found, None otherwise.
        """
        if self.token_count_column and self.token_count_column in df.columns:
            return self.token_count_column

        # Look for length analyzer token count columns
        # Pattern: {source_column}__length__token_count
        for col in df.columns:
            info = parse_analyzer_column_name(col)
            if (
                info
                and info.analyzer_id == "length"
                and info.metric_name == "token_count"
            ):
                return col

        # Fallback: look for any column with 'token_count' in the name
        for col in df.columns:
            if "token_count" in col.lower() and df[col].dtype in [
                "int64",
                "float64",
                "Int64",
            ]:
                return col

        return None

    def _aggregate_tokens_by_role(
        self,
        df: pd.DataFrame,
        token_col: str,
        role_col: str,
        conversation_id_col: str,
    ) -> dict[str, dict[str, int]]:
        """Aggregate tokens by role for each conversation.

        Args:
            df: Message-level DataFrame.
            token_col: Name of the token count column.
            role_col: Name of the role column.
            conversation_id_col: Name of the conversation ID column.

        Returns:
            Dictionary mapping conversation_id to metrics dict with:
            - system_tokens: Sum of system message tokens
            - user_tokens: Sum of user message tokens
            - input_tokens: Sum of system + user message tokens (total input to model)
            - output_tokens: Sum of assistant message tokens
        """
        conv_metrics = {}

        for conv_id, group in df.groupby(conversation_id_col):
            # Filter out NaN token counts
            group = group.dropna(subset=[token_col])

            # Sum tokens by role
            system_tokens = 0
            user_tokens = 0
            assistant_tokens = 0

            for _, row in group.iterrows():
                role = str(row[role_col]).lower()
                token_val = row[token_col]
                # Handle NaN values - iterrows() returns scalars, but type checker
                # doesn't know this, so we use try/except
                try:
                    # Convert to float first to handle NaN, then to int
                    token_count = int(float(token_val))
                except (ValueError, TypeError, OverflowError):
                    token_count = 0

                if role == "system":
                    system_tokens += token_count
                elif role == "user":
                    user_tokens += token_count
                elif role == "assistant":
                    assistant_tokens += token_count

            # Calculate input tokens (system + user)
            # This is what the model sees during training
            input_tokens = system_tokens + user_tokens

            conv_metrics[conv_id] = {
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
                "input_tokens": input_tokens,
                "output_tokens": assistant_tokens,
            }

        return conv_metrics

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: dict | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze token counts by role and aggregate to conversation level.

        This analyzer works on message-level DataFrames and requires:
        - A conversation_id column to group messages
        - A role column to identify message roles
        - A token_count column (typically from the length analyzer)

        Args:
            df: Input DataFrame with messages (must be message-level).
            schema: Column schema dict (optional, not used but kept for interface).

        Returns:
            Tuple of (DataFrame with added conversation-level token metrics,
            generated column schema dict).
        """
        result_df = df.copy()
        generated_schema = {}

        # Check for required columns
        if "conversation_id" not in df.columns:
            # If no conversation_id, cannot aggregate - return unchanged
            return result_df, generated_schema

        # Find role column
        role_column = None
        if schema:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and col in df.columns
                    and "role" in col.lower()
                ):
                    role_column = col
                    break

        # Fallback to common column names
        if role_column is None and "role" in df.columns:
            role_column = "role"

        if role_column is None:
            # Cannot analyze without role column
            return result_df, generated_schema

        # Find token count column
        token_count_column = self._find_token_count_column(df)
        if token_count_column is None:
            # Cannot analyze without token count column
            # This is expected if length analyzer hasn't run yet
            return result_df, generated_schema

        # Aggregate tokens by role for each conversation
        conv_metrics = self._aggregate_tokens_by_role(
            df, token_count_column, role_column, "conversation_id"
        )

        # Add metrics to each row based on its conversation_id
        analyzer_id = getattr(self, "analyzer_id", "token_stats")
        source_col = "conversation"

        metric_schemas = {
            "system_tokens": {
                "type": ColumnType.INT,
                "content_type": ContentType.NUMERIC,
                "description": (
                    "Sum of tokens in all system messages in the conversation"
                ),
            },
            "user_tokens": {
                "type": ColumnType.INT,
                "content_type": ContentType.NUMERIC,
                "description": "Sum of tokens in all user messages in the conversation",
            },
            "input_tokens": {
                "type": ColumnType.INT,
                "content_type": ContentType.NUMERIC,
                "description": (
                    "Sum of tokens in system and user messages "
                    "(total input to the model during training/fine-tuning)"
                ),
            },
            "output_tokens": {
                "type": ColumnType.INT,
                "content_type": ContentType.NUMERIC,
                "description": (
                    "Sum of tokens in all assistant messages (model output/generation)"
                ),
            },
        }

        # Add all metrics in a specific order for clarity
        metric_names = [
            "system_tokens",
            "user_tokens",
            "input_tokens",
            "output_tokens",
        ]
        for metric_name in metric_names:
            metric_schema = metric_schemas[metric_name]
            col_name = make_analyzer_column_name(source_col, analyzer_id, metric_name)
            result_df[col_name] = df["conversation_id"].map(
                lambda cid: conv_metrics.get(cid, {}).get(metric_name, 0)
            )
            generated_schema[col_name] = metric_schema

        return result_df, generated_schema
