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

"""Conversation structure analyzer for analyzing turn patterns.

Based on findings from "Fixing It in Post" paper which showed that
conversation structure (single-turn vs multi-turn) significantly
impacts model capabilities.
"""

from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.column_utils import make_analyzer_column_name
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("conversation_structure")
class ConversationStructureAnalyzer(SampleAnalyzer):
    """Analyzer for conversation structure and turn patterns.

    This analyzer evaluates conversation structure to help understand
    dataset composition. The paper found that Tulu is 95% single-turn
    vs SmolTalk 70% multi-turn, which impacts model behavior.

    Metrics computed (conversation-level):
        - turn_count: Total number of turns in conversation
        - user_turn_count: Number of user messages
        - assistant_turn_count: Number of assistant messages
        - is_single_turn: Whether it's a single-turn conversation (<=2 messages)
        - is_multi_turn: Whether it's a multi-turn conversation (>2 messages)
        - avg_turn_length: Average message length in conversation
        - turn_length_variance: Variance in message lengths
        - conversation_depth: How many back-and-forth exchanges
        - role_balance: Ratio of user to assistant messages (0-1)
        - has_system_prompt: Whether conversation has a system message
    """

    def __init__(
        self,
        *,
        single_turn_threshold: int = 2,
        compute_length_stats: bool = True,
    ):
        """Initialize the ConversationStructureAnalyzer.

        Args:
            single_turn_threshold: Max messages to consider single-turn.
            compute_length_stats: Whether to compute length statistics.
        """
        self.single_turn_threshold = single_turn_threshold
        self.compute_length_stats = compute_length_stats

    def get_output_schema(
        self,
        df: pd.DataFrame | None = None,
        schema: dict | None = None,
        analyzer_id: str | None = None,
    ) -> dict:
        """Return the schema this analyzer will produce.

        Args:
            df: Not used - this analyzer produces conversation-level metrics.
            schema: Not used - this analyzer produces conversation-level metrics.
            analyzer_id: The analyzer ID for column naming. Defaults to "conversation_structure".

        Returns:
            Schema dict mapping column names to their type/description.
        """
        if analyzer_id is None:
            analyzer_id = getattr(self, "analyzer_id", "conversation_structure")

        # This analyzer uses "conversation" as the source column
        source_col = "conversation"

        schema = {
            make_analyzer_column_name(source_col, analyzer_id, "turn_count"): {
                "type": ColumnType.INT,
                "content_type": ContentType.NUMERIC,
                "description": "Total number of turns in conversation",
            },
            make_analyzer_column_name(source_col, analyzer_id, "user_turn_count"): {
                "type": ColumnType.INT,
                "content_type": ContentType.NUMERIC,
                "description": "Number of user turns",
            },
            make_analyzer_column_name(source_col, analyzer_id, "assistant_turn_count"): {
                "type": ColumnType.INT,
                "content_type": ContentType.NUMERIC,
                "description": "Number of assistant turns",
            },
            make_analyzer_column_name(source_col, analyzer_id, "is_single_turn"): {
                "type": ColumnType.BOOL,
                "content_type": ContentType.BOOLEAN,
                "description": "Whether conversation is single-turn",
            },
            make_analyzer_column_name(source_col, analyzer_id, "is_multi_turn"): {
                "type": ColumnType.BOOL,
                "content_type": ContentType.BOOLEAN,
                "description": "Whether conversation is multi-turn",
            },
            make_analyzer_column_name(source_col, analyzer_id, "conversation_depth"): {
                "type": ColumnType.INT,
                "content_type": ContentType.NUMERIC,
                "description": "Maximum conversation depth",
            },
            make_analyzer_column_name(source_col, analyzer_id, "role_balance"): {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": "Balance between user and assistant turns (0.0-1.0)",
            },
            make_analyzer_column_name(source_col, analyzer_id, "has_system_prompt"): {
                "type": ColumnType.BOOL,
                "content_type": ContentType.BOOLEAN,
                "description": "Whether conversation has system prompt",
            },
        }

        if self.compute_length_stats:
            schema[make_analyzer_column_name(source_col, analyzer_id, "avg_turn_length")] = {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": "Average turn length in conversation",
            }
            schema[make_analyzer_column_name(source_col, analyzer_id, "turn_length_variance")] = {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": "Variance in turn lengths",
            }

        return schema

    def _analyze_conversation(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze a single conversation's structure.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.

        Returns:
            Dictionary of structure metrics.
        """
        if not messages:
            return {
                "turn_count": 0,
                "user_turn_count": 0,
                "assistant_turn_count": 0,
                "is_single_turn": True,
                "is_multi_turn": False,
                "avg_turn_length": 0.0,
                "turn_length_variance": 0.0,
                "conversation_depth": 0,
                "role_balance": 0.0,
                "has_system_prompt": False,
            }

        # Count turns by role
        turn_count = len(messages)
        user_turns = sum(1 for m in messages if m.get("role", "").lower() == "user")
        assistant_turns = sum(
            1 for m in messages if m.get("role", "").lower() == "assistant"
        )
        system_turns = sum(
            1 for m in messages if m.get("role", "").lower() == "system"
        )

        # Determine conversation type
        is_single_turn = turn_count <= self.single_turn_threshold
        is_multi_turn = turn_count > self.single_turn_threshold

        # Calculate conversation depth (number of complete exchanges)
        # An exchange is a user message followed by assistant message
        depth = min(user_turns, assistant_turns)

        # Role balance (0.5 is perfectly balanced, 0 or 1 is all one role)
        non_system_turns = user_turns + assistant_turns
        role_balance = (
            user_turns / non_system_turns if non_system_turns > 0 else 0.0
        )

        result = {
            "turn_count": turn_count,
            "user_turn_count": user_turns,
            "assistant_turn_count": assistant_turns,
            "is_single_turn": is_single_turn,
            "is_multi_turn": is_multi_turn,
            "conversation_depth": depth,
            "role_balance": round(role_balance, 3),
            "has_system_prompt": system_turns > 0,
        }

        # Compute length statistics if enabled
        if self.compute_length_stats:
            lengths = [
                len(str(m.get("content", "")).split())
                for m in messages
                if m.get("role", "").lower() != "system"
            ]

            if lengths:
                avg_length = sum(lengths) / len(lengths)
                variance = (
                    sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
                    if len(lengths) > 1
                    else 0.0
                )
                result["avg_turn_length"] = round(avg_length, 2)
                result["turn_length_variance"] = round(variance, 2)
            else:
                result["avg_turn_length"] = 0.0
                result["turn_length_variance"] = 0.0

        return result

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze conversation structure.

        This analyzer works at the conversation level, not message level.
        It expects a DataFrame where each row is a message, with a
        conversation_id column to group messages.

        Args:
            df: Input DataFrame with messages.
            schema: Column schema dict.

        Returns:
            DataFrame with added conversation structure columns.
        """
        result_df = df.copy()
        analyzer_id = getattr(self, "analyzer_id", "conversation_structure")
        source_col = "conversation"

        # Check for conversation_id column
        if "conversation_id" not in df.columns:
            # If no conversation_id, treat each row as its own conversation
            # This handles flat instruction-response format
            result_df[make_analyzer_column_name(source_col, analyzer_id, "turn_count")] = 2
            result_df[make_analyzer_column_name(source_col, analyzer_id, "user_turn_count")] = 1
            result_df[make_analyzer_column_name(source_col, analyzer_id, "assistant_turn_count")] = 1
            result_df[make_analyzer_column_name(source_col, analyzer_id, "is_single_turn")] = True
            result_df[make_analyzer_column_name(source_col, analyzer_id, "is_multi_turn")] = False
            result_df[make_analyzer_column_name(source_col, analyzer_id, "conversation_depth")] = 1
            result_df[make_analyzer_column_name(source_col, analyzer_id, "role_balance")] = 0.5
            result_df[make_analyzer_column_name(source_col, analyzer_id, "has_system_prompt")] = False
            return result_df

        # Find role and content columns
        role_column = None
        content_column = None

        if schema:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and col in df.columns
                    and "role" in col.lower()
                ):
                    role_column = col
                if (
                    config.get("content_type") == ContentType.TEXT
                    and col in df.columns
                ):
                    content_column = col

        # Fallback to common column names
        if role_column is None and "role" in df.columns:
            role_column = "role"
        if content_column is None and "text_content" in df.columns:
            content_column = "text_content"
        elif content_column is None and "content" in df.columns:
            content_column = "content"

        if role_column is None or content_column is None:
            return result_df

        # Build conversation metrics
        conv_metrics = {}
        for conv_id, group in df.groupby("conversation_id"):
            messages = [
                {"role": row[role_column], "content": row[content_column]}
                for _, row in group.iterrows()
            ]
            conv_metrics[conv_id] = self._analyze_conversation(messages)

        # Add metrics to each row based on its conversation_id
        for metric_name in [
            "turn_count",
            "user_turn_count",
            "assistant_turn_count",
            "is_single_turn",
            "is_multi_turn",
            "conversation_depth",
            "role_balance",
            "has_system_prompt",
        ]:
            col_name = make_analyzer_column_name(source_col, analyzer_id, metric_name)
            result_df[col_name] = df["conversation_id"].map(
                lambda cid, m=metric_name: conv_metrics.get(cid, {}).get(m)
            )

        if self.compute_length_stats:
            col_name = make_analyzer_column_name(source_col, analyzer_id, "avg_turn_length")
            result_df[col_name] = df["conversation_id"].map(
                lambda cid: conv_metrics.get(cid, {}).get("avg_turn_length")
            )
            col_name = make_analyzer_column_name(source_col, analyzer_id, "turn_length_variance")
            result_df[col_name] = df["conversation_id"].map(
                lambda cid: conv_metrics.get(cid, {}).get("turn_length_variance")
            )

        return result_df
