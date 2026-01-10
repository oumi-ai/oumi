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

"""Conversation structure analyzer for analyzing multi-turn conversation metrics."""

from typing import Optional

import numpy as np
import pandas as pd

from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("conversation_structure")
class ConversationStructureAnalyzer(SampleAnalyzer):
    """Analyzer that computes conversation structure metrics.

    Analyzes turn counts, message lengths, and conversation balance.
    """

    def __init__(
        self,
        *,
        conversation_id_column: str = "conversation_id",
        role_column: str = "role",
        content_column: str = "text_content",
        min_turns: int = 2,
        max_turns: int = 100,
        tokenizer=None,
    ):
        """Initialize the ConversationStructureAnalyzer.

        Args:
            conversation_id_column: Column for grouping by conversation.
            role_column: Column containing role values.
            content_column: Column containing message content.
            min_turns: Minimum turns for a valid conversation.
            max_turns: Maximum turns before flagging as too long.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.conversation_id_column = conversation_id_column
        self.role_column = role_column
        self.content_column = content_column
        self.min_turns = min_turns
        self.max_turns = max_turns

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze conversation structure.

        Args:
            df: Input DataFrame with conversation data.
            schema: Column schema (optional).

        Returns:
            DataFrame with added columns:
            - conv_turn_count: Total turns in conversation
            - conv_user_turn_count: User message count
            - conv_assistant_turn_count: Assistant message count
            - conv_avg_turn_length: Average message length (words)
            - conv_user_assistant_ratio: User/assistant message ratio
            - conv_is_too_short: Below minimum turns
            - conv_is_too_long: Above maximum turns
        """
        result_df = df.copy()

        # Verify required columns exist
        required = [self.conversation_id_column, self.role_column, self.content_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Compute per-conversation metrics
        conv_metrics = {}
        for conv_id, group in df.groupby(self.conversation_id_column):
            roles = group[self.role_column].str.lower()
            contents = group[self.content_column].astype(str)

            turn_count = len(group)
            user_turns = (roles == "user").sum()
            assistant_turns = (roles == "assistant").sum()

            # Word counts per message
            word_counts = contents.apply(lambda x: len(x.split()))
            avg_length = word_counts.mean()
            length_variance = word_counts.var() if len(word_counts) > 1 else 0

            # User/assistant ratio
            ua_ratio = user_turns / assistant_turns if assistant_turns > 0 else float("inf")

            conv_metrics[conv_id] = {
                "turn_count": turn_count,
                "user_turn_count": user_turns,
                "assistant_turn_count": assistant_turns,
                "avg_turn_length": avg_length,
                "turn_length_variance": length_variance,
                "user_assistant_ratio": ua_ratio,
                "is_too_short": turn_count < self.min_turns,
                "is_too_long": turn_count > self.max_turns,
            }

        # Map metrics back to rows
        for metric in [
            "turn_count",
            "user_turn_count",
            "assistant_turn_count",
            "avg_turn_length",
            "turn_length_variance",
            "user_assistant_ratio",
            "is_too_short",
            "is_too_long",
        ]:
            result_df[f"conv_{metric}"] = df[self.conversation_id_column].map(
                lambda x: conv_metrics.get(x, {}).get(metric, np.nan)
            )

        return result_df, {}
