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

"""Role sequence analyzer for validating conversation turn structure."""

from typing import Optional

import pandas as pd

from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("role_sequence")
class RoleSequenceAnalyzer(SampleAnalyzer):
    """Analyzer that validates conversation role sequences.

    Checks that conversations follow expected turn-taking patterns.
    """

    def __init__(
        self,
        *,
        role_column: str = "role",
        conversation_id_column: str = "conversation_id",
        valid_roles: Optional[set[str]] = None,
        require_alternating: bool = True,
        require_assistant_end: bool = True,
    ):
        """Initialize the RoleSequenceAnalyzer.

        Args:
            role_column: Column containing role values.
            conversation_id_column: Column for grouping by conversation.
            valid_roles: Set of valid role values. Default: {system, user, assistant}.
            require_alternating: Require user/assistant to alternate.
            require_assistant_end: Require conversation to end with assistant.
        """
        self.role_column = role_column
        self.conversation_id_column = conversation_id_column
        self.valid_roles = valid_roles or {"system", "user", "assistant"}
        self.require_alternating = require_alternating
        self.require_assistant_end = require_assistant_end

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze role sequences in conversations.

        Args:
            df: Input DataFrame with role column.
            schema: Column schema (not used but required by interface).

        Returns:
            DataFrame with added columns:
            - role_is_valid: Role is in valid_roles set
            - role_sequence_valid: Conversation has valid role sequence
            - role_has_consecutive_same: Same role appears consecutively
        """
        result_df = df.copy()

        if self.role_column not in df.columns:
            raise ValueError(f"Role column '{self.role_column}' not found in DataFrame.")

        roles = df[self.role_column].astype(str).str.lower()

        # Check if each role is valid
        result_df["role_is_valid"] = roles.isin(self.valid_roles)

        # Check for consecutive same roles (excluding system)
        prev_roles = roles.shift(1)
        same_as_prev = (roles == prev_roles) & (roles != "system")

        # Handle conversation boundaries
        if self.conversation_id_column in df.columns:
            conv_ids = df[self.conversation_id_column]
            prev_conv = conv_ids.shift(1)
            # Don't flag first message of each conversation
            same_as_prev = same_as_prev & (conv_ids == prev_conv)

        result_df["role_has_consecutive_same"] = same_as_prev

        # Compute conversation-level validity
        if self.conversation_id_column in df.columns:
            conv_valid = self._compute_conversation_validity(df, roles)
            result_df["role_sequence_valid"] = df[self.conversation_id_column].map(
                conv_valid
            )
        else:
            # Single conversation - check overall validity
            is_valid = self._check_sequence_valid(roles.tolist())
            result_df["role_sequence_valid"] = is_valid

        return result_df

    def _compute_conversation_validity(
        self, df: pd.DataFrame, roles: pd.Series
    ) -> dict[str, bool]:
        """Compute validity for each conversation."""
        validity = {}
        for conv_id, group in df.groupby(self.conversation_id_column):
            role_list = roles.loc[group.index].tolist()
            validity[conv_id] = self._check_sequence_valid(role_list)
        return validity

    def _check_sequence_valid(self, roles: list[str]) -> bool:
        """Check if a role sequence is valid."""
        if not roles:
            return False

        # Filter out system messages for alternation check
        non_system = [r for r in roles if r != "system"]

        if not non_system:
            return False

        # Check starts with user (after any system messages)
        if non_system[0] != "user":
            return False

        # Check ends with assistant
        if self.require_assistant_end and non_system[-1] != "assistant":
            return False

        # Check alternation
        if self.require_alternating:
            for i in range(1, len(non_system)):
                if non_system[i] == non_system[i - 1]:
                    return False

        return True
