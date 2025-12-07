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

"""Field mapping utilities for customer onboarding.

This module provides tools to automatically map customer data columns
to Oumi configuration placeholders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from oumi.onboarding.data_analyzer import DataSchema


@dataclass
class FieldMapping:
    """Represents a mapping from customer column to Oumi placeholder."""

    customer_column: str
    oumi_placeholder: str
    confidence: float  # 0.0 to 1.0
    reason: str


class FieldMapper:
    """Map customer data columns to Oumi configuration placeholders.

    This class analyzes column names and content to suggest mappings
    from customer data to Oumi's expected placeholder names.

    Example:
        >>> from oumi.onboarding import DataAnalyzer, FieldMapper
        >>> analyzer = DataAnalyzer()
        >>> schema = analyzer.analyze("./data.csv")
        >>> mapper = FieldMapper()
        >>> mappings = mapper.suggest_mappings(schema)
        >>> for m in mappings:
        ...     print(f"{m.customer_column} -> {{{m.oumi_placeholder}}}")
    """

    # Patterns to match column names to Oumi placeholders
    # Each pattern maps to (oumi_placeholder, confidence_boost)
    COLUMN_PATTERNS: dict[str, list[tuple[str, float]]] = {
        # Question/Query patterns
        "question": [
            (r"^question$", 1.0),
            (r"^query$", 0.9),
            (r"^prompt$", 0.8),
            (r"^input$", 0.7),
            (r"^user_input$", 0.85),
            (r"^request$", 0.7),
            (r"question", 0.6),
            (r"query", 0.5),
        ],
        # Answer/Response patterns
        "answer": [
            (r"^answer$", 1.0),
            (r"^response$", 0.95),
            (r"^output$", 0.8),
            (r"^reply$", 0.85),
            (r"^completion$", 0.8),
            (r"^assistant$", 0.7),
            (r"answer", 0.6),
            (r"response", 0.5),
        ],
        # Context/Document patterns
        "context": [
            (r"^context$", 1.0),
            (r"^document$", 0.95),
            (r"^passage$", 0.9),
            (r"^text$", 0.6),
            (r"^content$", 0.5),
            (r"context", 0.6),
            (r"document", 0.5),
            (r"passage", 0.5),
        ],
        # Conversation patterns
        "conversation": [
            (r"^conversation$", 1.0),
            (r"^chat$", 0.9),
            (r"^dialogue$", 0.9),
            (r"^messages$", 0.85),
            (r"^turns$", 0.8),
            (r"^transcript$", 0.75),
            (r"conversation", 0.6),
            (r"chat", 0.5),
        ],
        # System instruction patterns
        "system_instruction": [
            (r"^system$", 0.9),
            (r"^system_prompt$", 1.0),
            (r"^instruction$", 0.85),
            (r"^persona$", 0.7),
            (r"^role$", 0.5),
            (r"system", 0.5),
        ],
        # Topic/Category patterns
        "topic": [
            (r"^topic$", 1.0),
            (r"^category$", 0.9),
            (r"^domain$", 0.85),
            (r"^subject$", 0.8),
            (r"^type$", 0.5),
            (r"topic", 0.5),
        ],
        # Label/Target patterns
        "label": [
            (r"^label$", 1.0),
            (r"^target$", 0.9),
            (r"^class$", 0.85),
            (r"^classification$", 0.8),
            (r"^status$", 0.6),
            (r"^flag$", 0.5),
            (r"label", 0.5),
        ],
    }

    # Minimum confidence to include a mapping
    MIN_CONFIDENCE = 0.3

    def __init__(self, min_confidence: float = 0.3):
        """Initialize the FieldMapper.

        Args:
            min_confidence: Minimum confidence score to include a mapping.
        """
        self.min_confidence = min_confidence

    def suggest_mappings(
        self,
        schema: DataSchema,
        goal: Optional[str] = None,
    ) -> list[FieldMapping]:
        """Suggest mappings from customer columns to Oumi placeholders.

        Args:
            schema: The analyzed data schema.
            goal: Optional synthesis goal to prioritize relevant mappings.

        Returns:
            List of suggested FieldMappings, sorted by confidence.
        """
        mappings = []
        used_placeholders: set[str] = set()

        # Sort columns by text likelihood (prefer text columns for mapping)
        sorted_columns = sorted(
            schema.columns,
            key=lambda c: (c.is_text or c.is_conversation, c.avg_length or 0),
            reverse=True,
        )

        for col in sorted_columns:
            mapping = self._find_best_mapping(
                col.name,
                col,
                goal,
                used_placeholders,
            )
            if mapping and mapping.confidence >= self.min_confidence:
                mappings.append(mapping)
                used_placeholders.add(mapping.oumi_placeholder)

        # Sort by confidence
        return sorted(mappings, key=lambda m: m.confidence, reverse=True)

    def _find_best_mapping(
        self,
        col_name: str,
        col_info: "ColumnInfo",
        goal: Optional[str],
        used_placeholders: set[str],
    ) -> Optional[FieldMapping]:
        """Find the best mapping for a column."""
        from oumi.onboarding.data_analyzer import ColumnInfo

        col_lower = col_name.lower().strip()
        best_mapping: Optional[FieldMapping] = None
        best_score = 0.0

        for placeholder, patterns in self.COLUMN_PATTERNS.items():
            # Skip if placeholder already used
            if placeholder in used_placeholders:
                continue

            for pattern, base_confidence in patterns:
                if re.search(pattern, col_lower, re.IGNORECASE):
                    score = base_confidence

                    # Boost confidence based on column characteristics
                    if placeholder in ("question", "answer", "context") and col_info.is_text:
                        score = min(1.0, score + 0.1)

                    if placeholder == "conversation" and col_info.is_conversation:
                        score = min(1.0, score + 0.2)

                    if placeholder == "label" and col_info.is_categorical:
                        score = min(1.0, score + 0.15)

                    # Boost based on goal alignment
                    if goal:
                        score = self._adjust_for_goal(score, placeholder, goal)

                    if score > best_score:
                        best_score = score
                        best_mapping = FieldMapping(
                            customer_column=col_name,
                            oumi_placeholder=placeholder,
                            confidence=score,
                            reason=self._generate_reason(col_name, placeholder, pattern),
                        )

        return best_mapping

    def _adjust_for_goal(
        self, score: float, placeholder: str, goal: str
    ) -> float:
        """Adjust confidence score based on synthesis goal."""
        goal_boosts = {
            "qa": {"question": 0.1, "answer": 0.1, "context": 0.05},
            "conversation": {"conversation": 0.15, "system_instruction": 0.05},
            "augmentation": {"context": 0.1, "topic": 0.05},
            "instruction": {"system_instruction": 0.15, "context": 0.05},
        }

        if goal in goal_boosts and placeholder in goal_boosts[goal]:
            score = min(1.0, score + goal_boosts[goal][placeholder])

        return score

    def _generate_reason(
        self, col_name: str, placeholder: str, pattern: str
    ) -> str:
        """Generate a human-readable reason for the mapping."""
        if pattern.startswith("^") and pattern.endswith("$"):
            return f"Column '{col_name}' exactly matches expected placeholder '{placeholder}'"
        return f"Column '{col_name}' matches pattern for '{placeholder}'"

    def get_mapping_dict(
        self,
        mappings: list[FieldMapping],
    ) -> dict[str, str]:
        """Convert mappings to a simple dictionary.

        Args:
            mappings: List of FieldMappings.

        Returns:
            Dictionary mapping customer columns to Oumi placeholders.
        """
        return {m.customer_column: m.oumi_placeholder for m in mappings}

    def apply_mappings_to_template(
        self,
        template: str,
        mappings: list[FieldMapping],
    ) -> str:
        """Apply mappings to a template string.

        Replaces Oumi placeholders with customer column references.

        Args:
            template: Template string with {placeholder} syntax.
            mappings: List of FieldMappings to apply.

        Returns:
            Template with placeholders replaced by column references.
        """
        result = template
        for mapping in mappings:
            # Replace {placeholder} with {customer_column}
            result = result.replace(
                f"{{{mapping.oumi_placeholder}}}",
                f"{{{mapping.customer_column}}}",
            )
        return result

    def get_unmapped_placeholders(
        self,
        template: str,
        mappings: list[FieldMapping],
    ) -> list[str]:
        """Find placeholders in template that are not mapped.

        Args:
            template: Template string with {placeholder} syntax.
            mappings: List of existing mappings.

        Returns:
            List of unmapped placeholder names.
        """
        # Find all placeholders in template
        placeholders = re.findall(r"\{(\w+)\}", template)

        # Get mapped placeholders
        mapped = {m.oumi_placeholder for m in mappings}

        # Return unmapped
        return [p for p in placeholders if p not in mapped]

    def suggest_for_template(
        self,
        schema: DataSchema,
        template: str,
        goal: Optional[str] = None,
    ) -> tuple[list[FieldMapping], list[str]]:
        """Suggest mappings specifically for a template's placeholders.

        Args:
            schema: The analyzed data schema.
            template: Template string with {placeholder} syntax.
            goal: Optional synthesis goal.

        Returns:
            Tuple of (mappings, unmapped_placeholders).
        """
        # Find required placeholders
        required_placeholders = set(re.findall(r"\{(\w+)\}", template))

        # Get all possible mappings
        all_mappings = self.suggest_mappings(schema, goal)

        # Filter to only mappings for required placeholders
        relevant_mappings = [
            m for m in all_mappings if m.oumi_placeholder in required_placeholders
        ]

        # Find unmapped
        mapped = {m.oumi_placeholder for m in relevant_mappings}
        unmapped = [p for p in required_placeholders if p not in mapped]

        return relevant_mappings, unmapped
