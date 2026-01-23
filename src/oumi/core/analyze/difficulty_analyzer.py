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

"""Difficulty analyzer for estimating sample difficulty.

This analyzer estimates the difficulty of instruction-response pairs
using heuristic signals, useful for curriculum learning and dataset
stratification.
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.column_utils import make_analyzer_column_name
from oumi.core.analyze.sample_analyzer import DEFAULT_TEXT_COLUMNS, SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("difficulty")
class DifficultyAnalyzer(SampleAnalyzer):
    """Analyzer for estimating sample difficulty.

    This analyzer estimates difficulty based on multiple signals:
        - Instruction complexity (length, specificity, constraints)
        - Domain signals (technical terms, specialized vocabulary)
        - Response requirements (expected depth, structure)
        - Reasoning requirements (multi-step, logical chains)

    Metrics computed:
        - difficulty_score: Overall difficulty score (0-1)
        - difficulty_tier: Difficulty tier (easy, medium, hard, expert)
        - requires_reasoning: Whether multi-step reasoning is needed
        - requires_domain_knowledge: Whether specialized knowledge needed
        - constraint_count: Number of explicit constraints in instruction
    """

    # Patterns indicating reasoning requirements
    _REASONING_PATTERNS = [
        re.compile(
            r"\b(?:why|explain\s+why|reason|because|therefore|"
            r"consequently|thus|hence|analyze|evaluate)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:step[- ]by[- ]step|first|then|next|finally|"
            r"after\s+that|following\s+that)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:compare|contrast|differentiate|distinguish|"
            r"pros\s+and\s+cons|advantages|disadvantages)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:if|when|assuming|given\s+that|suppose|"
            r"in\s+case|provided\s+that)\b",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating domain-specific knowledge
    _DOMAIN_PATTERNS = {
        "programming": re.compile(
            r"\b(?:algorithm|function|class|API|database|SQL|"
            r"HTTP|REST|JSON|XML|regex|recursion|complexity|"
            r"runtime|memory|pointer|thread|async)\b",
            re.IGNORECASE,
        ),
        "math": re.compile(
            r"\b(?:theorem|proof|equation|derivative|integral|"
            r"matrix|vector|polynomial|logarithm|exponential|"
            r"probability|statistics|variance|correlation)\b",
            re.IGNORECASE,
        ),
        "science": re.compile(
            r"\b(?:hypothesis|experiment|theory|molecule|atom|"
            r"chemical|biological|physics|quantum|relativity|"
            r"genome|evolution|ecosystem|thermodynamics)\b",
            re.IGNORECASE,
        ),
        "legal": re.compile(
            r"\b(?:statute|regulation|liability|jurisdiction|"
            r"precedent|litigation|plaintiff|defendant|tort|"
            r"contract|compliance|constitutional)\b",
            re.IGNORECASE,
        ),
        "medical": re.compile(
            r"\b(?:diagnosis|symptom|treatment|medication|"
            r"pathology|anatomy|physiology|clinical|surgical|"
            r"pharmaceutical|therapeutic|prognosis)\b",
            re.IGNORECASE,
        ),
        "finance": re.compile(
            r"\b(?:portfolio|investment|derivative|equity|"
            r"bond|yield|dividend|valuation|arbitrage|"
            r"hedge|liquidity|amortization)\b",
            re.IGNORECASE,
        ),
    }

    # Patterns indicating constraints
    _CONSTRAINT_PATTERNS = [
        re.compile(
            r"\b(?:must|should|need\s+to|have\s+to|required|"
            r"mandatory|essential|necessary)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:at\s+least|at\s+most|maximum|minimum|"
            r"exactly|only|no\s+more\s+than|no\s+less\s+than)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:without|except|excluding|not\s+including|"
            r"avoid|don't|do\s+not|never)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:in\s+\d+\s+(?:words|sentences|paragraphs)|"
            r"using\s+(?:only|just)|within\s+\d+)\b",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating multi-part questions
    _MULTI_PART_PATTERNS = [
        re.compile(r"\b(?:1\)|2\)|3\)|a\)|b\)|c\))\s*\w+", re.IGNORECASE),
        re.compile(
            r"\b(?:first|second|third|also|additionally|"
            r"furthermore|moreover)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\band\s+(?:also|then|finally)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        *,
        analyze_user_only: bool = True,
        include_component_scores: bool = True,
    ):
        """Initialize the DifficultyAnalyzer.

        Args:
            analyze_user_only: If True, only analyze user messages.
            include_component_scores: Include individual difficulty components.
        """
        self.analyze_user_only = analyze_user_only
        self.include_component_scores = include_component_scores

    def _count_constraints(self, text: str) -> int:
        """Count explicit constraints in the instruction.

        Args:
            text: Instruction text.

        Returns:
            Number of constraints.
        """
        count = 0
        for pattern in self._CONSTRAINT_PATTERNS:
            count += len(pattern.findall(text))
        return count

    def _requires_reasoning(self, text: str) -> bool:
        """Check if instruction requires multi-step reasoning.

        Args:
            text: Instruction text.

        Returns:
            True if reasoning is required.
        """
        reasoning_count = sum(
            len(pattern.findall(text)) for pattern in self._REASONING_PATTERNS
        )
        return reasoning_count >= 2

    def _detect_domains(self, text: str) -> list[str]:
        """Detect specialized domains in the instruction.

        Args:
            text: Instruction text.

        Returns:
            List of detected domain names.
        """
        domains = []
        for domain, pattern in self._DOMAIN_PATTERNS.items():
            if pattern.search(text):
                domains.append(domain)
        return domains

    def _is_multi_part(self, text: str) -> bool:
        """Check if instruction has multiple parts/questions.

        Args:
            text: Instruction text.

        Returns:
            True if multi-part.
        """
        for pattern in self._MULTI_PART_PATTERNS:
            if len(pattern.findall(text)) >= 2:
                return True
        return False

    def _compute_difficulty_score(
        self,
        text: str,
        constraint_count: int,
        requires_reasoning: bool,
        domains: list[str],
        is_multi_part: bool,
    ) -> float:
        """Compute overall difficulty score.

        Args:
            text: Instruction text.
            constraint_count: Number of constraints.
            requires_reasoning: Whether reasoning is required.
            domains: List of detected domains.
            is_multi_part: Whether instruction is multi-part.

        Returns:
            Difficulty score (0-1).
        """
        score = 0.3  # Base difficulty

        # Word count contribution (longer = harder)
        word_count = len(text.split())
        if word_count > 100:
            score += 0.15
        elif word_count > 50:
            score += 0.1
        elif word_count < 10:
            score -= 0.1

        # Constraint contribution
        score += min(0.2, constraint_count * 0.05)

        # Reasoning contribution
        if requires_reasoning:
            score += 0.15

        # Domain knowledge contribution
        score += min(0.2, len(domains) * 0.1)

        # Multi-part contribution
        if is_multi_part:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _get_difficulty_tier(self, score: float) -> str:
        """Convert difficulty score to tier label.

        Args:
            score: Difficulty score (0-1).

        Returns:
            Tier label.
        """
        if score >= 0.75:
            return "expert"
        elif score >= 0.5:
            return "hard"
        elif score >= 0.3:
            return "medium"
        else:
            return "easy"

    def _analyze_instruction(self, text: str) -> dict[str, Any]:
        """Analyze an instruction for difficulty.

        Args:
            text: Instruction text.

        Returns:
            Dictionary of difficulty metrics.
        """
        if not text or not text.strip():
            return {
                "difficulty_score": 0.0,
                "difficulty_tier": "easy",
                "requires_reasoning": False,
                "requires_domain_knowledge": False,
                "constraint_count": 0,
            }

        constraint_count = self._count_constraints(text)
        requires_reasoning = self._requires_reasoning(text)
        domains = self._detect_domains(text)
        is_multi_part = self._is_multi_part(text)

        difficulty_score = self._compute_difficulty_score(
            text, constraint_count, requires_reasoning, domains, is_multi_part
        )

        result = {
            "difficulty_score": round(difficulty_score, 3),
            "difficulty_tier": self._get_difficulty_tier(difficulty_score),
        }

        if self.include_component_scores:
            result["requires_reasoning"] = requires_reasoning
            result["requires_domain_knowledge"] = len(domains) > 0
            result["constraint_count"] = constraint_count

        return result

    def get_output_schema(
        self,
        df: pd.DataFrame | None = None,
        schema: dict | None = None,
        analyzer_id: str | None = None,
    ) -> dict:
        """Return the schema this analyzer will produce."""
        aid: str = analyzer_id or getattr(self, "analyzer_id", "difficulty")

        if schema is not None and df is not None:
            text_columns = [
                col
                for col, config in schema.items()
                if config.get("content_type") == ContentType.TEXT and col in df.columns
            ]
        else:
            text_columns = DEFAULT_TEXT_COLUMNS

        output_schema = {}
        for column in text_columns:
            col_name = make_analyzer_column_name(column, aid, "score")
            output_schema[col_name] = {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": "Difficulty score (0-1)",
            }
            col_name = make_analyzer_column_name(column, aid, "tier")
            output_schema[col_name] = {
                "type": ColumnType.STRING,
                "content_type": ContentType.CATEGORICAL,
                "description": "Difficulty tier (easy, medium, hard, expert)",
            }

            if self.include_component_scores:
                col_name = make_analyzer_column_name(column, aid, "requires_reasoning")
                output_schema[col_name] = {
                    "type": ColumnType.BOOL,
                    "content_type": ContentType.BOOLEAN,
                    "description": "Whether task requires reasoning",
                }
                col_name = make_analyzer_column_name(
                    column, aid, "requires_domain_knowledge"
                )
                output_schema[col_name] = {
                    "type": ColumnType.BOOL,
                    "content_type": ContentType.BOOLEAN,
                    "description": "Whether task requires domain knowledge",
                }
                col_name = make_analyzer_column_name(column, aid, "constraint_count")
                output_schema[col_name] = {
                    "type": ColumnType.INT,
                    "content_type": ContentType.NUMERIC,
                    "description": "Number of constraints in task",
                }

        return output_schema

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for difficulty estimation.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added difficulty analysis columns.
            generated column schema dict).
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for difficulty "
                "analysis. Please provide a column schema dict that specifies "
                "which columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df

        # Find the role column if needed
        role_column = None
        if self.analyze_user_only:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and col in df.columns
                    and "role" in col.lower()
                ):
                    role_column = col
                    break

        analyzer_id = getattr(self, "analyzer_id", "difficulty")

        for column in text_columns:
            if self.analyze_user_only and role_column is not None:
                # Only analyze user messages
                analysis_results = df.apply(
                    lambda row: (
                        self._analyze_instruction(str(row[column]))
                        if str(row.get(role_column, "")).lower() == "user"
                        else {
                            "difficulty_score": None,
                            "difficulty_tier": None,
                            "requires_reasoning": None,
                            "requires_domain_knowledge": None,
                            "constraint_count": None,
                        }
                    ),
                    axis=1,
                )
            else:
                analysis_results = (
                    df[column].astype(str).apply(self._analyze_instruction)
                )

            # Extract results to columns
            col_name = make_analyzer_column_name(column, analyzer_id, "score")
            result_df[col_name] = analysis_results.apply(
                lambda r: r.get("difficulty_score")
            )

            col_name = make_analyzer_column_name(column, analyzer_id, "tier")
            result_df[col_name] = analysis_results.apply(
                lambda r: r.get("difficulty_tier")
            )

            if self.include_component_scores:
                col_name = make_analyzer_column_name(column, analyzer_id, "requires_reasoning")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("requires_reasoning")
                )

                col_name = make_analyzer_column_name(column, analyzer_id, "requires_domain_knowledge")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("requires_domain_knowledge")
                )

                col_name = make_analyzer_column_name(column, analyzer_id, "constraint_count")
                result_df[col_name] = analysis_results.apply(
                    lambda r: r.get("constraint_count")
                )

        return result_df
