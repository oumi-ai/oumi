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

from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeResponseFormat,
)


@dataclass
class RuleJudgeParams(BaseParams):
    r"""Parameters for rule-based judge evaluation.

    This class defines the configuration for a rule-based judge that uses
    deterministic rules.

    Examples:
        Regex pattern matching:
        >>> rule_params = RuleJudgeParams(  # doctest: +SKIP
        ...     rule_type="regex_match",
        ...     input_fields=["text"],
        ...     rule_config={"pattern": r"\\d{3}-\\d{4}", "match_mode": "contains"},
        ...     response_format=JudgeResponseFormat.XML,
        ...     judgment_type=JudgeOutputType.BOOL
        ... )
    """

    rule_type: str
    """Type of rule to apply (e.g., 'exact_match', 'regex_match', 'contains', etc.)"""

    input_fields: list[str]
    """List of input field names that the rule will operate on.

    These fields must be present in the input data passed to the judge.
    Example: ["expected_answer", "actual_answer"] for comparison rules.
    """

    rule_config: dict[str, Any] = field(default_factory=dict)
    """Configuration specific to the rule type.

    Different rule types require different configuration parameters.
    Examples:
        - regex_match: {"pattern": r"\\d{3}-\\d{4}", "match_mode": "fullmatch"}
    """

    response_format: JudgeResponseFormat = field(default=JudgeResponseFormat.XML)
    """The format in which the judge output should be formatted."""

    judgment_type: JudgeOutputType = field(default=JudgeOutputType.BOOL)
    """The type of output that the judgment produces."""

    judgment_scores: dict[str, float] | None = field(default=None)
    """For ENUM judgment_type, the mapping from category names to numeric scores.

    Example:
        {"excellent": 1.0, "good": 0.7, "poor": 0.3}
    """

    def __post_init__(self):
        """Validate the parameters after initialization."""
        self._validate_params()

    def _validate_params(self):
        """Validate the parameters for consistency and completeness.

        Raises:
            ValueError: If parameters are invalid
        """
        if not self.rule_type or not self.rule_type.strip():
            raise ValueError("rule_type cannot be empty")

        if not self.input_fields:
            raise ValueError("input_fields cannot be empty")

        if not all(
            isinstance(field, str) and field.strip() for field in self.input_fields
        ):
            raise ValueError("All input_fields must be non-empty strings")

        if self.judgment_type == JudgeOutputType.ENUM and not self.judgment_scores:
            raise ValueError("judgment_scores must be provided for ENUM judgment_type")

        if self.judgment_scores:
            if not all(
                isinstance(score, int | float)
                for score in self.judgment_scores.values()
            ):
                raise ValueError("All judgment_scores values must be numeric")
            if len(self.judgment_scores) == 0:
                raise ValueError("judgment_scores cannot be empty when provided")
