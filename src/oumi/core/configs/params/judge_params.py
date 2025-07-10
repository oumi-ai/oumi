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
from enum import Enum
from typing import Optional

from oumi.core.configs.params.base_params import BaseParams
from oumi.utils.placeholders import resolve_placeholders


class JudgeResponseFormat(str, Enum):
    """Enumeration of possible response formats for the judge output."""

    JSON = "json"
    """JSON structured response format."""

    XML = "xml"
    """XML-tagged response format."""

    RAW = "raw"
    """Plain text response format."""


class JudgeOutputType(str, Enum):
    """Enumeration of possible output types for the judge's output fields."""

    TEXT = "text"
    """Free-form text judgment."""

    ENUM = "enum"
    """Categorical judgment from predefined options."""

    INT = "int"
    """Integer value judgment."""

    FLOAT = "float"
    """Floating-point value judgment."""

    BOOL = "bool"
    """Boolean judgment (True/False, Yes/No)."""


@dataclass
class JudgeParams(BaseParams):
    """Parameters for the Judge prompt and response format.

    This class holds the parameters for a single-attribute judge,
    including the prompt template and response format.

    Examples:
        Basic boolean judgment:
        >>> judge_params = JudgeParams( # doctest: +SKIP
        ...     prompt_template="Is the following answer helpful? Question: {question},
        ...                      Answer: {answer}. Respond with True or False.",
        ...     response_format=JudgeResponseFormat.XML,
        ...     judgment_type=JudgeOutputType.BOOL,
        ...     include_explanation=False
        ... )

        Categorical judgment with scores:
        >>> judge_params = JudgeParams( # doctest: +SKIP
        ...     prompt_template="Rate the quality of this text: {text}.
        ..                       Respond with 'excellent', 'good', or 'poor'.",
        ...     response_format=JudgeResponseFormat.JSON,
        ...     judgment_type=JudgeOutputType.ENUM,
        ...     judgment_scores={"excellent": 1.0, "good": 0.7, "poor": 0.3},
        ...     include_explanation=True
        ... )
    """

    prompt_template: str
    """Template for the judge prompt with placeholders, such as {question}, {answer}."""

    system_instruction: Optional[str] = field(default=None)
    """Optional system message to guide judge behavior."""

    template_variables: dict[str, str] = field(default_factory=dict)
    """Variables to be replaced in `prompt_template` and `system_instruction`.

    This dictionary contains variable names and their corresponding values that should
    be replaced in the `prompt_template` and `system_instruction` fields, before the
    dataset-based placeholders are processed. These variables have the following format:
    {variable_name}."""

    response_format: JudgeResponseFormat = field(default=JudgeResponseFormat.XML)
    """The format in which the judge should respond."""

    judgment_type: JudgeOutputType = field(default=JudgeOutputType.BOOL)
    """The type of output that the judgment should be provided with."""

    judgment_scores: Optional[dict[str, float]] = field(default=None)
    """For ENUM judgment_type, the mapping from category names to numeric scores.

    Example:
        {"excellent": 1.0, "good": 0.7, "poor": 0.3}
    """

    include_explanation: bool = field(default=False)
    """Whether the judge should provide an explanation before the judgment."""

    examples: list[dict[str, str]] = field(default_factory=list)
    """Few-shot examples for the judge as a list of field value dictionaries.

    Each dictionary should contain values for all template placeholders and
    expected output fields. Used to provide examples of how the judge should respond.

    Example:
        [
            {
                "question": "What is 2+2?",                      # placeholder value
                "answer": "4",                                   # placeholder value
                "judgment": "Correct",                           # output field value
                "explanation": "It is mathematically correct."   # output field value
            },
            {
                "question": "What is the capital of Mars?",      # placeholder value
                "answer": "New York",                            # placeholder value
                "judgment": "Incorrect",                         # output field value
                "explanation": "Mars does not have capitals."    # output field value
            }
        ]
    """

    def __post_init__(self):
        """Validate the parameters after initialization."""
        self._validate_params()

    def _validate_params(self):
        """Validate the parameters for consistency and completeness.

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate prompt template is not empty
        if not self.prompt_template.strip():
            raise ValueError("prompt_template cannot be empty")

        # Validate judgment scores for ENUM judgment type
        if self.judgment_type == JudgeOutputType.ENUM and not self.judgment_scores:
            raise ValueError("judgment_scores must be provided for ENUM judgment_type")

        # Validate judgment scores are numeric if provided
        if self.judgment_scores:
            if not all(
                isinstance(score, (int, float))
                for score in self.judgment_scores.values()
            ):
                raise ValueError("All judgment_scores values must be numeric")
            if not self.judgment_scores:
                raise ValueError("judgment_scores cannot be empty when provided")

    def replace_template_variables(self):
        """Apply template variables to prompt_template and system_instruction."""
        if not self.template_variables:
            return

        self.prompt_template = resolve_placeholders(
            self.prompt_template, self.template_variables, missing_values_allowed=True
        )
        if self.system_instruction:
            self.system_instruction = resolve_placeholders(
                self.system_instruction,
                self.template_variables,
                missing_values_allowed=True,
            )
