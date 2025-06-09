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

from oumi.core.configs import BaseConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams


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
class JudgeConfig(BaseConfig):
    """Configuration for the Judge.

    This class holds the configuration for a single-attribute judge,
    including the prompt template, response format, and model parameters.

    Examples:
        Basic boolean judgment:
        >>> judge_config = JudgeConfig( # doctest: +SKIP
        ...     prompt_template="Is the following answer helpful? Question: {question},
        ...                      Answer: {answer}. Respond with True or False.",
        ...     response_format=JudgeResponseFormat.XML,
        ...     judgment_type=JudgeOutputType.BOOL,
        ...     include_explanation=False
        ... )

        Categorical judgment with scores:
        >>> judge_config = JudgeConfig( # doctest: +SKIP
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

    response_format: JudgeResponseFormat = field(default=JudgeResponseFormat.XML)
    """The format in which the judge should respond."""

    include_explanation: bool = field(default=False)
    """Whether the judge should provide an explanation before the judgment."""

    judgment_type: JudgeOutputType = field(default=JudgeOutputType.BOOL)
    """The type of output that the judgment should be provided with."""

    judgment_scores: Optional[dict[str, float]] = field(default=None)
    """For ENUM judgment_type, the mapping from category names to numeric scores.

    Example:
        {"excellent": 1.0, "good": 0.7, "poor": 0.3}
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the underlying judge model used in inference."""

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during inference."""

    engine: InferenceEngineType = field(default=InferenceEngineType.NATIVE)
    """The inference engine to use for generation."""

    remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""

    def __post_init__(self):
        """Validate the configuration after initialization."""
        self._resolve_prompt_template()
        self._validate_config()

    def _resolve_prompt_template(self):
        """Resolve prompt_template from registry if it's a registry key."""
        from oumi.core.registry import REGISTRY

        # Check if prompt_template is a registry key (simple heuristic: no curly braces)
        if (
            isinstance(self.prompt_template, str)
            and self.prompt_template.strip()
            and "{" not in self.prompt_template
            and "}" not in self.prompt_template
        ):
            # Try to load from registry
            registered_template = REGISTRY.get_prompt_template(self.prompt_template)
            if registered_template is not None:
                # Replace with the registered template
                self.prompt_template = registered_template

    def _validate_config(self):
        """Validate the configuration for consistency and completeness.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate prompt template is not empty
        if not self.prompt_template.strip():
            raise ValueError("prompt_template cannot be empty")

        # Validate judgment scores are numeric if provided
        if self.judgment_scores:
            if not all(
                isinstance(score, (int, float))
                for score in self.judgment_scores.values()
            ):
                raise ValueError("All judgment_scores values must be numeric")
            if not self.judgment_scores:
                raise ValueError("judgment_scores cannot be empty when provided")
