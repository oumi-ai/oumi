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
    """Enumeration of possible response formats for judge outputs."""

    JSON = "json"
    """JSON structured response format."""

    XML = "xml"
    """XML-tagged response format."""

    RAW = "raw"
    """Plain text response format."""


class JudgeOutputType(str, Enum):
    """Enumeration of possible judgment types for judge evaluations."""

    TEXT = "text"
    """Free-form text judgment."""

    ENUM = "enum"
    """Categorical judgment from predefined options."""

    INT = "int"
    """Integer value judgment."""

    FLOAT = "float"
    """Floating-point value judgment."""

    BOOL = "bool"
    """Boolean judgment (True/False)."""


@dataclass
class JudgeConfig(BaseConfig):
    """Configuration for the Judge.

    This class holds the configuration for a single-attribute judge,
    including the prompt template, response format, and model parameters.

    Examples:
        >>> judge_config = JudgeConfig( # doctest: +SKIP
        ...     prompt_template="Is the following answer helpful? Question: {question}, Answer: {answer}. Respond with True or False.",
        ...     response_format=JudgeResponseFormat.XML,
        ...     judgment_type=JudgeOutputType.BOOL
        ... )
    """

    prompt_template: str
    """The template for the judge prompt with placeholders like {question}, {answer}, etc."""

    response_format: JudgeResponseFormat = field(default=JudgeResponseFormat.XML)
    """The format in which the judge should respond."""

    include_explanation: bool = field(default=False)
    """Whether the judge should provide an explanation before the judgment."""

    judgment_type: JudgeOutputType = field(default=JudgeOutputType.BOOL)
    """The type of judgment the judge should make."""

    judgment_scores: Optional[dict[str, float]] = field(default=None)
    """For ENUM judgment_type, mapping from category names to scores."""

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model used in inference."""

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during inference."""

    engine: InferenceEngineType = field(default=InferenceEngineType.NATIVE)
    """The inference engine to use for generation."""

    remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""
