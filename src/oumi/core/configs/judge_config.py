from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

from oumi.core.configs import BaseConfig, GenerationConfig, ModelParams


class JudgeAttributeValueType(str, Enum):
    """The type of the attribute."""

    BOOL = "bool"
    """The attribute is a boolean."""

    CATEGORICAL = "categorical"
    """The attribute is a categorical."""

    LIKERT_5 = "likert-5"
    """The attribute is a Likert scale."""


@dataclass
class JudgeConfig(BaseConfig):
    attributes: Dict[str, JudgeAttribute] = field(default_factory=dict)
    """The attributes to judge."""

    model: ModelParams = field(default_factory=ModelParams)
    """Configuration parameters for the model used in inference."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    """Configuration parameters for text generation during inference."""
