from dataclasses import dataclass, field

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.generation_config import GenerationConfig
from oumi.core.configs.params.model_params import ModelParams


@dataclass
class InferenceConfig(BaseConfig):
    model: ModelParams = field(default_factory=ModelParams)
    """Configuration parameters for the model used in inference."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    """Configuration parameters for text generation during inference."""
