from dataclasses import dataclass, field

from lema.core.configs.base_config import BaseConfig
from lema.core.configs.generation_config import GenerationConfig
from lema.core.configs.params.model_params import ModelParams


@dataclass
class InferenceConfig(BaseConfig):
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
