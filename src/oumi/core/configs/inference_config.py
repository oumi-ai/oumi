from dataclasses import dataclass, field
from typing import Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams


@dataclass
class InferenceConfig(BaseConfig):
    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model used in inference."""

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during inference."""

    engine: Optional[str] = None
    """The inference engine to use for generation.

    Supported values:
    - "vllm"
    - "llamacpp"
    - "anthropic"
    - "remote"
    - "native"

    If not specified, the "native" engine will be used.
    """
