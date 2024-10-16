from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams


class InferenceEngineType(str, Enum):
    """The supported inference engines."""

    NATIVE = "NATIVE"
    """The native inference engine using a local forward pass."""

    VLLM = "VLLM"
    """The vLLM inference engine."""

    LLAMACPP = "LLAMACPP"
    """The LlamaCPP inference engine."""

    REMOTE = "REMOTE"
    """The inference engine for APIs that implement the OpenAI Chat API interface."""

    ANTHROPIC = "ANTHROPIC"
    """The inference engine for Anthropic's API."""


@dataclass
class InferenceConfig(BaseConfig):
    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model used in inference."""

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during inference."""

    engine: Optional[InferenceEngineType] = None
    """The inference engine to use for generation.

    Options:

        - NATIVE: Use the native inference engine via a local forward pass.
        - VLLM: Use the vLLM inference engine.
        - LLAMACPP: Use LlamaCPP inference engine.
        - REMOTE: Use the inference engine for APIs that implement the OpenAI Chat API
          interface.
        - ANTHROPIC: Use the inference engine for Anthropic's API.

    If not specified, the "NATIVE" engine will be used.
    """
