from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams


class InferenceEngineType(str, Enum):
    """The supported inference engines."""

    NATIVE = "NATIVE"
    """The native inference engine using a local forward pass."""

    VLLM = "VLLM"
    """The vLLM inference engine started locally by oumi using vLLM library."""

    REMOTE_VLLM = "REMOTE_VLLM"
    """The external vLLM inference engine."""

    SGLANG = "SGLANG"
    """The SGLang inference engine."""

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

    input_path: Optional[str] = None
    """Path to the input file containing prompts for text generation.

    The input file should be in JSONL format, where each line is a JSON representation
    of an Oumi `Conversation` object.
    """

    output_path: Optional[str] = None
    """Path to the output file where the generated text will be saved."""

    engine: Optional[InferenceEngineType] = None
    """The inference engine to use for generation.

    Options:

        - NATIVE: Use the native inference engine via a local forward pass.
        - VLLM: Use the vLLM inference engine started locally by oumi.
        - REMOTE_VLLM: Use the external vLLM inference engine.
        - SGLANG: Use the SGLang inference engine.
        - LLAMACPP: Use LlamaCPP inference engine.
        - REMOTE: Use the inference engine for APIs that implement the OpenAI Chat API
          interface.
        - ANTHROPIC: Use the inference engine for Anthropic's API.

    If not specified, the "NATIVE" engine will be used.
    """

    remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""
