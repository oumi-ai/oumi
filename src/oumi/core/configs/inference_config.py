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

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams


@dataclass
class InferenceConfig(BaseConfig):
    """Configuration for model inference.

    Example:
        Creating config for a local model::

            config = InferenceConfig.for_model("meta-llama/Llama-3.2-1B")

        Creating config with custom generation parameters::

            config = InferenceConfig.for_model(
                "gpt2",
                max_new_tokens=100,
                temperature=0.7,
            )

        Creating config for a remote API::

            config = InferenceConfig.for_model(
                "gpt-4",
                engine=InferenceEngineType.OPENAI,
                api_key="sk-...",
            )
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model used in inference."""

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during inference."""

    input_path: str | None = None
    """Path to the input file containing prompts for text generation.

    The input file should be in JSONL format, where each line is a JSON representation
    of an Oumi `Conversation` object.
    """

    output_path: str | None = None
    """Path to the output file where the generated text will be saved."""

    engine: InferenceEngineType | None = None
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

    remote_params: RemoteParams | None = None
    """Parameters for running inference against a remote API."""

    @classmethod
    def for_model(
        cls,
        model_name: str,
        *,
        engine: InferenceEngineType | None = None,
        torch_dtype: str = "auto",
        device_map: str | None = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 256,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "InferenceConfig":
        """Create an InferenceConfig for a specific model with sensible defaults.

        This is a convenience factory method for quickly setting up inference
        without manually constructing nested configuration objects.

        Args:
            model_name: Name or path of the model (HuggingFace model ID, local path,
                or remote model name depending on engine).
            engine: The inference engine to use. If None, defaults to NATIVE for
                local models.
            torch_dtype: Data type for model parameters ("auto", "float16", "bfloat16").
            device_map: Device placement strategy ("auto", None, or specific device).
            trust_remote_code: Whether to trust remote code when loading the model.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling parameter.
            api_key: API key for remote inference engines.
            base_url: Base URL for remote inference engines.
            **kwargs: Additional keyword arguments for advanced configuration.

        Returns:
            A configured InferenceConfig instance.

        Example:
            >>> config = InferenceConfig.for_model("gpt2", max_new_tokens=50)
            >>> config.model.model_name
            'gpt2'
            >>> config.generation.max_new_tokens
            50

            >>> config = InferenceConfig.for_model(
            ...     "gpt-4",
            ...     engine=InferenceEngineType.OPENAI,
            ...     api_key="sk-...",
            ... )
        """
        # Build model params
        model_kwargs = {
            k: v
            for k, v in kwargs.items()
            if hasattr(ModelParams, "__dataclass_fields__")
            and k in ModelParams.__dataclass_fields__
        }
        model_params = ModelParams(
            model_name=model_name,
            torch_dtype_str=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        # Build generation params
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        # Add any extra generation kwargs from kwargs
        for k, v in kwargs.items():
            if (
                hasattr(GenerationParams, "__dataclass_fields__")
                and k in GenerationParams.__dataclass_fields__
            ):
                gen_kwargs[k] = v

        generation_params = GenerationParams(**gen_kwargs)

        # Build remote params if needed
        remote_params = None
        if api_key is not None or base_url is not None:
            remote_params = RemoteParams(
                api_key=api_key,
                api_url=base_url,
            )

        return cls(
            model=model_params,
            generation=generation_params,
            engine=engine,
            remote_params=remote_params,
        )
