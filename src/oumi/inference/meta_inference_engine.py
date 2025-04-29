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

"""A unified interface for multiple inference engines and models.

The MetaInferenceEngine provides a simplified interface to run inference across
different models and inference engines without needing to create separate
configurations for each model. It automatically selects the appropriate engine
based on the model name.

Examples:
    Basic usage with multiple models:

    ```python
    from oumi.inference import MetaInferenceEngine
    from oumi.core.types.conversation import Conversation, Message, Role

    # Create a simple conversation
    conversation = Conversation(messages=[
        Message(role=Role.USER, content="Explain quantum computing in simple terms.")
    ])

    # Initialize the engine with generation parameters
    engine = MetaInferenceEngine(temperature=0.7, max_tokens=1000)

    # Run inference with different models
    for model_name in ["gpt-4o", "claude-3-sonnet", "gemini-pro"]:
        response = engine.infer([conversation], model_name=model_name)
        print(f"\\n=== {model_name} response ===")
        print(response[0].messages[-1].content)
    ```

    With custom API keys:

    ```python
    # For models requiring API keys
    engine = MetaInferenceEngine(temperature=0.7)

    # For OpenAI
    response = engine.infer(
        [conversation],
        model_name="gpt-4",
        remote_params={"api_key": "your-openai-key"}
    )

    # For Anthropic
    response = engine.infer(
        [conversation],
        model_name="claude-3-opus",
        remote_params={"api_key": "your-anthropic-key"}
    )
    ```
"""

import re
from typing import Any, Optional

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
    RemoteParams,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation


class MetaInferenceEngine:
    """A unified interface for running inference with different models.

    This class provides a simplified interface to run inference across different
    models and inference engines without needing to create separate configurations
    for each model.

    Example:
        >>> conversations = [...]
        >>> engine = MetaInferenceEngine(temperature=0.7)
        >>> for model_name in ["gpt-4o", "gemini-pro", "claude-3-sonnet", "meta-llama/Llama-3-70b"]:
        >>>     response = engine.infer(conversations, model_name=model_name)
    """

    def __init__(self, **generation_kwargs: Any):
        """Initialize a MetaInferenceEngine with common generation parameters.

        Args:
            **generation_kwargs: Keyword arguments to configure generation parameters
                (e.g., temperature, max_new_tokens, top_p, etc.)
        """
        # Convert common API parameter names to Oumi parameters
        if "max_tokens" in generation_kwargs:
            generation_kwargs["max_new_tokens"] = generation_kwargs.pop("max_tokens")

        self.generation_params = GenerationParams(**generation_kwargs)
        self._engines: dict[str, BaseInferenceEngine] = {}

    def _get_engine_for_model(
        self, model_name: str, **engine_kwargs
    ) -> BaseInferenceEngine:
        """Get or create an inference engine for the specified model.

        Args:
            model_name: The name of the model to use
            **engine_kwargs: Additional configuration parameters for the engine

        Returns:
            An inference engine configured for the model
        """
        if model_name in self._engines:
            return self._engines[model_name]

        # Determine appropriate engine based on model name pattern
        engine_type = self._select_engine_type(model_name)

        # Create model params
        model_params = ModelParams(model_name=model_name)

        # Create remote params if needed
        remote_params = None
        if engine_type in [
            InferenceEngineType.ANTHROPIC,
            InferenceEngineType.OPENAI,
            InferenceEngineType.GOOGLE_GEMINI,
            InferenceEngineType.TOGETHER,
        ]:
            remote_params = RemoteParams(**engine_kwargs.get("remote_params", {}))

        # Build the engine
        from oumi.builders.inference_engines import build_inference_engine

        engine = build_inference_engine(
            engine_type=engine_type,
            model_params=model_params,
            remote_params=remote_params,
            generation_params=self.generation_params,
        )

        # Cache the engine
        self._engines[model_name] = engine
        return engine

    def _select_engine_type(self, model_name: str) -> InferenceEngineType:
        """Select the appropriate engine type based on the model name.

        Args:
            model_name: The name of the model

        Returns:
            The most appropriate inference engine type for the model
        """
        model_name_lower = model_name.lower()

        # OpenAI models
        if (
            model_name_lower.startswith("gpt")
            or "openai" in model_name_lower
            or model_name_lower.startswith("text-")
            or model_name_lower.startswith("o1-")
        ):
            return InferenceEngineType.OPENAI

        # Anthropic models
        if "claude" in model_name_lower:
            return InferenceEngineType.ANTHROPIC

        # Google models
        if "gemini" in model_name_lower:
            return InferenceEngineType.GOOGLE_GEMINI

        # LLaMA models - prefer VLLM for local inference
        if (
            "llama" in model_name_lower
            or "meta" in model_name_lower
            or re.search(r"mistral|mixtral|phi|qwen|falcon", model_name_lower)
        ):
            try:
                # Try VLLM first (for better performance)
                return InferenceEngineType.VLLM
            except ImportError:
                # Fall back to native inference
                return InferenceEngineType.NATIVE

        # Default to native inference
        return InferenceEngineType.NATIVE

    def infer(
        self,
        conversations: list[Conversation],
        model_name: str,
        inference_config: Optional[InferenceConfig] = None,
        **kwargs: Any,
    ) -> list[Conversation]:
        """Run inference on conversations using the specified model.

        Args:
            conversations: List of conversations to run inference on
            model_name: Name of the model to use
            inference_config: Optional inference configuration
                (will override other parameters)
            **kwargs: Additional parameters to pass to the inference engine

        Returns:
            List of conversations with model responses
        """
        if inference_config is None:
            # Get or create engine for this model
            engine = self._get_engine_for_model(model_name, **kwargs)
            # Run inference directly
            return engine.infer_online(conversations)
        else:
            # If a custom inference config is provided, override the model name
            config = inference_config.copy()
            config.model.model_name = model_name

            # Get or create engine for this model using the config
            engine_type = config.engine or self._select_engine_type(model_name)

            from oumi.builders.inference_engines import build_inference_engine

            engine = build_inference_engine(
                engine_type=engine_type,
                model_params=config.model,
                remote_params=config.remote_params,
                generation_params=config.generation or self.generation_params,
            )

            # Run inference with the config
            return engine.infer(conversations, config)
