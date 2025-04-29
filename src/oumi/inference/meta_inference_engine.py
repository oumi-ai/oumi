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

    Using fully qualified model names and CLI aliases:

    ```python
    # Using specific engines with fully qualified names
    vllm_response = engine.infer([conversation], model_name="vllm/llama3.1-8b")
    together_response = engine.infer([conversation], model_name="together/llama3.1-70b")

    # Using CLI aliases defined in oumi.cli.alias
    alias_response = engine.infer([conversation], model_name="claude-3-7-sonnet")
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
            model_name: The name of the model to use (can be fully qualified: "engine/model")
            **engine_kwargs: Additional configuration parameters for the engine

        Returns:
            An inference engine configured for the model
        """
        if model_name in self._engines:
            return self._engines[model_name]

        # Determine appropriate engine based on model name pattern
        engine_type = self._select_engine_type(model_name)

        # Extract the actual model name if using fully qualified format
        actual_model_name = model_name
        if "/" in model_name and not (
            model_name.startswith("meta-llama/")
            or model_name.startswith("huggingface/")
            or model_name.startswith("mistralai/")
        ):
            # Extract the actual model name part
            _, actual_model_name = model_name.split("/", 1)

        # Create model params
        model_params = ModelParams(model_name=actual_model_name)

        # Create remote params if needed
        remote_params = None
        if engine_type in [
            InferenceEngineType.ANTHROPIC,
            InferenceEngineType.OPENAI,
            InferenceEngineType.GOOGLE_GEMINI,
            InferenceEngineType.GOOGLE_VERTEX,
            InferenceEngineType.TOGETHER,
            InferenceEngineType.REMOTE,
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

        Supports three formats:
        1. Fully qualified name: "engine_type/model_name" (e.g., "vllm/llama3.1-8b")
        2. CLI aliases: Names defined in oumi/cli/alias.py (e.g., "llama4-scout-instruct")
        3. Plain model names: Automatically selects based on model name pattern

        Args:
            model_name: The name of the model

        Returns:
            The most appropriate inference engine type for the model
        """
        # Check for fully qualified name in format "engine_type/model_name"
        if "/" in model_name and not (
            model_name.startswith("meta-llama/")
            or model_name.startswith("huggingface/")
            or model_name.startswith("mistralai/")
        ):
            engine_part, _ = model_name.split("/", 1)
            engine_part = engine_part.upper()

            # Try to match with InferenceEngineType
            try:
                return InferenceEngineType(engine_part)
            except ValueError:
                # If not a direct match, try some common mappings
                engine_mappings = {
                    "OPENAI": InferenceEngineType.OPENAI,
                    "ANTHROPIC": InferenceEngineType.ANTHROPIC,
                    "CLAUDE": InferenceEngineType.ANTHROPIC,
                    "GEMINI": InferenceEngineType.GOOGLE_GEMINI,
                    "GOOGLE": InferenceEngineType.GOOGLE_GEMINI,
                    "TOGETHER": InferenceEngineType.TOGETHER,
                    "LOCAL": InferenceEngineType.NATIVE,
                    "NATIVE": InferenceEngineType.NATIVE,
                }
                if engine_part in engine_mappings:
                    return engine_mappings[engine_part]

        # Check for CLI aliases in oumi.cli.alias module
        from oumi.cli.alias import _ALIASES, AliasType

        if model_name in _ALIASES and AliasType.INFER in _ALIASES[model_name]:
            # We have an alias for inference, but we need to determine the engine type
            # Let's look at the string pattern in the YAML file
            yaml_path = _ALIASES[model_name][AliasType.INFER]

            # Map YAML paths to engine types
            if "apis/anthropic" in yaml_path:
                return InferenceEngineType.ANTHROPIC
            elif "apis/openai" in yaml_path:
                return InferenceEngineType.OPENAI
            elif "apis/gemini" in yaml_path:
                return InferenceEngineType.GOOGLE_GEMINI
            elif "apis/vertex" in yaml_path:
                return InferenceEngineType.GOOGLE_VERTEX

        # Fall back to pattern matching for the model name
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

        # Together models
        if "together" in model_name_lower:
            return InferenceEngineType.TOGETHER

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
