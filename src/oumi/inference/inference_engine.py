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

"""Unified inference engine with simplified API."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import override

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.params.base_params import BaseParams
from oumi.core.inference import BaseInferenceEngine

if TYPE_CHECKING:
    from oumi.core.types.conversation import Conversation


# =============================================================================
# Helper functions
# =============================================================================

T = TypeVar("T", bound=BaseParams)


def _merge_params(
    config_class: type[T],
    config_obj: T | None,
    flat_overrides: dict[str, Any],
) -> T:
    """Merge flat params over config object, using dataclass defaults for missing.

    Priority order: flat_overrides > config_obj > dataclass defaults

    Args:
        config_class: The dataclass type to create (e.g., GenerationParams)
        config_obj: Optional existing config object to use as base
        flat_overrides: Dict of flat param names to values (None values ignored)

    Returns:
        New instance of config_class with merged values
    """
    kwargs: dict[str, Any] = {}

    for field in fields(config_class):
        field_name = field.name
        flat_value = flat_overrides.get(field_name)

        if flat_value is not None:
            # Flat param provided and not None - use it
            kwargs[field_name] = flat_value
        elif config_obj is not None:
            # Use value from config object
            kwargs[field_name] = getattr(config_obj, field_name)
        # else: let dataclass use its default

    return config_class(**kwargs)


def _get_provider_map() -> dict[str, type[BaseInferenceEngine]]:
    """Build provider name -> engine class mapping from ENGINE_MAP.

    Derives the mapping from the canonical ENGINE_MAP to avoid duplication.
    Provider names are lowercased for case-insensitive matching.
    """
    # Import here to avoid circular imports
    from oumi.builders.inference_engines import ENGINE_MAP

    provider_map: dict[str, type[BaseInferenceEngine]] = {}

    for engine_type, engine_class in ENGINE_MAP.items():
        # Use enum name (lowercased) as the canonical provider name
        # e.g., InferenceEngineType.FIREWORKS -> "fireworks"
        provider_name = engine_type.name.lower()
        provider_map[provider_name] = engine_class

        # Also add by enum value if different from name
        # e.g., GOOGLE_GEMINI has value "GEMINI", so also add "gemini"
        if engine_type.value.lower() != provider_name:
            provider_map[engine_type.value.lower()] = engine_class

    return provider_map


def _is_remote_engine(engine_class: type[BaseInferenceEngine]) -> bool:
    """Check if an engine class is a remote engine (needs RemoteParams)."""
    # Import here to avoid circular imports
    from oumi.inference.remote_inference_engine import RemoteInferenceEngine

    return issubclass(engine_class, RemoteInferenceEngine)


def _resolve_provider(provider: str) -> tuple[str, type[BaseInferenceEngine]]:
    """Resolve provider name to canonical name and engine class.

    Args:
        provider: User-provided provider name (case-insensitive)

    Returns:
        Tuple of (canonical_provider_name, engine_class)

    Raises:
        ValueError: If provider is not recognized
    """
    provider_map = _get_provider_map()
    provider_lower = provider.lower()

    if provider_lower not in provider_map:
        supported = sorted(provider_map.keys())
        raise ValueError(
            f"Unknown provider: {provider!r}. Supported providers: {supported}"
        )

    return provider_lower, provider_map[provider_lower]


# =============================================================================
# Main class
# =============================================================================


class InferenceEngine(BaseInferenceEngine):
    """Unified inference engine with simplified API.

    This class provides a simplified interface for creating inference engines
    by specifying a provider name and model, with optional configuration through
    flat parameters or full config objects.

    The precedence for parameters is: flat params > config objects > defaults.

    Examples:
        Simple usage with flat parameters:

        >>> from oumi.inference import InferenceEngine
        >>> engine = InferenceEngine(
        ...     provider="fireworks",
        ...     model="accounts/fireworks/models/llama-v3-70b",
        ...     temperature=0.7,
        ... )
        >>> result = engine.infer(conversations)

        Using OpenRouter:

        >>> engine = InferenceEngine(
        ...     provider="openrouter",
        ...     model="meta-llama/llama-3-70b",
        ...     api_key="sk-or-...",
        ... )

        Advanced usage with config objects:

        >>> from oumi.core.configs import ModelParams, GenerationParams
        >>> engine = InferenceEngine(
        ...     provider="openai",
        ...     model_params=ModelParams(model_name="gpt-4"),
        ...     generation_params=GenerationParams(temperature=0.9, top_p=0.95),
        ... )

        Mixing flat params and config objects (flat params override):

        >>> engine = InferenceEngine(
        ...     provider="anthropic",
        ...     model_params=ModelParams(model_name="claude-3-opus-20240229"),
        ...     temperature=0.5,  # Overrides any temperature in generation_params
        ... )

    Attributes:
        provider: The canonical name of the inference provider.
        engine: The underlying provider-specific engine instance.
    """

    @classmethod
    def supported_providers(cls) -> list[str]:
        """Return list of supported provider names."""
        return sorted(_get_provider_map().keys())

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        *,
        # --- Curated generation params ---
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        top_p: float | None = None,
        stop_strings: list[str] | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        # --- Curated remote params ---
        api_key: str | None = None,
        api_url: str | None = None,
        num_workers: int | None = None,
        max_retries: int | None = None,
        # --- Curated model params ---
        trust_remote_code: bool | None = None,
        chat_template: str | None = None,
        # --- Full config objects (flat params override these) ---
        model_params: ModelParams | None = None,
        generation_params: GenerationParams | None = None,
        remote_params: RemoteParams | None = None,
    ):
        """Initialize the unified inference engine.

        Args:
            provider: The inference provider to use. Call `supported_providers()`
                for available options.
            model: The model name/identifier. Required if model_params is not provided.

            temperature: Controls randomness in output. Higher values (e.g., 1.0) make
                output more random, lower values (e.g., 0.2) more deterministic.
            max_new_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling threshold (0.0-1.0).
            stop_strings: List of sequences where generation stops.
            seed: Random seed for deterministic generation.
            frequency_penalty: Penalize tokens based on frequency in text so far.
            presence_penalty: Penalize tokens based on presence in text so far.

            api_key: API key for authentication. Overrides environment variable.
            api_url: Custom API endpoint URL.
            num_workers: Number of parallel workers for inference.
            max_retries: Maximum retry attempts on failure.

            trust_remote_code: Allow loading remote code from model repos.
            chat_template: Chat template name to use for formatting.

            model_params: Full ModelParams config object. Flat params override.
            generation_params: Full GenerationParams config. Flat params override.
            remote_params: Full RemoteParams config object. Flat params override.

        Raises:
            ValueError: If provider is not supported or if model is not specified.
        """
        # Resolve provider name and get engine class
        self._provider, self._engine_class = _resolve_provider(provider)

        # Collect flat overrides for each config type
        generation_overrides = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "stop_strings": stop_strings,
            "seed": seed,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        remote_overrides = {
            "api_key": api_key,
            "api_url": api_url,
            "num_workers": num_workers,
            "max_retries": max_retries,
        }

        model_overrides = {
            "trust_remote_code": trust_remote_code,
            "chat_template": chat_template,
        }

        # Build ModelParams
        if model is not None:
            model_overrides["model_name"] = model
        elif model_params is None or model_params.model_name is None:
            raise ValueError("Either 'model' or 'model_params' must be provided.")

        final_model_params = _merge_params(ModelParams, model_params, model_overrides)

        # Build GenerationParams
        final_generation_params = _merge_params(
            GenerationParams, generation_params, generation_overrides
        )

        # Build RemoteParams if needed
        final_remote_params: RemoteParams | None = None
        if _is_remote_engine(self._engine_class):
            final_remote_params = _merge_params(
                RemoteParams, remote_params, remote_overrides
            )

        # Create the underlying engine
        self._engine = self._create_engine(
            final_model_params,
            final_generation_params,
            final_remote_params,
        )

        # Initialize base class with the same params
        super().__init__(
            model_params=final_model_params,
            generation_params=final_generation_params,
        )

    @property
    def provider(self) -> str:
        """The canonical name of the inference provider."""
        return self._provider

    @property
    def engine(self) -> BaseInferenceEngine:
        """The underlying provider-specific engine instance."""
        return self._engine

    def _create_engine(
        self,
        model_params: ModelParams,
        generation_params: GenerationParams,
        remote_params: RemoteParams | None,
    ) -> BaseInferenceEngine:
        """Create the underlying provider-specific engine."""
        if _is_remote_engine(self._engine_class):
            return self._engine_class(
                model_params=model_params,
                generation_params=generation_params,
                remote_params=remote_params,
            )
        else:
            return self._engine_class(
                model_params=model_params,
                generation_params=generation_params,
            )

    @override
    def get_supported_params(self) -> set[str]:
        """Returns the supported generation parameters for the underlying engine."""
        return self._engine.get_supported_params()

    @override
    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Delegates inference to the underlying engine."""
        return self._engine._infer_online(input, inference_config)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying engine.

        This allows access to provider-specific methods like `infer_batch`,
        `get_batch_status`, etc.
        """
        # Avoid infinite recursion for attributes accessed during __init__
        if name in ("_engine", "_engine_class", "_provider"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self._engine, name)

    def __repr__(self) -> str:
        """Return a string representation of the engine."""
        return (
            f"InferenceEngine(provider={self._provider!r}, "
            f"model={self._model_params.model_name!r})"
        )
