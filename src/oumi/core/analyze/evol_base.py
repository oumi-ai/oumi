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

"""Base class for evolution-based analyzers (DEITA-style scoring).

This module provides the shared infrastructure for analyzers that use the
Evol-Instruct approach: generating evolved variants of text and using
comparative ranking to compute quality/complexity scores.

Reference: "What Makes Good Data for Alignment?" (Liu et al., 2023)
https://arxiv.org/abs/2312.15685
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.utils.logging import logger


class EvolBaseAnalyzer(SampleAnalyzer, ABC):
    """Base class for evolution-based analyzers using comparative ranking.

    This class provides shared infrastructure for analyzers that:
    1. Generate evolved variants of input text (more complex/higher quality)
    2. Use comparative ranking to score the original against variants
    3. Normalize rankings to scores

    The key insight from the DEITA paper is that comparative ranking produces
    better score discrimination than direct LLM scoring (e.g., "rate 1-10").

    Subclasses implement:
    - _get_evolution_prompt(): Prompt to generate evolved variants
    - _get_ranking_prompt(): Prompt to rank original among variants
    - _process_evolution_result(): Process results into scores
    """

    def __init__(
        self,
        *,
        # Model configuration
        model_type: str = "api",
        api_provider: str = "anthropic",
        api_model: str = "claude-4-5-haiku",
        local_model: Optional[str] = None,
        inference_config: Optional[dict[str, Any]] = None,
        # Evolution configuration
        num_evolutions: int = 3,
        # Performance
        batch_size: int = 8,
        max_text_length: int = 4000,
        temperature: float = 0.7,
        max_retries: int = 2,
        cache_responses: bool = True,
        show_progress: bool = True,
    ):
        """Initialize the EvolBaseAnalyzer.

        Args:
            model_type: Type of model to use: "api" or "local".
                - "api": Use OpenAI/Anthropic API (requires API key)
                - "local": Use local model via Oumi inference
            api_provider: API provider when model_type="api":
                - "openai": OpenAI API (default)
                - "anthropic": Anthropic API
            api_model: Model name for API provider (e.g., "gpt-4o-mini", "claude-3-haiku").
            local_model: Model name/path for local inference when model_type="local"
                (e.g., "meta-llama/Llama-3-8B-Instruct").
            inference_config: Additional inference configuration options.
                Keys depend on engine type but commonly include:
                - engine: Inference engine type (e.g., "vllm", "native")
                - trust_remote_code: Whether to trust remote code
                - device_map: Device mapping for local models
            num_evolutions: Number of evolved variants to generate (1-6).
                More variants give better score discrimination but increase cost.
                Default is 3 (good balance of cost vs. precision).
            batch_size: Number of samples to process per batch.
            max_text_length: Maximum character length of text to process.
                Longer texts are truncated.
            temperature: Sampling temperature for generation (0.0-1.0).
                Higher values produce more diverse evolutions.
            max_retries: Maximum retries for failed LLM calls.
            cache_responses: Whether to cache LLM responses for identical inputs.
            show_progress: Whether to show progress during analysis.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate parameters
        if model_type not in ("api", "local"):
            raise ValueError(
                f"Invalid model_type: '{model_type}'. Must be 'api' or 'local'."
            )

        if model_type == "api" and api_provider not in ("openai", "anthropic"):
            raise ValueError(
                f"Invalid api_provider: '{api_provider}'. Must be 'openai' or 'anthropic'."
            )

        if model_type == "local" and not local_model:
            raise ValueError("local_model must be specified when model_type='local'.")

        if not 1 <= num_evolutions <= 6:
            raise ValueError(f"num_evolutions must be 1-6, got {num_evolutions}.")

        self.model_type = model_type
        self.api_provider = api_provider
        self.api_model = api_model
        self.local_model = local_model
        self.inference_config = inference_config or {}
        self.num_evolutions = num_evolutions
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_responses = cache_responses
        self.show_progress = show_progress

        # Lazy-initialized inference engine
        self._inference_engine = None
        self._inference_config_obj = None

        # Response cache
        self._response_cache: dict[str, Any] = {}

    def _initialize_inference(self) -> None:
        """Initialize the inference engine based on configuration."""
        if self._inference_engine is not None:
            return

        try:
            from oumi.builders.inference_engines import build_inference_engine
            from oumi.core.configs import (
                GenerationParams,
                InferenceConfig,
                InferenceEngineType,
                ModelParams,
                RemoteParams,
            )

            # Determine model name and engine type
            if self.model_type == "api":
                model_name = self.api_model
                if self.api_provider == "anthropic":
                    engine_type = InferenceEngineType.ANTHROPIC
                    api_key_env = "ANTHROPIC_API_KEY"
                else:  # openai
                    engine_type = InferenceEngineType.REMOTE
                    api_key_env = "OPENAI_API_KEY"
            else:  # local
                model_name = self.local_model
                engine_str = self.inference_config.get("engine", "native")
                engine_type = InferenceEngineType(engine_str.upper())
                api_key_env = None

            # Build model params
            model_params = ModelParams(
                model_name=model_name,
                trust_remote_code=self.inference_config.get("trust_remote_code", False),
            )

            # Build generation params
            generation_params = GenerationParams(
                temperature=self.temperature,
                max_new_tokens=self.inference_config.get("max_tokens", 1024),
                top_p=self.inference_config.get("top_p", 1.0),
            )

            # Build remote params for API models
            remote_params = None
            if self.model_type == "api":
                remote_params = RemoteParams(
                    api_url=self.inference_config.get("api_base"),
                    api_key_env_varname=self.inference_config.get(
                        "api_key_env", api_key_env
                    ),
                )

            # Build inference config
            self._inference_config_obj = InferenceConfig(
                model=model_params,
                generation=generation_params,
                engine=engine_type,
                remote_params=remote_params,
            )

            # Build inference engine
            self._inference_engine = build_inference_engine(
                engine_type=engine_type,
                model_params=model_params,
                remote_params=remote_params,
                generation_params=generation_params,
            )

            logger.info(
                f"Initialized Evol analyzer with model: {model_name}, "
                f"engine: {engine_type.value}"
            )

        except ImportError as e:
            raise ImportError(
                f"Failed to import inference components: {e}. "
                "Make sure oumi inference dependencies are installed."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize inference engine: {e}")

    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum length.

        Args:
            text: Input text.

        Returns:
            Truncated text with note if truncated.
        """
        if len(text) <= self.max_text_length:
            return text
        return text[: self.max_text_length] + "\n[... truncated ...]"

    def _call_llm(self, prompt: str, use_cache: bool = True) -> str:
        """Call the LLM with a prompt and get the response.

        Args:
            prompt: The prompt to send to the LLM.
            use_cache: Whether to use cached responses.

        Returns:
            The LLM's response text.
        """
        # Check cache
        if use_cache and self.cache_responses:
            cache_key = str(hash(prompt))
            if cache_key in self._response_cache:
                return self._response_cache[cache_key]

        # Initialize inference if needed
        self._initialize_inference()

        if self._inference_engine is None:
            raise RuntimeError("Inference engine not initialized")

        from oumi.core.types.conversation import Conversation, Message, Role

        conversation = Conversation(messages=[Message(role=Role.USER, content=prompt)])

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                results = self._inference_engine.infer(
                    input=[conversation],
                    inference_config=self._inference_config_obj,
                )

                if results and len(results) > 0:
                    result_conv = results[0]
                    for msg in result_conv.messages:
                        if msg.role == Role.ASSISTANT:
                            if isinstance(msg.content, str):
                                response = msg.content
                                # Cache the response
                                if use_cache and self.cache_responses:
                                    self._response_cache[str(hash(prompt))] = response
                                return response

                raise RuntimeError("No response from LLM")

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                    continue
                break

        raise RuntimeError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _parse_json_list(self, response: str) -> list[str]:
        """Parse a JSON list from LLM response.

        Args:
            response: LLM response text.

        Returns:
            List of strings from the JSON array.
        """
        # Try to find JSON array in response
        try:
            # Look for JSON array
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract numbered items
        items = []
        for line in response.split("\n"):
            line = line.strip()
            # Match numbered items like "1. ...", "1) ...", "- ..."
            match = re.match(r"^(?:\d+[\.\)]\s*|-\s*|•\s*)(.+)$", line)
            if match:
                items.append(match.group(1).strip())

        return items

    def _parse_json_dict(self, response: str) -> dict[str, Any]:
        """Parse a JSON dict from LLM response.

        Args:
            response: LLM response text.

        Returns:
            Dictionary parsed from JSON.
        """
        try:
            # Look for JSON object
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
        except json.JSONDecodeError:
            pass

        return {}

    def _generate_evolutions(self, text: str) -> list[str]:
        """Generate evolved variants of the input text.

        Args:
            text: Original text to evolve.

        Returns:
            List of evolved variants (does NOT include original).
        """
        prompt = self._get_evolution_prompt(text)
        response = self._call_llm(prompt)
        variants = self._parse_json_list(response)

        # Ensure we have the right number of evolutions
        if len(variants) < self.num_evolutions:
            logger.warning(
                f"Expected {self.num_evolutions} evolutions, got {len(variants)}. "
                "Padding with duplicates."
            )
            while len(variants) < self.num_evolutions:
                variants.append(variants[-1] if variants else text)

        return variants[: self.num_evolutions]

    def _rank_variants(self, original: str, variants: list[str]) -> dict[str, int]:
        """Rank the original text among evolved variants.

        Args:
            original: Original text.
            variants: List of evolved variants.

        Returns:
            Dictionary mapping labels (A, B, C, ...) to ranks (1 = lowest, N = highest).
            "A" is always the original.
        """
        prompt = self._get_ranking_prompt(original, variants)
        response = self._call_llm(prompt)
        rankings = self._parse_json_dict(response)

        # Ensure we have rankings for all items
        labels = ["A"] + [chr(ord("B") + i) for i in range(len(variants))]
        result = {}

        for label in labels:
            if label in rankings:
                try:
                    result[label] = int(rankings[label])
                except (ValueError, TypeError):
                    # Assign middle rank as fallback
                    result[label] = (len(labels) + 1) // 2
            else:
                # Assign middle rank as fallback
                result[label] = (len(labels) + 1) // 2

        return result

    def _compute_normalized_score(
        self, rank: int, total_items: int, invert: bool = False
    ) -> float:
        """Convert a rank to a normalized score between 0 and 1.

        Args:
            rank: Rank position (1 = lowest/simplest, N = highest/most complex).
            total_items: Total number of items being ranked.
            invert: If True, invert the score (lower rank = higher score).

        Returns:
            Normalized score between 0 and 1.
        """
        if total_items <= 1:
            return 0.5

        # Normalize rank to 0-1 range
        # rank 1 -> 0, rank N -> 1
        normalized = (rank - 1) / (total_items - 1)

        if invert:
            normalized = 1.0 - normalized

        return normalized

    @abstractmethod
    def _get_evolution_prompt(self, text: str) -> str:
        """Get the prompt for generating evolved variants.

        Args:
            text: Original text to evolve.

        Returns:
            Prompt string for evolution generation.
        """
        pass

    @abstractmethod
    def _get_ranking_prompt(self, original: str, variants: list[str]) -> str:
        """Get the prompt for ranking original among variants.

        Args:
            original: Original text (labeled as "A").
            variants: List of evolved variants (labeled B, C, ...).

        Returns:
            Prompt string for ranking.
        """
        pass
