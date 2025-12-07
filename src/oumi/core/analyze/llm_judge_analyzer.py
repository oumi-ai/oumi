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

"""LLM Judge analyzer for evaluating samples using an LLM."""

import json
import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("llm_judge")
class LLMJudgeAnalyzer(SampleAnalyzer):
    """Analyzer that uses an LLM to evaluate text samples.

    This analyzer enables flexible, custom evaluation of dataset samples using
    a language model as a judge. Users can specify custom prompts to evaluate
    samples for quality, safety, relevance, or any other criteria.

    The analyzer supports:
        - Custom evaluation prompts with {text} placeholder
        - Configurable inference settings (model, temperature, etc.)
        - Structured output parsing (JSON scores or text labels)
        - Batch processing for efficiency
        - Multiple evaluation criteria in a single pass

    Output metrics:
        - llm_judge_score: Numeric score from 0-10 (if structured output)
        - llm_judge_label: Text label/category assigned by the LLM
        - llm_judge_reasoning: Optional explanation from the LLM
        - llm_judge_raw_response: Full raw response from the LLM

    Example prompts:
        - Quality: "Rate the quality of this response from 0-10: {text}"
        - Safety: "Is this text safe? Reply YES or NO: {text}"
        - Relevance: "Does this answer the question correctly? {text}"

    Note:
        This analyzer requires an inference engine to be configured.
        It supports both local models and remote APIs (OpenAI, Anthropic, etc.).
    """

    # Default evaluation prompt
    DEFAULT_PROMPT = """Evaluate the following text for quality on a scale of 0-10.
Consider factors like:
- Coherence and clarity
- Relevance and accuracy
- Completeness
- Grammar and style

Text to evaluate:
{text}

Respond with a JSON object containing:
- "score": a number from 0-10
- "label": one of "excellent", "good", "average", "poor"
- "reasoning": a brief explanation (1-2 sentences)

JSON response:"""

    def __init__(
        self,
        *,
        prompt: Optional[str] = None,
        inference_config: Optional[dict[str, Any]] = None,
        batch_size: int = 10,
        max_text_length: int = 4000,
        parse_json_response: bool = True,
        score_field: str = "score",
        label_field: str = "label",
        reasoning_field: str = "reasoning",
        default_score: float = 5.0,
        default_label: str = "unknown",
        cache_responses: bool = True,
    ):
        """Initialize the LLMJudgeAnalyzer.

        Args:
            prompt: Custom evaluation prompt. Use {text} as placeholder for the
                sample text. If None, uses the default quality evaluation prompt.
            inference_config: Dictionary with inference configuration. Keys:
                - model_name: Name/path of the model to use
                - engine: Inference engine type ("remote", "vllm", "native", etc.)
                - api_base: Base URL for remote API (if using remote engine)
                - api_key: API key for remote API (can also use env variable)
                - temperature: Sampling temperature (default: 0.1)
                - max_tokens: Maximum tokens in response (default: 256)
                Additional keys are passed to the inference config.
            batch_size: Number of samples to process in each batch.
            max_text_length: Maximum character length of text to send to LLM.
                Longer texts are truncated with a note.
            parse_json_response: Whether to parse JSON from LLM response.
            score_field: Field name for numeric score in JSON response.
            label_field: Field name for label in JSON response.
            reasoning_field: Field name for reasoning in JSON response.
            default_score: Default score when parsing fails.
            default_label: Default label when parsing fails.
            cache_responses: Whether to cache LLM responses for identical texts.
        """
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.inference_config = inference_config or {}
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.parse_json_response = parse_json_response
        self.score_field = score_field
        self.label_field = label_field
        self.reasoning_field = reasoning_field
        self.default_score = default_score
        self.default_label = default_label
        self.cache_responses = cache_responses

        # Initialize inference engine lazily
        self._inference_engine = None
        self._inference_config_obj = None
        self._response_cache: dict[str, dict[str, Any]] = {}

        # Validate prompt has placeholder
        if "{text}" not in self.prompt:
            raise ValueError(
                "Prompt must contain {text} placeholder for sample text. "
                f"Got prompt: {self.prompt[:100]}..."
            )

    def _initialize_inference(self) -> None:
        """Initialize the inference engine from config."""
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

            # Build model params
            model_name = self.inference_config.get("model_name", "gpt-4o-mini")
            model_params = ModelParams(
                model_name=model_name,
                trust_remote_code=self.inference_config.get("trust_remote_code", False),
            )

            # Build generation params
            generation_params = GenerationParams(
                temperature=self.inference_config.get("temperature", 0.1),
                max_new_tokens=self.inference_config.get("max_tokens", 256),
                top_p=self.inference_config.get("top_p", 1.0),
            )

            # Determine engine type
            engine_str = self.inference_config.get("engine", "remote")
            engine_type = InferenceEngineType(engine_str.upper())

            # Build remote params if using remote engine
            remote_params = None
            if engine_type in (
                InferenceEngineType.REMOTE,
                InferenceEngineType.ANTHROPIC,
            ):
                remote_params = RemoteParams(
                    api_url=self.inference_config.get("api_base"),
                    api_key_env_varname=self.inference_config.get(
                        "api_key_env", "OPENAI_API_KEY"
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
                f"Initialized LLM Judge with model: {model_name}, engine: {engine_str}"
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

    def _format_prompt(self, text: str) -> str:
        """Format the evaluation prompt with the sample text.

        Args:
            text: Sample text to evaluate.

        Returns:
            Formatted prompt string.
        """
        truncated_text = self._truncate_text(text)
        return self.prompt.format(text=truncated_text)

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed dictionary with score, label, and reasoning.
        """
        result = {
            "score": self.default_score,
            "label": self.default_label,
            "reasoning": "",
            "raw_response": response,
        }

        if not self.parse_json_response:
            result["raw_response"] = response
            return result

        # Try to find JSON in response
        try:
            # Look for JSON object in response
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                # Extract fields
                if self.score_field in parsed:
                    score = parsed[self.score_field]
                    if isinstance(score, (int, float)):
                        result["score"] = float(score)
                    elif isinstance(score, str):
                        # Try to parse numeric string
                        try:
                            result["score"] = float(score)
                        except ValueError:
                            pass

                if self.label_field in parsed:
                    result["label"] = str(parsed[self.label_field])

                if self.reasoning_field in parsed:
                    result["reasoning"] = str(parsed[self.reasoning_field])

        except json.JSONDecodeError:
            # Try to extract score from text
            score_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", response)
            if score_match:
                result["score"] = float(score_match.group(1))

            # Try to extract simple yes/no or label
            response_lower = response.lower().strip()
            if response_lower.startswith("yes"):
                result["label"] = "yes"
                result["score"] = 10.0
            elif response_lower.startswith("no"):
                result["label"] = "no"
                result["score"] = 0.0

        return result

    def _evaluate_single(self, text: str) -> dict[str, Any]:
        """Evaluate a single text sample.

        Args:
            text: Text to evaluate.

        Returns:
            Dictionary with evaluation results.
        """
        # Check cache
        if self.cache_responses:
            cache_key = str(hash(text))
            if cache_key in self._response_cache:
                return self._response_cache[cache_key]

        # Initialize inference if needed
        self._initialize_inference()

        if self._inference_engine is None:
            return {
                "score": self.default_score,
                "label": self.default_label,
                "reasoning": "Inference engine not initialized",
                "raw_response": "",
            }

        # Build conversation
        from oumi.core.types.conversation import Conversation, Message, Role

        prompt = self._format_prompt(text)
        conversation = Conversation(
            messages=[Message(role=Role.USER, content=prompt)]
        )

        try:
            # Run inference
            results = self._inference_engine.infer(
                input=[conversation],
                inference_config=self._inference_config_obj,
            )

            if results and len(results) > 0:
                # Get assistant response
                result_conv = results[0]
                response = ""
                for msg in result_conv.messages:
                    if msg.role == Role.ASSISTANT:
                        if isinstance(msg.content, str):
                            response = msg.content
                        break

                parsed = self._parse_json_response(response)
            else:
                parsed = {
                    "score": self.default_score,
                    "label": self.default_label,
                    "reasoning": "No response from LLM",
                    "raw_response": "",
                }

        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            parsed = {
                "score": self.default_score,
                "label": self.default_label,
                "reasoning": f"Error: {str(e)}",
                "raw_response": "",
            }

        # Cache result
        if self.cache_responses:
            self._response_cache[str(hash(text))] = parsed

        return parsed

    def _evaluate_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Evaluate a batch of texts.

        Args:
            texts: List of texts to evaluate.

        Returns:
            List of evaluation results.
        """
        # Initialize inference if needed
        self._initialize_inference()

        if self._inference_engine is None:
            return [
                {
                    "score": self.default_score,
                    "label": self.default_label,
                    "reasoning": "Inference engine not initialized",
                    "raw_response": "",
                }
                for _ in texts
            ]

        from oumi.core.types.conversation import Conversation, Message, Role

        # Build conversations, checking cache
        conversations = []
        cached_results: dict[int, dict[str, Any]] = {}
        texts_to_evaluate: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            if self.cache_responses:
                cache_key = str(hash(text))
                if cache_key in self._response_cache:
                    cached_results[i] = self._response_cache[cache_key]
                    continue

            prompt = self._format_prompt(text)
            conversations.append(
                Conversation(messages=[Message(role=Role.USER, content=prompt)])
            )
            texts_to_evaluate.append((i, text))

        # Run inference on non-cached items
        results: list[dict[str, Any]] = []
        if conversations:
            try:
                inference_results = self._inference_engine.infer(
                    input=conversations,
                    inference_config=self._inference_config_obj,
                )

                for (orig_idx, text), result_conv in zip(
                    texts_to_evaluate, inference_results
                ):
                    response = ""
                    for msg in result_conv.messages:
                        if msg.role == Role.ASSISTANT:
                            if isinstance(msg.content, str):
                                response = msg.content
                            break

                    parsed = self._parse_json_response(response)
                    cached_results[orig_idx] = parsed

                    # Update cache
                    if self.cache_responses:
                        self._response_cache[str(hash(text))] = parsed

            except Exception as e:
                logger.warning(f"Batch LLM evaluation failed: {e}")
                for orig_idx, text in texts_to_evaluate:
                    cached_results[orig_idx] = {
                        "score": self.default_score,
                        "label": self.default_label,
                        "reasoning": f"Error: {str(e)}",
                        "raw_response": "",
                    }

        # Reconstruct results in original order
        for i in range(len(texts)):
            results.append(cached_results.get(i, {
                "score": self.default_score,
                "label": self.default_label,
                "reasoning": "Missing result",
                "raw_response": "",
            }))

        return results

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields using LLM evaluation.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added LLM judge columns.
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for LLM judge analysis. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df

        analyzer_id = getattr(self, "analyzer_id", "llm_judge")

        for column in text_columns:
            texts = df[column].astype(str).tolist()

            # Process in batches
            all_results = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_results = self._evaluate_batch(batch)
                all_results.extend(batch_results)

            # Add columns
            result_df[f"{column}_{analyzer_id}_score"] = [
                r["score"] for r in all_results
            ]
            result_df[f"{column}_{analyzer_id}_label"] = [
                r["label"] for r in all_results
            ]
            result_df[f"{column}_{analyzer_id}_reasoning"] = [
                r["reasoning"] for r in all_results
            ]
            result_df[f"{column}_{analyzer_id}_raw_response"] = [
                r["raw_response"] for r in all_results
            ]

        return result_df
