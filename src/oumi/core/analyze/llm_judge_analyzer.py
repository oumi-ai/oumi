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

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.column_utils import make_analyzer_column_name
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

    requires_llm: bool = True
    requires_remote_llm: bool = True  # Uses remote API by default

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

    # Preset prompts for common evaluation scenarios
    PRESET_PROMPTS = {
        "instruction_quality": """Evaluate this instruction/question for clarity and training quality (0-10).

A high-quality instruction should:
- Have a clear, specific goal or question
- Use action verbs (write, explain, create, list, etc.)
- Include enough context to understand the request
- Avoid vague language ("something", "stuff", "whatever")
- Be appropriately detailed (not too short, not overly verbose)

Instruction to evaluate:
{text}

Respond with JSON:
- "score": 0-10 (10 = perfectly clear and actionable)
- "label": one of "excellent", "good", "needs_improvement", "poor"
- "reasoning": brief explanation

JSON response:""",
        "response_quality": """Evaluate this assistant response for training quality (0-10).

A high-quality response should:
- Directly address the user's request
- Be complete and not truncated
- Have appropriate structure (lists, code blocks when needed)
- Provide accurate and helpful information
- Be appropriately detailed for the question

Response to evaluate:
{text}

Respond with JSON:
- "score": 0-10 (10 = excellent training example)
- "label": one of "excellent", "good", "needs_improvement", "poor"
- "reasoning": brief explanation

JSON response:""",
        "conversation_coherence": """Evaluate this multi-turn conversation for coherence (0-10).

A coherent conversation should:
- Have logical flow between turns
- Reference previous context appropriately
- Show consistent topic threading
- Have natural turn-taking
- Avoid repetition or contradictions

Conversation to evaluate:
{text}

Respond with JSON:
- "score": 0-10 (10 = perfectly coherent)
- "label": one of "coherent", "mostly_coherent", "somewhat_disjointed", "incoherent"
- "reasoning": brief explanation

JSON response:""",
        "safety": """Evaluate this text for safety concerns (0-10, higher = safer).

Check for:
- Harmful or dangerous content
- Personal information exposure
- Inappropriate language
- Misleading or false claims
- Potential policy violations

Text to evaluate:
{text}

Respond with JSON:
- "score": 0-10 (10 = completely safe)
- "label": one of "safe", "caution", "concerning", "unsafe"
- "reasoning": brief explanation of any concerns

JSON response:""",
        "helpfulness": """Evaluate how helpful this response is to the user (0-10).

A helpful response should:
- Actually answer the user's question
- Provide actionable information
- Be easy to understand
- Include relevant details
- Not be unnecessarily verbose

Response to evaluate:
{text}

Respond with JSON:
- "score": 0-10 (10 = maximally helpful)
- "label": one of "very_helpful", "helpful", "somewhat_helpful", "not_helpful"
- "reasoning": brief explanation

JSON response:""",
        "factuality": """Evaluate this text for factual accuracy (0-10).

Consider:
- Are claims verifiable and accurate?
- Are there any obvious errors or misconceptions?
- Is hedging appropriate when uncertain?
- Are sources or reasoning provided when needed?

Text to evaluate:
{text}

Respond with JSON:
- "score": 0-10 (10 = completely factual)
- "label": one of "factual", "mostly_factual", "contains_errors", "unreliable"
- "reasoning": brief explanation of any issues

JSON response:""",
    }

    @classmethod
    def list_presets(cls) -> list[str]:
        """List available prompt presets.

        Returns:
            List of available preset names.
        """
        return list(cls.PRESET_PROMPTS.keys())

    # Role filtering defaults for presets
    # None means evaluate all messages regardless of role
    PRESET_ROLE_FILTERS = {
        "instruction_quality": "system",  # Only evaluate system instructions
        "response_quality": "assistant",  # Only evaluate assistant responses
        "conversation_coherence": None,  # Evaluate all messages
        "safety": None,  # Evaluate all messages for safety
        "helpfulness": "assistant",  # Only evaluate assistant responses
        "factuality": "assistant",  # Only evaluate assistant responses
    }

    # Which data levels to analyze for each preset
    # Format: (analyze_message_level, analyze_conversation_level)
    PRESET_ANALYSIS_LEVELS = {
        "instruction_quality": (True, False),  # Only message-level (system prompts)
        "response_quality": (True, False),  # Only message-level (assistant responses)
        "conversation_coherence": (False, True),  # Only conversation-level
        "safety": (True, True),  # Both levels (check all content)
        "helpfulness": (True, False),  # Only message-level (assistant responses)
        "factuality": (True, False),  # Only message-level (assistant responses)
    }

    @classmethod
    def get_preset_prompt(cls, preset_name: str) -> str:
        """Get a preset prompt by name.

        Args:
            preset_name: Name of the preset.

        Returns:
            The preset prompt string.

        Raises:
            ValueError: If preset name is not found.
        """
        if preset_name not in cls.PRESET_PROMPTS:
            available = ", ".join(cls.PRESET_PROMPTS.keys())
            raise ValueError(f"Unknown preset: '{preset_name}'. Available: {available}")
        return cls.PRESET_PROMPTS[preset_name]

    def __init__(
        self,
        *,
        prompt: Optional[str] = None,
        prompt_preset: Optional[str] = None,
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
        filter_role: Optional[str] = None,
        analyze_message_level: Optional[bool] = None,
        analyze_conversation_level: Optional[bool] = None,
    ):
        """Initialize the LLMJudgeAnalyzer.

        Args:
            prompt: Custom evaluation prompt. Use {text} as placeholder for the
                sample text. If None, uses the default quality evaluation prompt.
            prompt_preset: Use a preset prompt instead of custom. Available presets:
                - "instruction_quality": Evaluate instruction clarity for SFT
                - "response_quality": Evaluate assistant response quality
                - "conversation_coherence": Evaluate multi-turn conversation flow
                - "safety": Check for safety concerns
                - "helpfulness": Evaluate how helpful a response is
                - "factuality": Check for factual accuracy
                Takes precedence over `prompt` if both are provided.
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
            filter_role: Only evaluate messages from this role (e.g., "user", "assistant").
                If None, uses the preset's default role filter (if any), or evaluates
                all messages. Set to "none" to explicitly disable role filtering.
            analyze_message_level: Whether to analyze message-level data.
                If None, uses the preset's default. If False, skips message-level analysis.
            analyze_conversation_level: Whether to analyze conversation-level data.
                If None, uses the preset's default. If False, skips conversation-level
                analysis to save API calls.
        """
        # Handle preset prompts
        if prompt_preset is not None:
            if prompt_preset not in self.PRESET_PROMPTS:
                available = ", ".join(self.PRESET_PROMPTS.keys())
                raise ValueError(
                    f"Unknown prompt preset: '{prompt_preset}'. "
                    f"Available presets: {available}"
                )
            self.prompt = self.PRESET_PROMPTS[prompt_preset]
        else:
            self.prompt = prompt or self.DEFAULT_PROMPT

        self.prompt_preset = prompt_preset
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

        # Determine role filtering: explicit > preset default > None
        if filter_role == "none":
            self.filter_role = None  # Explicitly disable filtering
        elif filter_role is not None:
            self.filter_role = filter_role  # Explicit role specified
        elif prompt_preset and prompt_preset in self.PRESET_ROLE_FILTERS:
            self.filter_role = self.PRESET_ROLE_FILTERS[
                prompt_preset
            ]  # Use preset default
        else:
            self.filter_role = None  # No filtering

        # Determine analysis levels: explicit > preset default > (True, True)
        if prompt_preset and prompt_preset in self.PRESET_ANALYSIS_LEVELS:
            preset_msg_level, preset_conv_level = self.PRESET_ANALYSIS_LEVELS[
                prompt_preset
            ]
        else:
            preset_msg_level, preset_conv_level = (True, True)  # Default: analyze both

        # Apply explicit overrides if provided
        self.analyze_message_level = (
            analyze_message_level
            if analyze_message_level is not None
            else preset_msg_level
        )
        self.analyze_conversation_level = (
            analyze_conversation_level
            if analyze_conversation_level is not None
            else preset_conv_level
        )

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
                InferenceEngineType.OPENAI,
            ):
                # Extract nested remote_params config
                remote_params_config = self.inference_config.get("remote_params", {})
                remote_params_kwargs = {**remote_params_config}

                # Allow top-level overrides for backward compatibility
                if "api_base" in self.inference_config:
                    remote_params_kwargs["api_url"] = self.inference_config["api_base"]
                if "api_key_env" in self.inference_config:
                    remote_params_kwargs["api_key_env_varname"] = self.inference_config[
                        "api_key_env"
                    ]
                else:
                    # Set default if not specified
                    remote_params_kwargs.setdefault(
                        "api_key_env_varname", "OPENAI_API_KEY"
                    )

                remote_params = RemoteParams(**remote_params_kwargs)

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
        conversation = Conversation(messages=[Message(role=Role.USER, content=prompt)])

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

        # Build conversations, checking cache and deduplicating within batch
        conversations = []
        cached_results: dict[int, dict[str, Any]] = {}
        texts_to_evaluate: list[tuple[int, str]] = []
        text_to_indices: dict[
            str, list[int]
        ] = {}  # Map text to indices for deduplication

        for i, text in enumerate(texts):
            # Check global cache first
            if self.cache_responses:
                cache_key = str(hash(text))
                if cache_key in self._response_cache:
                    cached_results[i] = self._response_cache[cache_key]
                    logger.debug(f"Cache hit for text {i} (hash: {cache_key[:8]}...)")
                    continue

            # Check for duplicates within this batch
            if text in text_to_indices:
                # Duplicate within batch - will copy result later
                text_to_indices[text].append(i)
                logger.debug(f"Duplicate text at index {i}, will reuse result")
                continue

            # New unique text to evaluate
            text_to_indices[text] = [i]
            prompt = self._format_prompt(text)
            conversations.append(
                Conversation(messages=[Message(role=Role.USER, content=prompt)])
            )
            texts_to_evaluate.append((i, text))

        # Log cache/deduplication stats
        num_cached = len([i for i in range(len(texts)) if i in cached_results])
        num_unique = len(conversations)
        num_duplicates = len(texts) - num_cached - num_unique
        if num_cached > 0 or num_duplicates > 0:
            logger.info(
                f"Batch of {len(texts)}: {num_unique} unique to evaluate, "
                f"{num_duplicates} duplicates, {num_cached} from cache"
            )

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

                    # Copy result to all indices with this text (handles duplicates)
                    for idx in text_to_indices[text]:
                        cached_results[idx] = parsed

                    # Update cache for future batches
                    if self.cache_responses:
                        cache_key = str(hash(text))
                        self._response_cache[cache_key] = parsed
                        logger.debug(
                            f"Cached result for text (hash: {cache_key[:8]}...)"
                        )

            except Exception as e:
                logger.warning(f"Batch LLM evaluation failed: {e}")
                for orig_idx, text in texts_to_evaluate:
                    error_result = {
                        "score": self.default_score,
                        "label": self.default_label,
                        "reasoning": f"Error: {str(e)}",
                        "raw_response": "",
                    }
                    # Copy error result to all indices with this text
                    for idx in text_to_indices[text]:
                        cached_results[idx] = error_result

        # Reconstruct results in original order
        for i in range(len(texts)):
            results.append(
                cached_results.get(
                    i,
                    {
                        "score": self.default_score,
                        "label": self.default_label,
                        "reasoning": "Missing result",
                        "raw_response": "",
                    },
                )
            )

        return results

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields using LLM evaluation.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added LLM judge columns.
            generated column schema dict).
        """
        result_df = df.copy()
        generated_schema = {}

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for LLM judge analysis. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        # Check if this is conversation-level or message-level data
        has_role_column = any(
            "role" in col.lower() and col in df.columns for col in schema.keys()
        )
        is_message_level = has_role_column
        is_conversation_level = not has_role_column

        # Skip based on analysis level flags
        if is_message_level and not self.analyze_message_level:
            logger.info(
                f"Skipping message-level analysis (analyze_message_level=False). "
                f"Set analyze_message_level=True to enable."
            )
            return result_df, generated_schema

        if is_conversation_level and not self.analyze_conversation_level:
            logger.info(
                f"Skipping conversation-level analysis (analyze_conversation_level=False). "
                f"Set analyze_conversation_level=True to enable."
            )
            return result_df, generated_schema

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df, generated_schema

        analyzer_id = getattr(self, "analyzer_id", "llm_judge")

        # Find role column for filtering if needed
        role_column = None
        if self.filter_role:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and "role" in col.lower()
                    and col in df.columns
                ):
                    role_column = col
                    break

            if role_column is None:
                logger.warning(
                    f"Role filtering requested (filter_role='{self.filter_role}') "
                    f"but no role column found in schema. Evaluating all messages."
                )

        for column in text_columns:
            # Determine which rows to evaluate
            if self.filter_role and role_column:
                # Filter to only specified role
                role_mask = df[role_column].str.lower() == self.filter_role.lower()
                indices_to_evaluate = df[role_mask].index.tolist()
                logger.info(
                    f"Evaluating {len(indices_to_evaluate)} '{self.filter_role}' messages "
                    f"(filtered from {len(df)} total)"
                )
            else:
                # Evaluate all rows
                indices_to_evaluate = df.index.tolist()

            # Initialize results for all rows (None for filtered-out rows)
            all_results = [None] * len(df)

            if indices_to_evaluate:
                # Get texts for rows to evaluate
                texts_to_evaluate = [
                    str(df.loc[idx, column]) for idx in indices_to_evaluate
                ]

                # Process in batches
                evaluated_results = []
                for i in range(0, len(texts_to_evaluate), self.batch_size):
                    batch = texts_to_evaluate[i : i + self.batch_size]
                    batch_results = self._evaluate_batch(batch)
                    evaluated_results.extend(batch_results)

                # Place results in correct positions
                for idx, result in zip(indices_to_evaluate, evaluated_results):
                    all_results[idx] = result

            # Add columns with schema (None for filtered-out rows)
            col_name = make_analyzer_column_name(column, analyzer_id, "score")
            result_df[col_name] = [r["score"] if r else None for r in all_results]
            generated_schema[col_name] = {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": "LLM judge score (0-10, higher = better quality)",
            }

            col_name = make_analyzer_column_name(column, analyzer_id, "label")
            result_df[col_name] = [r["label"] if r else None for r in all_results]
            generated_schema[col_name] = {
                "type": ColumnType.STRING,
                "content_type": ContentType.CATEGORICAL,
                "description": "LLM judge label/category for the sample",
            }

            col_name = make_analyzer_column_name(column, analyzer_id, "reasoning")
            result_df[col_name] = [r["reasoning"] if r else None for r in all_results]
            generated_schema[col_name] = {
                "type": ColumnType.STRING,
                "content_type": ContentType.TEXT,
                "description": "LLM judge reasoning/explanation",
            }

            col_name = make_analyzer_column_name(column, analyzer_id, "raw_response")
            result_df[col_name] = [
                r["raw_response"] if r else None for r in all_results
            ]
            generated_schema[col_name] = {
                "type": ColumnType.STRING,
                "content_type": ContentType.TEXT,
                "description": "Raw LLM response before parsing",
            }

        return result_df, generated_schema
