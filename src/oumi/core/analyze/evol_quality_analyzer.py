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

"""Evol Quality analyzer for measuring response quality.

This analyzer implements the DEITA Evol Quality scoring approach, which
measures response quality through comparative ranking of evolved variants.

By generating improved versions of a response and ranking the original among
them, we get fine-grained quality scores that capture subtle quality differences
better than direct LLM scoring.

Reference: "What Makes Good Data for Alignment?" (Liu et al., 2023)
https://arxiv.org/abs/2312.15685
"""

from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.evol_base import EvolBaseAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("evol_quality")
class EvolQualityAnalyzer(EvolBaseAnalyzer):
    """Analyzer that scores response quality via evolved variant ranking.

    This analyzer measures how good/helpful a response is by:
    1. Generating N improved versions of the response
    2. Asking an LLM to rank the original among the improved variants
    3. Using the rank position as a quality score

    A score of 0 means the original is the lowest quality among variants.
    A score of 1 means the original is already high quality (comparable to improved).

    Key metrics:
    - evol_quality_score: Normalized score 0-1 (higher = better quality)
    - evol_quality_rank: Raw rank position (1 = lowest quality)
    - evol_quality_improvement_potential: How much better it could be

    This analyzer is particularly useful for:
    - Evaluating SFT dataset response quality
    - Identifying low-quality responses that need filtering
    - Understanding the quality distribution of a dataset

    Note: This analyzer requires LLM inference (API or local model).
    For best results, it should also have access to the instruction/question
    that the response is answering (via conversation context or instruction column).
    """

    # Quality aspects for response evolution
    QUALITY_ASPECTS = [
        "helpfulness",
        "depth",
        "accuracy",
        "structure",
        "clarity",
        "completeness",
    ]

    def __init__(
        self,
        *,
        # Model configuration
        model_type: str = "api",
        api_provider: str = "anthropic",
        api_model: str = "claude-3-5-haiku-20241022",
        local_model: Optional[str] = None,
        inference_config: Optional[dict[str, Any]] = None,
        # Evolution configuration
        num_evolutions: int = 3,
        quality_aspects: Optional[list[str]] = None,
        # Context configuration
        instruction_column: Optional[str] = None,
        use_conversation_context: bool = True,
        # Analysis scope
        analyze_role: str = "assistant",
        # Performance
        batch_size: int = 8,
        max_text_length: int = 4000,
        temperature: float = 0.7,
        max_retries: int = 2,
        cache_responses: bool = True,
        show_progress: bool = True,
    ):
        """Initialize the EvolQualityAnalyzer.

        Args:
            model_type: Type of model to use: "api" or "local".
            api_provider: API provider: "openai" or "anthropic".
            api_model: Model name for API provider.
            local_model: Model name/path for local inference.
            inference_config: Additional inference configuration.
            num_evolutions: Number of evolved variants to generate (1-6).
            quality_aspects: List of aspects for quality evolution.
                Available: "helpfulness", "depth", "accuracy", "structure",
                "clarity", "completeness".
                If None, uses ["helpfulness", "depth", "accuracy", "structure"].
            instruction_column: Name of the column containing the instruction
                that the response is answering. If provided, this context
                improves evolution and ranking quality.
            use_conversation_context: If True, attempts to find instruction
                from previous messages in the conversation. Only works when
                analyzing assistant messages with a preceding user message.
            analyze_role: Which role's messages to analyze:
                - "assistant": Only analyze assistant responses (default)
                - "user": Only analyze user messages
                - "all": Analyze all text columns
            batch_size: Number of samples to process per batch.
            max_text_length: Maximum character length of text to process.
            temperature: Sampling temperature for generation.
            max_retries: Maximum retries for failed LLM calls.
            cache_responses: Whether to cache LLM responses.
            show_progress: Whether to show progress during analysis.
        """
        super().__init__(
            model_type=model_type,
            api_provider=api_provider,
            api_model=api_model,
            local_model=local_model,
            inference_config=inference_config,
            num_evolutions=num_evolutions,
            batch_size=batch_size,
            max_text_length=max_text_length,
            temperature=temperature,
            max_retries=max_retries,
            cache_responses=cache_responses,
            show_progress=show_progress,
        )

        # Validate quality aspects
        if quality_aspects is not None:
            for aspect in quality_aspects:
                if aspect not in self.QUALITY_ASPECTS:
                    valid_aspects = ", ".join(self.QUALITY_ASPECTS)
                    raise ValueError(
                        f"Invalid quality aspect: '{aspect}'. "
                        f"Available aspects: {valid_aspects}"
                    )
            self.quality_aspects = quality_aspects
        else:
            self.quality_aspects = ["helpfulness", "depth", "accuracy", "structure"]

        self.instruction_column = instruction_column
        self.use_conversation_context = use_conversation_context

        if analyze_role not in ("user", "assistant", "all"):
            raise ValueError(
                f"Invalid analyze_role: '{analyze_role}'. "
                "Must be 'user', 'assistant', or 'all'."
            )
        self.analyze_role = analyze_role

        # Store current instruction context (set during analysis)
        self._current_instruction: Optional[str] = None

        # Store dataset-level metrics
        self._dataset_metrics: dict[str, Any] = {}

    def _get_evolution_prompt(self, text: str) -> str:
        """Get the prompt for generating improved response variants.

        Args:
            text: Original response text.

        Returns:
            Prompt string for quality evolution.
        """
        truncated = self._truncate_text(text)

        aspects_desc = {
            "helpfulness": "Make it more helpful and directly useful",
            "depth": "Add more depth, detail, and thoroughness",
            "accuracy": "Improve accuracy and correctness",
            "structure": "Better organize with clear structure",
            "clarity": "Make it clearer and easier to understand",
            "completeness": "Make it more complete, covering all aspects",
        }

        selected_aspects = "\n".join(
            f"- {aspects_desc[aspect]}" for aspect in self.quality_aspects
        )

        # Include instruction context if available
        instruction_context = ""
        if self._current_instruction:
            truncated_instruction = self._truncate_text(self._current_instruction)
            instruction_context = f"""This response is answering the following instruction/question:
"{truncated_instruction}"

"""

        return f"""{instruction_context}Given this response:
"{truncated}"

Generate {self.num_evolutions} progressively BETTER/HIGHER QUALITY versions of this response.
Each version should be an improvement over the previous.

Improve quality by:
{selected_aspects}

Important:
- Each version must be a complete, standalone response
- Maintain the same intent and topic as the original
- Focus on genuine quality improvements, not just length
- Order from slightly better to much better

Return ONLY a JSON array of {self.num_evolutions} strings, nothing else:
["improved version 1", "improved version 2", ...]

JSON array:"""

    def _get_ranking_prompt(self, original: str, variants: list[str]) -> str:
        """Get the prompt for ranking original among improved variants.

        Args:
            original: Original response (labeled as "A").
            variants: List of improved variants.

        Returns:
            Prompt string for quality ranking.
        """
        truncated_original = self._truncate_text(original)

        # Build the list of options
        options = [f"A: {truncated_original}"]
        for i, variant in enumerate(variants):
            label = chr(ord("B") + i)
            truncated_variant = self._truncate_text(variant)
            options.append(f"{label}: {truncated_variant}")

        options_text = "\n\n".join(options)

        # Include instruction context if available
        instruction_context = ""
        if self._current_instruction:
            truncated_instruction = self._truncate_text(self._current_instruction)
            instruction_context = f"""These responses are answering:
"{truncated_instruction}"

"""

        return f"""{instruction_context}Rank these responses from LOWEST quality (1) to HIGHEST quality ({len(variants) + 1}).

Consider quality factors:
- Helpfulness and relevance to the task
- Accuracy and correctness
- Completeness and depth
- Clarity and structure
- Overall usefulness

Responses to rank:

{options_text}

Return ONLY a JSON object mapping each letter to its quality rank (1 = lowest quality):
Example: {{"A": 2, "B": 1, "C": 3}}

JSON ranking:"""

    def _find_role_column(
        self, df: pd.DataFrame, schema: Optional[dict]
    ) -> Optional[str]:
        """Find the role column in the DataFrame."""
        if not schema:
            return None

        for col, config in schema.items():
            if (
                config.get("content_type") == ContentType.CATEGORICAL
                and col in df.columns
                and "role" in col.lower()
            ):
                return col
        return None

    def _get_instruction_for_response(
        self,
        df: pd.DataFrame,
        idx: int,
        text_column: str,
        role_column: Optional[str],
    ) -> Optional[str]:
        """Get the instruction/question that a response is answering.

        Args:
            df: The DataFrame.
            idx: Index of the response row.
            text_column: Name of the text column.
            role_column: Name of the role column, if available.

        Returns:
            The instruction text, or None if not found.
        """
        # Check explicit instruction column first
        if self.instruction_column and self.instruction_column in df.columns:
            return str(df.iloc[idx][self.instruction_column])

        # Try to find instruction from conversation context
        if self.use_conversation_context and role_column:
            # Look for preceding user message in the conversation
            # This assumes conversation_index column exists for grouping
            conv_idx_col = None
            for col in df.columns:
                if "conversation" in col.lower() and "index" in col.lower():
                    conv_idx_col = col
                    break

            if conv_idx_col:
                current_conv = df.iloc[idx][conv_idx_col]
                conv_messages = df[df[conv_idx_col] == current_conv]

                # Find the user message before this assistant message
                current_msg_idx = conv_messages.index.get_loc(idx)
                if current_msg_idx > 0:
                    for prev_idx in range(current_msg_idx - 1, -1, -1):
                        prev_row = conv_messages.iloc[prev_idx]
                        if (
                            role_column in conv_messages.columns
                            and str(prev_row[role_column]).lower() == "user"
                        ):
                            return str(prev_row[text_column])

        return None

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze response quality using evolved variant ranking.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added quality analysis columns:
            - {col}_evol_quality_score: Normalized score 0-1
            - {col}_evol_quality_rank: Raw rank position
            - {col}_evol_quality_improvement_potential: How much better it could be
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for quality analysis."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df

        analyzer_id = getattr(self, "analyzer_id", "evol_quality")
        role_column = self._find_role_column(df, schema)

        for column in text_columns:
            all_texts = df[column].astype(str).tolist()
            n_samples = len(all_texts)

            # Determine which rows to analyze based on role
            if self.analyze_role == "all" or role_column is None:
                analyze_indices = list(range(n_samples))
            else:
                role_mask = df[role_column].str.lower() == self.analyze_role
                analyze_indices = [i for i in range(n_samples) if role_mask.iloc[i]]

            if len(analyze_indices) == 0:
                logger.warning(
                    f"No samples to analyze for role '{self.analyze_role}' "
                    f"in column '{column}'."
                )
                continue

            logger.info(
                f"Computing quality scores for {len(analyze_indices)} samples "
                f"in column '{column}'..."
            )

            # Initialize result arrays
            all_scores = [None] * n_samples
            all_ranks = [None] * n_samples
            all_improvement = [None] * n_samples

            # Process samples
            iterator = analyze_indices
            if self.show_progress:
                iterator = tqdm(
                    analyze_indices,
                    desc=f"Analyzing quality ({column})",
                )

            for idx in iterator:
                text = all_texts[idx]

                # Get instruction context for this response
                self._current_instruction = self._get_instruction_for_response(
                    df, idx, column, role_column
                )

                try:
                    # Generate improved variants
                    variants = self._generate_evolutions(text)

                    # Rank original among variants
                    rankings = self._rank_variants(text, variants)

                    # Original is always "A"
                    original_rank = rankings.get("A", 1)
                    total_items = len(variants) + 1

                    # Compute normalized score (higher rank = higher quality)
                    score = self._compute_normalized_score(
                        original_rank, total_items, invert=False
                    )

                    # Improvement potential: how much better it could be
                    # If original ranks 1 (lowest quality), potential = 1
                    # If original ranks highest, potential = 0
                    improvement_potential = 1.0 - score

                    all_scores[idx] = score
                    all_ranks[idx] = original_rank
                    all_improvement[idx] = improvement_potential

                except Exception as e:
                    logger.warning(f"Failed to analyze sample {idx}: {e}")
                    all_scores[idx] = 0.5  # Default middle score
                    all_ranks[idx] = (self.num_evolutions + 2) // 2
                    all_improvement[idx] = 0.5

            # Reset instruction context
            self._current_instruction = None

            # Add columns to result DataFrame
            result_df[f"{column}_{analyzer_id}_score"] = all_scores
            result_df[f"{column}_{analyzer_id}_rank"] = all_ranks
            result_df[f"{column}_{analyzer_id}_improvement_potential"] = all_improvement

            # Compute dataset-level metrics
            valid_scores = [s for s in all_scores if s is not None]
            if valid_scores:
                import numpy as np

                self._dataset_metrics[column] = {
                    "total_analyzed": len(valid_scores),
                    "mean_quality_score": round(float(np.mean(valid_scores)), 4),
                    "median_quality_score": round(float(np.median(valid_scores)), 4),
                    "std_quality_score": round(float(np.std(valid_scores)), 4),
                    "min_quality_score": round(float(np.min(valid_scores)), 4),
                    "max_quality_score": round(float(np.max(valid_scores)), 4),
                    "low_quality_ratio": round(
                        sum(1 for s in valid_scores if s < 0.33) / len(valid_scores), 4
                    ),
                    "high_quality_ratio": round(
                        sum(1 for s in valid_scores if s > 0.66) / len(valid_scores), 4
                    ),
                }

                logger.info(
                    f"Column '{column}': Mean quality score = "
                    f"{self._dataset_metrics[column]['mean_quality_score']:.3f}"
                )

        return result_df

    def compute_dataset_metrics(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Return dataset-level quality metrics.

        Returns:
            Dictionary with aggregate quality metrics per column.
        """
        return self._dataset_metrics.copy()
