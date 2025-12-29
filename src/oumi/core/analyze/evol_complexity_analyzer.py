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

"""Evol Complexity analyzer for measuring instruction complexity.

This analyzer implements the DEITA Evol Complexity scoring approach, which
measures instruction complexity through comparative ranking of evolved variants.

The key insight is that comparative ranking produces better score discrimination
than direct LLM scoring (e.g., "rate complexity 1-10"). By generating increasingly
complex variants and ranking the original among them, we get fine-grained
complexity scores.

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


@register_sample_analyzer("evol_complexity")
class EvolComplexityAnalyzer(EvolBaseAnalyzer):
    """Analyzer that scores instruction complexity via evolved variant ranking.

    This analyzer measures how complex/sophisticated an instruction is by:
    1. Generating N progressively more complex versions of the instruction
    2. Asking an LLM to rank the original among the evolved variants
    3. Using the rank position as a complexity score

    A score of 0 means the original is the simplest (least complex) among variants.
    A score of 1 means the original is already very complex (comparable to evolved).

    Key metrics:
    - evol_complexity_score: Normalized score 0-1 (higher = more complex)
    - evol_complexity_rank: Raw rank position (1 = simplest)
    - evol_complexity_headroom: How much more complex it could be evolved

    This differs from simple heuristics (like instruction length) by capturing
    semantic complexity: multi-step reasoning, constraints, edge cases, etc.

    Note: This analyzer requires LLM inference (API or local model).
    """

    # Evolution operators for generating more complex instructions
    EVOLUTION_OPERATORS = [
        "add_constraints",
        "require_reasoning",
        "increase_depth",
        "add_edge_cases",
        "require_specificity",
        "add_domain_knowledge",
    ]

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
        evolution_operators: Optional[list[str]] = None,
        # Analysis scope
        analyze_role: str = "user",
        # Performance
        batch_size: int = 8,
        max_text_length: int = 4000,
        temperature: float = 0.7,
        max_retries: int = 2,
        cache_responses: bool = True,
        show_progress: bool = True,
    ):
        """Initialize the EvolComplexityAnalyzer.

        Args:
            model_type: Type of model to use: "api" or "local".
            api_provider: API provider: "openai" or "anthropic".
            api_model: Model name for API provider.
            local_model: Model name/path for local inference.
            inference_config: Additional inference configuration.
            num_evolutions: Number of evolved variants to generate (1-6).
            evolution_operators: List of operators for complexity evolution.
                Available: "add_constraints", "require_reasoning", "increase_depth",
                "add_edge_cases", "require_specificity", "add_domain_knowledge".
                If None, uses ["add_constraints", "require_reasoning", "increase_depth"].
            analyze_role: Which role's messages to analyze:
                - "user": Only analyze user messages/instructions (default)
                - "assistant": Only analyze assistant responses
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

        # Validate evolution operators
        if evolution_operators is not None:
            for op in evolution_operators:
                if op not in self.EVOLUTION_OPERATORS:
                    valid_ops = ", ".join(self.EVOLUTION_OPERATORS)
                    raise ValueError(
                        f"Invalid evolution operator: '{op}'. "
                        f"Available operators: {valid_ops}"
                    )
            self.evolution_operators = evolution_operators
        else:
            self.evolution_operators = [
                "add_constraints",
                "require_reasoning",
                "increase_depth",
            ]

        if analyze_role not in ("user", "assistant", "all"):
            raise ValueError(
                f"Invalid analyze_role: '{analyze_role}'. "
                "Must be 'user', 'assistant', or 'all'."
            )
        self.analyze_role = analyze_role

        # Store dataset-level metrics
        self._dataset_metrics: dict[str, Any] = {}

    def _get_evolution_prompt(self, text: str) -> str:
        """Get the prompt for generating more complex instruction variants.

        Args:
            text: Original instruction text.

        Returns:
            Prompt string for complexity evolution.
        """
        truncated = self._truncate_text(text)

        operators_desc = {
            "add_constraints": "Add constraints or requirements (e.g., 'must use only...', 'without using...')",
            "require_reasoning": "Require multi-step reasoning or explanation",
            "increase_depth": "Increase depth by asking for more detailed analysis",
            "add_edge_cases": "Add edge cases or exceptional scenarios to handle",
            "require_specificity": "Make the task more specific and precise",
            "add_domain_knowledge": "Require domain-specific knowledge or expertise",
        }

        selected_ops = "\n".join(
            f"- {operators_desc[op]}" for op in self.evolution_operators
        )

        return f"""Given this instruction/question:
"{truncated}"

Generate {self.num_evolutions} progressively MORE COMPLEX versions of this instruction.
Each version should be harder/more sophisticated than the previous while staying on the same topic.

Make versions more complex by:
{selected_ops}

Important:
- Each version must be a complete, standalone instruction
- Maintain the core topic/intent of the original
- Do NOT just make it longer - make it genuinely more complex
- Order from slightly more complex to much more complex

Return ONLY a JSON array of {self.num_evolutions} strings, nothing else:
["more complex version 1", "more complex version 2", ...]

JSON array:"""

    def _get_ranking_prompt(self, original: str, variants: list[str]) -> str:
        """Get the prompt for ranking original among complex variants.

        Args:
            original: Original instruction (labeled as "A").
            variants: List of evolved complex variants.

        Returns:
            Prompt string for complexity ranking.
        """
        truncated_original = self._truncate_text(original)

        # Build the list of options
        options = [f"A: {truncated_original}"]
        for i, variant in enumerate(variants):
            label = chr(ord("B") + i)
            truncated_variant = self._truncate_text(variant)
            options.append(f"{label}: {truncated_variant}")

        options_text = "\n\n".join(options)

        return f"""Rank these instructions from LEAST complex (1) to MOST complex ({len(variants) + 1}).

Consider complexity factors:
- Multi-step reasoning required
- Constraints and requirements
- Edge cases to handle
- Domain knowledge needed
- Specificity of the task

Instructions to rank:

{options_text}

Return ONLY a JSON object mapping each letter to its complexity rank (1 = simplest):
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

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze instruction complexity using evolved variant ranking.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added complexity analysis columns,
            generated column schema dict).
            - {col}_evol_complexity_score: Normalized score 0-1
            - {col}_evol_complexity_rank: Raw rank position
            - {col}_evol_complexity_headroom: Potential for more complexity
        """
        result_df = df.copy()
        generated_schema = {}

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for complexity analysis."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df, generated_schema

        analyzer_id = getattr(self, "analyzer_id", "evol_complexity")
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
                f"Computing complexity scores for {len(analyze_indices)} samples "
                f"in column '{column}'..."
            )

            # Initialize result arrays
            all_scores = [None] * n_samples
            all_ranks = [None] * n_samples
            all_headroom = [None] * n_samples

            # Process samples
            iterator = analyze_indices
            if self.show_progress:
                iterator = tqdm(
                    analyze_indices,
                    desc=f"Analyzing complexity ({column})",
                )

            for idx in iterator:
                text = all_texts[idx]

                try:
                    # Generate evolved variants
                    variants = self._generate_evolutions(text)

                    # Rank original among variants
                    rankings = self._rank_variants(text, variants)

                    # Original is always "A"
                    original_rank = rankings.get("A", 1)
                    total_items = len(variants) + 1

                    # Compute normalized score (higher rank = higher complexity)
                    score = self._compute_normalized_score(
                        original_rank, total_items, invert=False
                    )

                    # Headroom: how much more complex it could be
                    # If original ranks 1 (simplest), headroom = 1
                    # If original ranks highest, headroom = 0
                    headroom = 1.0 - score

                    all_scores[idx] = score
                    all_ranks[idx] = original_rank
                    all_headroom[idx] = headroom

                except Exception as e:
                    logger.warning(f"Failed to analyze sample {idx}: {e}")
                    all_scores[idx] = 0.5  # Default middle score
                    all_ranks[idx] = (self.num_evolutions + 2) // 2
                    all_headroom[idx] = 0.5

            # Add columns to result DataFrame
            result_df[f"{column}_{analyzer_id}_score"] = all_scores
            result_df[f"{column}_{analyzer_id}_rank"] = all_ranks
            result_df[f"{column}_{analyzer_id}_headroom"] = all_headroom

            # Compute dataset-level metrics
            valid_scores = [s for s in all_scores if s is not None]
            if valid_scores:
                import numpy as np

                self._dataset_metrics[column] = {
                    "total_analyzed": len(valid_scores),
                    "mean_complexity_score": round(float(np.mean(valid_scores)), 4),
                    "median_complexity_score": round(float(np.median(valid_scores)), 4),
                    "std_complexity_score": round(float(np.std(valid_scores)), 4),
                    "min_complexity_score": round(float(np.min(valid_scores)), 4),
                    "max_complexity_score": round(float(np.max(valid_scores)), 4),
                    "low_complexity_ratio": round(
                        sum(1 for s in valid_scores if s < 0.33) / len(valid_scores), 4
                    ),
                    "high_complexity_ratio": round(
                        sum(1 for s in valid_scores if s > 0.66) / len(valid_scores), 4
                    ),
                }

                logger.info(
                    f"Column '{column}': Mean complexity score = "
                    f"{self._dataset_metrics[column]['mean_complexity_score']:.3f}"
                )

        return result_df, generated_schema

    def compute_dataset_metrics(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Return dataset-level complexity metrics.

        Returns:
            Dictionary with aggregate complexity metrics per column.
        """
        return self._dataset_metrics.copy()
