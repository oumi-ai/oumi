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

"""Representation diversity analyzer using nearest-neighbor distance metrics.

This analyzer implements the DEITA-style diversity scoring approach, which measures
sample diversity using embedding-based nearest-neighbor distances. Unlike the
EmbeddingAnalyzer (which focuses on finding duplicates) or QuestionDiversityAnalyzer
(which measures cluster distribution), this analyzer computes a per-sample diversity
score indicating how unique each sample is relative to the dataset.

Reference: "What Makes Good Data for Alignment?" (Liu et al., 2023)
https://arxiv.org/abs/2312.15685

Example usage:
    >>> from oumi.core.analyze import ReprDiversityAnalyzer
    >>> analyzer = ReprDiversityAnalyzer(
    ...     k_neighbors=5,
    ...     diversity_threshold=0.3,
    ... )
    >>> result_df, schema = analyzer.analyze_sample(df, schema)
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.embedding_base_analyzer import EmbeddingBasedAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("repr_diversity")
class ReprDiversityAnalyzer(EmbeddingBasedAnalyzer):
    """Analyzer that computes diversity scores using nearest-neighbor distances.

    This analyzer measures how unique/diverse each sample is by computing the
    distance to its nearest neighbors in embedding space. Samples with larger
    distances to their neighbors are more unique and contribute more diversity.

    Key metrics:
    - nn_distance: Distance to the single nearest neighbor (1-cosine similarity)
    - diversity_score: Mean distance to K nearest neighbors
    - is_redundant: Boolean flag if diversity score is below threshold
    - diversity_percentile: Percentile rank of diversity score in the dataset

    This implements the "Repr Filter" approach from the DEITA paper, which uses
    nearest-neighbor distances to ensure selected samples are diverse.

    Note: This analyzer requires sentence-transformers and scikit-learn.
    Install with: pip install 'oumi[analyze]'

    Example:
        >>> analyzer = ReprDiversityAnalyzer(k_neighbors=5, diversity_threshold=0.3)
        >>> result_df, schema = analyzer.analyze_sample(
        ...     df,
        ...     schema={"content": {"content_type": "text"}}
        ... )
    """

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        k_neighbors: int = 5,
        diversity_threshold: float = 0.3,
        role_specific_thresholds: Optional[dict[str, Optional[float]]] = None,
        embed_field: str = "all",
        batch_size: int = 32,
        device: Optional[str] = None,
        show_progress_bar: bool = True,
        store_embeddings: bool = False,
        similarity_chunk_size: int = 1000,
    ):
        """Initialize the ReprDiversityAnalyzer.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Default is "all-MiniLM-L6-v2" which is fast and effective.
                For higher quality embeddings, try "all-mpnet-base-v2".
            k_neighbors: Number of nearest neighbors to consider for diversity
                score computation. Must be >= 1. Default is 5. Higher values
                give smoother diversity estimates but are more computationally
                expensive.
            diversity_threshold: Threshold for flagging samples as redundant.
                Samples with diversity_score below this threshold are considered
                redundant (too similar to existing samples). Must be in range
                [0.0, 1.0]. Default 0.3 means samples with average neighbor
                distance < 0.3 are flagged. Value is in distance space
                (1 - cosine_similarity). Ignored if role_specific_thresholds
                is provided.
            role_specific_thresholds: Optional dict mapping role names to thresholds.
                If provided, different thresholds are applied to different roles.
                Use None as threshold value to exclude that role from analysis.
                Example: {"system": None, "user": 0.25, "assistant": 0.3}
            embed_field: Which field(s) to embed for diversity analysis:
                - "all": Embed all text columns (default)
                - "user": Only embed user messages (requires role column)
                - "assistant": Only embed assistant responses (requires role column)
            batch_size: Batch size for embedding computation.
            device: Device for embedding model ("cuda", "cpu", or None for auto).
            show_progress_bar: Whether to show progress bars during computation.
            store_embeddings: Whether to store embeddings in output DataFrame.
                Note: This can significantly increase memory usage.
            similarity_chunk_size: Chunk size for similarity matrix computation.
                Larger values are faster but use more memory. Default 1000.

        Raises:
            ImportError: If sentence-transformers or scikit-learn is not installed.
            ValueError: If parameters are invalid.
        """
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            show_progress_bar=show_progress_bar,
        )

        # Validate parameters
        if k_neighbors < 1:
            raise ValueError(
                f"k_neighbors must be >= 1, got {k_neighbors}"
            )
        if not 0.0 <= diversity_threshold <= 1.0:
            raise ValueError(
                f"diversity_threshold must be in range [0.0, 1.0], "
                f"got {diversity_threshold}"
            )

        valid_embed_fields = ("all", "user", "assistant")
        if embed_field not in valid_embed_fields:
            raise ValueError(
                f"Invalid embed_field: '{embed_field}'. "
                f"Must be one of: {valid_embed_fields}"
            )

        # Check sklearn dependency
        self._check_sklearn()

        self.k_neighbors = k_neighbors
        self.diversity_threshold = diversity_threshold
        self.role_specific_thresholds = role_specific_thresholds or {}
        self.embed_field = embed_field
        self.store_embeddings = store_embeddings
        self.similarity_chunk_size = similarity_chunk_size

    def _compute_diversity_scores(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute diversity scores using nearest-neighbor distances.

        Args:
            embeddings: Array of embeddings with shape (n_samples, embedding_dim).

        Returns:
            Tuple of (nn_distances, diversity_scores):
            - nn_distances: Distance to single nearest neighbor (n_samples,)
            - diversity_scores: Mean distance to K nearest neighbors (n_samples,)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        n_samples = len(embeddings)
        k = min(self.k_neighbors, n_samples - 1)

        if k <= 0:
            return np.zeros(n_samples), np.zeros(n_samples)

        logger.info(
            f"Computing nearest neighbor distances for {n_samples} samples (k={k})..."
        )

        # Process in chunks to manage memory for large datasets
        chunk_size = min(self.similarity_chunk_size, n_samples)
        nn_distances = np.zeros(n_samples)
        diversity_scores = np.zeros(n_samples)

        for i in range(0, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)
            chunk_embeddings = embeddings[i:end_i]

            # Compute similarity of this chunk against all embeddings
            similarities = cosine_similarity(chunk_embeddings, embeddings)

            for local_idx in range(end_i - i):
                global_idx = i + local_idx
                sim_row = similarities[local_idx]

                # Set self-similarity to -inf so it's not selected as a neighbor
                sim_row[global_idx] = -np.inf

                # Convert to distances (1 - similarity)
                dist_row = 1 - sim_row

                # Find K nearest neighbors (smallest distances)
                # np.partition is O(n) vs O(n log n) for full sort
                k_smallest_indices = np.argpartition(dist_row, k)[:k]
                k_smallest_distances = dist_row[k_smallest_indices]

                # Nearest neighbor distance (single nearest)
                nn_distances[global_idx] = np.min(k_smallest_distances)

                # Diversity score (mean of K nearest)
                diversity_scores[global_idx] = np.mean(k_smallest_distances)

        return nn_distances, diversity_scores

    def _get_role_value(self, df: pd.DataFrame, idx: int, role_column: str) -> str:
        """Safely get the role value from a DataFrame row.

        Args:
            df: DataFrame containing the role column.
            idx: Row index.
            role_column: Name of the role column.

        Returns:
            Lowercase role value as string.
        """
        role_val = df.loc[idx, role_column]
        if isinstance(role_val, str):
            return role_val.lower()
        return str(role_val).lower()

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields using embedding-based diversity scoring.

        This method computes diversity scores for each sample based on their
        nearest neighbor distances in embedding space.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (result DataFrame, generated schema dict).
            The result DataFrame contains the original columns plus:
            - {col}_repr_diversity_nn_distance: Distance to nearest neighbor
            - {col}_repr_diversity_score: Mean distance to K nearest neighbors
            - {col}_repr_diversity_is_redundant: True if below threshold
            - {col}_repr_diversity_percentile: Percentile rank of diversity score
        """
        self._validate_schema_required(schema)
        result_df = df.copy()
        generated_schema: dict = {}

        text_columns = self._find_text_columns(df, schema)
        if not text_columns:
            return result_df, generated_schema

        analyzer_id = getattr(self, "analyzer_id", "repr_diversity")
        role_column = self._find_role_column(df, schema)

        for column in text_columns:
            all_texts = df[column].astype(str).tolist()
            n_samples = len(all_texts)

            if self.embed_field == "all" or role_column is None:
                analyze_indices = list(range(n_samples))
                texts_to_embed = all_texts
            else:
                role_mask = df[role_column].str.lower() == self.embed_field
                analyze_indices = df[role_mask].index.tolist()
                texts_to_embed = [all_texts[i] for i in analyze_indices]

            if len(texts_to_embed) < 2:
                logger.warning(
                    f"Not enough samples ({len(texts_to_embed)}) to analyze "
                    f"diversity for column '{column}'. Need at least 2 samples."
                )
                continue

            logger.info(
                f"Computing diversity scores for {len(texts_to_embed)} samples "
                f"in column '{column}'..."
            )

            embeddings = self._compute_embeddings(texts_to_embed)
            nn_distances, diversity_scores = self._compute_diversity_scores(embeddings)

            # Compute percentiles
            diversity_percentiles = np.zeros(len(diversity_scores))
            for i, score in enumerate(diversity_scores):
                diversity_percentiles[i] = (
                    np.sum(diversity_scores <= score) / len(diversity_scores) * 100
                )

            # Determine redundancy
            is_redundant = self._compute_redundancy(
                diversity_scores, analyze_indices, df, role_column
            )

            # Build result arrays and add to DataFrame
            result_df, col_schema = self._add_result_columns(
                result_df,
                nn_distances,
                diversity_scores,
                is_redundant,
                diversity_percentiles,
                embeddings,
                analyze_indices,
                n_samples,
                column,
                analyzer_id,
                df,
                role_column,
            )
            generated_schema.update(col_schema)

            # Compute and store dataset-level metrics
            self._compute_dataset_metrics(
                column, texts_to_embed, diversity_scores, nn_distances,
                is_redundant, analyze_indices, df, role_column
            )

        return result_df, generated_schema

    def _compute_redundancy(
        self,
        diversity_scores: np.ndarray,
        analyze_indices: list[int],
        df: pd.DataFrame,
        role_column: Optional[str],
    ) -> np.ndarray:
        """Compute redundancy flags based on thresholds."""
        if self.role_specific_thresholds and role_column:
            is_redundant = np.zeros(len(diversity_scores), dtype=bool)
            for local_idx, global_idx in enumerate(analyze_indices):
                role = self._get_role_value(df, global_idx, role_column)
                threshold = self.role_specific_thresholds.get(
                    role, self.diversity_threshold
                )
                if threshold is not None:
                    is_redundant[local_idx] = diversity_scores[local_idx] < threshold
        else:
            is_redundant = diversity_scores < self.diversity_threshold
        return is_redundant

    def _add_result_columns(
        self,
        result_df: pd.DataFrame,
        nn_distances: np.ndarray,
        diversity_scores: np.ndarray,
        is_redundant: np.ndarray,
        diversity_percentiles: np.ndarray,
        embeddings: np.ndarray,
        analyze_indices: list[int],
        n_samples: int,
        column: str,
        analyzer_id: str,
        original_df: pd.DataFrame,
        role_column: Optional[str],
    ) -> tuple[pd.DataFrame, dict]:
        """Add computed columns to the result DataFrame."""
        # Initialize arrays with NaN/None for unanalyzed rows
        all_nn_distances = [np.nan] * n_samples
        all_diversity_scores = [np.nan] * n_samples
        all_is_redundant: list[Optional[bool]] = [None] * n_samples
        all_percentiles = [np.nan] * n_samples

        for local_idx, global_idx in enumerate(analyze_indices):
            all_nn_distances[global_idx] = float(nn_distances[local_idx])
            all_diversity_scores[global_idx] = float(diversity_scores[local_idx])

            # Handle role-specific exclusion (threshold = None)
            if self.role_specific_thresholds and role_column:
                role = self._get_role_value(original_df, global_idx, role_column)
                threshold = self.role_specific_thresholds.get(
                    role, self.diversity_threshold
                )
                if threshold is None:
                    all_is_redundant[global_idx] = None
                else:
                    all_is_redundant[global_idx] = bool(is_redundant[local_idx])
            else:
                all_is_redundant[global_idx] = bool(is_redundant[local_idx])

            all_percentiles[global_idx] = float(diversity_percentiles[local_idx])

        # Column names
        nn_col = f"{column}_{analyzer_id}_nn_distance"
        score_col = f"{column}_{analyzer_id}_score"
        redundant_col = f"{column}_{analyzer_id}_is_redundant"
        percentile_col = f"{column}_{analyzer_id}_percentile"

        result_df[nn_col] = all_nn_distances
        result_df[score_col] = all_diversity_scores
        result_df[redundant_col] = all_is_redundant
        result_df[percentile_col] = all_percentiles

        generated_schema = {
            nn_col: {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": f"Distance to nearest neighbor for {column}",
            },
            score_col: {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": f"Mean K-NN diversity score for {column}",
            },
            redundant_col: {
                "type": ColumnType.BOOL,
                "content_type": ContentType.CATEGORICAL,
                "description": f"Whether {column} is redundant (low diversity)",
            },
            percentile_col: {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": f"Diversity percentile rank for {column}",
            },
        }

        # Optionally store embeddings
        if self.store_embeddings:
            emb_col = f"{column}_{analyzer_id}_embedding"
            all_embeddings: list[Optional[list]] = [None] * n_samples
            for local_idx, global_idx in enumerate(analyze_indices):
                all_embeddings[global_idx] = embeddings[local_idx].tolist()
            result_df[emb_col] = all_embeddings
            generated_schema[emb_col] = {
                "type": ColumnType.OBJECT,
                "content_type": ContentType.METADATA,
                "description": f"Embedding vector for {column}",
            }

        return result_df, generated_schema

    def _compute_dataset_metrics(
        self,
        column: str,
        texts_to_embed: list[str],
        diversity_scores: np.ndarray,
        nn_distances: np.ndarray,
        is_redundant: np.ndarray,
        analyze_indices: list[int],
        df: pd.DataFrame,
        role_column: Optional[str],
    ) -> None:
        """Compute and store dataset-level metrics."""
        redundant_count = int(np.sum(is_redundant))
        total_samples = len(texts_to_embed)

        metrics: dict[str, Any] = {
            "total_samples": total_samples,
            "redundant_samples": redundant_count,
            "redundant_ratio": round(redundant_count / total_samples, 4),
            "mean_diversity_score": round(float(np.mean(diversity_scores)), 4),
            "median_diversity_score": round(float(np.median(diversity_scores)), 4),
            "std_diversity_score": round(float(np.std(diversity_scores)), 4),
            "min_diversity_score": round(float(np.min(diversity_scores)), 4),
            "max_diversity_score": round(float(np.max(diversity_scores)), 4),
            "mean_nn_distance": round(float(np.mean(nn_distances)), 4),
            "diversity_threshold": self.diversity_threshold,
            "k_neighbors": self.k_neighbors,
        }

        # Add role-specific metrics if configured
        if self.role_specific_thresholds and role_column:
            role_metrics = self._compute_role_specific_metrics(
                diversity_scores, analyze_indices, df, role_column
            )
            metrics["role_specific_metrics"] = role_metrics

        # Add threshold warning if high redundancy
        redundant_ratio = redundant_count / total_samples
        if redundant_ratio > 0.5:
            metrics["threshold_warning"] = (
                f"High redundancy rate ({redundant_ratio*100:.1f}%). "
                f"Consider increasing threshold or using role-specific thresholds."
            )

        self._dataset_metrics[column] = metrics

        logger.info(
            f"Column '{column}': {redundant_count}/{total_samples} samples "
            f"({redundant_ratio*100:.1f}%) are redundant"
        )

    def _compute_role_specific_metrics(
        self,
        diversity_scores: np.ndarray,
        analyze_indices: list[int],
        df: pd.DataFrame,
        role_column: str,
    ) -> dict[str, Any]:
        """Compute metrics for each role."""
        role_metrics: dict[str, Any] = {}

        for role_name, threshold in self.role_specific_thresholds.items():
            role_indices = [
                local_idx
                for local_idx, global_idx in enumerate(analyze_indices)
                if self._get_role_value(df, global_idx, role_column) == role_name
            ]

            if role_indices and threshold is not None:
                role_scores = diversity_scores[role_indices]
                role_redundant = int(np.sum(role_scores < threshold))
                role_metrics[role_name] = {
                    "threshold": threshold,
                    "total_samples": len(role_indices),
                    "redundant_samples": role_redundant,
                    "redundant_ratio": round(role_redundant / len(role_indices), 4),
                    "mean_score": round(float(np.mean(role_scores)), 4),
                    "median_score": round(float(np.median(role_scores)), 4),
                }
            elif threshold is None:
                excluded_count = len([
                    1 for local_idx, global_idx in enumerate(analyze_indices)
                    if self._get_role_value(df, global_idx, role_column) == role_name
                ])
                role_metrics[role_name] = {
                    "threshold": None,
                    "total_samples": excluded_count,
                    "excluded": True,
                }

        return role_metrics
