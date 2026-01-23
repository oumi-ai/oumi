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
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.column_utils import make_analyzer_column_name
from oumi.core.analyze.sample_analyzer import DEFAULT_TEXT_COLUMNS, SampleAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("repr_diversity")
class ReprDiversityAnalyzer(SampleAnalyzer):
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
    Install with: pip install 'oumi[analyze_advanced]'
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
    ):
        """Initialize the ReprDiversityAnalyzer.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Default is "all-MiniLM-L6-v2" which is fast and effective.
                For higher quality embeddings, try "all-mpnet-base-v2".
            k_neighbors: Number of nearest neighbors to consider for diversity
                score computation. Default is 5. Higher values give smoother
                diversity estimates but are more computationally expensive.
            diversity_threshold: Threshold for flagging samples as redundant.
                Samples with diversity_score below this threshold are considered
                redundant (too similar to existing samples). Default 0.3 means
                samples with average neighbor distance < 0.3 are flagged.
                Value is in distance space (1 - cosine_similarity).
                Ignored if role_specific_thresholds is provided.
            role_specific_thresholds: Optional dict mapping role names to thresholds.
                If provided, different thresholds are applied to different roles.
                Use None as threshold value to exclude that role from diversity analysis.
                Example: {"system": None, "user": 0.25, "assistant": 0.3}
                This is useful for datasets where some roles (e.g., system prompts)
                are intentionally identical and should not be analyzed for diversity.
            embed_field: Which field(s) to embed for diversity analysis:
                - "all": Embed all text columns (default)
                - "user": Only embed user messages (requires role column)
                - "assistant": Only embed assistant responses (requires role column)
            batch_size: Batch size for embedding computation.
            device: Device for embedding model ("cuda", "cpu", or None for auto).
            show_progress_bar: Whether to show progress bars during computation.
            store_embeddings: Whether to store embeddings in output DataFrame.
                Note: This can significantly increase memory usage.

        Raises:
            ImportError: If sentence-transformers or scikit-learn is not installed.
            ValueError: If embed_field is not a valid option.
        """
        self._check_dependencies()

        valid_embed_fields = ("all", "user", "assistant")
        if embed_field not in valid_embed_fields:
            raise ValueError(
                f"Invalid embed_field: '{embed_field}'. "
                f"Must be one of: {valid_embed_fields}"
            )

        self.model_name = model_name
        self.k_neighbors = k_neighbors
        self.diversity_threshold = diversity_threshold
        self.role_specific_thresholds = role_specific_thresholds or {}
        self.embed_field = embed_field
        self.batch_size = batch_size
        self.device = device
        self.show_progress_bar = show_progress_bar
        self.store_embeddings = store_embeddings

        # Lazy-load the model
        self._model = None

        # Store dataset-level metrics
        self._dataset_metrics: dict[str, Any] = {}

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed.

        Raises:
            ImportError: If dependencies are not installed.
        """
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "ReprDiversityAnalyzer requires sentence-transformers. "
                "Install with: pip install 'oumi[analyze_advanced]'"
            )

        try:
            import sklearn  # noqa: F401
        except ImportError:
            raise ImportError(
                "ReprDiversityAnalyzer requires scikit-learn. "
                "Install with: pip install 'oumi[analyze_advanced]'"
            )

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers model.

        Returns:
            SentenceTransformer model instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        model = self._get_model()

        if not self.show_progress_bar:
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings

        # Use our own progress bar for better terminal compatibility
        all_embeddings = []

        with tqdm(total=len(texts), desc="Computing embeddings") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_embeddings.append(batch_embeddings)
                pbar.update(len(batch_texts))

        return np.vstack(all_embeddings)

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
        k = min(
            self.k_neighbors, n_samples - 1
        )  # Can't have more neighbors than samples

        if k <= 0:
            # Not enough samples for neighbor computation
            return np.zeros(n_samples), np.zeros(n_samples)

        logger.info(
            f"Computing nearest neighbor distances for {n_samples} samples (k={k})..."
        )

        # Compute cosine similarity matrix
        # For large datasets, we process in chunks to manage memory
        chunk_size = min(1000, n_samples)
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

    def _find_role_column(
        self, df: pd.DataFrame, schema: Optional[dict]
    ) -> Optional[str]:
        """Find the role column in the DataFrame using the schema.

        Args:
            df: Input DataFrame.
            schema: Column schema dict.

        Returns:
            Name of the role column if found, None otherwise.
        """
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
    ) -> pd.DataFrame:
        """Analyze text fields using embedding-based diversity scoring.

        This method computes diversity scores for each sample based on their
        nearest neighbor distances in embedding space.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added diversity analysis columns,
            generated column schema dict).
            - {col}_repr_diversity_nn_distance: Distance to nearest neighbor
            - {col}_repr_diversity_score: Mean distance to K nearest neighbors
            - {col}_repr_diversity_is_redundant: True if below threshold
            - {col}_repr_diversity_percentile: Percentile rank of diversity score
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for diversity analysis. "
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

        # Get analyzer ID for column naming
        analyzer_id = getattr(self, "analyzer_id", "repr_diversity")

        # Find role column for role-based filtering
        role_column = self._find_role_column(df, schema)

        for column in text_columns:
            # Get texts to analyze based on embed_field setting
            all_texts = df[column].astype(str).tolist()
            n_samples = len(all_texts)

            if self.embed_field == "all" or role_column is None:
                # Analyze all rows
                analyze_indices = list(range(n_samples))
                texts_to_embed = all_texts
            else:
                # Filter by role
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

            # Compute embeddings
            embeddings = self._compute_embeddings(texts_to_embed)

            # Compute diversity scores
            nn_distances, diversity_scores = self._compute_diversity_scores(embeddings)

            # Compute percentiles
            diversity_percentiles = np.zeros(len(diversity_scores))
            for i, score in enumerate(diversity_scores):
                diversity_percentiles[i] = (
                    np.sum(diversity_scores <= score) / len(diversity_scores) * 100
                )

            # Determine redundancy - apply role-specific thresholds if configured
            if self.role_specific_thresholds and role_column:
                # Role-specific threshold mode
                is_redundant = np.zeros(len(diversity_scores), dtype=bool)
                for local_idx, global_idx in enumerate(analyze_indices):
                    role = (
                        df.loc[global_idx, role_column].lower()
                        if isinstance(df.loc[global_idx, role_column], str)
                        else str(df.loc[global_idx, role_column]).lower()
                    )
                    threshold = self.role_specific_thresholds.get(
                        role,
                        self.diversity_threshold,  # Fall back to default
                    )
                    if threshold is not None:
                        is_redundant[local_idx] = (
                            diversity_scores[local_idx] < threshold
                        )
            else:
                # Single threshold mode (original behavior)
                is_redundant = diversity_scores < self.diversity_threshold

            # Create result arrays for all rows
            # Use np.nan for numerical columns to avoid negative near-zero values in CSV
            # Use None for boolean columns (proper null representation)
            all_nn_distances = [np.nan] * n_samples
            all_diversity_scores = [np.nan] * n_samples
            all_is_redundant = [None] * n_samples
            all_percentiles = [np.nan] * n_samples

            # Fill in values for analyzed rows
            for local_idx, global_idx in enumerate(analyze_indices):
                all_nn_distances[global_idx] = float(nn_distances[local_idx])
                all_diversity_scores[global_idx] = float(diversity_scores[local_idx])

                # Handle role-specific exclusion (threshold = None)
                if self.role_specific_thresholds and role_column:
                    role = (
                        df.loc[global_idx, role_column].lower()
                        if isinstance(df.loc[global_idx, role_column], str)
                        else str(df.loc[global_idx, role_column]).lower()
                    )
                    threshold = self.role_specific_thresholds.get(
                        role, self.diversity_threshold
                    )
                    if threshold is None:
                        # Exclude this role from redundancy analysis
                        all_is_redundant[global_idx] = None
                    else:
                        all_is_redundant[global_idx] = bool(is_redundant[local_idx])
                else:
                    all_is_redundant[global_idx] = bool(is_redundant[local_idx])

                all_percentiles[global_idx] = float(diversity_percentiles[local_idx])

            # Add columns to result DataFrame
            col_name = make_analyzer_column_name(column, analyzer_id, "nn_distance")
            result_df[col_name] = all_nn_distances

            col_name = make_analyzer_column_name(column, analyzer_id, "score")
            result_df[col_name] = all_diversity_scores

            col_name = make_analyzer_column_name(column, analyzer_id, "is_redundant")
            result_df[col_name] = all_is_redundant

            col_name = make_analyzer_column_name(column, analyzer_id, "percentile")
            result_df[col_name] = all_percentiles

            # Optionally store embeddings
            if self.store_embeddings:
                all_embeddings = [None] * n_samples
                for local_idx, global_idx in enumerate(analyze_indices):
                    all_embeddings[global_idx] = embeddings[local_idx].tolist()
                col_name = f"{column}_{analyzer_id}_embedding"
                result_df[col_name] = all_embeddings

            # Store dataset-level metrics
            redundant_count = int(np.sum(is_redundant))
            metrics = {
                "total_samples": len(texts_to_embed),
                "redundant_samples": redundant_count,
                "redundant_ratio": round(redundant_count / len(texts_to_embed), 4),
                "mean_diversity_score": round(float(np.mean(diversity_scores)), 4),
                "median_diversity_score": round(float(np.median(diversity_scores)), 4),
                "std_diversity_score": round(float(np.std(diversity_scores)), 4),
                "min_diversity_score": round(float(np.min(diversity_scores)), 4),
                "max_diversity_score": round(float(np.max(diversity_scores)), 4),
                "mean_nn_distance": round(float(np.mean(nn_distances)), 4),
                "diversity_threshold": self.diversity_threshold,
                "k_neighbors": self.k_neighbors,
            }

            # Add role-specific metrics if role_specific_thresholds is used
            if self.role_specific_thresholds and role_column:
                role_metrics = {}
                for role_name, threshold in self.role_specific_thresholds.items():
                    role_indices = [
                        local_idx
                        for local_idx, global_idx in enumerate(analyze_indices)
                        if (
                            df.loc[global_idx, role_column].lower()
                            if isinstance(df.loc[global_idx, role_column], str)
                            else str(df.loc[global_idx, role_column]).lower()
                        )
                        == role_name
                    ]
                    if role_indices and threshold is not None:
                        role_scores = diversity_scores[role_indices]
                        role_redundant = np.sum(role_scores < threshold)
                        role_metrics[role_name] = {
                            "threshold": threshold,
                            "total_samples": len(role_indices),
                            "redundant_samples": int(role_redundant),
                            "redundant_ratio": round(
                                float(role_redundant / len(role_indices)), 4
                            ),
                            "mean_score": round(float(np.mean(role_scores)), 4),
                            "median_score": round(float(np.median(role_scores)), 4),
                        }
                    elif threshold is None:
                        # Count excluded samples
                        excluded_count = len(
                            [
                                local_idx
                                for local_idx, global_idx in enumerate(analyze_indices)
                                if (
                                    df.loc[global_idx, role_column].lower()
                                    if isinstance(df.loc[global_idx, role_column], str)
                                    else str(df.loc[global_idx, role_column]).lower()
                                )
                                == role_name
                            ]
                        )
                        role_metrics[role_name] = {
                            "threshold": None,
                            "total_samples": excluded_count,
                            "excluded": True,
                        }
                metrics["role_specific_metrics"] = role_metrics

            # Add threshold appropriateness warning
            redundant_ratio = redundant_count / len(texts_to_embed)
            if redundant_ratio > 0.5:
                metrics["threshold_warning"] = (
                    f"High redundancy rate ({redundant_ratio * 100:.1f}%). "
                    f"Consider increasing threshold or using role-specific thresholds."
                )

            self._dataset_metrics[column] = metrics

            logger.info(
                f"Column '{column}': {redundant_count}/{len(texts_to_embed)} samples "
                f"({redundant_count / len(texts_to_embed) * 100:.1f}%) are redundant"
            )

        return result_df

    def compute_dataset_metrics(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Return dataset-level diversity metrics computed during analyze_sample.

        This method returns aggregate diversity metrics including:
        - total_samples: Number of samples analyzed
        - redundant_samples: Number of samples flagged as redundant
        - redundant_ratio: Fraction of samples that are redundant
        - mean_diversity_score: Average diversity score across samples
        - median_diversity_score: Median diversity score
        - std_diversity_score: Standard deviation of diversity scores
        - min/max_diversity_score: Range of diversity scores

        Args:
            df: Analyzed DataFrame (output of analyze_sample).
            schema: Column schema dict.

        Returns:
            Dictionary of dataset-level diversity metrics.
        """
        return self._dataset_metrics.copy()
