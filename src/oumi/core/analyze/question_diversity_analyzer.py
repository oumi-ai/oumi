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

"""Question diversity analyzer for detecting narrow user question distributions.

This analyzer helps detect when a dataset has too many similar user questions,
or when the distribution of questions is too narrow compared to the potential
space of questions.

Note: This only analyzes user messages (role="user"), not system prompts or
assistant responses. System prompts are expected to be similar/identical.

Example usage:
    >>> from oumi.core.analyze import QuestionDiversityAnalyzer
    >>> analyzer = QuestionDiversityAnalyzer(
    ...     cluster_questions=True,
    ...     clustering_method="dbscan",
    ... )
    >>> result_df, schema = analyzer.analyze_sample(df, schema)
"""

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.embedding_base_analyzer import EmbeddingBasedAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("question_diversity")
class QuestionDiversityAnalyzer(EmbeddingBasedAnalyzer):
    """Analyzer for measuring user question diversity in datasets.

    This analyzer focuses specifically on user messages (questions) to detect:
    - Concentrated clusters of similar questions
    - Low diversity in question types
    - Narrow distribution compared to potential question space

    Important: This analyzer ONLY looks at user messages (role="user").
    System prompts and assistant responses are ignored. System prompts
    being identical is expected and fine.

    Key metrics:
    - question_entropy: Shannon entropy of cluster distribution (higher = more diverse)
    - question_gini: Gini coefficient (0 = uniform, 1 = concentrated)
    - largest_cluster_ratio: % of questions in the largest cluster
    - diversity_rating: "low" / "medium" / "high" based on entropy

    Unlike EmbeddingAnalyzer which finds duplicates, this analyzer focuses on
    the overall distribution and diversity of question types.

    Note: This analyzer requires sentence-transformers and scikit-learn.
    Install with: pip install 'oumi[analyze]'

    Example:
        >>> analyzer = QuestionDiversityAnalyzer(clustering_method="dbscan")
        >>> result_df, schema = analyzer.analyze_sample(
        ...     df,
        ...     schema={"content": {"content_type": "text"}, "role": {"content_type": "categorical"}}
        ... )
        >>> print(analyzer.compute_dataset_metrics(result_df))
    """

    def __init__(
        self,
        *,
        # Clustering configuration
        cluster_questions: bool = True,
        clustering_method: str = "dbscan",
        eps: float = 0.15,
        min_samples: int = 2,
        n_clusters: int = 10,
        # Embedding configuration
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
        # Distribution analysis
        compute_entropy: bool = True,
        compute_concentration: bool = True,
        flag_concentrated_clusters: bool = True,
        concentration_threshold: float = 0.5,
        # Progress display
        show_progress_bar: bool = True,
    ):
        """Initialize the QuestionDiversityAnalyzer.

        Args:
            cluster_questions: Whether to cluster questions for analysis.
            clustering_method: Clustering method: "dbscan" or "kmeans".
                DBSCAN automatically determines the number of clusters based on
                density, while k-means requires specifying n_clusters.
            eps: DBSCAN epsilon (neighborhood size). Smaller values create more
                clusters and require higher similarity to group. Default 0.15
                requires ~99% cosine similarity to link questions directly.
            min_samples: DBSCAN minimum samples to form a cluster.
            n_clusters: Number of clusters for k-means (ignored for DBSCAN).
            model_name: Sentence-transformers model for embeddings.
                "all-MiniLM-L6-v2" is fast; "all-mpnet-base-v2" is more accurate.
            batch_size: Batch size for embedding computation.
            device: Device for embedding model ("cuda", "cpu", or None for auto).
            compute_entropy: Whether to compute Shannon entropy of distribution.
            compute_concentration: Whether to compute Gini coefficient.
            flag_concentrated_clusters: Whether to flag samples in large clusters.
            concentration_threshold: Threshold for flagging concentrated clusters.
                A cluster with more than this fraction of samples is considered
                concentrated. Default 0.5 means >50% in one cluster triggers flag.
            show_progress_bar: Whether to show progress bars.

        Raises:
            ImportError: If required dependencies are not installed.
            ValueError: If clustering_method is invalid or concentration_threshold
                is out of range.
        """
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            show_progress_bar=show_progress_bar,
        )

        # Validate clustering method
        if clustering_method not in ("dbscan", "kmeans"):
            raise ValueError(
                f"Unknown clustering method: {clustering_method}. "
                "Supported methods: 'dbscan', 'kmeans'."
            )

        # Validate concentration threshold
        if not 0.0 <= concentration_threshold <= 1.0:
            raise ValueError(
                f"concentration_threshold must be in range [0.0, 1.0], "
                f"got {concentration_threshold}"
            )

        # Check sklearn dependency
        self._check_sklearn()

        self.cluster_questions = cluster_questions
        self.clustering_method = clustering_method
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self.compute_entropy = compute_entropy
        self.compute_concentration = compute_concentration
        self.flag_concentrated_clusters = flag_concentrated_clusters
        self.concentration_threshold = concentration_threshold

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster question embeddings.

        Args:
            embeddings: Array of embeddings.

        Returns:
            Array of cluster labels (-1 for noise in DBSCAN).
        """
        if self.clustering_method == "dbscan":
            from sklearn.cluster import DBSCAN

            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = clustering.fit_predict(embeddings)
        else:  # kmeans
            from sklearn.cluster import KMeans

            n_clusters_to_use = min(self.n_clusters, len(embeddings))

            # Suppress ConvergenceWarning when n_clusters >= n_samples
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Number of distinct clusters.*",
                    category=Warning,
                )
                clustering = KMeans(
                    n_clusters=n_clusters_to_use,
                    random_state=42,
                    n_init=10,
                )
                labels = clustering.fit_predict(embeddings)

        return labels

    def _compute_shannon_entropy(self, cluster_labels: np.ndarray) -> float:
        """Compute Shannon entropy of cluster distribution.

        Higher entropy = more diverse distribution.
        Maximum entropy occurs when all clusters have equal size.

        Args:
            cluster_labels: Array of cluster assignments.

        Returns:
            Shannon entropy value (non-negative, in bits).
        """
        # Count samples per cluster (excluding noise label -1 for DBSCAN)
        unique, counts = np.unique(cluster_labels, return_counts=True)

        # For DBSCAN, exclude noise points from entropy calculation
        mask = unique != -1
        if not np.any(mask):
            return 0.0

        counts = counts[mask]
        total = counts.sum()

        if total == 0:
            return 0.0

        # Compute probabilities
        probs = counts / total

        # Shannon entropy: H = -sum(p * log(p))
        # Use log2 for bits, add small epsilon to avoid log(0)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return float(entropy)

    def _compute_gini_coefficient(self, cluster_labels: np.ndarray) -> float:
        """Compute Gini coefficient of cluster distribution.

        Gini = 0 means perfectly uniform distribution.
        Gini = 1 means maximum inequality (all in one cluster).

        Args:
            cluster_labels: Array of cluster assignments.

        Returns:
            Gini coefficient between 0 and 1.
        """
        # Count samples per cluster (excluding noise label -1)
        unique, counts = np.unique(cluster_labels, return_counts=True)

        mask = unique != -1
        if not np.any(mask):
            return 1.0  # No valid clusters = maximally concentrated

        counts = counts[mask].astype(float)

        if len(counts) <= 1:
            return 1.0  # Single cluster = maximally concentrated

        # Sort counts
        counts = np.sort(counts)
        n = len(counts)

        # Gini coefficient formula
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * counts) - (n + 1) * np.sum(counts)) / (
            n * np.sum(counts)
        )

        return float(gini)

    def _compute_cluster_sizes(self, cluster_labels: np.ndarray) -> dict[int, int]:
        """Compute size of each cluster.

        Args:
            cluster_labels: Array of cluster assignments.

        Returns:
            Dict mapping cluster_id to count.
        """
        unique, counts = np.unique(cluster_labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def _get_diversity_rating(
        self, entropy: float, n_clusters: int, noise_ratio: float = 0.0
    ) -> str:
        """Get a qualitative diversity rating based on entropy and noise ratio.

        Args:
            entropy: Shannon entropy value.
            n_clusters: Number of clusters found (excluding noise).
            noise_ratio: Fraction of samples that are noise (unique/diverse).

        Returns:
            Rating string: "low", "medium", or "high".
        """
        # High noise ratio means high diversity (questions are unique)
        if noise_ratio > 0.7:
            return "high"
        elif noise_ratio > 0.4:
            return "medium"

        if n_clusters <= 1:
            return "low"

        # Maximum possible entropy for n clusters
        max_entropy = np.log2(n_clusters)

        if max_entropy == 0:
            return "low"

        # Normalized entropy (0 to 1)
        normalized = entropy / max_entropy

        if normalized < 0.5:
            return "low"
        elif normalized < 0.8:
            return "medium"
        else:
            return "high"

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze question diversity in the dataset.

        This analyzer focuses on user messages (questions) to measure
        how diverse the questions are across the dataset.

        Note: Only user messages (role="user") are analyzed. System prompts
        and assistant responses are ignored.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text and role fields.

        Returns:
            Tuple of (result DataFrame, generated schema dict).
            The result DataFrame contains the original columns plus:
            - {col}_question_diversity_cluster_id: Cluster assignment
            - {col}_question_diversity_cluster_size: Size of assigned cluster
            - {col}_question_diversity_is_concentrated: True if in large cluster
        """
        self._validate_schema_required(schema)
        result_df = df.copy()
        generated_schema: dict = {}

        text_columns = self._find_text_columns(df, schema)
        if not text_columns:
            return result_df, generated_schema

        # Find role column to filter for user messages
        role_column = self._find_role_column(df, schema)
        analyzer_id = getattr(self, "analyzer_id", "question_diversity")

        for column in text_columns:
            all_texts = df[column].astype(str).tolist()

            # Filter for user messages (questions) if role column exists
            if role_column is not None:
                user_mask = df[role_column].str.lower() == "user"
                question_indices = df[user_mask].index.tolist()
                questions = [all_texts[i] for i in range(len(df)) if user_mask.iloc[i]]
            else:
                # No role column - analyze all messages as questions
                question_indices = list(range(len(df)))
                questions = all_texts

            if len(questions) < 2:
                logger.warning(
                    f"Not enough questions ({len(questions)}) to analyze "
                    "diversity. Need at least 2 samples."
                )
                continue

            # Compute embeddings for questions only
            logger.info(f"Computing embeddings for {len(questions)} user questions...")
            embeddings = self._compute_embeddings(questions)

            # Cluster questions
            if self.cluster_questions:
                result_df, col_schema = self._process_clustering(
                    result_df,
                    embeddings,
                    questions,
                    question_indices,
                    column,
                    analyzer_id,
                    len(df),
                )
                generated_schema.update(col_schema)

        return result_df, generated_schema

    def _process_clustering(
        self,
        result_df: pd.DataFrame,
        embeddings: np.ndarray,
        questions: list[str],
        question_indices: list[int],
        column: str,
        analyzer_id: str,
        total_rows: int,
    ) -> tuple[pd.DataFrame, dict]:
        """Process clustering and compute diversity metrics."""
        logger.info(
            f"Clustering {len(questions)} questions using {self.clustering_method}..."
        )
        cluster_labels = self._cluster_embeddings(embeddings)
        cluster_sizes = self._compute_cluster_sizes(cluster_labels)

        # Log clustering results
        n_clusters = len([c for c in cluster_sizes.keys() if c != -1])
        noise_count = cluster_sizes.get(-1, 0)
        if noise_count > 0:
            logger.info(
                f"Found {n_clusters} clusters, {noise_count} unique/diverse "
                f"questions (not similar to others)"
            )
        else:
            logger.info(f"Found {n_clusters} clusters")

        # Create result arrays for all rows (None for non-user rows)
        all_cluster_ids = [None] * total_rows
        all_cluster_sizes = [None] * total_rows
        all_concentrated_flags = [None] * total_rows

        total_questions = len(questions)
        for idx, (orig_idx, cluster_id) in enumerate(
            zip(question_indices, cluster_labels)
        ):
            all_cluster_ids[orig_idx] = int(cluster_id)
            cluster_size = cluster_sizes.get(cluster_id, 0)
            all_cluster_sizes[orig_idx] = cluster_size

            # Flag if in concentrated cluster
            # Note: cluster_id == -1 means noise (not similar to others)
            # Noise points are NOT concentrated - they're diverse!
            if self.flag_concentrated_clusters:
                if cluster_id == -1:
                    all_concentrated_flags[orig_idx] = False
                else:
                    concentration_ratio = cluster_size / total_questions
                    all_concentrated_flags[orig_idx] = (
                        concentration_ratio > self.concentration_threshold
                    )

        # Add columns
        cluster_id_col = f"{column}_{analyzer_id}_cluster_id"
        cluster_size_col = f"{column}_{analyzer_id}_cluster_size"
        concentrated_col = f"{column}_{analyzer_id}_is_concentrated"

        result_df[cluster_id_col] = all_cluster_ids
        result_df[cluster_size_col] = all_cluster_sizes

        generated_schema = {
            cluster_id_col: {
                "type": ColumnType.INT,
                "content_type": ContentType.CATEGORICAL,
                "description": f"Question cluster ID for {column} (-1 = unique)",
            },
            cluster_size_col: {
                "type": ColumnType.INT,
                "content_type": ContentType.NUMERIC,
                "description": f"Size of question cluster for {column}",
            },
        }

        if self.flag_concentrated_clusters:
            result_df[concentrated_col] = all_concentrated_flags
            generated_schema[concentrated_col] = {
                "type": ColumnType.BOOL,
                "content_type": ContentType.CATEGORICAL,
                "description": f"Whether {column} is in a concentrated cluster",
            }

        # Compute dataset-level metrics
        diversity_ratio = noise_count / total_questions if total_questions > 0 else 0

        self._dataset_metrics[column] = {
            "num_question_clusters": n_clusters,
            "num_noise_samples": noise_count,
            "diversity_ratio": round(diversity_ratio, 4),
            "cluster_distribution": {
                str(k): v for k, v in cluster_sizes.items() if k != -1
            },
        }

        if self.compute_entropy:
            entropy = self._compute_shannon_entropy(cluster_labels)
            self._dataset_metrics[column]["question_entropy"] = round(entropy, 4)
            self._dataset_metrics[column]["diversity_rating"] = self._get_diversity_rating(
                entropy, n_clusters, diversity_ratio
            )

        if self.compute_concentration:
            gini = self._compute_gini_coefficient(cluster_labels)
            self._dataset_metrics[column]["question_gini"] = round(gini, 4)

            # Largest cluster ratio
            max_cluster_size = max(
                (v for k, v in cluster_sizes.items() if k != -1), default=0
            )
            largest_ratio = max_cluster_size / total_questions if total_questions > 0 else 0
            self._dataset_metrics[column]["largest_cluster_ratio"] = round(
                largest_ratio, 4
            )

        return result_df, generated_schema
