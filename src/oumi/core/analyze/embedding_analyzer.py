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

"""Embedding analyzer for semantic analysis of text content."""

from typing import Any, Optional

import numpy as np
import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("embedding")
class EmbeddingAnalyzer(SampleAnalyzer):
    """Analyzer that computes embeddings for semantic analysis of text content.

    This analyzer uses sentence-transformers to compute dense vector embeddings
    for text content, enabling semantic analysis such as:
    - Detecting semantic duplicates (similar meaning but different wording)
    - Clustering similar samples
    - Computing semantic similarity scores

    Note: This analyzer requires sentence-transformers to be installed.
    Install with: pip install 'oumi[analyze_advanced]'
    """

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        detect_duplicates: bool = True,
        duplicate_threshold: float = 0.95,
        cluster_samples: bool = False,
        clustering_method: str = "dbscan",
        n_clusters: Optional[int] = None,
        eps: float = 0.5,
        min_samples: int = 2,
        batch_size: int = 32,
        device: Optional[str] = None,
        store_embeddings: bool = False,
    ):
        """Initialize the EmbeddingAnalyzer.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Default is "all-MiniLM-L6-v2" which is fast and effective.
            detect_duplicates: Whether to detect semantic duplicates.
                Duplicates are identified by cosine similarity above threshold.
            duplicate_threshold: Cosine similarity threshold for detecting
                duplicates. Values closer to 1.0 require higher similarity.
            cluster_samples: Whether to cluster samples by semantic similarity.
            clustering_method: Clustering method to use: "dbscan" or "kmeans".
                DBSCAN automatically determines the number of clusters.
            n_clusters: Number of clusters for k-means. Required if using kmeans.
            eps: DBSCAN epsilon parameter (maximum distance between samples).
            min_samples: DBSCAN minimum samples in a neighborhood.
            batch_size: Batch size for embedding computation.
            device: Device to use for embedding model ("cuda", "cpu", or None for auto).
            store_embeddings: Whether to store embeddings in the output DataFrame.
                Note: This can significantly increase memory usage.

        Raises:
            ImportError: If sentence-transformers is not installed.
            ValueError: If clustering_method is "kmeans" but n_clusters is not set.
        """
        self._check_dependencies()

        self.model_name = model_name
        self.detect_duplicates = detect_duplicates
        self.duplicate_threshold = duplicate_threshold
        self.cluster_samples = cluster_samples
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.batch_size = batch_size
        self.device = device
        self.store_embeddings = store_embeddings

        # Validate parameters
        if self.cluster_samples and self.clustering_method == "kmeans":
            if self.n_clusters is None:
                raise ValueError(
                    "n_clusters must be specified when using kmeans clustering. "
                    "Either set n_clusters or use clustering_method='dbscan'."
                )

        # Lazy-load the model
        self._model = None

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "EmbeddingAnalyzer requires sentence-transformers. "
                "Install with: pip install 'oumi[analyze_advanced]'"
            )

        try:
            import sklearn  # noqa: F401
        except ImportError:
            raise ImportError(
                "EmbeddingAnalyzer requires scikit-learn for clustering. "
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
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    def _detect_semantic_duplicates(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect semantic duplicates based on cosine similarity.

        Args:
            embeddings: Array of embeddings.

        Returns:
            Tuple of (duplicate_group_ids, is_duplicate flags).
            - duplicate_group_ids: Integer IDs grouping duplicates together
            - is_duplicate: Boolean array indicating if sample has duplicates
        """
        from sklearn.metrics.pairwise import cosine_similarity

        n_samples = len(embeddings)

        # Initialize arrays
        duplicate_group_ids = np.arange(n_samples)  # Each sample starts in own group
        is_duplicate = np.zeros(n_samples, dtype=bool)

        # Compute similarity matrix (can be memory intensive for large datasets)
        # For very large datasets, we would use batch processing
        if n_samples > 10000:
            logger.warning(
                f"Computing similarity matrix for {n_samples} samples. "
                "This may take a while and use significant memory."
            )

        # Process in chunks to manage memory
        chunk_size = min(1000, n_samples)
        processed = set()

        for i in range(0, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)
            chunk_embeddings = embeddings[i:end_i]

            # Compute similarity of this chunk against all embeddings
            similarities = cosine_similarity(chunk_embeddings, embeddings)

            for local_idx, global_idx in enumerate(range(i, end_i)):
                if global_idx in processed:
                    continue

                # Find duplicates (high similarity above threshold)
                sim_row = similarities[local_idx]
                duplicate_indices = np.where(sim_row >= self.duplicate_threshold)[0]

                # Exclude self
                duplicate_indices = duplicate_indices[duplicate_indices != global_idx]

                if len(duplicate_indices) > 0:
                    # Mark as having duplicates
                    is_duplicate[global_idx] = True
                    is_duplicate[duplicate_indices] = True

                    # Assign same group ID
                    min_group = min(
                        duplicate_group_ids[global_idx],
                        *[duplicate_group_ids[j] for j in duplicate_indices],
                    )
                    duplicate_group_ids[global_idx] = min_group
                    for j in duplicate_indices:
                        duplicate_group_ids[j] = min_group

                processed.add(global_idx)

        return duplicate_group_ids, is_duplicate

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings using the specified method.

        Args:
            embeddings: Array of embeddings.

        Returns:
            Array of cluster labels.
        """
        if self.clustering_method == "dbscan":
            from sklearn.cluster import DBSCAN

            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = clustering.fit_predict(embeddings)
        elif self.clustering_method == "kmeans":
            from sklearn.cluster import KMeans

            clustering = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = clustering.fit_predict(embeddings)
        else:
            raise ValueError(
                f"Unknown clustering method: {self.clustering_method}. "
                f"Supported methods: 'dbscan', 'kmeans'."
            )

        return labels

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields using embeddings for semantic analysis.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added embedding analysis columns.
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for embedding analysis. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            # No text columns to analyze in this DataFrame, return unchanged
            return result_df

        # Get analyzer ID for column naming
        analyzer_id = getattr(self, "analyzer_id", "embedding")

        for column in text_columns:
            # Get texts and compute embeddings
            texts = df[column].astype(str).tolist()

            if len(texts) == 0:
                continue

            logger.info(f"Computing embeddings for {len(texts)} samples...")
            embeddings = self._compute_embeddings(texts)

            # Detect semantic duplicates
            if self.detect_duplicates:
                logger.info("Detecting semantic duplicates...")
                duplicate_groups, is_duplicate = self._detect_semantic_duplicates(
                    embeddings
                )
                dup_group_col = f"{column}_{analyzer_id}_duplicate_group"
                result_df[dup_group_col] = duplicate_groups
                has_dup_col = f"{column}_{analyzer_id}_has_semantic_duplicate"
                result_df[has_dup_col] = is_duplicate

            # Cluster samples
            if self.cluster_samples:
                logger.info(
                    f"Clustering samples using {self.clustering_method}..."
                )
                cluster_labels = self._cluster_embeddings(embeddings)
                result_df[f"{column}_{analyzer_id}_cluster"] = cluster_labels

            # Optionally store embeddings (can be large)
            if self.store_embeddings:
                # Store as list per row (not ideal for large datasets)
                result_df[f"{column}_{analyzer_id}_embedding"] = [
                    emb.tolist() for emb in embeddings
                ]

        return result_df
