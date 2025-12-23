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

"""Embedding analyzer for semantic analysis of text content.

This module provides the EmbeddingAnalyzer class for detecting semantic
duplicates, fuzzy duplicates (using MinHash LSH), and clustering samples
based on their semantic similarity.

Example usage:
    >>> from oumi.core.analyze import EmbeddingAnalyzer
    >>> analyzer = EmbeddingAnalyzer(
    ...     detect_duplicates=True,
    ...     duplicate_threshold=0.95,
    ... )
    >>> result_df, schema = analyzer.analyze_sample(df, schema)
"""

from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.analyze.embedding_base_analyzer import EmbeddingBasedAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("embedding")
class EmbeddingAnalyzer(EmbeddingBasedAnalyzer):
    """Analyzer that computes embeddings for semantic analysis of text content.

    This analyzer uses sentence-transformers to compute dense vector embeddings
    for text content, enabling semantic analysis such as:
    - Detecting semantic duplicates (similar meaning but different wording)
    - Detecting fuzzy duplicates (using MinHash LSH for O(n) scalability)
    - Clustering similar samples using DBSCAN or K-means

    Note: This analyzer requires sentence-transformers to be installed.
    Install with: pip install 'oumi[analyze]'

    Example:
        >>> analyzer = EmbeddingAnalyzer(
        ...     detect_duplicates=True,
        ...     duplicate_threshold=0.95,
        ...     cluster_samples=True,
        ...     clustering_method="dbscan",
        ... )
        >>> result_df, schema = analyzer.analyze_sample(
        ...     df,
        ...     schema={"content": {"content_type": "text"}}
        ... )
    """

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        detect_duplicates: bool = True,
        duplicate_threshold: float = 0.95,
        duplicate_scope: str = "all",
        detect_fuzzy_duplicates: bool = False,
        fuzzy_threshold: float = 0.8,
        fuzzy_ngram_size: int = 3,
        fuzzy_num_perm: int = 128,
        cluster_samples: bool = False,
        clustering_method: str = "dbscan",
        n_clusters: Optional[int] = None,
        eps: float = 0.5,
        min_samples: int = 2,
        batch_size: int = 32,
        device: Optional[str] = None,
        store_embeddings: bool = False,
        show_progress_bar: bool = True,
        similarity_chunk_size: int = 1000,
    ):
        """Initialize the EmbeddingAnalyzer.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Default is "all-MiniLM-L6-v2" which is fast and effective.
            detect_duplicates: Whether to detect semantic duplicates.
                Duplicates are identified by cosine similarity above threshold.
            duplicate_threshold: Cosine similarity threshold for detecting
                duplicates. Must be in range [0.0, 1.0]. Values closer to 1.0
                require higher similarity. Default 0.95.
            duplicate_scope: Controls which messages are compared for duplicate
                detection. Options:
                - "all": Compare all messages together (default)
                - "by_role": Only compare messages with the same role
                - "user": Only detect duplicates among user messages
                - "assistant": Only detect duplicates among assistant messages
            detect_fuzzy_duplicates: Whether to detect fuzzy (near) duplicates
                using MinHash LSH. Much faster than semantic duplicate detection
                for large datasets. Requires datasketch package.
            fuzzy_threshold: Jaccard similarity threshold for fuzzy duplicates.
                Must be in range [0.0, 1.0]. Default 0.8.
            fuzzy_ngram_size: N-gram size for fuzzy duplicate detection.
                Smaller values are more sensitive to small changes. Default 3.
            fuzzy_num_perm: Number of permutations for MinHash. Higher values
                are more accurate but slower. Default 128.
            cluster_samples: Whether to cluster samples by semantic similarity.
            clustering_method: Clustering method: "dbscan" or "kmeans".
            n_clusters: Number of clusters for k-means. Required if using kmeans.
            eps: DBSCAN epsilon parameter (maximum distance between samples).
            min_samples: DBSCAN minimum samples in a neighborhood.
            batch_size: Batch size for embedding computation.
            device: Device for embedding model ("cuda", "cpu", or None for auto).
            store_embeddings: Whether to store embeddings in the output DataFrame.
            show_progress_bar: Whether to show progress bars during computation.
            similarity_chunk_size: Chunk size for similarity matrix computation.
                Larger values are faster but use more memory. Default 1000.

        Raises:
            ImportError: If required dependencies are not installed.
            ValueError: If parameters are invalid.
        """
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            show_progress_bar=show_progress_bar,
        )

        # Validate thresholds
        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError(
                f"duplicate_threshold must be in range [0.0, 1.0], "
                f"got {duplicate_threshold}"
            )
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError(
                f"fuzzy_threshold must be in range [0.0, 1.0], "
                f"got {fuzzy_threshold}"
            )

        # Validate duplicate_scope
        valid_scopes = ("all", "by_role", "user", "assistant")
        if duplicate_scope not in valid_scopes:
            raise ValueError(
                f"Invalid duplicate_scope: '{duplicate_scope}'. "
                f"Must be one of: {valid_scopes}"
            )

        # Validate clustering parameters
        if cluster_samples:
            self._check_sklearn()
            if clustering_method == "kmeans" and n_clusters is None:
                raise ValueError(
                    "n_clusters must be specified when using kmeans clustering. "
                    "Either set n_clusters or use clustering_method='dbscan'."
                )
            if clustering_method not in ("dbscan", "kmeans"):
                raise ValueError(
                    f"Unknown clustering_method: '{clustering_method}'. "
                    f"Supported methods: 'dbscan', 'kmeans'."
                )

        # Check fuzzy duplicate dependencies
        if detect_fuzzy_duplicates:
            self._check_datasketch()

        self.detect_duplicates = detect_duplicates
        self.duplicate_threshold = duplicate_threshold
        self.duplicate_scope = duplicate_scope
        self.detect_fuzzy_duplicates = detect_fuzzy_duplicates
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_ngram_size = fuzzy_ngram_size
        self.fuzzy_num_perm = fuzzy_num_perm
        self.cluster_samples = cluster_samples
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.store_embeddings = store_embeddings
        self.similarity_chunk_size = similarity_chunk_size

    def _detect_semantic_duplicates(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect semantic duplicates based on cosine similarity.

        Uses chunked processing to manage memory for large datasets.

        Args:
            embeddings: Array of normalized embeddings.

        Returns:
            Tuple of (duplicate_group_ids, is_duplicate flags).
            - duplicate_group_ids: Integer IDs grouping duplicates together
            - is_duplicate: Boolean array indicating if sample has duplicates
        """
        from sklearn.metrics.pairwise import cosine_similarity

        n_samples = len(embeddings)

        # Initialize arrays
        duplicate_group_ids = np.arange(n_samples)
        is_duplicate = np.zeros(n_samples, dtype=bool)

        if n_samples > 10000:
            logger.warning(
                f"Computing similarity matrix for {n_samples} samples. "
                "This may take a while and use significant memory. "
                "Consider using fuzzy duplicates (MinHash LSH) for faster processing."
            )

        # Process in chunks to manage memory
        chunk_size = min(self.similarity_chunk_size, n_samples)
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

                    # Assign same group ID (use minimum for consistency)
                    min_group = min(
                        duplicate_group_ids[global_idx],
                        *[duplicate_group_ids[j] for j in duplicate_indices],
                    )
                    duplicate_group_ids[global_idx] = min_group
                    for j in duplicate_indices:
                        duplicate_group_ids[j] = min_group

                processed.add(global_idx)

        return duplicate_group_ids, is_duplicate

    def _get_ngrams(self, text: str) -> set[str]:
        """Extract character n-grams from text.

        Args:
            text: Input text string.

        Returns:
            Set of n-gram strings. For texts shorter than n-gram size,
            returns a set containing just the text itself.
        """
        text = text.lower().strip()
        n = self.fuzzy_ngram_size
        if len(text) < n:
            return {text} if text else set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def _detect_fuzzy_duplicates(
        self, texts: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect fuzzy (near) duplicates using MinHash LSH.

        This method is much faster than semantic duplicate detection for
        large datasets, as it uses locality-sensitive hashing with O(n)
        complexity for candidate generation.

        Args:
            texts: List of text strings.

        Returns:
            Tuple of (fuzzy_group_ids, is_fuzzy_duplicate, jaccard_scores).
        """
        from datasketch import MinHash, MinHashLSH

        n_samples = len(texts)

        # Initialize arrays
        fuzzy_group_ids = np.arange(n_samples)
        is_fuzzy_duplicate = np.zeros(n_samples, dtype=bool)
        jaccard_scores = np.zeros(n_samples, dtype=float)

        # Create MinHash for each text
        logger.info(f"Creating MinHash signatures for {n_samples} samples...")
        minhashes: list[MinHash] = []

        text_iterator = (
            tqdm(texts, desc="Creating MinHash signatures")
            if self.show_progress_bar
            else texts
        )
        for text in text_iterator:
            m = MinHash(num_perm=self.fuzzy_num_perm)
            for ngram in self._get_ngrams(text):
                m.update(ngram.encode("utf-8"))
            minhashes.append(m)

        # Create LSH index
        lsh = MinHashLSH(threshold=self.fuzzy_threshold, num_perm=self.fuzzy_num_perm)
        for i, m in enumerate(minhashes):
            lsh.insert(str(i), m)

        # Find near-duplicates
        logger.info("Finding fuzzy duplicates using LSH...")
        processed_groups: dict[int, int] = {}
        current_group = 0

        minhash_iterator = (
            tqdm(enumerate(minhashes), total=len(minhashes), desc="Finding duplicates")
            if self.show_progress_bar
            else enumerate(minhashes)
        )
        for i, m in minhash_iterator:
            candidates = lsh.query(m)
            candidate_indices = [int(c) for c in candidates if int(c) != i]

            if candidate_indices:
                is_fuzzy_duplicate[i] = True

                # Compute actual Jaccard similarity with nearest candidate
                max_jaccard = 0.0
                for j in candidate_indices:
                    jaccard = minhashes[i].jaccard(minhashes[j])
                    max_jaccard = max(max_jaccard, jaccard)
                    is_fuzzy_duplicate[j] = True
                jaccard_scores[i] = max_jaccard

                # Assign group IDs
                existing_groups = [
                    processed_groups[j]
                    for j in candidate_indices
                    if j in processed_groups
                ]

                if i in processed_groups:
                    group_id = processed_groups[i]
                elif existing_groups:
                    group_id = min(existing_groups)
                else:
                    group_id = current_group
                    current_group += 1

                processed_groups[i] = group_id
                fuzzy_group_ids[i] = group_id

                for j in candidate_indices:
                    if j not in processed_groups:
                        processed_groups[j] = group_id
                    fuzzy_group_ids[j] = min(fuzzy_group_ids[j], group_id)

        return fuzzy_group_ids, is_fuzzy_duplicate, jaccard_scores

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings using the specified method.

        Args:
            embeddings: Array of embeddings.

        Returns:
            Array of cluster labels. For DBSCAN, -1 indicates noise points.
        """
        if self.clustering_method == "dbscan":
            from sklearn.cluster import DBSCAN

            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = clustering.fit_predict(embeddings)
        else:  # kmeans
            from sklearn.cluster import KMeans

            clustering = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = clustering.fit_predict(embeddings)

        return labels

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields using embeddings for semantic analysis.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (result DataFrame, generated schema dict).
            The result DataFrame contains the original columns plus:
            - {col}_embedding_duplicate_group: Duplicate group ID
            - {col}_embedding_has_semantic_duplicate: Boolean flag
            - {col}_embedding_fuzzy_duplicate_group: Fuzzy group ID (if enabled)
            - {col}_embedding_has_fuzzy_duplicate: Boolean flag (if enabled)
            - {col}_embedding_fuzzy_jaccard_score: Jaccard score (if enabled)
            - {col}_embedding_cluster: Cluster label (if enabled)
            - {col}_embedding_embedding: Raw embedding (if store_embeddings=True)
        """
        self._validate_schema_required(schema)
        result_df = df.copy()
        generated_schema: dict = {}

        text_columns = self._find_text_columns(df, schema)
        if not text_columns:
            return result_df, generated_schema

        analyzer_id = getattr(self, "analyzer_id", "embedding")
        role_column = self._find_role_column(df, schema)

        for column in text_columns:
            texts = df[column].astype(str).tolist()

            if len(texts) == 0:
                continue

            logger.info(f"Computing embeddings for {len(texts)} samples...")
            embeddings = self._compute_embeddings(texts)

            # Detect semantic duplicates
            if self.detect_duplicates:
                result_df, col_schema = self._process_semantic_duplicates(
                    result_df, df, embeddings, texts, column, analyzer_id,
                    role_column, schema
                )
                generated_schema.update(col_schema)

            # Detect fuzzy duplicates
            if self.detect_fuzzy_duplicates:
                result_df, col_schema = self._process_fuzzy_duplicates(
                    result_df, df, texts, column, analyzer_id, role_column
                )
                generated_schema.update(col_schema)

            # Cluster samples
            if self.cluster_samples:
                logger.info(f"Clustering samples using {self.clustering_method}...")
                cluster_labels = self._cluster_embeddings(embeddings)
                cluster_col = f"{column}_{analyzer_id}_cluster"
                result_df[cluster_col] = cluster_labels
                generated_schema[cluster_col] = {
                    "type": ColumnType.INT,
                    "content_type": ContentType.CATEGORICAL,
                    "description": f"Cluster label for {column} (-1 = noise for DBSCAN)",
                }

            # Store embeddings if requested
            if self.store_embeddings:
                emb_col = f"{column}_{analyzer_id}_embedding"
                result_df[emb_col] = [emb.tolist() for emb in embeddings]
                generated_schema[emb_col] = {
                    "type": ColumnType.OBJECT,
                    "content_type": ContentType.METADATA,
                    "description": f"Embedding vector for {column}",
                }

        return result_df, generated_schema

    def _process_semantic_duplicates(
        self,
        result_df: pd.DataFrame,
        original_df: pd.DataFrame,
        embeddings: np.ndarray,
        texts: list[str],
        column: str,
        analyzer_id: str,
        role_column: Optional[str],
        schema: Optional[dict],
    ) -> tuple[pd.DataFrame, dict]:
        """Process semantic duplicate detection with role-aware support."""
        dup_group_col = f"{column}_{analyzer_id}_duplicate_group"
        has_dup_col = f"{column}_{analyzer_id}_has_semantic_duplicate"

        use_role_aware = self.duplicate_scope != "all" and role_column is not None

        if not use_role_aware:
            logger.info("Detecting semantic duplicates...")
            duplicate_groups, is_duplicate = self._detect_semantic_duplicates(
                embeddings
            )
            result_df[dup_group_col] = duplicate_groups
            result_df[has_dup_col] = is_duplicate
        else:
            logger.info(
                f"Detecting semantic duplicates with scope='{self.duplicate_scope}'..."
            )
            result_df[dup_group_col] = np.arange(len(original_df))
            result_df[has_dup_col] = False

            if self.duplicate_scope == "by_role":
                roles_to_process = (
                    original_df[role_column].str.lower().unique().tolist()
                )
            else:
                roles_to_process = [self.duplicate_scope]

            group_offset = 0
            for role in roles_to_process:
                role_mask = original_df[role_column].str.lower() == role
                role_indices = original_df[role_mask].index.tolist()

                if len(role_indices) < 2:
                    continue

                role_embeddings = embeddings[role_mask]
                logger.info(f"  Processing {len(role_indices)} '{role}' messages...")

                role_groups, role_is_dup = self._detect_semantic_duplicates(
                    role_embeddings
                )

                role_groups_offset = role_groups + group_offset
                group_offset = role_groups_offset.max() + 1

                for local_idx, global_idx in enumerate(role_indices):
                    result_df.loc[global_idx, dup_group_col] = role_groups_offset[
                        local_idx
                    ]
                    result_df.loc[global_idx, has_dup_col] = role_is_dup[local_idx]

        generated_schema = {
            dup_group_col: {
                "type": ColumnType.INT,
                "content_type": ContentType.METADATA,
                "description": f"Duplicate group ID for {column}",
            },
            has_dup_col: {
                "type": ColumnType.BOOL,
                "content_type": ContentType.CATEGORICAL,
                "description": f"Whether {column} has semantic duplicates",
            },
        }

        return result_df, generated_schema

    def _process_fuzzy_duplicates(
        self,
        result_df: pd.DataFrame,
        original_df: pd.DataFrame,
        texts: list[str],
        column: str,
        analyzer_id: str,
        role_column: Optional[str],
    ) -> tuple[pd.DataFrame, dict]:
        """Process fuzzy duplicate detection with role-aware support."""
        fuzzy_group_col = f"{column}_{analyzer_id}_fuzzy_duplicate_group"
        has_fuzzy_col = f"{column}_{analyzer_id}_has_fuzzy_duplicate"
        jaccard_col = f"{column}_{analyzer_id}_fuzzy_jaccard_score"

        use_role_aware = self.duplicate_scope != "all" and role_column is not None

        if not use_role_aware:
            logger.info("Detecting fuzzy duplicates using MinHash LSH...")
            fuzzy_groups, is_fuzzy_dup, jaccard_scores = self._detect_fuzzy_duplicates(
                texts
            )
            result_df[fuzzy_group_col] = fuzzy_groups
            result_df[has_fuzzy_col] = is_fuzzy_dup
            result_df[jaccard_col] = jaccard_scores
        else:
            logger.info(
                f"Detecting fuzzy duplicates with scope='{self.duplicate_scope}'..."
            )
            result_df[fuzzy_group_col] = np.arange(len(original_df))
            result_df[has_fuzzy_col] = False
            result_df[jaccard_col] = 0.0

            if self.duplicate_scope == "by_role":
                roles_to_process = (
                    original_df[role_column].str.lower().unique().tolist()
                )
            else:
                roles_to_process = [self.duplicate_scope]

            group_offset = 0
            for role in roles_to_process:
                role_mask = original_df[role_column].str.lower() == role
                role_indices = original_df[role_mask].index.tolist()

                if len(role_indices) < 2:
                    continue

                role_texts = [texts[i] for i in role_indices]
                logger.info(f"  Processing {len(role_indices)} '{role}' messages...")

                role_groups, role_is_dup, role_jaccard = self._detect_fuzzy_duplicates(
                    role_texts
                )

                role_groups_offset = role_groups + group_offset
                group_offset = role_groups_offset.max() + 1

                for local_idx, global_idx in enumerate(role_indices):
                    result_df.loc[global_idx, fuzzy_group_col] = role_groups_offset[
                        local_idx
                    ]
                    result_df.loc[global_idx, has_fuzzy_col] = role_is_dup[local_idx]
                    result_df.loc[global_idx, jaccard_col] = role_jaccard[local_idx]

        generated_schema = {
            fuzzy_group_col: {
                "type": ColumnType.INT,
                "content_type": ContentType.METADATA,
                "description": f"Fuzzy duplicate group ID for {column}",
            },
            has_fuzzy_col: {
                "type": ColumnType.BOOL,
                "content_type": ContentType.CATEGORICAL,
                "description": f"Whether {column} has fuzzy duplicates",
            },
            jaccard_col: {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": f"Jaccard similarity score for {column}",
            },
        }

        return result_df, generated_schema
