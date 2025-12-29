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
from tqdm import tqdm

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
    ):
        """Initialize the EmbeddingAnalyzer.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Default is "all-MiniLM-L6-v2" which is fast and effective.
            detect_duplicates: Whether to detect semantic duplicates.
                Duplicates are identified by cosine similarity above threshold.
            duplicate_threshold: Cosine similarity threshold for detecting
                duplicates. Values closer to 1.0 require higher similarity.
            duplicate_scope: Controls which messages are compared for duplicate
                detection. Options:
                - "all": Compare all messages together (default, original behavior)
                - "by_role": Only compare messages with the same role (user vs user,
                  assistant vs assistant, etc.)
                - "user": Only detect duplicates among user messages
                - "assistant": Only detect duplicates among assistant messages
                Requires a "role" column in the schema. Falls back to "all" if
                no role column is found.
            detect_fuzzy_duplicates: Whether to detect fuzzy (near) duplicates
                using MinHash LSH. Much faster than semantic duplicate detection
                for large datasets. Requires datasketch package.
            fuzzy_threshold: Jaccard similarity threshold for fuzzy duplicates.
                Values closer to 1.0 require higher similarity. Default 0.8.
            fuzzy_ngram_size: N-gram size for fuzzy duplicate detection.
                Smaller values are more sensitive to small changes. Default 3.
            fuzzy_num_perm: Number of permutations for MinHash. Higher values
                are more accurate but slower. Default 128.
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
            show_progress_bar: Whether to show progress bars during embedding
                computation and fuzzy duplicate detection. Default True.

        Raises:
            ImportError: If sentence-transformers is not installed.
            ValueError: If clustering_method is "kmeans" but n_clusters is not set.
        """
        self._check_dependencies()

        self.model_name = model_name
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
        self.batch_size = batch_size
        self.device = device
        self.store_embeddings = store_embeddings
        self.show_progress_bar = show_progress_bar

        # Validate parameters
        valid_scopes = ("all", "by_role", "user", "assistant")
        if self.duplicate_scope not in valid_scopes:
            raise ValueError(
                f"Invalid duplicate_scope: '{self.duplicate_scope}'. "
                f"Must be one of: {valid_scopes}"
            )

        if self.cluster_samples and self.clustering_method == "kmeans":
            if self.n_clusters is None:
                raise ValueError(
                    "n_clusters must be specified when using kmeans clustering. "
                    "Either set n_clusters or use clustering_method='dbscan'."
                )

        # Check fuzzy duplicate dependencies
        if self.detect_fuzzy_duplicates:
            self._check_fuzzy_dependencies()

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

    def _check_fuzzy_dependencies(self) -> None:
        """Check if datasketch is installed for fuzzy duplicate detection.

        Raises:
            ImportError: If datasketch is not installed.
        """
        try:
            import datasketch  # noqa: F401
        except ImportError:
            raise ImportError(
                "Fuzzy duplicate detection requires datasketch. "
                "Install with: pip install 'oumi[analyze]' or pip install datasketch"
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
            # No progress bar - encode all at once
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings

        # Use our own progress bar for better terminal compatibility
        # Process in batches with tqdm
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

    def _get_ngrams(self, text: str) -> set[str]:
        """Extract character n-grams from text.

        Args:
            text: Input text.

        Returns:
            Set of n-gram strings.
        """
        text = text.lower().strip()
        n = self.fuzzy_ngram_size
        if len(text) < n:
            return {text}
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def _detect_fuzzy_duplicates(
        self, texts: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect fuzzy (near) duplicates using MinHash LSH.

        This method is much faster than semantic duplicate detection for
        large datasets, as it uses locality-sensitive hashing.

        Args:
            texts: List of text strings.

        Returns:
            Tuple of (fuzzy_group_ids, is_fuzzy_duplicate, jaccard_scores).
            - fuzzy_group_ids: Integer IDs grouping near-duplicates together
            - is_fuzzy_duplicate: Boolean array indicating if sample has near-duplicates
            - jaccard_scores: Estimated Jaccard similarity to nearest duplicate
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
        processed_groups: dict[int, int] = {}  # Maps sample index to group ID
        current_group = 0

        minhash_iterator = (
            tqdm(enumerate(minhashes), total=len(minhashes), desc="Finding duplicates")
            if self.show_progress_bar
            else enumerate(minhashes)
        )
        for i, m in minhash_iterator:
            # Query LSH for similar items
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
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields using embeddings for semantic analysis.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added embedding analysis columns.
            generated column schema dict).
        """
        result_df = df.copy()
        generated_schema = {}

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
            return result_df, generated_schema

        # Get analyzer ID for column naming
        analyzer_id = getattr(self, "analyzer_id", "embedding")

        # Find role column for role-aware duplicate detection
        role_column = self._find_role_column(df, schema)

        for column in text_columns:
            # Get texts and compute embeddings
            texts = df[column].astype(str).tolist()

            if len(texts) == 0:
                continue

            logger.info(f"Computing embeddings for {len(texts)} samples...")
            embeddings = self._compute_embeddings(texts)

            # Detect semantic duplicates
            if self.detect_duplicates:
                dup_group_col = f"{column}_{analyzer_id}_duplicate_group"
                has_dup_col = f"{column}_{analyzer_id}_has_semantic_duplicate"

                # Determine if we should use role-aware detection
                use_role_aware = (
                    self.duplicate_scope != "all" and role_column is not None
                )

                if not use_role_aware:
                    # Original behavior: compare all messages together
                    logger.info("Detecting semantic duplicates...")
                    duplicate_groups, is_duplicate = self._detect_semantic_duplicates(
                        embeddings
                    )
                    result_df[dup_group_col] = duplicate_groups
                    result_df[has_dup_col] = is_duplicate
                else:
                    # Role-aware duplicate detection
                    logger.info(
                        f"Detecting semantic duplicates with scope='{self.duplicate_scope}'..."
                    )
                    result_df[dup_group_col] = np.arange(len(df))
                    result_df[has_dup_col] = False

                    # Determine which roles to process
                    if self.duplicate_scope == "by_role":
                        roles_to_process = df[role_column].str.lower().unique().tolist()
                    else:
                        # "user" or "assistant" - only process that role
                        roles_to_process = [self.duplicate_scope]

                    # Track group ID offset to ensure unique groups across roles
                    group_offset = 0

                    for role in roles_to_process:
                        role_mask = df[role_column].str.lower() == role
                        role_indices = df[role_mask].index.tolist()

                        if len(role_indices) < 2:
                            continue

                        # Get embeddings for this role
                        role_embeddings = embeddings[role_mask]

                        logger.info(
                            f"  Processing {len(role_indices)} '{role}' messages..."
                        )

                        # Detect duplicates within this role
                        role_groups, role_is_dup = self._detect_semantic_duplicates(
                            role_embeddings
                        )

                        # Assign results back to the full DataFrame
                        # Offset group IDs to be unique across roles
                        role_groups_offset = role_groups + group_offset
                        group_offset = role_groups_offset.max() + 1

                        for local_idx, global_idx in enumerate(role_indices):
                            result_df.loc[global_idx, dup_group_col] = (
                                role_groups_offset[local_idx]
                            )
                            result_df.loc[global_idx, has_dup_col] = role_is_dup[
                                local_idx
                            ]

            # Detect fuzzy (near) duplicates using MinHash
            if self.detect_fuzzy_duplicates:
                fuzzy_group_col = f"{column}_{analyzer_id}_fuzzy_duplicate_group"
                has_fuzzy_col = f"{column}_{analyzer_id}_has_fuzzy_duplicate"
                jaccard_col = f"{column}_{analyzer_id}_fuzzy_jaccard_score"

                # Determine if we should use role-aware detection
                use_role_aware = (
                    self.duplicate_scope != "all" and role_column is not None
                )

                if not use_role_aware:
                    # Original behavior: compare all messages together
                    logger.info("Detecting fuzzy duplicates using MinHash LSH...")
                    fuzzy_groups, is_fuzzy_dup, jaccard_scores = (
                        self._detect_fuzzy_duplicates(texts)
                    )
                    result_df[fuzzy_group_col] = fuzzy_groups
                    result_df[has_fuzzy_col] = is_fuzzy_dup
                    result_df[jaccard_col] = jaccard_scores
                else:
                    # Role-aware fuzzy duplicate detection
                    logger.info(
                        f"Detecting fuzzy duplicates with scope='{self.duplicate_scope}'..."
                    )
                    result_df[fuzzy_group_col] = np.arange(len(df))
                    result_df[has_fuzzy_col] = False
                    result_df[jaccard_col] = 0.0

                    # Determine which roles to process
                    if self.duplicate_scope == "by_role":
                        roles_to_process = df[role_column].str.lower().unique().tolist()
                    else:
                        roles_to_process = [self.duplicate_scope]

                    group_offset = 0

                    for role in roles_to_process:
                        role_mask = df[role_column].str.lower() == role
                        role_indices = df[role_mask].index.tolist()

                        if len(role_indices) < 2:
                            continue

                        role_texts = [texts[i] for i in role_indices]

                        logger.info(
                            f"  Processing {len(role_indices)} '{role}' messages..."
                        )

                        role_groups, role_is_dup, role_jaccard = (
                            self._detect_fuzzy_duplicates(role_texts)
                        )

                        role_groups_offset = role_groups + group_offset
                        group_offset = role_groups_offset.max() + 1

                        for local_idx, global_idx in enumerate(role_indices):
                            result_df.loc[global_idx, fuzzy_group_col] = (
                                role_groups_offset[local_idx]
                            )
                            result_df.loc[global_idx, has_fuzzy_col] = role_is_dup[
                                local_idx
                            ]
                            result_df.loc[global_idx, jaccard_col] = role_jaccard[
                                local_idx
                            ]

            # Cluster samples
            if self.cluster_samples:
                logger.info(f"Clustering samples using {self.clustering_method}...")
                cluster_labels = self._cluster_embeddings(embeddings)
                result_df[f"{column}_{analyzer_id}_cluster"] = cluster_labels

            # Optionally store embeddings (can be large)
            if self.store_embeddings:
                # Store as list per row (not ideal for large datasets)
                result_df[f"{column}_{analyzer_id}_embedding"] = [
                    emb.tolist() for emb in embeddings
                ]

        return result_df, generated_schema
