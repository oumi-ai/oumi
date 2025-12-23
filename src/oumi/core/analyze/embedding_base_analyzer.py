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

"""Base class for embedding-based analyzers.

This module provides a base class that encapsulates common functionality
for analyzers that use sentence embeddings, including:
- Lazy model loading
- Batch embedding computation
- Role column detection
- Dependency checking

Analyzers that use embeddings (EmbeddingAnalyzer, QuestionDiversityAnalyzer,
ReprDiversityAnalyzer) should inherit from this class.
"""

from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.utils.logging import logger


class EmbeddingBasedAnalyzer(SampleAnalyzer):
    """Base class for analyzers that use sentence embeddings.

    This class provides common functionality for embedding-based analysis:
    - Lazy loading of sentence-transformer models
    - Batched embedding computation with progress bars
    - Role column detection from schema
    - Dependency checking for optional packages

    Subclasses must implement the analyze_sample method.

    Attributes:
        model_name: Name of the sentence-transformers model.
        batch_size: Batch size for embedding computation.
        device: Device to use ("cuda", "cpu", or None for auto).
        show_progress_bar: Whether to show progress during computation.
    """

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
        show_progress_bar: bool = True,
    ):
        """Initialize the embedding-based analyzer.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Default is "all-MiniLM-L6-v2" which is fast and effective.
                For higher quality, try "all-mpnet-base-v2".
            batch_size: Batch size for embedding computation. Larger values
                are faster but use more memory.
            device: Device for embedding model ("cuda", "cpu", or None for auto).
                If None, will use CUDA if available, otherwise CPU.
            show_progress_bar: Whether to show progress bars during embedding
                computation. Disable for cleaner logs in batch processing.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        self._check_sentence_transformers()

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.show_progress_bar = show_progress_bar

        # Lazy-load the model
        self._model = None

        # Store dataset-level metrics (populated by subclasses)
        self._dataset_metrics: dict[str, Any] = {}

    def _check_sentence_transformers(self) -> None:
        """Check if sentence-transformers is installed.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                f"{self.__class__.__name__} requires sentence-transformers. "
                "Install with: pip install 'oumi[analyze]'"
            )

    def _check_sklearn(self) -> None:
        """Check if scikit-learn is installed.

        Raises:
            ImportError: If scikit-learn is not installed.
        """
        try:
            import sklearn  # noqa: F401
        except ImportError:
            raise ImportError(
                f"{self.__class__.__name__} requires scikit-learn. "
                "Install with: pip install 'oumi[analyze]'"
            )

    def _check_datasketch(self) -> None:
        """Check if datasketch is installed for MinHash LSH.

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

        The model is loaded on first use and cached for subsequent calls.
        This avoids loading the model if the analyzer is never used.

        Returns:
            SentenceTransformer model instance.

        Raises:
            Exception: If model loading fails (e.g., model not found).
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {e}. "
                    "Check that the model name is correct and you have internet access."
                ) from e
        return self._model

    def warm_up(self) -> None:
        """Pre-load the embedding model.

        Call this method to load the model before analysis to avoid
        delays during the first analyze_sample call. Useful for
        benchmarking or when you need predictable latency.
        """
        _ = self._get_model()
        logger.info(f"Model '{self.model_name}' loaded and ready")

    def _compute_embeddings(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute embeddings for a list of texts.

        Args:
            texts: List of text strings to embed. Empty strings are handled
                gracefully but may produce less meaningful embeddings.
            normalize: Whether to L2-normalize embeddings. Normalized embeddings
                allow using dot product as cosine similarity. Default True.

        Returns:
            Numpy array of shape (n_texts, embedding_dim). If normalize=True,
            each embedding vector has unit L2 norm.

        Note:
            For very long texts (>512 tokens for most models), the text will
            be truncated. Consider chunking long documents before embedding.
        """
        if not texts:
            return np.array([])

        model = self._get_model()

        # Filter empty texts and track indices
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            # All texts were empty - return zero embeddings
            embedding_dim = model.get_sentence_embedding_dimension()
            return np.zeros((len(texts), embedding_dim))

        if len(non_empty_texts) != len(texts):
            logger.warning(
                f"{len(texts) - len(non_empty_texts)} empty texts will receive "
                "zero embeddings"
            )

        # Compute embeddings
        if not self.show_progress_bar or len(non_empty_texts) <= self.batch_size:
            # No progress bar needed for small batches
            embeddings = model.encode(
                non_empty_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
        else:
            # Use our own progress bar for better terminal compatibility
            all_embeddings = []

            with tqdm(total=len(non_empty_texts), desc="Computing embeddings") as pbar:
                for i in range(0, len(non_empty_texts), self.batch_size):
                    batch_texts = non_empty_texts[i : i + self.batch_size]
                    batch_embeddings = model.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=normalize,
                    )
                    all_embeddings.append(batch_embeddings)
                    pbar.update(len(batch_texts))

            embeddings = np.vstack(all_embeddings)

        # If some texts were empty, create full result array with zeros
        if len(non_empty_texts) != len(texts):
            embedding_dim = embeddings.shape[1]
            full_embeddings = np.zeros((len(texts), embedding_dim))
            for new_idx, orig_idx in enumerate(non_empty_indices):
                full_embeddings[orig_idx] = embeddings[new_idx]
            return full_embeddings

        return embeddings

    def _find_role_column(
        self, df: pd.DataFrame, schema: Optional[dict]
    ) -> Optional[str]:
        """Find the role column in the DataFrame using the schema.

        Looks for a column that:
        1. Has content_type == ContentType.CATEGORICAL
        2. Exists in the DataFrame
        3. Has "role" in its name (case-insensitive)

        Args:
            df: Input DataFrame.
            schema: Column schema dict mapping column names to config dicts.

        Returns:
            Name of the role column if found, None otherwise.
        """
        if not schema:
            return None

        for col, config in schema.items():
            if col not in df.columns:
                continue

            content_type = config.get("content_type")
            # Handle both string and enum content types
            is_categorical = (
                content_type == ContentType.CATEGORICAL
                or content_type == "categorical"
            )

            if is_categorical and "role" in col.lower():
                return col

        return None

    def _find_text_columns(
        self, df: pd.DataFrame, schema: Optional[dict]
    ) -> list[str]:
        """Find text columns in the DataFrame using the schema.

        Args:
            df: Input DataFrame.
            schema: Column schema dict mapping column names to config dicts.

        Returns:
            List of column names that contain text content.
        """
        if not schema:
            return []

        text_columns = []
        for col, config in schema.items():
            if col not in df.columns:
                continue

            content_type = config.get("content_type")
            # Handle both string and enum content types
            is_text = content_type == ContentType.TEXT or content_type == "text"

            if is_text:
                text_columns.append(col)

        return text_columns

    def _validate_schema_required(self, schema: Optional[dict]) -> None:
        """Validate that schema is provided.

        Args:
            schema: Column schema dict.

        Raises:
            ValueError: If schema is None or empty.
        """
        if not schema:
            raise ValueError(
                f"schema is required for {self.__class__.__name__}. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content (content_type='text')."
            )

    @abstractmethod
    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze the DataFrame and return results with schema.

        Subclasses must implement this method.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (result DataFrame, generated schema dict).
        """
        pass

    def compute_dataset_metrics(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Return dataset-level metrics computed during analyze_sample.

        This method returns aggregate metrics that were computed during
        the last call to analyze_sample. The specific metrics depend on
        the subclass implementation.

        Args:
            df: Analyzed DataFrame (output of analyze_sample).
            schema: Column schema dict.

        Returns:
            Dictionary of dataset-level metrics.
        """
        return self._dataset_metrics.copy()
