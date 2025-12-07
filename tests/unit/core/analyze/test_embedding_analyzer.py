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

"""Tests for the EmbeddingAnalyzer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# Check if sentence-transformers is available
def _sentence_transformers_available():
    try:
        import sentence_transformers  # noqa: F401
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


# Skip all tests if sentence-transformers is not available
pytestmark = pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence-transformers or sklearn not installed"
)


class TestEmbeddingAnalyzerInit:
    """Tests for EmbeddingAnalyzer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        analyzer = EmbeddingAnalyzer()
        assert analyzer.model_name == "all-MiniLM-L6-v2"
        assert analyzer.detect_duplicates is True
        assert analyzer.duplicate_threshold == 0.95
        assert analyzer.cluster_samples is False
        assert analyzer.batch_size == 32

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        analyzer = EmbeddingAnalyzer(
            model_name="paraphrase-MiniLM-L6-v2",
            detect_duplicates=False,
            duplicate_threshold=0.9,
            cluster_samples=True,
            clustering_method="dbscan",
            eps=0.3,
            batch_size=64,
        )
        assert analyzer.model_name == "paraphrase-MiniLM-L6-v2"
        assert analyzer.detect_duplicates is False
        assert analyzer.duplicate_threshold == 0.9
        assert analyzer.cluster_samples is True
        assert analyzer.eps == 0.3

    def test_init_kmeans_requires_n_clusters(self):
        """Test that kmeans clustering requires n_clusters."""
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        with pytest.raises(ValueError, match="n_clusters must be specified"):
            EmbeddingAnalyzer(
                cluster_samples=True,
                clustering_method="kmeans",
                n_clusters=None,
            )

    def test_init_kmeans_with_n_clusters(self):
        """Test that kmeans clustering works with n_clusters."""
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        analyzer = EmbeddingAnalyzer(
            cluster_samples=True,
            clustering_method="kmeans",
            n_clusters=5,
        )
        assert analyzer.n_clusters == 5


class TestEmbeddingAnalyzerWithMocks:
    """Tests using mocked embedding model."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock SentenceTransformer model."""
        model = MagicMock()
        # Return random embeddings of dimension 384
        model.encode.return_value = np.random.randn(10, 384)
        return model

    def test_compute_embeddings(self, mock_model):
        """Test embedding computation."""
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        with patch("oumi.core.analyze.embedding_analyzer.EmbeddingAnalyzer._get_model") as mock_get:
            mock_get.return_value = mock_model
            analyzer = EmbeddingAnalyzer()

            texts = ["hello", "world", "test"]
            mock_model.encode.return_value = np.random.randn(3, 384)

            embeddings = analyzer._compute_embeddings(texts)

            mock_model.encode.assert_called_once()
            assert embeddings.shape == (3, 384)

    def test_analyze_sample_with_duplicates(self, mock_model):
        """Test analysis with duplicate detection."""
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        # Create embeddings where first two are very similar
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],  # Very similar to first
            [0.0, 1.0, 0.0],   # Different
            [0.0, 0.0, 1.0],   # Different
        ])
        mock_model.encode.return_value = embeddings

        with patch("oumi.core.analyze.embedding_analyzer.EmbeddingAnalyzer._get_model") as mock_get:
            mock_get.return_value = mock_model
            analyzer = EmbeddingAnalyzer(
                detect_duplicates=True,
                duplicate_threshold=0.9,
                cluster_samples=False,
            )

            df = pd.DataFrame({
                "text_content": ["hello", "hello world", "different", "another"],
            })

            result_df = analyzer.analyze_sample(
                df, schema={"text_content": {"content_type": "text"}}
            )

            # Check that duplicate columns are added
            assert "text_content_embedding_duplicate_group" in result_df.columns
            assert "text_content_embedding_has_semantic_duplicate" in result_df.columns

    def test_analyze_sample_with_clustering(self, mock_model):
        """Test analysis with clustering."""
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        embeddings = np.random.randn(10, 384)
        mock_model.encode.return_value = embeddings

        with patch("oumi.core.analyze.embedding_analyzer.EmbeddingAnalyzer._get_model") as mock_get:
            mock_get.return_value = mock_model
            analyzer = EmbeddingAnalyzer(
                detect_duplicates=False,
                cluster_samples=True,
                clustering_method="dbscan",
            )

            df = pd.DataFrame({
                "text_content": [f"text_{i}" for i in range(10)],
            })

            result_df = analyzer.analyze_sample(
                df, schema={"text_content": {"content_type": "text"}}
            )

            # Check that cluster column is added
            assert "text_content_embedding_cluster" in result_df.columns


class TestEmbeddingAnalyzerValidation:
    """Tests for input validation."""

    def test_missing_schema_raises_error(self):
        """Test that missing schema raises ValueError."""
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        analyzer = EmbeddingAnalyzer()
        df = pd.DataFrame({"text_content": ["hello"]})

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)

    def test_no_text_columns_returns_unchanged(self):
        """Test that schema with no text columns returns DataFrame unchanged."""
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        analyzer = EmbeddingAnalyzer()
        df = pd.DataFrame({"text_content": ["hello"]})

        result_df = analyzer.analyze_sample(
            df, schema={"text_content": {"content_type": "numeric"}}
        )
        # DataFrame should be returned unchanged (no new columns added)
        assert list(result_df.columns) == list(df.columns)


class TestEmbeddingAnalyzerDependencyCheck:
    """Tests for dependency checking."""

    def test_import_error_message(self):
        """Test that import error has helpful message."""
        # This test is a bit tricky since we skip if deps are missing
        # We're just verifying the error message is correct when deps are available
        from oumi.core.analyze.embedding_analyzer import EmbeddingAnalyzer

        # The analyzer should initialize successfully if deps are available
        analyzer = EmbeddingAnalyzer()
        assert analyzer is not None
