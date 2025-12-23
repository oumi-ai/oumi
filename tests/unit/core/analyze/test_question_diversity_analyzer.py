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

"""Tests for the QuestionDiversityAnalyzer."""

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
    reason="sentence-transformers or sklearn not installed",
)


def _create_test_df(texts: list[str], roles: list[str] = None) -> pd.DataFrame:
    """Create a test DataFrame with text content and optional roles."""
    data = {"text_content": texts}
    if roles:
        data["role"] = roles
    return pd.DataFrame(data)


def _get_schema(with_role: bool = False) -> dict:
    """Get the standard schema for testing."""
    schema = {"text_content": {"content_type": "text"}}
    if with_role:
        schema["role"] = {"content_type": "categorical"}
    return schema


class TestQuestionDiversityAnalyzerInit:
    """Tests for QuestionDiversityAnalyzer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        assert analyzer.cluster_questions is True
        assert analyzer.clustering_method == "dbscan"
        assert analyzer.eps == 0.15
        assert analyzer.min_samples == 2
        assert analyzer.model_name == "all-MiniLM-L6-v2"
        assert analyzer.compute_entropy is True
        assert analyzer.compute_concentration is True
        assert analyzer.concentration_threshold == 0.5

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer(
            clustering_method="kmeans",
            n_clusters=5,
            eps=0.5,
            concentration_threshold=0.4,
            compute_entropy=False,
        )

        assert analyzer.clustering_method == "kmeans"
        assert analyzer.n_clusters == 5
        assert analyzer.eps == 0.5
        assert analyzer.concentration_threshold == 0.4
        assert analyzer.compute_entropy is False

    def test_init_invalid_clustering_method(self):
        """Test that invalid clustering method raises error."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        with pytest.raises(ValueError, match="Unknown clustering method"):
            QuestionDiversityAnalyzer(clustering_method="invalid")

    def test_init_invalid_concentration_threshold(self):
        """Test that invalid concentration threshold raises error."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        with pytest.raises(ValueError, match="concentration_threshold must be in range"):
            QuestionDiversityAnalyzer(concentration_threshold=1.5)

        with pytest.raises(ValueError, match="concentration_threshold must be in range"):
            QuestionDiversityAnalyzer(concentration_threshold=-0.1)


class TestEntropyCalculation:
    """Tests for Shannon entropy calculation."""

    def test_uniform_distribution_high_entropy(self):
        """Test that uniform distribution has high entropy."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        # 4 clusters with equal sizes
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        entropy = analyzer._compute_shannon_entropy(labels)

        # Max entropy for 4 clusters is log2(4) = 2.0
        assert entropy == pytest.approx(2.0, abs=0.01)

    def test_concentrated_distribution_low_entropy(self):
        """Test that concentrated distribution has low entropy."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        # 1 dominant cluster
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        entropy = analyzer._compute_shannon_entropy(labels)

        # Should be low entropy (close to 0)
        assert entropy < 1.0

    def test_single_cluster_zero_entropy(self):
        """Test that single cluster has zero entropy."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        labels = np.array([0, 0, 0, 0, 0])
        entropy = analyzer._compute_shannon_entropy(labels)

        assert entropy == pytest.approx(0.0, abs=0.01)

    def test_noise_labels_excluded(self):
        """Test that DBSCAN noise labels (-1) are excluded from entropy."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        # -1 is noise in DBSCAN, should be ignored
        labels = np.array([0, 0, 1, 1, -1, -1, -1])
        entropy = analyzer._compute_shannon_entropy(labels)

        # Should be entropy of [0,0,1,1] = log2(2) = 1.0
        assert entropy == pytest.approx(1.0, abs=0.01)


class TestGiniCalculation:
    """Tests for Gini coefficient calculation."""

    def test_uniform_distribution_zero_gini(self):
        """Test that uniform distribution has Gini close to 0."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        # 4 clusters with equal sizes
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        gini = analyzer._compute_gini_coefficient(labels)

        # Should be close to 0 for uniform
        assert gini < 0.1

    def test_concentrated_distribution_high_gini(self):
        """Test that concentrated distribution has higher Gini than uniform."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        # Very unequal clusters
        concentrated_labels = np.array([0] * 99 + [1])  # 99:1 distribution
        uniform_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # 2:2:2:2 distribution

        concentrated_gini = analyzer._compute_gini_coefficient(concentrated_labels)
        uniform_gini = analyzer._compute_gini_coefficient(uniform_labels)

        # Concentrated should have higher Gini than uniform
        assert concentrated_gini > uniform_gini
        # And concentrated should be notably above 0
        assert concentrated_gini > 0.3

    def test_single_cluster_max_gini(self):
        """Test that single cluster has Gini of 1."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        labels = np.array([0, 0, 0, 0, 0])
        gini = analyzer._compute_gini_coefficient(labels)

        assert gini == 1.0


class TestDiversityRating:
    """Tests for diversity rating calculation."""

    def test_low_diversity_rating(self):
        """Test low diversity rating."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        # Low entropy relative to max
        rating = analyzer._get_diversity_rating(entropy=0.5, n_clusters=8)
        assert rating == "low"

    def test_medium_diversity_rating(self):
        """Test medium diversity rating."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        # Medium entropy (0.5-0.8 normalized)
        rating = analyzer._get_diversity_rating(entropy=2.0, n_clusters=8)
        assert rating == "medium"

    def test_high_diversity_rating(self):
        """Test high diversity rating."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        # High entropy (>0.8 normalized)
        rating = analyzer._get_diversity_rating(entropy=2.8, n_clusters=8)
        assert rating == "high"

    def test_single_cluster_low_rating(self):
        """Test that single cluster returns low rating."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        rating = analyzer._get_diversity_rating(entropy=0.0, n_clusters=1)
        assert rating == "low"


class TestAnalyzeSampleWithMocks:
    """Tests using mocked embedding model."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock SentenceTransformer model."""
        model = MagicMock()
        return model

    def test_analyze_sample_basic(self, mock_model):
        """Test basic sample analysis with clustering."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        # Create embeddings that form 2 clusters
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ])
        mock_model.encode.return_value = embeddings

        with patch(
            "oumi.core.analyze.embedding_base_analyzer."
            "EmbeddingBasedAnalyzer._get_model"
        ) as mock_get:
            mock_get.return_value = mock_model
            analyzer = QuestionDiversityAnalyzer(
                clustering_method="kmeans",
                n_clusters=2,
            )

            df = _create_test_df([
                "Write a poem",
                "Create a story",
                "Calculate sum",
                "Compute average",
            ])

            result_df, generated_schema = analyzer.analyze_sample(df, _get_schema())

            # Check columns are added
            assert "text_content_question_diversity_cluster_id" in result_df.columns
            assert "text_content_question_diversity_cluster_size" in result_df.columns

            # Check generated schema
            assert "text_content_question_diversity_cluster_id" in generated_schema
            assert "text_content_question_diversity_cluster_size" in generated_schema

    def test_analyze_sample_with_roles(self, mock_model):
        """Test that only user messages are analyzed when role column exists."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        # Only 2 embeddings for user messages
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
        ])
        mock_model.encode.return_value = embeddings

        with patch(
            "oumi.core.analyze.embedding_base_analyzer."
            "EmbeddingBasedAnalyzer._get_model"
        ) as mock_get:
            mock_get.return_value = mock_model
            analyzer = QuestionDiversityAnalyzer(
                clustering_method="kmeans",
                n_clusters=2,
            )

            df = _create_test_df(
                texts=["User question 1", "Response 1", "User question 2", "Response 2"],
                roles=["user", "assistant", "user", "assistant"],
            )

            result_df, _ = analyzer.analyze_sample(df, _get_schema(with_role=True))

            # User messages should have cluster IDs
            cluster_col = "text_content_question_diversity_cluster_id"
            assert pd.notna(result_df.iloc[0][cluster_col])  # user
            assert pd.isna(result_df.iloc[1][cluster_col])  # assistant
            assert pd.notna(result_df.iloc[2][cluster_col])  # user
            assert pd.isna(result_df.iloc[3][cluster_col])  # assistant

    def test_analyze_sample_flags_concentrated_clusters(self, mock_model):
        """Test that concentrated clusters are flagged."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        # Embeddings that will form one big cluster and one small
        embeddings = np.array([
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
        ])
        mock_model.encode.return_value = embeddings

        with patch(
            "oumi.core.analyze.embedding_base_analyzer."
            "EmbeddingBasedAnalyzer._get_model"
        ) as mock_get:
            mock_get.return_value = mock_model
            analyzer = QuestionDiversityAnalyzer(
                clustering_method="kmeans",
                n_clusters=2,
                concentration_threshold=0.5,  # >50% in cluster = concentrated
            )

            df = _create_test_df(["q1", "q2", "q3", "q4"])

            result_df, generated_schema = analyzer.analyze_sample(df, _get_schema())

            # Check concentrated flag column exists
            assert "text_content_question_diversity_is_concentrated" in result_df.columns
            assert "text_content_question_diversity_is_concentrated" in generated_schema

    def test_compute_dataset_metrics(self, mock_model):
        """Test dataset-level metrics computation."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ])
        mock_model.encode.return_value = embeddings

        with patch(
            "oumi.core.analyze.embedding_base_analyzer."
            "EmbeddingBasedAnalyzer._get_model"
        ) as mock_get:
            mock_get.return_value = mock_model
            analyzer = QuestionDiversityAnalyzer(
                clustering_method="kmeans",
                n_clusters=2,
            )

            df = _create_test_df(["q1", "q2", "q3", "q4"])
            analyzer.analyze_sample(df, _get_schema())

            metrics = analyzer.compute_dataset_metrics(df, _get_schema())

            # Check metrics are computed
            assert "text_content" in metrics
            assert "num_question_clusters" in metrics["text_content"]
            assert "question_entropy" in metrics["text_content"]
            assert "question_gini" in metrics["text_content"]
            assert "largest_cluster_ratio" in metrics["text_content"]
            assert "diversity_rating" in metrics["text_content"]


    def test_analyze_sample_returns_tuple(self, mock_model):
        """Test that analyze_sample returns a tuple of (DataFrame, dict)."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        mock_model.encode.return_value = np.random.randn(3, 2)

        with patch(
            "oumi.core.analyze.embedding_base_analyzer."
            "EmbeddingBasedAnalyzer._get_model"
        ) as mock_get:
            mock_get.return_value = mock_model
            analyzer = QuestionDiversityAnalyzer(
                clustering_method="kmeans",
                n_clusters=2,
            )

            df = _create_test_df(["q1", "q2", "q3"])
            result = analyzer.analyze_sample(df, _get_schema())

            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], pd.DataFrame)
            assert isinstance(result[1], dict)


class TestValidation:
    """Tests for input validation."""

    def test_missing_schema_raises_error(self):
        """Test that missing schema raises ValueError."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()
        df = _create_test_df(["test"])

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)

    def test_no_text_columns_returns_unchanged(self):
        """Test that schema with no text columns returns DataFrame unchanged."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()
        df = _create_test_df(["test"])

        result_df, generated_schema = analyzer.analyze_sample(
            df, schema={"text_content": {"content_type": "numeric"}}
        )

        # DataFrame should be returned unchanged
        assert list(result_df.columns) == list(df.columns)
        assert generated_schema == {}


class TestClusterSizeComputation:
    """Tests for cluster size computation."""

    def test_compute_cluster_sizes(self):
        """Test cluster size computation."""
        from oumi.core.analyze.question_diversity_analyzer import (
            QuestionDiversityAnalyzer,
        )

        analyzer = QuestionDiversityAnalyzer()

        labels = np.array([0, 0, 0, 1, 1, 2, -1, -1])
        sizes = analyzer._compute_cluster_sizes(labels)

        assert sizes[0] == 3
        assert sizes[1] == 2
        assert sizes[2] == 1
        assert sizes[-1] == 2  # noise
