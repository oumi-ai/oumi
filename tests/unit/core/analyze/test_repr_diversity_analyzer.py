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

"""Tests for the ReprDiversityAnalyzer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType


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


class TestReprDiversityAnalyzerInit:
    """Tests for ReprDiversityAnalyzer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer()
        assert analyzer.model_name == "all-MiniLM-L6-v2"
        assert analyzer.k_neighbors == 5
        assert analyzer.diversity_threshold == 0.3
        assert analyzer.embed_field == "all"
        assert analyzer.batch_size == 32

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer(
            model_name="paraphrase-MiniLM-L6-v2",
            k_neighbors=10,
            diversity_threshold=0.5,
            embed_field="user",
            batch_size=64,
        )
        assert analyzer.model_name == "paraphrase-MiniLM-L6-v2"
        assert analyzer.k_neighbors == 10
        assert analyzer.diversity_threshold == 0.5
        assert analyzer.embed_field == "user"
        assert analyzer.batch_size == 64

    def test_init_invalid_embed_field(self):
        """Test that invalid embed_field raises error."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        with pytest.raises(ValueError, match="Invalid embed_field"):
            ReprDiversityAnalyzer(embed_field="invalid")


class TestReprDiversityAnalyzerWithMocks:
    """Tests using mocked embedding model."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock sentence transformer model."""
        model = MagicMock()
        # Return different embeddings for different texts
        # Use normalized embeddings so cosine similarity works correctly

        def mock_encode(texts, **kwargs):
            embeddings = []
            for i, text in enumerate(texts):
                # Create embeddings that vary based on text content
                if "similar" in text.lower():
                    # Similar texts get similar embeddings
                    base = np.array([1.0, 0.0, 0.0])
                elif "different" in text.lower():
                    # Different texts get orthogonal embeddings
                    base = np.array([0.0, 1.0, 0.0])
                else:
                    # Default embedding with some variation
                    base = np.array([0.5, 0.5, 0.0])
                # Add small variation based on index
                embedding = base + np.array([0.01 * i, -0.01 * i, 0.02 * i])
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            return np.array(embeddings)

        model.encode = mock_encode
        return model

    @pytest.fixture
    def sample_schema(self):
        """Create a sample schema for text fields."""
        return {
            "text_content": {"content_type": ContentType.TEXT},
            "role": {"content_type": ContentType.CATEGORICAL},
        }

    def test_analyze_sample_basic(self, mock_model, sample_schema):
        """Test basic diversity analysis."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer(
            k_neighbors=2,
            diversity_threshold=0.3,
            show_progress_bar=False,
        )

        # Mock the model loading
        with patch.object(analyzer, "_get_model", return_value=mock_model):
            df = pd.DataFrame(
                {
                    "text_content": [
                        "This is similar text 1",
                        "This is similar text 2",
                        "This is different text",
                        "Another similar text",
                    ],
                    "role": ["user", "user", "assistant", "user"],
                }
            )

            result_df = analyzer.analyze_sample(df, sample_schema)

            # Check that output columns were added
            assert "text_content_repr_diversity_nn_distance" in result_df.columns
            assert "text_content_repr_diversity_score" in result_df.columns
            assert "text_content_repr_diversity_is_redundant" in result_df.columns
            assert "text_content_repr_diversity_percentile" in result_df.columns

            # Check that values are reasonable
            assert all(
                result_df["text_content_repr_diversity_nn_distance"].notna()
            )
            assert all(result_df["text_content_repr_diversity_score"].notna())
            assert all(result_df["text_content_repr_diversity_is_redundant"].notna())

    def test_analyze_sample_role_filter(self, mock_model, sample_schema):
        """Test diversity analysis with role filtering."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer(
            k_neighbors=1,
            embed_field="user",
            show_progress_bar=False,
        )

        with patch.object(analyzer, "_get_model", return_value=mock_model):
            df = pd.DataFrame(
                {
                    "text_content": [
                        "User message 1",
                        "Assistant response 1",
                        "User message 2",
                        "Assistant response 2",
                    ],
                    "role": ["user", "assistant", "user", "assistant"],
                }
            )

            result_df = analyzer.analyze_sample(df, sample_schema)

            # Only user messages should have diversity scores
            assert pd.notna(result_df.loc[0, "text_content_repr_diversity_score"])
            assert pd.isna(result_df.loc[1, "text_content_repr_diversity_score"])
            assert pd.notna(result_df.loc[2, "text_content_repr_diversity_score"])
            assert pd.isna(result_df.loc[3, "text_content_repr_diversity_score"])

    def test_analyze_sample_no_schema_raises(self):
        """Test that missing schema raises error."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer()
        df = pd.DataFrame({"text_content": ["Text 1", "Text 2"]})

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)

    def test_compute_diversity_scores(self, mock_model):
        """Test diversity score computation."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer(
            k_neighbors=2,
            show_progress_bar=False,
        )

        # Create embeddings where some are similar and some are different
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Sample 0
                [0.99, 0.1, 0.0],  # Sample 1 - similar to 0
                [0.0, 1.0, 0.0],  # Sample 2 - different
                [0.0, 0.0, 1.0],  # Sample 3 - different
            ]
        )
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        nn_distances, diversity_scores = analyzer._compute_diversity_scores(embeddings)

        # Sample 0 and 1 are similar, should have low nn_distance
        assert nn_distances[0] < nn_distances[2]
        assert nn_distances[1] < nn_distances[3]

        # All scores should be between 0 and 2 (max cosine distance)
        assert all(0 <= d <= 2 for d in nn_distances)
        assert all(0 <= s <= 2 for s in diversity_scores)

    def test_compute_dataset_metrics(self, mock_model, sample_schema):
        """Test that dataset metrics are computed."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer(
            k_neighbors=2,
            diversity_threshold=0.3,
            show_progress_bar=False,
        )

        with patch.object(analyzer, "_get_model", return_value=mock_model):
            df = pd.DataFrame(
                {
                    "text_content": [
                        "Text 1",
                        "Text 2",
                        "Text 3",
                    ],
                }
            )

            analyzer.analyze_sample(df, sample_schema)
            metrics = analyzer.compute_dataset_metrics(df, sample_schema)

            assert "text_content" in metrics
            assert "total_samples" in metrics["text_content"]
            assert "redundant_samples" in metrics["text_content"]
            assert "mean_diversity_score" in metrics["text_content"]

    def test_store_embeddings(self, mock_model, sample_schema):
        """Test that embeddings are stored when requested."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer(
            store_embeddings=True,
            show_progress_bar=False,
        )

        with patch.object(analyzer, "_get_model", return_value=mock_model):
            df = pd.DataFrame(
                {
                    "text_content": ["Text 1", "Text 2"],
                }
            )

            result_df = analyzer.analyze_sample(df, sample_schema)

            assert "text_content_repr_diversity_embedding" in result_df.columns
            # Check that embeddings are lists
            assert isinstance(
                result_df.loc[0, "text_content_repr_diversity_embedding"], list
            )


class TestReprDiversityAnalyzerEdgeCases:
    """Test edge cases for ReprDiversityAnalyzer."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock sentence transformer model."""
        model = MagicMock()

        def mock_encode(texts, **kwargs):
            embeddings = np.random.randn(len(texts), 3)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings

        model.encode = mock_encode
        return model

    @pytest.fixture
    def sample_schema(self):
        return {"text_content": {"content_type": ContentType.TEXT}}

    def test_single_sample(self, mock_model, sample_schema):
        """Test with only one sample."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer(show_progress_bar=False)

        with patch.object(analyzer, "_get_model", return_value=mock_model):
            df = pd.DataFrame({"text_content": ["Only one sample"]})

            # Should handle gracefully (not enough samples for KNN)
            result_df = analyzer.analyze_sample(df, sample_schema)

            # Should return original df without diversity columns added
            # or with None values
            assert len(result_df) == 1

    def test_k_neighbors_larger_than_samples(self, mock_model, sample_schema):
        """Test when k_neighbors is larger than number of samples."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer(
            k_neighbors=10,  # More than sample count
            show_progress_bar=False,
        )

        with patch.object(analyzer, "_get_model", return_value=mock_model):
            df = pd.DataFrame(
                {
                    "text_content": ["Text 1", "Text 2", "Text 3"],
                }
            )

            result_df = analyzer.analyze_sample(df, sample_schema)

            # Should adapt k to available samples
            assert "text_content_repr_diversity_score" in result_df.columns

    def test_empty_dataframe(self, mock_model, sample_schema):
        """Test with empty DataFrame."""
        from oumi.core.analyze.repr_diversity_analyzer import ReprDiversityAnalyzer

        analyzer = ReprDiversityAnalyzer(show_progress_bar=False)

        with patch.object(analyzer, "_get_model", return_value=mock_model):
            df = pd.DataFrame({"text_content": []})

            result_df = analyzer.analyze_sample(df, sample_schema)
            assert len(result_df) == 0
