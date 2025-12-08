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

"""Tests for the EvolComplexityAnalyzer."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType


class TestEvolComplexityAnalyzerInit:
    """Tests for EvolComplexityAnalyzer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        analyzer = EvolComplexityAnalyzer()
        assert analyzer.model_type == "api"
        assert analyzer.api_provider == "anthropic"
        assert analyzer.api_model == "claude-3-5-haiku-20241022"
        assert analyzer.num_evolutions == 3
        assert analyzer.analyze_role == "user"

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        analyzer = EvolComplexityAnalyzer(
            model_type="api",
            api_provider="anthropic",
            api_model="claude-3-haiku-20240307",
            num_evolutions=5,
            analyze_role="all",
            evolution_operators=["add_constraints", "require_reasoning"],
        )
        assert analyzer.api_provider == "anthropic"
        assert analyzer.api_model == "claude-3-haiku-20240307"
        assert analyzer.num_evolutions == 5
        assert analyzer.analyze_role == "all"
        assert len(analyzer.evolution_operators) == 2

    def test_init_invalid_model_type(self):
        """Test that invalid model_type raises error."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        with pytest.raises(ValueError, match="Invalid model_type"):
            EvolComplexityAnalyzer(model_type="invalid")

    def test_init_invalid_api_provider(self):
        """Test that invalid api_provider raises error."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        with pytest.raises(ValueError, match="Invalid api_provider"):
            EvolComplexityAnalyzer(model_type="api", api_provider="invalid")

    def test_init_local_requires_model(self):
        """Test that local model_type requires local_model."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        with pytest.raises(ValueError, match="local_model must be specified"):
            EvolComplexityAnalyzer(model_type="local", local_model=None)

    def test_init_invalid_num_evolutions(self):
        """Test that invalid num_evolutions raises error."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        with pytest.raises(ValueError, match="num_evolutions must be 1-6"):
            EvolComplexityAnalyzer(num_evolutions=10)

    def test_init_invalid_evolution_operators(self):
        """Test that invalid evolution operators raise error."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        with pytest.raises(ValueError, match="Invalid evolution operator"):
            EvolComplexityAnalyzer(evolution_operators=["invalid_operator"])

    def test_init_invalid_analyze_role(self):
        """Test that invalid analyze_role raises error."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        with pytest.raises(ValueError, match="Invalid analyze_role"):
            EvolComplexityAnalyzer(analyze_role="invalid")


class TestEvolComplexityAnalyzerWithMocks:
    """Tests using mocked LLM inference."""

    @pytest.fixture
    def sample_schema(self):
        """Create a sample schema for text fields."""
        return {
            "text_content": {"content_type": ContentType.TEXT},
            "role": {"content_type": ContentType.CATEGORICAL},
        }

    @pytest.fixture
    def mock_llm_response(self):
        """Create mock LLM responses."""
        evolution_response = '["More complex version 1", "More complex version 2", "Most complex version"]'
        ranking_response = '{"A": 1, "B": 2, "C": 3, "D": 4}'
        return evolution_response, ranking_response

    def test_analyze_sample_basic(self, sample_schema, mock_llm_response):
        """Test basic complexity analysis with mocked LLM."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        evolution_resp, ranking_resp = mock_llm_response

        analyzer = EvolComplexityAnalyzer(
            num_evolutions=3,
            show_progress=False,
        )

        # Mock the _call_llm method
        call_count = [0]

        def mock_call_llm(prompt, use_cache=True):
            call_count[0] += 1
            if "Generate" in prompt:
                return evolution_resp
            else:
                return ranking_resp

        df = pd.DataFrame(
            {
                "text_content": [
                    "Write a function to calculate factorial",
                    "Explain recursion",
                ],
                "role": ["user", "user"],
            }
        )

        with patch.object(analyzer, "_call_llm", side_effect=mock_call_llm):
            result_df = analyzer.analyze_sample(df, sample_schema)

        # Check that output columns were added
        assert "text_content_evol_complexity_score" in result_df.columns
        assert "text_content_evol_complexity_rank" in result_df.columns
        assert "text_content_evol_complexity_headroom" in result_df.columns

        # Check that values are in expected range
        for idx in range(len(result_df)):
            score = result_df.loc[idx, "text_content_evol_complexity_score"]
            assert pd.notna(score)
            assert 0.0 <= score <= 1.0

            rank = result_df.loc[idx, "text_content_evol_complexity_rank"]
            assert pd.notna(rank)
            assert int(rank) == rank  # Check it's an integer value

    def test_analyze_sample_with_role_filter(self, sample_schema, mock_llm_response):
        """Test complexity analysis with role filtering."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        evolution_resp, ranking_resp = mock_llm_response

        analyzer = EvolComplexityAnalyzer(
            num_evolutions=3,
            analyze_role="user",
            show_progress=False,
        )

        def mock_call_llm(prompt, use_cache=True):
            if "Generate" in prompt:
                return evolution_resp
            else:
                return ranking_resp

        df = pd.DataFrame(
            {
                "text_content": [
                    "User instruction",
                    "Assistant response",
                    "Another user instruction",
                ],
                "role": ["user", "assistant", "user"],
            }
        )

        with patch.object(analyzer, "_call_llm", side_effect=mock_call_llm):
            result_df = analyzer.analyze_sample(df, sample_schema)

        # Only user messages should have scores
        assert pd.notna(result_df.loc[0, "text_content_evol_complexity_score"])
        assert pd.isna(result_df.loc[1, "text_content_evol_complexity_score"])
        assert pd.notna(result_df.loc[2, "text_content_evol_complexity_score"])

    def test_no_schema_raises(self):
        """Test that missing schema raises error."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        analyzer = EvolComplexityAnalyzer()
        df = pd.DataFrame({"text_content": ["Test"]})

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)


class TestEvolComplexityAnalyzerPrompts:
    """Tests for prompt generation."""

    def test_evolution_prompt_format(self):
        """Test that evolution prompt is properly formatted."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        analyzer = EvolComplexityAnalyzer(num_evolutions=3)
        prompt = analyzer._get_evolution_prompt("Write a hello world program")

        assert "Write a hello world program" in prompt
        assert "3" in prompt or "three" in prompt.lower()  # Number of evolutions
        assert "json" in prompt.lower()

    def test_ranking_prompt_format(self):
        """Test that ranking prompt is properly formatted."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        analyzer = EvolComplexityAnalyzer(num_evolutions=2)
        prompt = analyzer._get_ranking_prompt(
            "Original instruction",
            ["Complex version 1", "Complex version 2"],
        )

        assert "Original instruction" in prompt
        assert "Complex version 1" in prompt
        assert "A:" in prompt  # Original labeled as A
        assert "B:" in prompt  # First variant
        assert "C:" in prompt  # Second variant


class TestEvolComplexityAnalyzerScoring:
    """Tests for score computation."""

    def test_normalized_score_computation(self):
        """Test that scores are normalized correctly."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        analyzer = EvolComplexityAnalyzer()

        # Rank 1 out of 4 should give score close to 0
        score_low = analyzer._compute_normalized_score(1, 4)
        assert score_low == 0.0

        # Rank 4 out of 4 should give score 1.0
        score_high = analyzer._compute_normalized_score(4, 4)
        assert score_high == 1.0

        # Middle rank should give middle score
        score_mid = analyzer._compute_normalized_score(2, 3)
        assert 0.0 < score_mid < 1.0

    def test_headroom_computation(self):
        """Test headroom (improvement potential) computation."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        analyzer = EvolComplexityAnalyzer()

        # If score is 0 (simplest), headroom should be 1
        assert 1.0 - 0.0 == 1.0

        # If score is 1 (most complex), headroom should be 0
        assert 1.0 - 1.0 == 0.0


class TestEvolComplexityAnalyzerMetrics:
    """Tests for dataset metrics computation."""

    def test_dataset_metrics(self, tmp_path):
        """Test that dataset metrics are computed correctly."""
        from oumi.core.analyze.evol_complexity_analyzer import EvolComplexityAnalyzer

        analyzer = EvolComplexityAnalyzer(show_progress=False)

        # Simulate internal state after analysis
        analyzer._dataset_metrics = {
            "text_content": {
                "total_analyzed": 10,
                "mean_complexity_score": 0.5,
                "median_complexity_score": 0.45,
                "std_complexity_score": 0.2,
                "low_complexity_ratio": 0.3,
                "high_complexity_ratio": 0.2,
            }
        }

        metrics = analyzer.compute_dataset_metrics(pd.DataFrame(), None)

        assert "text_content" in metrics
        assert metrics["text_content"]["total_analyzed"] == 10
        assert metrics["text_content"]["mean_complexity_score"] == 0.5
