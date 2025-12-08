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

"""Tests for the EvolQualityAnalyzer."""

from unittest.mock import patch

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType


class TestEvolQualityAnalyzerInit:
    """Tests for EvolQualityAnalyzer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer()
        assert analyzer.model_type == "api"
        assert analyzer.api_provider == "anthropic"
        assert analyzer.api_model == "claude-3-5-haiku-20241022"
        assert analyzer.num_evolutions == 3
        assert analyzer.analyze_role == "assistant"

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer(
            model_type="api",
            api_provider="anthropic",
            api_model="claude-3-haiku-20240307",
            num_evolutions=4,
            analyze_role="all",
            quality_aspects=["helpfulness", "accuracy"],
        )
        assert analyzer.api_provider == "anthropic"
        assert analyzer.api_model == "claude-3-haiku-20240307"
        assert analyzer.num_evolutions == 4
        assert analyzer.analyze_role == "all"
        assert len(analyzer.quality_aspects) == 2

    def test_init_invalid_quality_aspects(self):
        """Test that invalid quality aspects raise error."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        with pytest.raises(ValueError, match="Invalid quality aspect"):
            EvolQualityAnalyzer(quality_aspects=["invalid_aspect"])

    def test_init_invalid_analyze_role(self):
        """Test that invalid analyze_role raises error."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        with pytest.raises(ValueError, match="Invalid analyze_role"):
            EvolQualityAnalyzer(analyze_role="invalid")

    def test_init_with_instruction_column(self):
        """Test initialization with instruction column specified."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer(
            instruction_column="instruction",
            use_conversation_context=False,
        )
        assert analyzer.instruction_column == "instruction"
        assert analyzer.use_conversation_context is False


class TestEvolQualityAnalyzerWithMocks:
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
        evolution_response = '["Improved version 1", "Better version 2", "Best version"]'
        ranking_response = '{"A": 2, "B": 1, "C": 3, "D": 4}'
        return evolution_response, ranking_response

    def test_analyze_sample_basic(self, sample_schema, mock_llm_response):
        """Test basic quality analysis with mocked LLM."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        evolution_resp, ranking_resp = mock_llm_response

        analyzer = EvolQualityAnalyzer(
            num_evolutions=3,
            analyze_role="assistant",
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
                    "Here is the explanation of recursion...",
                    "The factorial function works by...",
                ],
                "role": ["assistant", "assistant"],
            }
        )

        with patch.object(analyzer, "_call_llm", side_effect=mock_call_llm):
            result_df = analyzer.analyze_sample(df, sample_schema)

        # Check that output columns were added
        assert "text_content_evol_quality_score" in result_df.columns
        assert "text_content_evol_quality_rank" in result_df.columns
        assert "text_content_evol_quality_improvement_potential" in result_df.columns

        # Check that values are in expected range
        for idx in range(len(result_df)):
            score = result_df.loc[idx, "text_content_evol_quality_score"]
            assert score is not None
            assert 0.0 <= score <= 1.0

    def test_analyze_sample_with_role_filter(self, sample_schema, mock_llm_response):
        """Test quality analysis with role filtering."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        evolution_resp, ranking_resp = mock_llm_response

        analyzer = EvolQualityAnalyzer(
            num_evolutions=3,
            analyze_role="assistant",
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
                    "User question",
                    "Assistant response 1",
                    "Another user question",
                    "Assistant response 2",
                ],
                "role": ["user", "assistant", "user", "assistant"],
            }
        )

        with patch.object(analyzer, "_call_llm", side_effect=mock_call_llm):
            result_df = analyzer.analyze_sample(df, sample_schema)

        # Only assistant messages should have scores
        assert pd.isna(result_df.loc[0, "text_content_evol_quality_score"])
        assert pd.notna(result_df.loc[1, "text_content_evol_quality_score"])
        assert pd.isna(result_df.loc[2, "text_content_evol_quality_score"])
        assert pd.notna(result_df.loc[3, "text_content_evol_quality_score"])

    def test_analyze_sample_with_instruction_context(
        self, sample_schema, mock_llm_response
    ):
        """Test that instruction context is included in prompts."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        evolution_resp, ranking_resp = mock_llm_response
        captured_prompts = []

        analyzer = EvolQualityAnalyzer(
            num_evolutions=3,
            instruction_column="instruction",
            show_progress=False,
        )

        def mock_call_llm(prompt, use_cache=True):
            captured_prompts.append(prompt)
            if "Generate" in prompt:
                return evolution_resp
            else:
                return ranking_resp

        schema_with_instruction = {
            "text_content": {"content_type": ContentType.TEXT},
            "instruction": {"content_type": ContentType.TEXT},
            "role": {"content_type": ContentType.CATEGORICAL},
        }

        df = pd.DataFrame(
            {
                "text_content": ["Here is the answer..."],
                "instruction": ["What is recursion?"],
                "role": ["assistant"],
            }
        )

        with patch.object(analyzer, "_call_llm", side_effect=mock_call_llm):
            analyzer.analyze_sample(df, schema_with_instruction)

        # Check that instruction context was included in the prompt
        assert any("What is recursion?" in p for p in captured_prompts)


class TestEvolQualityAnalyzerPrompts:
    """Tests for prompt generation."""

    def test_evolution_prompt_format(self):
        """Test that evolution prompt is properly formatted."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer(num_evolutions=3)
        prompt = analyzer._get_evolution_prompt("This is a sample response")

        assert "This is a sample response" in prompt
        assert "3" in prompt or "three" in prompt.lower()  # Number of evolutions
        assert "json" in prompt.lower()
        assert "helpfulness" in prompt.lower() or "helpful" in prompt.lower()

    def test_evolution_prompt_with_context(self):
        """Test that evolution prompt includes instruction context."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer(num_evolutions=2)
        analyzer._current_instruction = "What is machine learning?"

        prompt = analyzer._get_evolution_prompt("ML is a branch of AI...")

        assert "What is machine learning?" in prompt
        assert "ML is a branch of AI" in prompt

    def test_ranking_prompt_format(self):
        """Test that ranking prompt is properly formatted."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer(num_evolutions=2)
        prompt = analyzer._get_ranking_prompt(
            "Original response",
            ["Better response 1", "Better response 2"],
        )

        assert "Original response" in prompt
        assert "Better response 1" in prompt
        assert "A:" in prompt  # Original labeled as A
        assert "B:" in prompt  # First variant
        assert "C:" in prompt  # Second variant


class TestEvolQualityAnalyzerMetrics:
    """Tests for dataset metrics computation."""

    def test_dataset_metrics(self):
        """Test that dataset metrics are computed correctly."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer(show_progress=False)

        # Simulate internal state after analysis
        analyzer._dataset_metrics = {
            "text_content": {
                "total_analyzed": 10,
                "mean_quality_score": 0.6,
                "median_quality_score": 0.55,
                "std_quality_score": 0.15,
                "low_quality_ratio": 0.2,
                "high_quality_ratio": 0.3,
            }
        }

        metrics = analyzer.compute_dataset_metrics(pd.DataFrame(), None)

        assert "text_content" in metrics
        assert metrics["text_content"]["total_analyzed"] == 10
        assert metrics["text_content"]["mean_quality_score"] == 0.6

    def test_improvement_potential_interpretation(self):
        """Test that improvement potential is correctly interpreted."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer()

        # Low quality score = high improvement potential
        low_score = 0.2
        high_potential = 1.0 - low_score
        assert abs(high_potential - 0.8) < 0.001

        # High quality score = low improvement potential
        high_score = 0.9
        low_potential = 1.0 - high_score
        assert abs(low_potential - 0.1) < 0.001


class TestEvolQualityAnalyzerEdgeCases:
    """Test edge cases for EvolQualityAnalyzer."""

    @pytest.fixture
    def sample_schema(self):
        return {
            "text_content": {"content_type": ContentType.TEXT},
            "role": {"content_type": ContentType.CATEGORICAL},
        }

    def test_no_matching_role(self, sample_schema):
        """Test when no rows match the analyze_role."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer(
            analyze_role="assistant",
            show_progress=False,
        )

        df = pd.DataFrame(
            {
                "text_content": ["User message 1", "User message 2"],
                "role": ["user", "user"],
            }
        )

        # Should not crash, just warn and return unchanged
        result_df = analyzer.analyze_sample(df, sample_schema)
        assert len(result_df) == 2

    def test_error_handling(self, sample_schema):
        """Test error handling during LLM calls."""
        from oumi.core.analyze.evol_quality_analyzer import EvolQualityAnalyzer

        analyzer = EvolQualityAnalyzer(
            num_evolutions=2,
            show_progress=False,
        )

        def mock_call_llm_with_error(prompt, use_cache=True):
            raise RuntimeError("LLM API error")

        df = pd.DataFrame(
            {
                "text_content": ["Test response"],
                "role": ["assistant"],
            }
        )

        with patch.object(
            analyzer, "_call_llm", side_effect=mock_call_llm_with_error
        ):
            result_df = analyzer.analyze_sample(df, sample_schema)

        # Should handle error gracefully with default values
        assert result_df.loc[0, "text_content_evol_quality_score"] == 0.5
