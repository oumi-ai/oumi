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

"""Unit tests for the Health Score module."""

import pandas as pd
import pytest

from oumi.core.analyze.health_score import (
    DatasetHealthScore,
    HealthScoreCalculator,
    HealthScoreComponent,
)
from oumi.core.analyze.recommendations import (
    Recommendation,
    RecommendationCategory,
    RecommendationSeverity,
)


@pytest.fixture
def sample_message_df():
    """Create a sample message DataFrame."""
    return pd.DataFrame(
        {
            "text_content": [
                "Hello, how are you?",
                "I am doing well, thank you for asking!",
                "What is the weather like today?",
                "The weather is sunny and warm.",
                "Can you help me with my homework?",
                "Of course, I'd be happy to help!",
            ],
            "role": ["user", "assistant", "user", "assistant", "user", "assistant"],
            "text_content_length_char_count": [19, 40, 33, 30, 35, 33],
            "text_content_length_word_count": [4, 8, 6, 5, 7, 7],
            "text_content_diversity_type_token_ratio": [1.0, 0.875, 1.0, 0.8, 0.857, 0.857],
            "text_content_quality_quality_score": [0.9, 0.85, 0.88, 0.92, 0.87, 0.91],
            "text_content_quality_has_pii": [False, False, False, False, False, False],
            "text_content_quality_has_encoding_issues": [
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "text_content_quality_has_special_tokens": [
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "text_content_quality_has_high_repetition": [
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        }
    )


@pytest.fixture
def sample_conversation_df():
    """Create a sample conversation DataFrame."""
    return pd.DataFrame(
        {
            "conversation_id": ["conv1", "conv2", "conv3"],
            "num_messages": [2, 2, 2],
        }
    )


@pytest.fixture
def sample_analysis_summary():
    """Create a sample analysis summary."""
    return {
        "dataset_overview": {
            "dataset_name": "test_dataset",
            "total_conversations": 3,
            "total_messages": 6,
        },
        "message_level_summary": {},
        "conversation_level_summary": {},
    }


class TestHealthScoreComponent:
    """Tests for HealthScoreComponent."""

    def test_component_creation(self):
        """Test component creation."""
        component = HealthScoreComponent(
            name="Test",
            score=85.0,
            weight=0.2,
            description="Test component",
            details={"key": "value"},
        )
        assert component.name == "Test"
        assert component.score == 85.0
        assert component.weight == 0.2
        assert component.description == "Test component"
        assert component.details == {"key": "value"}


class TestDatasetHealthScore:
    """Tests for DatasetHealthScore."""

    def test_health_score_creation(self):
        """Test health score creation."""
        components = [
            HealthScoreComponent(
                name="Test",
                score=85.0,
                weight=0.2,
                description="Test",
            )
        ]
        health_score = DatasetHealthScore(
            overall=85.0,
            grade="B",
            components=components,
            recommendations_count=2,
            high_severity_count=0,
            summary="Test summary",
        )
        assert health_score.overall == 85.0
        assert health_score.grade == "B"
        assert len(health_score.components) == 1
        assert health_score.recommendations_count == 2
        assert health_score.high_severity_count == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        components = [
            HealthScoreComponent(
                name="Test",
                score=85.5,
                weight=0.2,
                description="Test",
                details={"key": "value"},
            )
        ]
        health_score = DatasetHealthScore(
            overall=85.5,
            grade="B",
            components=components,
            recommendations_count=2,
            high_severity_count=1,
            summary="Test summary",
        )

        result = health_score.to_dict()

        assert result["overall"] == 85.5
        assert result["grade"] == "B"
        assert len(result["components"]) == 1
        assert result["components"][0]["name"] == "Test"
        assert result["components"][0]["score"] == 85.5
        assert result["recommendations_count"] == 2
        assert result["high_severity_count"] == 1
        assert result["summary"] == "Test summary"


class TestHealthScoreCalculator:
    """Tests for HealthScoreCalculator."""

    def test_default_initialization(self):
        """Test default initialization."""
        calculator = HealthScoreCalculator()
        assert calculator.recommendation_penalty == 2.0
        assert calculator.high_severity_penalty == 5.0
        assert sum(calculator.weights.values()) == pytest.approx(1.0)

    def test_custom_weights(self):
        """Test custom weights."""
        weights = {
            "diversity": 0.3,
            "balance": 0.2,
            "quality": 0.2,
            "consistency": 0.2,
            "length_distribution": 0.1,
        }
        calculator = HealthScoreCalculator(weights=weights)
        assert calculator.weights == weights

    def test_invalid_weights_raises_error(self):
        """Test that invalid weights raise error."""
        weights = {
            "diversity": 0.5,
            "balance": 0.5,
            "quality": 0.5,
            "consistency": 0.5,
            "length_distribution": 0.5,
        }
        with pytest.raises(ValueError, match="must sum to 1.0"):
            HealthScoreCalculator(weights=weights)

    def test_calculate_health_score(
        self,
        sample_message_df,
        sample_conversation_df,
        sample_analysis_summary,
    ):
        """Test health score calculation."""
        calculator = HealthScoreCalculator()
        recommendations = []

        health_score = calculator.calculate_health_score(
            message_df=sample_message_df,
            conversation_df=sample_conversation_df,
            analysis_summary=sample_analysis_summary,
            recommendations=recommendations,
        )

        assert isinstance(health_score, DatasetHealthScore)
        assert 0 <= health_score.overall <= 100
        assert health_score.grade in ["A", "B", "C", "D", "F"]
        assert len(health_score.components) == 5
        assert health_score.recommendations_count == 0
        assert health_score.high_severity_count == 0

    def test_recommendations_penalty(
        self,
        sample_message_df,
        sample_conversation_df,
        sample_analysis_summary,
    ):
        """Test that recommendations reduce the score."""
        calculator = HealthScoreCalculator(
            recommendation_penalty=5.0,
            high_severity_penalty=10.0,
        )

        # Score without recommendations
        no_rec_score = calculator.calculate_health_score(
            message_df=sample_message_df,
            conversation_df=sample_conversation_df,
            analysis_summary=sample_analysis_summary,
            recommendations=[],
        )

        # Score with recommendations
        recommendations = [
            Recommendation(
                category=RecommendationCategory.WARNING,
                severity=RecommendationSeverity.HIGH,
                title="Test warning",
                description="Test description",
                affected_samples=10,
            ),
            Recommendation(
                category=RecommendationCategory.INSIGHT,
                severity=RecommendationSeverity.LOW,
                title="Test insight",
                description="Test description",
                affected_samples=5,
            ),
        ]
        with_rec_score = calculator.calculate_health_score(
            message_df=sample_message_df,
            conversation_df=sample_conversation_df,
            analysis_summary=sample_analysis_summary,
            recommendations=recommendations,
        )

        # Score with recommendations should be lower
        assert with_rec_score.overall < no_rec_score.overall
        assert with_rec_score.recommendations_count == 2
        assert with_rec_score.high_severity_count == 1

    def test_grade_thresholds(
        self,
        sample_message_df,
        sample_conversation_df,
        sample_analysis_summary,
    ):
        """Test grade threshold logic."""
        calculator = HealthScoreCalculator()

        # Test that grades are assigned correctly based on thresholds
        health_score = calculator.calculate_health_score(
            message_df=sample_message_df,
            conversation_df=sample_conversation_df,
            analysis_summary=sample_analysis_summary,
            recommendations=[],
        )

        # The grade should correspond to the overall score
        if health_score.overall >= 90:
            assert health_score.grade == "A"
        elif health_score.overall >= 80:
            assert health_score.grade == "B"
        elif health_score.overall >= 70:
            assert health_score.grade == "C"
        elif health_score.overall >= 60:
            assert health_score.grade == "D"
        else:
            assert health_score.grade == "F"

    def test_component_names(
        self,
        sample_message_df,
        sample_conversation_df,
        sample_analysis_summary,
    ):
        """Test that all expected components are present."""
        calculator = HealthScoreCalculator()

        health_score = calculator.calculate_health_score(
            message_df=sample_message_df,
            conversation_df=sample_conversation_df,
            analysis_summary=sample_analysis_summary,
            recommendations=[],
        )

        component_names = {c.name for c in health_score.components}
        expected_names = {
            "Diversity",
            "Balance",
            "Quality",
            "Consistency",
            "Length Distribution",
        }
        assert component_names == expected_names

    def test_summary_generation(
        self,
        sample_message_df,
        sample_conversation_df,
        sample_analysis_summary,
    ):
        """Test summary string generation."""
        calculator = HealthScoreCalculator()

        health_score = calculator.calculate_health_score(
            message_df=sample_message_df,
            conversation_df=sample_conversation_df,
            analysis_summary=sample_analysis_summary,
            recommendations=[],
        )

        assert isinstance(health_score.summary, str)
        assert len(health_score.summary) > 0
        assert health_score.grade in health_score.summary
