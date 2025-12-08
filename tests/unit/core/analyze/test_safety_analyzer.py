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

"""Unit tests for SafetyAnalyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.safety_analyzer import SafetyAnalyzer


@pytest.fixture
def schema():
    """Create a sample schema."""
    return {
        "text_content": {"content_type": ContentType.TEXT},
        "role": {"content_type": ContentType.CATEGORICAL},
    }


@pytest.fixture
def analyzer():
    """Create a SafetyAnalyzer instance."""
    return SafetyAnalyzer()


class TestSafetyAnalyzer:
    """Tests for SafetyAnalyzer."""

    def test_safe_content(self, analyzer, schema):
        """Test that safe content is classified correctly."""
        df = pd.DataFrame({
            "text_content": ["The weather is nice today. Let's go for a walk."],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert "text_content_safety_is_safe" in result.columns
        assert result["text_content_safety_is_safe"].iloc[0] == True
        assert result["text_content_safety_score"].iloc[0] >= 0.9

    def test_violence_detection(self, analyzer, schema):
        """Test detection of violent content."""
        df = pd.DataFrame({
            "text_content": ["How to murder and attack people with weapons"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        # Violence should be detected as a category
        assert "violence" in result["text_content_safety_categories"].iloc[0]
        # With single pattern match, score is still high but category flagged
        assert result["text_content_safety_score"].iloc[0] < 1.0

    def test_illegal_activity_detection(self, analyzer, schema):
        """Test detection of illegal activity content."""
        df = pd.DataFrame({
            "text_content": ["How to hack into the system and steal credit card numbers"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        # Illegal activity should be detected as a category
        assert "illegal" in result["text_content_safety_categories"].iloc[0]
        # Score should be reduced
        assert result["text_content_safety_score"].iloc[0] < 1.0

    def test_self_harm_detection(self, analyzer, schema):
        """Test detection of self-harm content."""
        df = pd.DataFrame({
            "text_content": ["Ways to kill myself or commit suicide"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        # Self-harm should be detected as a category
        assert "self_harm" in result["text_content_safety_categories"].iloc[0]
        # Score should be reduced
        assert result["text_content_safety_score"].iloc[0] < 1.0

    def test_risk_levels(self, analyzer, schema):
        """Test that risk levels are assigned correctly."""
        df = pd.DataFrame({
            "text_content": [
                "Hello, how are you?",  # Safe
                "I'm angry at something",  # Low risk
            ],
            "role": ["user", "user"],
        })

        result = analyzer.analyze_sample(df, schema)

        # First should be safe
        assert result["text_content_safety_risk_level"].iloc[0] == "safe"

    def test_strict_mode(self, schema):
        """Test strict mode flags any pattern match."""
        analyzer = SafetyAnalyzer(strict_mode=True)
        df = pd.DataFrame({
            "text_content": ["The news mentioned a weapon found by police."],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        # In strict mode, any mention flags as unsafe
        assert result["text_content_safety_is_safe"].iloc[0] == False

    def test_empty_content(self, analyzer, schema):
        """Test handling of empty content."""
        df = pd.DataFrame({
            "text_content": [""],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_safety_is_safe"].iloc[0] == True
        assert result["text_content_safety_score"].iloc[0] == 1.0

    def test_multiple_categories(self, analyzer, schema):
        """Test detection of multiple unsafe categories."""
        df = pd.DataFrame({
            "text_content": [
                "How to make a weapon and steal money through fraud"
            ],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        categories = result["text_content_safety_categories"].iloc[0]
        # Should flag multiple categories
        assert len(categories.split(",")) >= 2

    def test_schema_required(self, analyzer):
        """Test that schema is required."""
        df = pd.DataFrame({"text_content": ["Test"]})

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)
