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

"""Unit tests for InputQualityAnalyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.input_quality_analyzer import InputQualityAnalyzer


@pytest.fixture
def schema():
    """Create a sample schema."""
    return {
        "text_content": {"content_type": ContentType.TEXT},
        "role": {"content_type": ContentType.CATEGORICAL},
    }


@pytest.fixture
def analyzer():
    """Create an InputQualityAnalyzer instance."""
    return InputQualityAnalyzer()


class TestInputQualityAnalyzer:
    """Tests for InputQualityAnalyzer."""

    def test_high_quality_input(self, analyzer, schema):
        """Test that high-quality inputs are rated correctly."""
        df = pd.DataFrame({
            "text_content": [
                "Write a Python function that takes a list of integers and "
                "returns the sum of all even numbers. Include type hints."
            ],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert "text_content_input_quality_tier" in result.columns
        tier = result["text_content_input_quality_tier"].iloc[0]
        assert tier in ["good", "excellent"]

    def test_low_quality_input(self, analyzer, schema):
        """Test that low-quality inputs are rated correctly."""
        df = pd.DataFrame({
            "text_content": ["hi"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        tier = result["text_content_input_quality_tier"].iloc[0]
        assert tier in ["very_poor", "poor"]
        assert result["text_content_input_quality_is_answerable"].iloc[0] == False

    def test_ambiguous_input(self, analyzer, schema):
        """Test that ambiguous inputs are flagged."""
        df = pd.DataFrame({
            "text_content": [
                "Do something with the stuff, maybe some things like that or whatever"
            ],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_input_quality_is_ambiguous"].iloc[0] == True

    def test_input_with_context(self, analyzer, schema):
        """Test that inputs with context are scored higher."""
        df = pd.DataFrame({
            "text_content": [
                'Given the following code: `def add(a, b): return a + b`, '
                "explain what it does in 3 sentences."
            ],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_input_quality_has_sufficient_context"].iloc[0] == True

    def test_only_analyzes_user_messages(self, analyzer, schema):
        """Test that only user messages are analyzed by default."""
        df = pd.DataFrame({
            "text_content": [
                "What is Python?",
                "Python is a programming language.",
            ],
            "role": ["user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        # User message should be analyzed
        assert result["text_content_input_quality_tier"].iloc[0] is not None
        # Assistant message should have None
        assert result["text_content_input_quality_tier"].iloc[1] is None

    def test_unanswerable_input(self, analyzer, schema):
        """Test detection of unanswerable inputs."""
        df = pd.DataFrame({
            "text_content": ["ok"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_input_quality_is_answerable"].iloc[0] == False

    def test_empty_input(self, analyzer, schema):
        """Test handling of empty inputs."""
        df = pd.DataFrame({
            "text_content": [""],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_input_quality_tier"].iloc[0] == "very_poor"
        assert result["text_content_input_quality_score"].iloc[0] == 0.0

    def test_schema_required(self, analyzer):
        """Test that schema is required."""
        df = pd.DataFrame({"text_content": ["Test"]})

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)
