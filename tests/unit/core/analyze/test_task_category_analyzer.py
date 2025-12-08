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

"""Unit tests for TaskCategoryAnalyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.task_category_analyzer import TaskCategoryAnalyzer


@pytest.fixture
def schema():
    """Create a sample schema."""
    return {
        "text_content": {"content_type": ContentType.TEXT},
        "role": {"content_type": ContentType.CATEGORICAL},
    }


@pytest.fixture
def analyzer():
    """Create a TaskCategoryAnalyzer instance."""
    return TaskCategoryAnalyzer()


class TestTaskCategoryAnalyzer:
    """Tests for TaskCategoryAnalyzer."""

    def test_classify_math_instruction(self, analyzer, schema):
        """Test that math instructions are classified correctly."""
        df = pd.DataFrame({
            "text_content": ["Calculate the derivative of x^2 + 3x"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert "text_content_task_category_category" in result.columns
        assert result["text_content_task_category_category"].iloc[0] == "math"
        assert result["text_content_task_category_is_stem"].iloc[0] == True

    def test_classify_coding_instruction(self, analyzer, schema):
        """Test that coding instructions are classified correctly."""
        df = pd.DataFrame({
            "text_content": ["Write a Python function to sort a list"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_task_category_category"].iloc[0] == "coding"
        assert result["text_content_task_category_is_stem"].iloc[0] == True

    def test_classify_creative_writing(self, analyzer, schema):
        """Test that creative writing instructions are classified correctly."""
        df = pd.DataFrame({
            "text_content": ["Write a creative story about a dragon with vivid characters"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_task_category_category"].iloc[0] == "creative_writing"
        assert result["text_content_task_category_is_stem"].iloc[0] == False

    def test_classify_information_seeking(self, analyzer, schema):
        """Test that information seeking instructions are classified correctly."""
        df = pd.DataFrame({
            "text_content": ["What is the capital of France?"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_task_category_category"].iloc[0] == "information_seeking"

    def test_classify_advice(self, analyzer, schema):
        """Test that advice instructions are classified correctly."""
        df = pd.DataFrame({
            "text_content": ["What should I do about my career change decision?"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_task_category_category"].iloc[0] == "advice"
        assert result["text_content_task_category_is_conversational"].iloc[0] == True

    def test_only_analyzes_user_messages(self, analyzer, schema):
        """Test that only user messages are analyzed by default."""
        df = pd.DataFrame({
            "text_content": [
                "Calculate 2 + 2",
                "The answer is 4",
            ],
            "role": ["user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        # User message should be classified
        assert result["text_content_task_category_category"].iloc[0] == "math"
        # Assistant message should have None
        assert result["text_content_task_category_category"].iloc[1] is None

    def test_analyze_all_messages(self, schema):
        """Test analyzing all messages when analyze_user_only is False."""
        analyzer = TaskCategoryAnalyzer(analyze_user_only=False)
        df = pd.DataFrame({
            "text_content": [
                "Calculate 2 + 2",
                "Let me solve this equation for you",
            ],
            "role": ["user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        # Both should be classified
        assert result["text_content_task_category_category"].iloc[0] is not None
        assert result["text_content_task_category_category"].iloc[1] is not None

    def test_empty_text(self, analyzer, schema):
        """Test handling of empty text."""
        df = pd.DataFrame({
            "text_content": [""],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_task_category_category"].iloc[0] == "other"
        assert result["text_content_task_category_confidence"].iloc[0] == 0.0

    def test_schema_required(self, analyzer):
        """Test that schema is required."""
        df = pd.DataFrame({"text_content": ["Test"]})

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)
