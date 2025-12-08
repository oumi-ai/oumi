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

"""Unit tests for InstructRewardAnalyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.instruct_reward_analyzer import InstructRewardAnalyzer


@pytest.fixture
def schema():
    """Create a sample schema."""
    return {
        "text_content": {"content_type": ContentType.TEXT},
        "role": {"content_type": ContentType.CATEGORICAL},
    }


@pytest.fixture
def analyzer():
    """Create an InstructRewardAnalyzer instance."""
    return InstructRewardAnalyzer()


class TestInstructRewardAnalyzer:
    """Tests for InstructRewardAnalyzer."""

    def test_high_quality_response(self, analyzer, schema):
        """Test that high-quality responses get high scores."""
        df = pd.DataFrame({
            "text_content": [
                "Here is a comprehensive answer to your question. "
                "First, let me explain the background. "
                "Then I'll provide a step-by-step solution. "
                "Finally, I'll summarize the key points. "
                "I hope this helps! Let me know if you have more questions."
            ],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert "text_content_instruct_reward_score" in result.columns
        score = result["text_content_instruct_reward_score"].iloc[0]
        assert score >= 3.0  # Good or excellent tier

    def test_low_quality_response(self, analyzer, schema):
        """Test that low-quality responses get low scores."""
        df = pd.DataFrame({
            "text_content": ["I don't know."],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        score = result["text_content_instruct_reward_score"].iloc[0]
        assert score < 2.5  # Below fair threshold

    def test_reward_tiers(self, analyzer, schema):
        """Test that reward tiers are assigned correctly."""
        df = pd.DataFrame({
            "text_content": [
                "N/A",  # Poor
                "Here is a brief answer.",  # Fair
                "Let me provide a detailed explanation with multiple points. "
                "First, the main concept. Second, the details. Finally, summary.",  # Good
            ],
            "role": ["assistant", "assistant", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        tiers = result["text_content_instruct_reward_tier"].tolist()
        # First should be poor, others should be better
        assert tiers[0] in ["poor", "fair"]

    def test_only_analyzes_assistant_messages(self, analyzer, schema):
        """Test that only assistant messages are analyzed by default."""
        df = pd.DataFrame({
            "text_content": [
                "What is the answer?",
                "Here is your answer with details.",
            ],
            "role": ["user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        # User message should have NaN/None
        assert pd.isna(result["text_content_instruct_reward_score"].iloc[0])
        # Assistant message should have a score
        assert not pd.isna(result["text_content_instruct_reward_score"].iloc[1])

    def test_component_scores(self, analyzer, schema):
        """Test that component scores are included."""
        df = pd.DataFrame({
            "text_content": ["Here is a detailed response with structure."],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert "text_content_instruct_reward_helpfulness" in result.columns
        assert "text_content_instruct_reward_completeness" in result.columns
        assert "text_content_instruct_reward_clarity" in result.columns

    def test_empty_response(self, analyzer, schema):
        """Test handling of empty responses."""
        df = pd.DataFrame({
            "text_content": [""],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_instruct_reward_score"].iloc[0] == 0.0
        assert result["text_content_instruct_reward_tier"].iloc[0] == "poor"

    def test_schema_required(self, analyzer):
        """Test that schema is required."""
        df = pd.DataFrame({"text_content": ["Test"]})

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)
