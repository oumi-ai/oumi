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

"""Unit tests for DifficultyAnalyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.difficulty_analyzer import DifficultyAnalyzer


@pytest.fixture
def schema():
    """Create a sample schema."""
    return {
        "text_content": {"content_type": ContentType.TEXT},
        "role": {"content_type": ContentType.CATEGORICAL},
    }


@pytest.fixture
def analyzer():
    """Create a DifficultyAnalyzer instance."""
    return DifficultyAnalyzer()


class TestDifficultyAnalyzer:
    """Tests for DifficultyAnalyzer."""

    def test_easy_instruction(self, analyzer, schema):
        """Test that simple instructions are rated as easy."""
        df = pd.DataFrame({
            "text_content": ["What is 2 + 2?"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert "text_content_difficulty_tier" in result.columns
        tier = result["text_content_difficulty_tier"].iloc[0]
        assert tier in ["easy", "medium"]

    def test_hard_instruction(self, analyzer, schema):
        """Test that complex instructions are rated as hard."""
        df = pd.DataFrame({
            "text_content": [
                "Implement a distributed consensus algorithm that must handle "
                "Byzantine faults, ensuring linearizability while maintaining "
                "at least 99.99% availability. The solution should include "
                "formal proofs of correctness using temporal logic. First, "
                "explain the theoretical foundations, then provide the "
                "implementation with complexity analysis."
            ],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        tier = result["text_content_difficulty_tier"].iloc[0]
        assert tier in ["hard", "expert"]
        assert result["text_content_difficulty_requires_reasoning"].iloc[0] == True

    def test_domain_knowledge_detection(self, analyzer, schema):
        """Test detection of domain-specific knowledge requirements."""
        df = pd.DataFrame({
            "text_content": [
                "Explain the use of derivatives and integrals in calculating "
                "the Black-Scholes option pricing model with variance correlation."
            ],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_difficulty_requires_domain_knowledge"].iloc[0] == True

    def test_constraint_counting(self, analyzer, schema):
        """Test counting of explicit constraints."""
        df = pd.DataFrame({
            "text_content": [
                "Write a function that must handle at least 1000 items, "
                "should return results within 100ms, and must not use "
                "more than 50MB of memory."
            ],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_difficulty_constraint_count"].iloc[0] >= 3

    def test_reasoning_detection(self, analyzer, schema):
        """Test detection of reasoning requirements."""
        df = pd.DataFrame({
            "text_content": [
                "Analyze the pros and cons of microservices vs monolithic "
                "architecture. First explain each approach, then compare them, "
                "and finally conclude with a recommendation."
            ],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_difficulty_requires_reasoning"].iloc[0] == True

    def test_only_analyzes_user_messages(self, analyzer, schema):
        """Test that only user messages are analyzed by default."""
        df = pd.DataFrame({
            "text_content": [
                "Explain quantum computing",
                "Quantum computing uses qubits...",
            ],
            "role": ["user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        # User message should be analyzed
        assert result["text_content_difficulty_tier"].iloc[0] is not None
        # Assistant message should have None
        assert result["text_content_difficulty_tier"].iloc[1] is None

    def test_empty_instruction(self, analyzer, schema):
        """Test handling of empty instructions."""
        df = pd.DataFrame({
            "text_content": [""],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_difficulty_tier"].iloc[0] == "easy"
        assert result["text_content_difficulty_score"].iloc[0] == 0.0

    def test_schema_required(self, analyzer):
        """Test that schema is required."""
        df = pd.DataFrame({"text_content": ["Test"]})

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)
