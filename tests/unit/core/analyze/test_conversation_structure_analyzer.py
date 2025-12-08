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

"""Unit tests for ConversationStructureAnalyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.conversation_structure_analyzer import ConversationStructureAnalyzer


@pytest.fixture
def schema():
    """Create a sample schema."""
    return {
        "text_content": {"content_type": ContentType.TEXT},
        "role": {"content_type": ContentType.CATEGORICAL},
    }


@pytest.fixture
def analyzer():
    """Create a ConversationStructureAnalyzer instance."""
    return ConversationStructureAnalyzer()


class TestConversationStructureAnalyzer:
    """Tests for ConversationStructureAnalyzer."""

    def test_single_turn_conversation(self, analyzer, schema):
        """Test that single-turn conversations are detected."""
        df = pd.DataFrame({
            "conversation_id": ["conv1", "conv1"],
            "text_content": ["What is Python?", "Python is a programming language."],
            "role": ["user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert "conversation_structure_is_single_turn" in result.columns
        assert result["conversation_structure_is_single_turn"].iloc[0] == True
        assert result["conversation_structure_turn_count"].iloc[0] == 2

    def test_multi_turn_conversation(self, analyzer, schema):
        """Test that multi-turn conversations are detected."""
        df = pd.DataFrame({
            "conversation_id": ["conv1"] * 6,
            "text_content": [
                "What is Python?",
                "Python is a programming language.",
                "Can you give an example?",
                "Sure, here's an example: print('Hello')",
                "How do I run it?",
                "You can run it with: python script.py",
            ],
            "role": ["user", "assistant", "user", "assistant", "user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["conversation_structure_is_multi_turn"].iloc[0] == True
        assert result["conversation_structure_turn_count"].iloc[0] == 6
        assert result["conversation_structure_conversation_depth"].iloc[0] == 3

    def test_conversation_with_system_prompt(self, analyzer, schema):
        """Test detection of system prompts."""
        df = pd.DataFrame({
            "conversation_id": ["conv1"] * 3,
            "text_content": [
                "You are a helpful assistant.",
                "What is Python?",
                "Python is a programming language.",
            ],
            "role": ["system", "user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["conversation_structure_has_system_prompt"].iloc[0] == True
        assert result["conversation_structure_turn_count"].iloc[0] == 3

    def test_role_balance(self, analyzer, schema):
        """Test role balance calculation."""
        df = pd.DataFrame({
            "conversation_id": ["conv1"] * 4,
            "text_content": ["Q1", "A1", "Q2", "A2"],
            "role": ["user", "assistant", "user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        # 2 user, 2 assistant = 0.5 balance
        assert result["conversation_structure_role_balance"].iloc[0] == 0.5

    def test_length_statistics(self, analyzer, schema):
        """Test length statistics computation."""
        df = pd.DataFrame({
            "conversation_id": ["conv1"] * 2,
            "text_content": ["Short", "This is a longer response with more words"],
            "role": ["user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert "conversation_structure_avg_turn_length" in result.columns
        assert result["conversation_structure_avg_turn_length"].iloc[0] > 0

    def test_flat_format_without_conversation_id(self, analyzer, schema):
        """Test handling of flat format without conversation_id."""
        df = pd.DataFrame({
            "text_content": ["What is Python?"],
            "role": ["user"],
        })

        result = analyzer.analyze_sample(df, schema)

        # Should default to single-turn
        assert result["conversation_structure_is_single_turn"].iloc[0] == True
        assert result["conversation_structure_turn_count"].iloc[0] == 2

    def test_multiple_conversations(self, analyzer, schema):
        """Test handling of multiple conversations."""
        df = pd.DataFrame({
            "conversation_id": ["conv1", "conv1", "conv2", "conv2", "conv2", "conv2"],
            "text_content": ["Q1", "A1", "Q2", "A2", "Q3", "A3"],
            "role": ["user", "assistant", "user", "assistant", "user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        # First conversation is single-turn
        conv1_mask = result["conversation_id"] == "conv1"
        assert result.loc[conv1_mask, "conversation_structure_is_single_turn"].iloc[0] == True

        # Second conversation is multi-turn
        conv2_mask = result["conversation_id"] == "conv2"
        assert result.loc[conv2_mask, "conversation_structure_is_multi_turn"].iloc[0] == True
