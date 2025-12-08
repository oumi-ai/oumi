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

"""Unit tests for ResponseCompletenessAnalyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.response_completeness_analyzer import ResponseCompletenessAnalyzer


@pytest.fixture
def schema():
    """Create a sample schema."""
    return {
        "text_content": {"content_type": ContentType.TEXT},
        "role": {"content_type": ContentType.CATEGORICAL},
    }


@pytest.fixture
def analyzer():
    """Create a ResponseCompletenessAnalyzer instance."""
    return ResponseCompletenessAnalyzer()


class TestResponseCompletenessAnalyzer:
    """Tests for ResponseCompletenessAnalyzer."""

    def test_complete_response(self, analyzer, schema):
        """Test that complete responses are detected."""
        df = pd.DataFrame({
            "text_content": [
                "Here is a complete answer to your question. "
                "The main points are explained above. "
                "I hope this helps!"
            ],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert "text_content_response_completeness_is_complete" in result.columns
        assert result["text_content_response_completeness_is_complete"].iloc[0] == True
        assert result["text_content_response_completeness_ends_naturally"].iloc[0] == True

    def test_truncated_mid_sentence(self, analyzer, schema):
        """Test detection of mid-sentence truncation."""
        df = pd.DataFrame({
            "text_content": [
                "Here is the answer. The main reason is because the"
            ],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_response_completeness_is_complete"].iloc[0] == False
        assert result["text_content_response_completeness_truncation_type"].iloc[0] == "mid_sentence"

    def test_truncated_with_ellipsis(self, analyzer, schema):
        """Test detection of ellipsis truncation."""
        df = pd.DataFrame({
            "text_content": ["The answer involves several steps..."],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_response_completeness_is_complete"].iloc[0] == False

    def test_incomplete_code_block(self, analyzer, schema):
        """Test detection of incomplete code blocks."""
        df = pd.DataFrame({
            "text_content": [
                "Here's the code:\n```python\ndef hello():\n    print('hello'"
            ],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_response_completeness_is_complete"].iloc[0] == False
        assert result["text_content_response_completeness_truncation_type"].iloc[0] == "incomplete_code"

    def test_response_with_conclusion(self, analyzer, schema):
        """Test detection of concluding statements."""
        # Note: conclusion detection checks the last 20% of the text
        df = pd.DataFrame({
            "text_content": [
                "Let me explain the concept in detail. First, we need to understand "
                "the basics of the problem. Then we can apply them to your specific case. "
                "After that, we analyze the results carefully. We compare options. "
                "We test hypotheses. We validate our approach. We run experiments. "
                "We document findings. We share insights. In conclusion, this "
                "approach is the most effective for your use case."
            ],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_response_completeness_has_conclusion"].iloc[0] == True

    def test_only_analyzes_assistant_messages(self, analyzer, schema):
        """Test that only assistant messages are analyzed by default."""
        df = pd.DataFrame({
            "text_content": [
                "What is the answer?",
                "Here is your complete answer.",
            ],
            "role": ["user", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        # User message should have None
        assert result["text_content_response_completeness_is_complete"].iloc[0] is None
        # Assistant message should be analyzed
        assert result["text_content_response_completeness_is_complete"].iloc[1] is not None

    def test_empty_response(self, analyzer, schema):
        """Test handling of empty responses."""
        df = pd.DataFrame({
            "text_content": [""],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_response_completeness_is_complete"].iloc[0] == False
        assert result["text_content_response_completeness_truncation_type"].iloc[0] == "empty"

    def test_natural_ending_detection(self, analyzer, schema):
        """Test natural ending detection."""
        df = pd.DataFrame({
            "text_content": [
                "The answer is 42.",  # Ends with period
                "Is that clear?",  # Ends with question mark
                "That's amazing!",  # Ends with exclamation
            ],
            "role": ["assistant", "assistant", "assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        for i in range(3):
            assert result["text_content_response_completeness_ends_naturally"].iloc[i] == True

    def test_strict_mode(self, schema):
        """Test strict mode requires natural endings."""
        analyzer = ResponseCompletenessAnalyzer(strict_mode=True)
        df = pd.DataFrame({
            "text_content": [
                "Here is an answer without proper ending and"
            ],
            "role": ["assistant"],
        })

        result = analyzer.analyze_sample(df, schema)

        assert result["text_content_response_completeness_is_complete"].iloc[0] == False

    def test_schema_required(self, analyzer):
        """Test that schema is required."""
        df = pd.DataFrame({"text_content": ["Test"]})

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, schema=None)
