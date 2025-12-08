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

"""Unit tests for the Content Pattern Analyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.content_pattern_analyzer import ContentPatternAnalyzer


@pytest.fixture
def sample_schema():
    """Create a sample schema for testing."""
    return {
        "text_content": {"content_type": ContentType.TEXT},
        "role": {"content_type": ContentType.CATEGORICAL},
    }


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "text_content": [
                "Hello, how are you?",
                "Dear [Name], welcome to [Company Name]!",
                "I had to make a difficult decision when I was working.",
                "<nooutput>",
                "I cannot provide that information as it is harmful.",
                "Visit https://www.example.com/ for more info.",
                "Normal clean text without any issues.",
                "Please fill in [Your Name] and [Your Email].",
            ],
            "role": [
                "user",
                "assistant",
                "assistant",
                "assistant",
                "assistant",
                "assistant",
                "assistant",
                "assistant",
            ],
        }
    )


class TestContentPatternAnalyzer:
    """Tests for ContentPatternAnalyzer."""

    def test_init_default(self):
        """Test default initialization."""
        analyzer = ContentPatternAnalyzer()
        assert analyzer.detect_placeholders is True
        assert analyzer.detect_hallucinated_experiences is True
        assert analyzer.detect_nooutput is True
        assert analyzer.detect_refusals is True
        assert analyzer.check_output_only is False

    def test_init_custom(self):
        """Test custom initialization."""
        analyzer = ContentPatternAnalyzer(
            detect_placeholders=False,
            detect_hallucinated_experiences=False,
            placeholder_whitelist=["[INPUT]", "[OUTPUT]"],
        )
        assert analyzer.detect_placeholders is False
        assert analyzer.detect_hallucinated_experiences is False
        assert "[INPUT]" in analyzer.placeholder_whitelist

    def test_detect_bracket_placeholders(self, sample_df, sample_schema):
        """Test detection of [Name], [Company Name], etc."""
        analyzer = ContentPatternAnalyzer(
            detect_placeholders=True,
            detect_hallucinated_experiences=False,
            detect_nooutput=False,
            detect_refusals=False,
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(sample_df, sample_schema)

        # Check placeholder detection columns exist
        assert "text_content_content_pattern_has_placeholder" in result_df.columns
        assert "text_content_content_pattern_placeholder_count" in result_df.columns

        # Row 1 has placeholders ([Name], [Company Name])
        assert result_df.loc[1, "text_content_content_pattern_has_placeholder"] == True  # noqa: E712
        assert result_df.loc[1, "text_content_content_pattern_placeholder_count"] == 2

        # Row 7 has placeholders ([Your Name], [Your Email])
        assert result_df.loc[7, "text_content_content_pattern_has_placeholder"] == True  # noqa: E712
        assert result_df.loc[7, "text_content_content_pattern_placeholder_count"] == 2

        # Row 0 has no placeholders
        assert result_df.loc[0, "text_content_content_pattern_has_placeholder"] == False  # noqa: E712

        # Row 6 has no placeholders
        assert result_df.loc[6, "text_content_content_pattern_has_placeholder"] == False  # noqa: E712

    def test_placeholder_whitelist(self, sample_schema):
        """Test that whitelisted placeholders are not flagged."""
        df = pd.DataFrame(
            {
                "text_content": [
                    "Use [INPUT] and [OUTPUT] for variables.",
                    "Dear [Name], please provide [Your Email].",
                ],
                "role": ["assistant", "assistant"],
            }
        )

        analyzer = ContentPatternAnalyzer(
            detect_placeholders=True,
            detect_hallucinated_experiences=False,
            detect_nooutput=False,
            detect_refusals=False,
            placeholder_whitelist=["[INPUT]", "[OUTPUT]"],
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(df, sample_schema)

        # Row 0 should have 0 placeholders (both are whitelisted)
        assert result_df.loc[0, "text_content_content_pattern_has_placeholder"] == False  # noqa: E712

        # Row 1 should have 2 placeholders (not whitelisted)
        assert result_df.loc[1, "text_content_content_pattern_has_placeholder"] == True  # noqa: E712
        assert result_df.loc[1, "text_content_content_pattern_placeholder_count"] == 2

    def test_detect_hallucinated_experience(self, sample_df, sample_schema):
        """Test detection of AI fabricated personal experiences."""
        analyzer = ContentPatternAnalyzer(
            detect_placeholders=False,
            detect_hallucinated_experiences=True,
            detect_nooutput=False,
            detect_refusals=False,
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(sample_df, sample_schema)

        assert (
            "text_content_content_pattern_has_hallucinated_experience"
            in result_df.columns
        )

        # Row 2 has hallucinated experience
        assert (
            result_df.loc[2, "text_content_content_pattern_has_hallucinated_experience"]  # noqa: E712
            == True  # noqa: E712
        )

        # Row 6 has no hallucinated experience
        assert (
            result_df.loc[6, "text_content_content_pattern_has_hallucinated_experience"]  # noqa: E712
            == False  # noqa: E712
        )

    def test_detect_hallucinated_experience_variations(self, sample_schema):
        """Test various hallucinated experience patterns."""
        df = pd.DataFrame(
            {
                "text_content": [
                    "When I was a software engineer at Google...",
                    "In my experience as a doctor, I've seen many cases...",
                    "During my career at Microsoft, I learned...",
                    "I recently visited the museum in my city.",
                    "The function returns a value.",
                ],
                "role": ["assistant"] * 5,
            }
        )

        analyzer = ContentPatternAnalyzer(
            detect_placeholders=False,
            detect_hallucinated_experiences=True,
            detect_nooutput=False,
            detect_refusals=False,
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(df, sample_schema)

        # Rows 0-3 should be flagged as hallucinated experiences
        assert (
            result_df.loc[0, "text_content_content_pattern_has_hallucinated_experience"]  # noqa: E712
            == True  # noqa: E712
        )
        assert (
            result_df.loc[1, "text_content_content_pattern_has_hallucinated_experience"]  # noqa: E712
            == True  # noqa: E712
        )
        assert (
            result_df.loc[2, "text_content_content_pattern_has_hallucinated_experience"]  # noqa: E712
            == True  # noqa: E712
        )
        assert (
            result_df.loc[3, "text_content_content_pattern_has_hallucinated_experience"]  # noqa: E712
            == True  # noqa: E712
        )

        # Row 4 should not be flagged
        assert (
            result_df.loc[4, "text_content_content_pattern_has_hallucinated_experience"]  # noqa: E712
            == False  # noqa: E712
        )

    def test_detect_nooutput(self, sample_df, sample_schema):
        """Test detection of <nooutput> and similar tags."""
        analyzer = ContentPatternAnalyzer(
            detect_placeholders=False,
            detect_hallucinated_experiences=False,
            detect_nooutput=True,
            detect_refusals=False,
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(sample_df, sample_schema)

        assert "text_content_content_pattern_has_nooutput" in result_df.columns

        # Row 3 has <nooutput>
        assert result_df.loc[3, "text_content_content_pattern_has_nooutput"] == True  # noqa: E712

        # Row 6 has no nooutput marker
        assert result_df.loc[6, "text_content_content_pattern_has_nooutput"] == False  # noqa: E712

    def test_detect_nooutput_variations(self, sample_schema):
        """Test various nooutput patterns."""
        df = pd.DataFrame(
            {
                "text_content": [
                    "<nooutput>",
                    "<no_output>",
                    "[nooutput]",
                    "N/A",
                    "None",
                    "-",
                    "This is a normal response.",
                ],
                "role": ["assistant"] * 7,
            }
        )

        analyzer = ContentPatternAnalyzer(
            detect_placeholders=False,
            detect_hallucinated_experiences=False,
            detect_nooutput=True,
            detect_refusals=False,
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(df, sample_schema)

        # Rows 0-5 should be flagged
        for i in range(6):
            assert (
                result_df.loc[i, "text_content_content_pattern_has_nooutput"] == True  # noqa: E712
            ), f"Row {i} should be flagged"

        # Row 6 should not be flagged
        assert result_df.loc[6, "text_content_content_pattern_has_nooutput"] == False  # noqa: E712

    def test_detect_refusal_patterns(self, sample_df, sample_schema):
        """Test detection of AI refusal patterns."""
        analyzer = ContentPatternAnalyzer(
            detect_placeholders=False,
            detect_hallucinated_experiences=False,
            detect_nooutput=False,
            detect_refusals=True,
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(sample_df, sample_schema)

        assert "text_content_content_pattern_has_refusal" in result_df.columns

        # Row 4 has refusal pattern
        assert result_df.loc[4, "text_content_content_pattern_has_refusal"] == True  # noqa: E712

        # Row 6 has no refusal
        assert result_df.loc[6, "text_content_content_pattern_has_refusal"] == False  # noqa: E712

    def test_detect_refusal_variations(self, sample_schema):
        """Test various refusal patterns."""
        df = pd.DataFrame(
            {
                "text_content": [
                    "I cannot provide that information.",
                    "I'm unable to help with that request.",
                    "I won't generate harmful content.",
                    "This request seems inappropriate.",
                    "Here is the information you requested.",
                ],
                "role": ["assistant"] * 5,
            }
        )

        analyzer = ContentPatternAnalyzer(
            detect_placeholders=False,
            detect_hallucinated_experiences=False,
            detect_nooutput=False,
            detect_refusals=True,
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(df, sample_schema)

        # Rows 0-3 should be flagged as refusals
        assert result_df.loc[0, "text_content_content_pattern_has_refusal"] == True  # noqa: E712
        assert result_df.loc[1, "text_content_content_pattern_has_refusal"] == True  # noqa: E712
        assert result_df.loc[2, "text_content_content_pattern_has_refusal"] == True  # noqa: E712
        assert result_df.loc[3, "text_content_content_pattern_has_refusal"] == True  # noqa: E712

        # Row 4 should not be flagged
        assert result_df.loc[4, "text_content_content_pattern_has_refusal"] == False  # noqa: E712

    def test_check_output_only(self, sample_schema):
        """Test that check_output_only only analyzes assistant messages."""
        df = pd.DataFrame(
            {
                "text_content": [
                    "Dear [Name], please help me.",  # User with placeholder
                    "Dear [Name], here is the answer.",  # Assistant with placeholder
                ],
                "role": ["user", "assistant"],
            }
        )

        analyzer = ContentPatternAnalyzer(
            detect_placeholders=True,
            detect_hallucinated_experiences=False,
            detect_nooutput=False,
            detect_refusals=False,
            check_output_only=True,
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(df, sample_schema)

        # Row 0 (user) should have None/empty results
        assert (
            result_df.loc[0, "text_content_content_pattern_has_placeholder"] is None
            or result_df.loc[0, "text_content_content_pattern_has_placeholder"] == {}
            or pd.isna(result_df.loc[0, "text_content_content_pattern_has_placeholder"])
        )

        # Row 1 (assistant) should be analyzed
        assert result_df.loc[1, "text_content_content_pattern_has_placeholder"] is True

    def test_no_schema_raises_error(self, sample_df):
        """Test that missing schema raises error."""
        analyzer = ContentPatternAnalyzer()

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(sample_df, None)

    def test_empty_dataframe(self, sample_schema):
        """Test with empty DataFrame."""
        analyzer = ContentPatternAnalyzer()
        analyzer.analyzer_id = "content_pattern"

        empty_df = pd.DataFrame({"text_content": [], "role": []})
        result_df = analyzer.analyze_sample(empty_df, sample_schema)

        assert len(result_df) == 0

    def test_no_text_columns(self):
        """Test with no text columns in schema."""
        analyzer = ContentPatternAnalyzer()
        analyzer.analyzer_id = "content_pattern"

        df = pd.DataFrame({"numeric_col": [1, 2, 3]})
        schema = {"numeric_col": {"content_type": ContentType.NUMERIC}}

        result_df = analyzer.analyze_sample(df, schema)

        # Should return unchanged DataFrame
        assert list(result_df.columns) == ["numeric_col"]

    def test_multiple_issues_in_same_text(self, sample_schema):
        """Test text with multiple issues."""
        df = pd.DataFrame(
            {
                "text_content": [
                    "I cannot provide [Name]'s info. When I was a manager, "
                    "I learned to refuse such requests."
                ],
                "role": ["assistant"],
            }
        )

        analyzer = ContentPatternAnalyzer(
            detect_placeholders=True,
            detect_hallucinated_experiences=True,
            detect_nooutput=True,
            detect_refusals=True,
        )
        analyzer.analyzer_id = "content_pattern"

        result_df = analyzer.analyze_sample(df, sample_schema)

        # Should detect all issues
        assert result_df.loc[0, "text_content_content_pattern_has_placeholder"] == True  # noqa: E712
        assert (
            result_df.loc[0, "text_content_content_pattern_has_hallucinated_experience"]  # noqa: E712
            == True  # noqa: E712
        )
        assert result_df.loc[0, "text_content_content_pattern_has_refusal"] == True  # noqa: E712
