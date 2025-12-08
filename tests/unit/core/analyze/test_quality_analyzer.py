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

"""Unit tests for the Quality Analyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.quality_analyzer import QualityAnalyzer


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
                "My email is test@example.com and phone is 555-123-4567.",
                "api_key=sk1234567890abcdef1234567890abcdef",
                "The The The same same same words words words repeated.",
                "Some text with content here.",
                "Normal clean text without any issues.",
                "Ã©ncoding issues in this text",
            ],
            "role": ["user", "user", "assistant", "assistant", "system", "user", "user"],
        }
    )


class TestQualityAnalyzer:
    """Tests for QualityAnalyzer."""

    def test_init_default(self):
        """Test default initialization."""
        analyzer = QualityAnalyzer()
        assert analyzer.detect_pii is True
        assert analyzer.detect_emails is True
        assert analyzer.detect_phones is True
        assert analyzer.detect_encoding_issues is True
        assert analyzer.detect_repetition is True

    def test_init_custom(self):
        """Test custom initialization."""
        analyzer = QualityAnalyzer(
            detect_pii=False,
            detect_emails=False,
            detect_language=False,
            repetition_threshold=0.5,
        )
        assert analyzer.detect_pii is False
        assert analyzer.detect_emails is False
        assert analyzer.repetition_threshold == 0.5

    def test_detect_email(self, sample_df, sample_schema):
        """Test email detection."""
        analyzer = QualityAnalyzer(
            detect_pii=True,
            detect_emails=True,
            detect_phones=False,
            detect_ssn=False,
            detect_credit_cards=False,
            detect_api_keys=False,
            detect_encoding_issues=False,
            detect_repetition=False,
        )
        analyzer.analyzer_id = "quality"

        result_df = analyzer.analyze_sample(sample_df, sample_schema)

        # Check email detection
        assert "text_content_quality_has_pii" in result_df.columns
        assert "text_content_quality_pii_types" in result_df.columns

        # Row 1 has an email
        assert result_df.loc[1, "text_content_quality_has_pii"] == True  # noqa: E712
        assert "email" in result_df.loc[1, "text_content_quality_pii_types"]

        # Row 0 has no PII
        assert result_df.loc[0, "text_content_quality_has_pii"] == False  # noqa: E712

    def test_detect_phone(self, sample_df, sample_schema):
        """Test phone number detection."""
        analyzer = QualityAnalyzer(
            detect_pii=True,
            detect_emails=False,
            detect_phones=True,
            detect_ssn=False,
            detect_credit_cards=False,
            detect_api_keys=False,
            detect_encoding_issues=False,
            detect_repetition=False,
        )
        analyzer.analyzer_id = "quality"

        result_df = analyzer.analyze_sample(sample_df, sample_schema)

        # Row 1 has a phone number
        assert result_df.loc[1, "text_content_quality_has_pii"] == True  # noqa: E712
        assert "phone" in result_df.loc[1, "text_content_quality_pii_types"]

    def test_detect_api_key(self, sample_df, sample_schema):
        """Test API key detection."""
        analyzer = QualityAnalyzer(
            detect_pii=True,
            detect_emails=False,
            detect_phones=False,
            detect_ssn=False,
            detect_credit_cards=False,
            detect_api_keys=True,
            detect_encoding_issues=False,
            detect_repetition=False,
        )
        analyzer.analyzer_id = "quality"

        result_df = analyzer.analyze_sample(sample_df, sample_schema)

        # Row 2 has an API key pattern
        # The pattern requires format like: api_key=xxx or secret: xxx
        assert result_df.loc[2, "text_content_quality_has_pii"] == True  # noqa: E712
        assert "api_key" in result_df.loc[2, "text_content_quality_pii_types"]

    def test_detect_encoding_issues(self, sample_df, sample_schema):
        """Test encoding issue detection."""
        analyzer = QualityAnalyzer(
            detect_pii=False,
            detect_encoding_issues=True,
            detect_repetition=False,
        )
        analyzer.analyzer_id = "quality"

        result_df = analyzer.analyze_sample(sample_df, sample_schema)

        assert "text_content_quality_has_encoding_issues" in result_df.columns

        # Row 6 has encoding issues (mojibake)
        assert result_df.loc[6, "text_content_quality_has_encoding_issues"] == True  # noqa: E712

        # Row 5 has no encoding issues
        assert result_df.loc[5, "text_content_quality_has_encoding_issues"] == False  # noqa: E712

    def test_detect_repetition(self, sample_df, sample_schema):
        """Test repetition detection."""
        analyzer = QualityAnalyzer(
            detect_pii=False,
            detect_encoding_issues=False,
            detect_repetition=True,
            repetition_ngram_size=2,
            repetition_threshold=0.3,
        )
        analyzer.analyzer_id = "quality"

        result_df = analyzer.analyze_sample(sample_df, sample_schema)

        assert "text_content_quality_repetition_ratio" in result_df.columns
        assert "text_content_quality_has_high_repetition" in result_df.columns

        # Row 3 has high repetition
        assert result_df.loc[3, "text_content_quality_has_high_repetition"] == True  # noqa: E712

        # Row 5 has low repetition
        assert result_df.loc[5, "text_content_quality_has_high_repetition"] == False  # noqa: E712

    def test_no_schema_raises_error(self, sample_df):
        """Test that missing schema raises error."""
        analyzer = QualityAnalyzer()

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(sample_df, None)

    def test_empty_dataframe(self, sample_schema):
        """Test with empty DataFrame."""
        analyzer = QualityAnalyzer()
        analyzer.analyzer_id = "quality"

        empty_df = pd.DataFrame({"text_content": [], "role": []})
        result_df = analyzer.analyze_sample(empty_df, sample_schema)

        assert len(result_df) == 0

    def test_no_text_columns(self):
        """Test with no text columns in schema."""
        analyzer = QualityAnalyzer()
        analyzer.analyzer_id = "quality"

        df = pd.DataFrame({"numeric_col": [1, 2, 3]})
        schema = {"numeric_col": {"content_type": ContentType.NUMERIC}}

        result_df = analyzer.analyze_sample(df, schema)

        # Should return unchanged DataFrame
        assert list(result_df.columns) == ["numeric_col"]
