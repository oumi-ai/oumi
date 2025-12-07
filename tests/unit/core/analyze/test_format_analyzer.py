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

"""Tests for the FormatAnalyzer."""

import pytest

from oumi.core.analyze.format_analyzer import FormatAnalyzer
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.analysis_utils import conversation_to_dataframes


def _single_message_conversation(text):
    return Conversation(messages=[Message(role=Role.USER, content=text)])


def _count_analysis_columns(df, analyzer_id="format"):
    """Count the number of analysis columns in a DataFrame."""
    return len([col for col in df.columns if f"_{analyzer_id}_" in col])


class TestFormatAnalyzerMarkdown:
    """Tests for markdown detection."""

    def test_detect_markdown_headers(self):
        """Test detection of markdown headers."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("# This is a header\n\nSome content")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_markdown"] == True  # noqa: E712

    def test_detect_markdown_h2_headers(self):
        """Test detection of h2 markdown headers."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("## Section\n\nContent here")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_markdown"] == True  # noqa: E712

    def test_detect_markdown_lists(self):
        """Test detection of markdown lists."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("- Item 1\n- Item 2\n- Item 3")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_markdown"] == True  # noqa: E712

    def test_detect_markdown_numbered_lists(self):
        """Test detection of numbered markdown lists."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("1. First\n2. Second\n3. Third")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_markdown"] == True  # noqa: E712

    def test_detect_markdown_bold(self):
        """Test detection of bold markdown."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("This is **bold** text")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_markdown"] == True  # noqa: E712

    def test_detect_markdown_links(self):
        """Test detection of markdown links."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("Check out [this link](https://example.com)")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_markdown"] == True  # noqa: E712

    def test_no_markdown(self):
        """Test plain text without markdown."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("This is plain text without any formatting.")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_markdown"] == False  # noqa: E712


class TestFormatAnalyzerJSON:
    """Tests for JSON detection."""

    def test_detect_json_code_block(self):
        """Test detection of JSON in code blocks."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=True,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        text = '```json\n{"key": "value"}\n```'
        conv = _single_message_conversation(text)
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_json"] == True  # noqa: E712

    def test_detect_inline_json_object(self):
        """Test detection of inline JSON objects."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=True,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        text = 'The response is {"status": "ok", "code": 200}'
        conv = _single_message_conversation(text)
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_json"] == True  # noqa: E712

    def test_detect_inline_json_array(self):
        """Test detection of inline JSON arrays."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=True,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        text = 'The items are [1, 2, 3]'
        conv = _single_message_conversation(text)
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_json"] == True  # noqa: E712

    def test_no_json(self):
        """Test text without JSON."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=True,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("This is plain text without JSON.")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_json"] == False  # noqa: E712

    def test_invalid_json_not_detected(self):
        """Test that invalid JSON is not detected."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=True,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        text = "This has braces {but not valid json}"
        conv = _single_message_conversation(text)
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_json"] == False  # noqa: E712


class TestFormatAnalyzerCodeBlocks:
    """Tests for code block detection."""

    def test_detect_code_block_with_language(self):
        """Test detection of code blocks with language specification."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=True,
            detect_urls=False,
            compute_complexity=False,
        )
        text = '```python\nprint("hello")\n```'
        conv = _single_message_conversation(text)
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_code_blocks"] == True  # noqa: E712
        assert result_df.iloc[0]["text_content_format_code_block_count"] == 1
        assert result_df.iloc[0]["text_content_format_code_block_languages"] == "python"

    def test_detect_multiple_code_blocks(self):
        """Test detection of multiple code blocks."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=True,
            detect_urls=False,
            compute_complexity=False,
        )
        text = '```python\nprint("hi")\n```\n\n```javascript\nconsole.log("hi")\n```'
        conv = _single_message_conversation(text)
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_code_blocks"] == True  # noqa: E712
        assert result_df.iloc[0]["text_content_format_code_block_count"] == 2
        assert (
            result_df.iloc[0]["text_content_format_code_block_languages"]
            == "python,javascript"
        )

    def test_detect_code_block_without_language(self):
        """Test detection of code blocks without language specification."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=True,
            detect_urls=False,
            compute_complexity=False,
        )
        text = "```\nsome code\n```"
        conv = _single_message_conversation(text)
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_code_blocks"] == True  # noqa: E712
        assert result_df.iloc[0]["text_content_format_code_block_count"] == 1
        assert result_df.iloc[0]["text_content_format_code_block_languages"] == ""

    def test_no_code_blocks(self):
        """Test text without code blocks."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=True,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("Plain text without code blocks")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_code_blocks"] == False  # noqa: E712
        assert result_df.iloc[0]["text_content_format_code_block_count"] == 0
        assert result_df.iloc[0]["text_content_format_code_block_languages"] == ""


class TestFormatAnalyzerURLs:
    """Tests for URL detection."""

    def test_detect_https_url(self):
        """Test detection of HTTPS URLs."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=True,
            compute_complexity=False,
        )
        conv = _single_message_conversation("Check out https://example.com/page")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_urls"] == True  # noqa: E712

    def test_detect_http_url(self):
        """Test detection of HTTP URLs."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=True,
            compute_complexity=False,
        )
        conv = _single_message_conversation("Visit http://test.org")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_urls"] == True  # noqa: E712

    def test_no_urls(self):
        """Test text without URLs."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=True,
            compute_complexity=False,
        )
        conv = _single_message_conversation("No links here, just text.")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_urls"] == False  # noqa: E712


class TestFormatAnalyzerEmails:
    """Tests for email detection."""

    def test_detect_email(self):
        """Test detection of email addresses."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            detect_emails=True,
            compute_complexity=False,
        )
        conv = _single_message_conversation("Contact us at test@example.com")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_emails"] == True  # noqa: E712

    def test_no_emails(self):
        """Test text without email addresses."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            detect_emails=True,
            compute_complexity=False,
        )
        conv = _single_message_conversation("No email addresses here.")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_emails"] == False  # noqa: E712


class TestFormatAnalyzerComplexity:
    """Tests for format complexity score."""

    def test_complexity_plain_text(self):
        """Test complexity score for plain text."""
        analyzer = FormatAnalyzer(compute_complexity=True)
        conv = _single_message_conversation("Just plain text.")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_format_complexity_score"] == 0.0

    def test_complexity_with_markdown(self):
        """Test complexity score increases with markdown."""
        analyzer = FormatAnalyzer(compute_complexity=True)
        conv = _single_message_conversation("# Header\n\nSome **bold** text")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        score = result_df.iloc[0]["text_content_format_format_complexity_score"]
        assert score > 0.0

    def test_complexity_with_code_blocks(self):
        """Test complexity score increases with code blocks."""
        analyzer = FormatAnalyzer(compute_complexity=True)
        text = '```python\nprint("hi")\n```'
        conv = _single_message_conversation(text)
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        score = result_df.iloc[0]["text_content_format_format_complexity_score"]
        assert score > 0.0

    def test_complexity_highly_formatted(self):
        """Test complexity score for highly formatted content."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=True,
            detect_code_blocks=True,
            detect_urls=True,
            detect_emails=True,
            compute_complexity=True,
        )
        text = (
            "# Header\n\n"
            "Check out [this link](https://example.com)\n\n"
            '```json\n{"key": "value"}\n```\n\n'
            "Contact: test@example.com"
        )
        conv = _single_message_conversation(text)
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        score = result_df.iloc[0]["text_content_format_format_complexity_score"]
        # Should have high complexity score
        assert score > 0.5


class TestFormatAnalyzerInstantiation:
    """Tests for analyzer instantiation and configuration."""

    def test_default_instantiation(self):
        """Test analyzer with default parameters."""
        analyzer = FormatAnalyzer()
        conv = _single_message_conversation("# Test\n\nHello world")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        # Default should include markdown, json, code_blocks, urls, complexity
        # but not emails
        assert "text_content_format_has_markdown" in result_df.columns
        assert "text_content_format_has_json" in result_df.columns
        assert "text_content_format_has_code_blocks" in result_df.columns
        assert "text_content_format_has_urls" in result_df.columns
        assert "text_content_format_format_complexity_score" in result_df.columns
        assert "text_content_format_has_emails" not in result_df.columns

    def test_all_features_enabled(self):
        """Test analyzer with all features enabled."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=True,
            detect_code_blocks=True,
            detect_urls=True,
            detect_emails=True,
            compute_complexity=True,
        )
        conv = _single_message_conversation("Test content")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        # All columns should be present
        assert "text_content_format_has_markdown" in result_df.columns
        assert "text_content_format_has_json" in result_df.columns
        assert "text_content_format_has_code_blocks" in result_df.columns
        assert "text_content_format_code_block_count" in result_df.columns
        assert "text_content_format_code_block_languages" in result_df.columns
        assert "text_content_format_has_urls" in result_df.columns
        assert "text_content_format_has_emails" in result_df.columns
        assert "text_content_format_format_complexity_score" in result_df.columns

    def test_no_features_enabled(self):
        """Test analyzer with no features enabled."""
        analyzer = FormatAnalyzer(
            detect_markdown=False,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            detect_emails=False,
            compute_complexity=False,
        )
        conv = _single_message_conversation("Test content")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert _count_analysis_columns(result_df) == 0


class TestFormatAnalyzerValidation:
    """Tests for input validation."""

    def test_missing_schema_raises_error(self):
        """Test that missing schema raises ValueError."""
        analyzer = FormatAnalyzer()
        conv = _single_message_conversation("hello world")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(test_df, schema=None)

    def test_no_text_columns_returns_unchanged(self):
        """Test that schema with no text columns returns DataFrame unchanged."""
        analyzer = FormatAnalyzer()
        conv = _single_message_conversation("hello world")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "numeric"}}
        )
        # DataFrame should be returned unchanged (no new columns added)
        assert list(result_df.columns) == list(test_df.columns)


class TestFormatAnalyzerEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Test handling of empty text."""
        analyzer = FormatAnalyzer()
        conv = _single_message_conversation("")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        # All detection should be False/0
        assert result_df.iloc[0]["text_content_format_has_markdown"] == False  # noqa: E712
        assert result_df.iloc[0]["text_content_format_has_json"] == False  # noqa: E712
        assert result_df.iloc[0]["text_content_format_has_code_blocks"] == False  # noqa: E712
        assert result_df.iloc[0]["text_content_format_format_complexity_score"] == 0.0

    def test_multiple_messages(self):
        """Test format analysis across multiple messages."""
        analyzer = FormatAnalyzer(
            detect_markdown=True,
            detect_json=False,
            detect_code_blocks=False,
            detect_urls=False,
            compute_complexity=False,
        )
        conv = Conversation(
            messages=[
                Message(role=Role.USER, content="# Header"),
                Message(role=Role.ASSISTANT, content="Plain text response"),
            ]
        )
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_format_has_markdown"] == True  # noqa: E712
        assert result_df.iloc[1]["text_content_format_has_markdown"] == False  # noqa: E712
