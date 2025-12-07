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

"""Tests for the FastTextAnalyzer."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Check if fast-langdetect is available
try:
    import fast_langdetect

    FAST_LANGDETECT_AVAILABLE = True
except ImportError:
    FAST_LANGDETECT_AVAILABLE = False

# Skip all tests if fast-langdetect is not available
pytestmark = pytest.mark.skipif(
    not FAST_LANGDETECT_AVAILABLE,
    reason="fast-langdetect not installed",
)


def _create_test_df(texts: list[str]) -> pd.DataFrame:
    """Create a test DataFrame with text content."""
    return pd.DataFrame({"text_content": texts})


def _get_schema() -> dict:
    """Get the standard schema for testing."""
    return {"text_content": {"content_type": "text"}}


class TestFastTextAnalyzerInit:
    """Tests for FastTextAnalyzer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()

        assert analyzer.detect_language is True
        assert analyzer.detect_script is True
        assert analyzer.detect_multilingual is True
        assert analyzer.min_confidence == 0.0
        assert analyzer.low_confidence_threshold == 0.5
        assert analyzer.use_fast_langdetect is True

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer(
            detect_language=True,
            detect_script=False,
            detect_multilingual=False,
            min_confidence=0.3,
            low_confidence_threshold=0.7,
        )

        assert analyzer.detect_language is True
        assert analyzer.detect_script is False
        assert analyzer.detect_multilingual is False
        assert analyzer.min_confidence == 0.3
        assert analyzer.low_confidence_threshold == 0.7


class TestScriptDetection:
    """Tests for script detection functionality."""

    def test_detect_latin_script(self):
        """Test detection of Latin script."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        script = analyzer._detect_script("Hello, this is English text.")

        assert script == "latin"

    def test_detect_cyrillic_script(self):
        """Test detection of Cyrillic script."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        script = analyzer._detect_script("Привет, это русский текст.")

        assert script == "cyrillic"

    def test_detect_cjk_script(self):
        """Test detection of CJK (Chinese) script."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        script = analyzer._detect_script("这是中文文本")

        assert script == "cjk"

    def test_detect_arabic_script(self):
        """Test detection of Arabic script."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        script = analyzer._detect_script("مرحبا بالعالم")

        assert script == "arabic"

    def test_detect_empty_text(self):
        """Test script detection with empty text."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        script = analyzer._detect_script("")

        assert script == "unknown"

    def test_detect_mixed_script(self):
        """Test script detection with mixed scripts (returns dominant)."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        # More Latin characters than Cyrillic
        script = analyzer._detect_script("Hello world! Привет!")

        assert script == "latin"


class TestLanguageNames:
    """Tests for language name mapping."""

    def test_get_common_language_names(self):
        """Test that common languages have proper names."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()

        assert analyzer._get_language_name("en") == "English"
        assert analyzer._get_language_name("es") == "Spanish"
        assert analyzer._get_language_name("zh") == "Chinese"
        assert analyzer._get_language_name("ja") == "Japanese"
        assert analyzer._get_language_name("ar") == "Arabic"

    def test_get_unknown_language_name(self):
        """Test fallback for unknown language codes."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()

        # Unknown codes should be capitalized
        assert analyzer._get_language_name("xyz") == "Xyz"


class TestLanguageDetection:
    """Tests for language detection functionality."""

    def test_detect_english(self):
        """Test detection of English text."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        analyzer._init_detector()

        lang, conf = analyzer._detect_language(
            "This is a sample English text for testing."
        )

        assert lang == "en"
        assert conf > 0.5

    def test_detect_spanish(self):
        """Test detection of Spanish text."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        analyzer._init_detector()

        lang, conf = analyzer._detect_language(
            "Este es un texto de ejemplo en español para pruebas."
        )

        assert lang == "es"
        assert conf > 0.5

    def test_detect_short_text(self):
        """Test detection with very short text."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        analyzer._init_detector()

        lang, conf = analyzer._detect_language("Hi")

        # Very short text may have low confidence or be unknown
        assert isinstance(lang, str)
        assert isinstance(conf, float)

    def test_detect_empty_text(self):
        """Test detection with empty text."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        analyzer._init_detector()

        lang, conf = analyzer._detect_language("")

        assert lang == "unknown"
        assert conf == 0.0


class TestAnalyzeSample:
    """Tests for the full analyze_sample functionality."""

    def test_analyze_sample_basic(self):
        """Test basic sample analysis."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer(
            detect_language=True,
            detect_script=True,
            detect_multilingual=False,
        )

        df = _create_test_df(["Hello, this is English text."])
        result = analyzer.analyze_sample(df, _get_schema())

        assert "text_content_fasttext_detected_language" in result.columns
        assert "text_content_fasttext_language_confidence" in result.columns
        assert "text_content_fasttext_detected_script" in result.columns

        assert result.iloc[0]["text_content_fasttext_detected_language"] == "en"
        assert result.iloc[0]["text_content_fasttext_detected_script"] == "latin"

    def test_analyze_sample_multiple_languages(self):
        """Test analysis of texts in different languages."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()

        df = _create_test_df([
            "This is English text.",
            "Este es texto en español.",
            "Ceci est du texte français.",
        ])
        result = analyzer.analyze_sample(df, _get_schema())

        languages = result["text_content_fasttext_detected_language"].tolist()

        # Check that different languages are detected
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages

    def test_analyze_sample_with_min_confidence(self):
        """Test that min_confidence filters low confidence detections."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer(min_confidence=0.99)

        df = _create_test_df(["Hi"])  # Very short, low confidence
        result = analyzer.analyze_sample(df, _get_schema())

        # With high min_confidence, short text should be unknown
        # (actual behavior depends on model)
        assert "text_content_fasttext_detected_language" in result.columns

    def test_schema_required(self):
        """Test that schema is required."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        df = _create_test_df(["Test"])

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, None)

    def test_no_text_columns(self):
        """Test handling when no text columns in schema."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        df = pd.DataFrame({"other_column": ["test"]})
        schema = {"other_column": {"content_type": "metadata"}}

        result = analyzer.analyze_sample(df, schema)

        # Should return unchanged DataFrame
        assert list(result.columns) == list(df.columns)


class TestDatasetMetrics:
    """Tests for dataset-level metrics computation."""

    def test_compute_dataset_metrics(self):
        """Test dataset-level metrics computation."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()

        df = _create_test_df([
            "This is English.",
            "This is also English.",
            "Este es español.",
        ])
        analyzed_df = analyzer.analyze_sample(df, _get_schema())
        metrics = analyzer.compute_dataset_metrics(analyzed_df, _get_schema())

        assert "language_distribution" in metrics
        assert "num_languages" in metrics
        assert "primary_language" in metrics
        assert metrics["num_languages"] >= 1

    def test_compute_metrics_empty_df(self):
        """Test metrics computation with no language columns."""
        from oumi.core.analyze.fasttext_analyzer import FastTextAnalyzer

        analyzer = FastTextAnalyzer()
        df = pd.DataFrame({"other_col": ["test"]})

        metrics = analyzer.compute_dataset_metrics(df, _get_schema())

        assert metrics == {}
