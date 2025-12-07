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

"""Unit tests for the analyzer presets module."""

import pytest

from oumi.core.analyze.presets import (
    get_preset,
    get_preset_with_language_detection,
    get_preset_with_tokenizer,
    list_presets,
)
from oumi.core.configs.analyze_config import SampleAnalyzerParams


class TestListPresets:
    """Tests for list_presets function."""

    def test_list_presets_returns_dict(self):
        """Test that list_presets returns a dictionary."""
        presets = list_presets()
        assert isinstance(presets, dict)

    def test_list_presets_contains_sft_quality(self):
        """Test that sft_quality preset is listed."""
        presets = list_presets()
        assert "sft_quality" in presets
        assert isinstance(presets["sft_quality"], str)
        assert len(presets["sft_quality"]) > 0


class TestGetPreset:
    """Tests for get_preset function."""

    def test_get_sft_quality_preset(self):
        """Test getting the sft_quality preset."""
        analyzers = get_preset("sft_quality")

        assert isinstance(analyzers, list)
        assert len(analyzers) > 0
        assert all(isinstance(a, SampleAnalyzerParams) for a in analyzers)

    def test_sft_quality_preset_contains_expected_analyzers(self):
        """Test that sft_quality preset contains expected analyzers."""
        analyzers = get_preset("sft_quality")
        analyzer_ids = [a.id for a in analyzers]

        # Should contain length, diversity, format, and quality
        assert "length" in analyzer_ids
        assert "diversity" in analyzer_ids
        assert "format" in analyzer_ids
        assert "quality" in analyzer_ids

    def test_sft_quality_preset_length_params(self):
        """Test length analyzer params in sft_quality preset."""
        analyzers = get_preset("sft_quality")
        length_analyzer = next(a for a in analyzers if a.id == "length")

        assert length_analyzer.params["char_count"] is True
        assert length_analyzer.params["word_count"] is True
        assert length_analyzer.params["sentence_count"] is True
        # Token count disabled by default (requires tokenizer)
        assert length_analyzer.params["token_count"] is False

    def test_sft_quality_preset_quality_params(self):
        """Test quality analyzer params in sft_quality preset."""
        analyzers = get_preset("sft_quality")
        quality_analyzer = next(a for a in analyzers if a.id == "quality")

        assert quality_analyzer.params["detect_pii"] is True
        assert quality_analyzer.params["detect_encoding_issues"] is True
        assert quality_analyzer.params["detect_special_tokens"] is True
        assert quality_analyzer.params["detect_repetition"] is True
        assert quality_analyzer.params["compute_quality_score"] is True
        # Language detection disabled by default (requires langdetect)
        assert quality_analyzer.params["detect_language"] is False

    def test_unknown_preset_raises_error(self):
        """Test that unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent_preset")

    def test_error_message_lists_available_presets(self):
        """Test that error message lists available presets."""
        with pytest.raises(ValueError) as exc_info:
            get_preset("nonexistent_preset")

        error_msg = str(exc_info.value)
        assert "sft_quality" in error_msg


class TestGetPresetWithTokenizer:
    """Tests for get_preset_with_tokenizer function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        result = get_preset_with_tokenizer("sft_quality", "gpt2")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_enables_token_count(self):
        """Test that token counting is enabled."""
        analyzers, tok_config = get_preset_with_tokenizer("sft_quality", "gpt2")
        length_analyzer = next(a for a in analyzers if a.id == "length")

        assert length_analyzer.params["token_count"] is True

    def test_returns_tokenizer_config(self):
        """Test that tokenizer config is returned."""
        analyzers, tok_config = get_preset_with_tokenizer(
            "sft_quality", "meta-llama/Llama-3.1-8B"
        )

        assert "tokenizer_name" in tok_config
        assert tok_config["tokenizer_name"] == "meta-llama/Llama-3.1-8B"
        assert "tokenizer_kwargs" in tok_config
        assert "trust_remote_code" in tok_config


class TestGetPresetWithLanguageDetection:
    """Tests for get_preset_with_language_detection function."""

    def test_enables_language_detection(self):
        """Test that language detection is enabled."""
        analyzers = get_preset_with_language_detection("sft_quality")
        quality_analyzer = next(a for a in analyzers if a.id == "quality")

        assert quality_analyzer.params["detect_language"] is True

    def test_preserves_other_params(self):
        """Test that other quality params are preserved."""
        analyzers = get_preset_with_language_detection("sft_quality")
        quality_analyzer = next(a for a in analyzers if a.id == "quality")

        # Other params should still be set
        assert quality_analyzer.params["detect_pii"] is True
        assert quality_analyzer.params["detect_encoding_issues"] is True
        assert quality_analyzer.params["compute_quality_score"] is True


class TestPresetImmutability:
    """Tests to ensure presets don't mutate when modified."""

    def test_modifications_dont_affect_original(self):
        """Test that modifying returned preset doesn't affect original."""
        # Get preset twice
        analyzers1 = get_preset("sft_quality")
        analyzers2 = get_preset("sft_quality")

        # Modify first one
        length1 = next(a for a in analyzers1 if a.id == "length")
        length1.params["char_count"] = False

        # Second one should be unchanged
        length2 = next(a for a in analyzers2 if a.id == "length")
        assert length2.params["char_count"] is True
