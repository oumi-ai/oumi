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

"""Tests for analyze config module."""

import pytest

from oumi.analyze.config import AnalyzerConfig, TypedAnalyzeConfig

# -----------------------------------------------------------------------------
# Tests: AnalyzerConfig
# -----------------------------------------------------------------------------


def test_analyzer_config_defaults_display_name_to_type():
    """display_name falls back to type when omitted."""
    config = AnalyzerConfig(type="length")
    assert config.type == "length"
    assert config.display_name == "length"


def test_analyzer_config_preserves_explicit_display_name():
    """Explicit display_name overrides the default."""
    config = AnalyzerConfig(type="length", display_name="length_custom")
    assert config.display_name == "length_custom"


def test_analyzer_config_requires_type():
    """type must be provided."""
    with pytest.raises(ValueError, match="type is required"):
        AnalyzerConfig()


# -----------------------------------------------------------------------------
# Tests: TypedAnalyzeConfig.from_dict
# -----------------------------------------------------------------------------


def test_from_dict_parses_analyzers():
    """Parse analyzers using new field names."""
    data = {
        "analyzers": [
            {"type": "length"},
            {"type": "quality", "display_name": "quality_check"},
            "turn_stats",  # String shorthand
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert len(config.analyzers) == 3
    assert config.analyzers[0].type == "length"
    assert config.analyzers[0].display_name == "length"
    assert config.analyzers[1].type == "quality"
    assert config.analyzers[1].display_name == "quality_check"
    assert config.analyzers[2].type == "turn_stats"


def test_from_dict_accepts_legacy_id_aliases():
    """Legacy id/instance_id keys map to type/display_name."""
    data = {
        "analyzers": [
            {"id": "length"},
            {"id": "quality", "instance_id": "quality_check"},
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert config.analyzers[0].type == "length"
    assert config.analyzers[0].display_name == "length"
    assert config.analyzers[1].type == "quality"
    assert config.analyzers[1].display_name == "quality_check"


def test_from_dict_raises_on_duplicate_display_names():
    """Duplicate display_name values raise."""
    data = {
        "analyzers": [
            {"type": "length"},
            {"type": "quality", "display_name": "length"},  # Duplicate!
        ]
    }

    with pytest.raises(ValueError, match="Duplicate analyzer display_name"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_raises_on_duplicate_default_display_names():
    """Duplicate default display_names (from same type) raise."""
    data = {
        "analyzers": [
            {"type": "length"},
            {"type": "length"},  # Same type -> same default display_name
        ]
    }

    with pytest.raises(ValueError, match="Duplicate analyzer display_name"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_allows_same_type_with_different_display_names():
    """Same analyzer type with different display_names is allowed."""
    data = {
        "analyzers": [
            {"type": "length", "display_name": "length_1"},
            {"type": "length", "display_name": "length_2"},
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert len(config.analyzers) == 2
    assert config.analyzers[0].display_name == "length_1"
    assert config.analyzers[1].display_name == "length_2"


def test_from_dict_empty_analyzers():
    """Empty analyzers list is valid."""
    data = {"analyzers": []}

    config = TypedAnalyzeConfig.from_dict(data)

    assert config.analyzers == []


# -----------------------------------------------------------------------------
# Tests: Custom Code Security
# -----------------------------------------------------------------------------


def test_from_dict_rejects_custom_code_by_default():
    """Custom metrics with code are rejected by default."""
    data = {
        "custom_metrics": [
            {
                "id": "my_metric",
                "function": "def compute(x): return {'value': 1}",
            }
        ]
    }

    with pytest.raises(ValueError, match="executable code"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_allows_custom_code_when_opted_in():
    """Custom metrics with code work when allow_custom_code=True."""
    data = {
        "custom_metrics": [
            {
                "id": "my_metric",
                "function": "def compute(x): return {'value': 1}",
            }
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data, allow_custom_code=True)

    assert len(config.custom_metrics) == 1
    assert config.custom_metrics[0].id == "my_metric"


def test_from_dict_allows_empty_function():
    """Custom metrics without function code are allowed."""
    data = {
        "custom_metrics": [
            {
                "id": "my_metric",
                "function": "",  # Empty function - no code execution
            }
        ]
    }

    # Should not raise - no executable code
    config = TypedAnalyzeConfig.from_dict(data)

    assert len(config.custom_metrics) == 1
