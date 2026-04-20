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
    assert config.id == "length"


def test_analyzer_config_defaults_id_to_display_name():
    """id falls back to display_name when omitted."""
    config = AnalyzerConfig(type="length", display_name="My Length")
    assert config.display_name == "My Length"
    assert config.id == "My Length"


def test_analyzer_config_preserves_explicit_id_and_display_name():
    """Explicit id and display_name both survive __post_init__."""
    config = AnalyzerConfig(type="length", id="asset-123", display_name="My Length")
    assert config.type == "length"
    assert config.id == "asset-123"
    assert config.display_name == "My Length"


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
    assert config.analyzers[0].id == "length"
    assert config.analyzers[1].type == "quality"
    assert config.analyzers[1].display_name == "quality_check"
    assert config.analyzers[1].id == "quality_check"
    assert config.analyzers[2].type == "turn_stats"


def test_from_dict_accepts_explicit_id():
    """An explicit id in YAML becomes the canonical identity."""
    data = {
        "analyzers": [
            {"type": "length", "id": "asset-1", "display_name": "Length A"},
            {"type": "length", "id": "asset-2", "display_name": "Length B"},
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert [a.id for a in config.analyzers] == ["asset-1", "asset-2"]
    assert [a.display_name for a in config.analyzers] == ["Length A", "Length B"]


def test_from_dict_allows_duplicate_display_names_when_ids_differ():
    """Two analyzers may share a display_name as long as ids differ."""
    data = {
        "analyzers": [
            {"type": "length", "id": "asset-1", "display_name": "Length"},
            {"type": "length", "id": "asset-2", "display_name": "Length"},
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert len(config.analyzers) == 2
    assert config.analyzers[0].display_name == "Length"
    assert config.analyzers[1].display_name == "Length"


def test_from_dict_raises_on_duplicate_ids():
    """Duplicate id values raise."""
    data = {
        "analyzers": [
            {"type": "length", "id": "dup"},
            {"type": "quality", "id": "dup"},
        ]
    }

    with pytest.raises(ValueError, match="Duplicate analyzer id"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_raises_on_duplicate_defaulted_ids():
    """Duplicate default ids (same type, no id/display_name) raise."""
    data = {
        "analyzers": [
            {"type": "length"},
            {"type": "length"},
        ]
    }

    with pytest.raises(ValueError, match="Duplicate analyzer id"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_accepts_legacy_instance_id_with_deprecation():
    """Legacy instance_id key still maps to display_name (with a warning)."""
    data = {
        "analyzers": [
            {"type": "length", "instance_id": "length_custom"},
        ]
    }

    with pytest.warns(DeprecationWarning, match="instance_id"):
        config = TypedAnalyzeConfig.from_dict(data)

    assert config.analyzers[0].display_name == "length_custom"
    assert config.analyzers[0].id == "length_custom"


def test_from_dict_handles_both_instance_id_and_display_name():
    """When both legacy and new keys are present, display_name wins."""
    data = {
        "analyzers": [
            {
                "type": "length",
                "instance_id": "legacy_name",
                "display_name": "new_name",
            },
        ]
    }

    with pytest.warns(DeprecationWarning, match="instance_id"):
        config = TypedAnalyzeConfig.from_dict(data)

    assert config.analyzers[0].display_name == "new_name"


def test_from_dict_rejects_legacy_id_as_type():
    """Pre-#2376 YAMLs that used `id` in place of `type` now fail loudly.

    `id` has new semantics (stable identity), so a YAML entry with only
    `id` and no `type` triggers the standard 'type is required' error.
    """
    data = {"analyzers": [{"id": "length"}]}

    with pytest.raises(ValueError, match="type is required"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_empty_analyzers():
    """Empty analyzers list is valid."""
    data = {"analyzers": []}

    config = TypedAnalyzeConfig.from_dict(data)

    assert config.analyzers == []


# -----------------------------------------------------------------------------
# Tests: TypedAnalyzeConfig.to_dict
# -----------------------------------------------------------------------------


def test_to_dict_includes_id_field():
    """to_dict round-trips type, id, and display_name."""
    config = TypedAnalyzeConfig(
        analyzers=[
            AnalyzerConfig(type="length", id="asset-1", display_name="Length"),
        ],
    )

    out = config.to_dict()

    assert out["analyzers"][0]["type"] == "length"
    assert out["analyzers"][0]["id"] == "asset-1"
    assert out["analyzers"][0]["display_name"] == "Length"


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
