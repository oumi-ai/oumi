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


def test_analyzer_config_auto_populates_display_name():
    """Test that display_name defaults to type if not provided."""
    config = AnalyzerConfig(type="length")
    assert config.display_name == "length"


def test_analyzer_config_preserves_explicit_display_name():
    """Test that explicit display_name is preserved."""
    config = AnalyzerConfig(type="length", display_name="Length")
    assert config.display_name == "Length"


# -----------------------------------------------------------------------------
# Tests: TypedAnalyzeConfig.from_dict
# -----------------------------------------------------------------------------


def test_from_dict_parses_analyzers():
    """Test parsing analyzers from dict."""
    data = {
        "analyzers": [
            {"type": "length", "display_name": "Length"},
            {"type": "quality", "display_name": "Quality Check"},
            "turn_stats",  # String shorthand
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert len(config.analyzers) == 3
    assert config.analyzers[0].type == "length"
    assert config.analyzers[0].display_name == "Length"
    assert config.analyzers[1].type == "quality"
    assert config.analyzers[1].display_name == "Quality Check"
    assert config.analyzers[2].type == "turn_stats"
    assert config.analyzers[2].display_name == "turn_stats"


def test_from_dict_backward_compat_id_and_instance_id():
    """Test that old id/instance_id field names still work."""
    data = {
        "analyzers": [
            {"id": "length"},
            {"id": "quality", "instance_id": "quality_check"},
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert len(config.analyzers) == 2
    assert config.analyzers[0].type == "length"
    assert config.analyzers[0].display_name == "length"
    assert config.analyzers[1].type == "quality"
    assert config.analyzers[1].display_name == "quality_check"


def test_from_dict_raises_on_duplicate_display_names():
    """Test that duplicate display_name values raise an error."""
    data = {
        "analyzers": [
            {"type": "length", "display_name": "Length"},
            {"type": "quality", "display_name": "Length"},  # Duplicate!
        ]
    }

    with pytest.raises(ValueError, match="Duplicate analyzer display_name"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_raises_on_duplicate_default_display_names():
    """Test that duplicate default display_names (from same type) raise an error."""
    data = {
        "analyzers": [
            {"type": "length"},
            {"type": "length"},  # Same type -> same default display_name
        ]
    }

    with pytest.raises(ValueError, match="Duplicate analyzer display_name"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_allows_same_type_with_different_display_names():
    """Test that same analyzer type with different display_names is allowed."""
    data = {
        "analyzers": [
            {"type": "length", "display_name": "Length 1"},
            {"type": "length", "display_name": "Length 2"},
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert len(config.analyzers) == 2
    assert config.analyzers[0].display_name == "Length 1"
    assert config.analyzers[1].display_name == "Length 2"


def test_from_dict_empty_analyzers():
    """Test that empty analyzers list is valid."""
    data = {"analyzers": []}

    config = TypedAnalyzeConfig.from_dict(data)

    assert config.analyzers == []


def test_from_dict_type_takes_precedence_over_id():
    """Test that 'type' takes precedence when both 'type' and 'id' are provided."""
    data = {
        "analyzers": [
            {"type": "quality", "id": "length"},  # type wins
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)
    assert config.analyzers[0].type == "quality"


def test_from_dict_display_name_takes_precedence_over_instance_id():
    """Test that 'display_name' takes precedence over 'instance_id'."""
    data = {
        "analyzers": [
            {
                "type": "length",
                "display_name": "New Name",
                "instance_id": "old_name",
            },
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)
    assert config.analyzers[0].display_name == "New Name"


def test_from_dict_test_display_name():
    """Test that YAML display_name is mapped to TestParams.title."""
    data = {
        "tests": [
            {
                "id": "check_tokens",
                "type": "threshold",
                "metric": "Length.total_tokens",
                "display_name": "Token count check",
            }
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)
    assert config.tests[0].title == "Token count check"


def test_from_dict_test_title_backward_compat():
    """Test that 'title' works directly in YAML test configs."""
    data = {
        "tests": [
            {
                "id": "check_tokens",
                "type": "threshold",
                "metric": "Length.total_tokens",
                "title": "Token count check",
            }
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)
    assert config.tests[0].title == "Token count check"


# -----------------------------------------------------------------------------
# Tests: Unknown YAML fields
# -----------------------------------------------------------------------------


def test_from_dict_unknown_analyzer_field_raises_value_error():
    """Test that typos in analyzer config raise ValueError."""
    data = {
        "analyzers": [
            {"type": "length", "typo_field": "oops"},
        ]
    }

    with pytest.raises(ValueError, match="Invalid analyzer config"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_unknown_test_field_raises_value_error():
    """Test that typos in test config raise ValueError."""
    data = {
        "tests": [
            {
                "id": "check",
                "type": "threshold",
                "metric": "Length.total_tokens",
                "oops": "bad_field",
            }
        ]
    }

    with pytest.raises(ValueError, match="Invalid test config"):
        TypedAnalyzeConfig.from_dict(data)
