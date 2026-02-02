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


def test_analyzer_config_auto_populates_instance_id():
    """Test that instance_id defaults to id if not provided."""
    config = AnalyzerConfig(id="length")
    assert config.instance_id == "length"


def test_analyzer_config_preserves_explicit_instance_id():
    """Test that explicit instance_id is preserved."""
    config = AnalyzerConfig(id="length", instance_id="length_custom")
    assert config.instance_id == "length_custom"


# -----------------------------------------------------------------------------
# Tests: TypedAnalyzeConfig.from_dict
# -----------------------------------------------------------------------------


def test_from_dict_parses_analyzers():
    """Test parsing analyzers from dict."""
    data = {
        "analyzers": [
            {"id": "length"},
            {"id": "quality", "instance_id": "quality_check"},
            "turn_stats",  # String shorthand
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert len(config.analyzers) == 3
    assert config.analyzers[0].id == "length"
    assert config.analyzers[0].instance_id == "length"
    assert config.analyzers[1].id == "quality"
    assert config.analyzers[1].instance_id == "quality_check"
    assert config.analyzers[2].id == "turn_stats"


def test_from_dict_raises_on_duplicate_instance_ids():
    """Test that duplicate instance_id values raise an error."""
    data = {
        "analyzers": [
            {"id": "length"},
            {"id": "quality", "instance_id": "length"},  # Duplicate!
        ]
    }

    with pytest.raises(ValueError, match="Duplicate analyzer instance_id"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_raises_on_duplicate_default_instance_ids():
    """Test that duplicate default instance_ids (from same id) raise an error."""
    data = {
        "analyzers": [
            {"id": "length"},
            {"id": "length"},  # Same id -> same default instance_id
        ]
    }

    with pytest.raises(ValueError, match="Duplicate analyzer instance_id"):
        TypedAnalyzeConfig.from_dict(data)


def test_from_dict_allows_same_type_with_different_instance_ids():
    """Test that same analyzer type with different instance_ids is allowed."""
    data = {
        "analyzers": [
            {"id": "length", "instance_id": "length_1"},
            {"id": "length", "instance_id": "length_2"},
        ]
    }

    config = TypedAnalyzeConfig.from_dict(data)

    assert len(config.analyzers) == 2
    assert config.analyzers[0].instance_id == "length_1"
    assert config.analyzers[1].instance_id == "length_2"


def test_from_dict_empty_analyzers():
    """Test that empty analyzers list is valid."""
    data = {"analyzers": []}

    config = TypedAnalyzeConfig.from_dict(data)

    assert config.analyzers == []


# -----------------------------------------------------------------------------
# Tests: Custom Code Security
# -----------------------------------------------------------------------------


def test_from_dict_rejects_custom_code_by_default():
    """Test that custom metrics with code are rejected by default."""
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
    """Test that custom metrics with code work when allow_custom_code=True."""
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
    """Test that custom metrics without function code are allowed."""
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
