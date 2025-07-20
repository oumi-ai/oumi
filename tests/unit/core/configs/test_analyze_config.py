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

import pytest

from oumi.core.configs.analyze_config import AnalyzeConfig, SampleAnalyzerParams


def test_sample_analyzer_param_validation_success():
    """Test successful validation of SampleAnalyzerParams."""
    # Should not raise any exception during __post_init__
    analyzer = SampleAnalyzerParams(id="test_analyzer")
    assert analyzer.id == "test_analyzer"


def test_sample_analyzer_param_validation_missing_id():
    """Test validation failure when id is missing."""
    with pytest.raises(ValueError, match="Analyzer 'id' must be provided"):
        AnalyzeConfig(
            dataset_name="test_dataset", analyzers=[SampleAnalyzerParams(id="")]
        )


def test_sample_analyzer_param_with_complex_config():
    """Test SampleAnalyzerParams with complex configuration."""
    complex_config = {
        "nested": {"key": "value"},
        "list": [1, 2, 3],
        "boolean": True,
        "number": 3.14,
    }

    analyzer = SampleAnalyzerParams(id="complex_analyzer", config=complex_config)

    assert analyzer.id == "complex_analyzer"
    assert analyzer.config == complex_config


def test_analyze_config_validation_missing_dataset_name():
    """Test validation failure when dataset_name is missing."""
    with pytest.raises(ValueError, match="'dataset_name' must be provided"):
        AnalyzeConfig(dataset_name=None)


def test_analyze_config_validation_empty_dataset_name():
    """Test validation failure when dataset_name is empty."""
    with pytest.raises(ValueError, match="'dataset_name' must be provided"):
        AnalyzeConfig(dataset_name="")


def test_analyze_config_validation_with_valid_analyzers():
    """Test validation with valid analyzer configurations."""
    analyzers = [
        SampleAnalyzerParams(id="analyzer1"),
        SampleAnalyzerParams(id="analyzer2"),
    ]

    # Should not raise any exception during __post_init__
    AnalyzeConfig(dataset_name="test_dataset", analyzers=analyzers)


def test_analyze_config_validation_duplicate_analyzer_ids():
    """Test validation failure with duplicate analyzer IDs."""
    analyzers = [
        SampleAnalyzerParams(id="duplicate_id"),
        SampleAnalyzerParams(id="duplicate_id"),
    ]

    with pytest.raises(ValueError, match="Duplicate analyzer ID found: 'duplicate_id'"):
        AnalyzeConfig(dataset_name="test_dataset", analyzers=analyzers)


def test_analyze_config_default_values():
    """Test that AnalyzeConfig has correct default values."""
    config = AnalyzeConfig(dataset_name="test_dataset")

    assert config.dataset_name == "test_dataset"
    assert config.split == "train"  # default value
    assert config.sample_count is None  # default value
    assert config.output_path == "."  # default value
    assert config.analyzers == []  # default value


def test_analyze_config_with_custom_values():
    """Test AnalyzeConfig with custom parameter values."""
    analyzers = [
        SampleAnalyzerParams(id="analyzer1", config={"param1": "value1"}),
        SampleAnalyzerParams(id="analyzer2", config={"param2": "value2"}),
    ]

    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="test",
        sample_count=100,
        output_path="/tmp/output",
        analyzers=analyzers,
    )

    assert config.dataset_name == "test_dataset"
    assert config.split == "test"
    assert config.sample_count == 100
    assert config.output_path == "/tmp/output"
    assert len(config.analyzers) == 2
    assert config.analyzers[0].id == "analyzer1"
    assert config.analyzers[1].id == "analyzer2"
