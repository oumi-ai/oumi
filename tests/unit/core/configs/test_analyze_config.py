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

from oumi.core.configs.analyze_config import (
    AnalyzeConfig,
    DatasetSource,
    SampleAnalyzerParams,
)


def test_dataset_source_required_field():
    """Test that dataset_source is a required field."""
    # Should work without dataset_source (MISSING becomes '???')
    config = AnalyzeConfig()
    assert config.dataset_source == "???"
    assert config.dataset_name == "Custom Dataset"

    # Should work with dataset_source
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG, dataset_name="test_dataset"
    )
    assert config.dataset_source == DatasetSource.CONFIG
    assert config.dataset_name == "test_dataset"


def test_dataset_source_validation_success():
    """Test successful validation of dataset_source values."""
    # Test CONFIG mode
    config_config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG, dataset_name="test_dataset"
    )
    assert config_config.dataset_source == DatasetSource.CONFIG
    assert config_config.dataset_name == "test_dataset"

    # Test DIRECT mode
    config_direct = AnalyzeConfig(
        dataset_source=DatasetSource.DIRECT, dataset_name="test_dataset"
    )
    assert config_direct.dataset_source == DatasetSource.DIRECT
    assert config_direct.dataset_name == "test_dataset"


def test_dataset_source_validation_invalid_value():
    """Test validation failure with invalid dataset_source value."""
    # The validation for invalid dataset_source values happens in DatasetAnalyzer,
    # not in AnalyzeConfig
    # So this should actually work (though it's not a valid enum value)
    config = AnalyzeConfig(dataset_source="invalid_value", dataset_name="test_dataset")  # type: ignore
    assert config.dataset_source == "invalid_value"

    # The validation will fail when trying to use this config in DatasetAnalyzer
    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

    with pytest.raises(ValueError, match="Invalid dataset_source: invalid_value"):
        DatasetAnalyzer(config)


def test_sample_analyzer_param_validation_success():
    """Test successful validation of SampleAnalyzerParams."""
    # Should not raise any exception during __post_init__
    analyzer = SampleAnalyzerParams(id="test_analyzer")
    assert analyzer.id == "test_analyzer"


def test_sample_analyzer_param_validation_missing_id():
    """Test validation failure when id is missing."""
    with pytest.raises(ValueError, match="Analyzer 'id' must be provided"):
        AnalyzeConfig(
            dataset_source=DatasetSource.CONFIG,  # Required field
            dataset_name="test_dataset",
            analyzers=[SampleAnalyzerParams(id="")],
        )


def test_sample_analyzer_param_with_language_detection_params():
    """Test SampleAnalyzerParams with language detection analyzer configuration."""
    language_detection_params = {
        "confidence_threshold": 0.2,
        "top_k": 3,
        "multilingual_flag": {
            "enabled": True,
            "min_num_languages": 2,
        },
    }

    analyzer = SampleAnalyzerParams(
        id="language_detection", params=language_detection_params
    )

    assert analyzer.id == "language_detection"
    assert analyzer.params == language_detection_params


def test_analyze_config_validation_missing_dataset_name():
    """Test validation failure when dataset_name is missing."""
    with pytest.raises(
        ValueError,
        match="Either 'dataset_name' or 'dataset_path' must be provided when "
        "dataset_source=DatasetSource.CONFIG",
    ):
        AnalyzeConfig(
            dataset_source=DatasetSource.CONFIG,  # Required field
            dataset_name=None,
        )


def test_analyze_config_validation_empty_dataset_name():
    """Test validation failure when dataset_name is empty."""
    with pytest.raises(
        ValueError, match="Either 'dataset_name' or 'dataset_path' must be provided"
    ):
        AnalyzeConfig(
            dataset_source=DatasetSource.CONFIG,  # Required field
            dataset_name="",
        )


def test_analyze_config_validation_missing_dataset_path():
    """Test validation failure when both dataset_name and dataset_path are missing."""
    with pytest.raises(
        ValueError, match="Either 'dataset_name' or 'dataset_path' must be provided"
    ):
        AnalyzeConfig(
            dataset_source=DatasetSource.CONFIG,  # Required field
            dataset_name=None,
            dataset_path=None,
        )


def test_analyze_config_validation_empty_dataset_path():
    """Test validation failure when both dataset_name and dataset_path are empty."""
    with pytest.raises(
        ValueError, match="Either 'dataset_name' or 'dataset_path' must be provided"
    ):
        AnalyzeConfig(
            dataset_source=DatasetSource.CONFIG,  # Required field
            dataset_name="",
            dataset_path="",
        )


def test_analyze_config_validation_missing_processor_when_multimodal():
    """Test validation failure when processor_name is missing but
    is_multimodal is True."""
    with pytest.raises(
        ValueError,
        match="'processor_name' must be specified when 'is_multimodal' is True",
    ):
        AnalyzeConfig(
            dataset_source=DatasetSource.CONFIG,  # Required field
            dataset_path="/path/to/dataset.json",
            dataset_format="oumi",
            is_multimodal=True,
            processor_name=None,
        )


def test_analyze_config_validation_empty_processor_when_multimodal():
    """Test validation failure when processor_name is empty but is_multimodal
    is True."""
    with pytest.raises(
        ValueError,
        match="'processor_name' must be specified when 'is_multimodal' is True",
    ):
        AnalyzeConfig(
            dataset_path="/path/to/dataset.json",
            dataset_format="oumi",
            is_multimodal=True,
            processor_name="",
        )


def test_analyze_config_validation_multimodal_wrong_format():
    """Test validation failure when is_multimodal is True but dataset_format is
    not 'oumi'."""
    with pytest.raises(
        ValueError, match="Multimodal datasets require dataset_format='oumi'"
    ):
        AnalyzeConfig(
            dataset_path="/path/to/dataset.json",
            dataset_format="alpaca",
            is_multimodal=True,
            processor_name="openai/clip-vit-base-patch32",
        )


def test_analyze_config_validation_missing_dataset_format():
    """Test validation failure when dataset_path is provided but dataset_format
    is missing."""
    with pytest.raises(
        ValueError, match="'dataset_format' must be specified when using 'dataset_path'"
    ):
        AnalyzeConfig(dataset_path="/path/to/dataset.json")


def test_analyze_config_validation_invalid_dataset_format():
    """Test validation failure when dataset_format is not 'oumi' or 'alpaca'."""
    with pytest.raises(
        ValueError, match="'dataset_format' must be either 'oumi' or 'alpaca'"
    ):
        AnalyzeConfig(
            dataset_path="/path/to/dataset.json",
            dataset_format="invalid_format",
        )


def test_analyze_config_validation_missing_is_multimodal():
    """Test validation failure when dataset_path is provided but is_multimodal is
    missing."""
    with pytest.raises(
        ValueError, match="'is_multimodal' must be specified when using 'dataset_path'"
    ):
        AnalyzeConfig(
            dataset_path="/path/to/dataset.json",
            dataset_format="oumi",
            # Missing is_multimodal
        )


def test_analyze_config_validation_is_multimodal_required():
    """Test that is_multimodal can be explicitly set to True or False for
    custom datasets."""
    # Should work with is_multimodal=True
    config = AnalyzeConfig(
        dataset_path="/path/to/dataset.json",
        dataset_format="oumi",
        is_multimodal=True,
        processor_name="openai/clip-vit-base-patch32",
    )
    assert config.is_multimodal is True

    # Should work with is_multimodal=False
    config = AnalyzeConfig(
        dataset_path="/path/to/dataset.json",
        dataset_format="oumi",
        is_multimodal=False,
    )
    assert config.is_multimodal is False


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
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG, dataset_name="test_dataset"
    )

    assert config.dataset_source == DatasetSource.CONFIG
    assert config.dataset_name == "test_dataset"
    assert config.split == "train"  # default value
    assert config.sample_count is None  # default value
    assert config.output_path == "."  # default value
    assert config.analyzers == []  # default value


def test_analyze_config_with_custom_values():
    """Test AnalyzeConfig with custom parameter values."""
    analyzers = [
        SampleAnalyzerParams(id="analyzer1", params={"param1": "value1"}),
        SampleAnalyzerParams(id="analyzer2", params={"param2": "value2"}),
    ]

    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="test",
        sample_count=100,
        output_path="/tmp/output",
        analyzers=analyzers,
        processor_name="Salesforce/blip2-opt-2.7b",
        processor_kwargs={"image_size": 224, "do_resize": True},
        trust_remote_code=True,
    )

    assert config.dataset_name == "test_dataset"
    assert config.split == "test"
    assert config.sample_count == 100
    assert config.output_path == "/tmp/output"
    assert len(config.analyzers) == 2
    assert config.analyzers[0].id == "analyzer1"
    assert config.analyzers[1].id == "analyzer2"
    assert config.processor_name == "Salesforce/blip2-opt-2.7b"
    assert config.processor_kwargs == {"image_size": 224, "do_resize": True}
    assert config.trust_remote_code is True


def test_analyze_config_processor_fields_custom_values():
    """Test AnalyzeConfig with custom processor parameter values."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        processor_name="Salesforce/blip2-opt-2.7b",
        processor_kwargs={"image_size": 224, "do_resize": True},
        trust_remote_code=True,
    )

    assert config.processor_name == "Salesforce/blip2-opt-2.7b"
    assert config.processor_kwargs == {"image_size": 224, "do_resize": True}
    assert config.trust_remote_code is True


def test_analyze_config_sample_count_zero():
    """Test validation failure when sample_count is zero."""
    with pytest.raises(ValueError, match="`sample_count` must be greater than 0."):
        AnalyzeConfig(dataset_name="test_dataset", sample_count=0)


def test_analyze_config_sample_count_negative():
    """Test validation failure when sample_count is negative."""
    with pytest.raises(ValueError, match="`sample_count` must be greater than 0."):
        AnalyzeConfig(dataset_name="test_dataset", sample_count=-5)


def test_analyze_config_sample_count_valid():
    """Test that valid sample_count values are accepted."""
    # Should not raise any exception
    config = AnalyzeConfig(dataset_name="test_dataset", sample_count=1)
    assert config.sample_count == 1

    config = AnalyzeConfig(dataset_name="test_dataset", sample_count=100)
    assert config.sample_count == 100

    # None should also be valid
    config = AnalyzeConfig(dataset_name="test_dataset", sample_count=None)
    assert config.sample_count is None
