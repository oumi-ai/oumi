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
    SampleAnalyzerParams,
)


def test_sample_analyzer_param_validation_success():
    """Test successful validation of SampleAnalyzerParams."""
    # Should not raise any exception during __post_init__
    analyzer = SampleAnalyzerParams(id="test_analyzer")
    assert analyzer.id == "test_analyzer"


def test_sample_analyzer_param_validation_missing_id():
    """Test validation failure when id is missing."""
    with pytest.raises(ValueError, match="Analyzer 'id' must be provided"):
        AnalyzeConfig(
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


def test_analyze_config_deprecated_is_multimodal_warning():
    """Test that using is_multimodal emits a deprecation warning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        AnalyzeConfig(
            dataset_path="/path/to/dataset.json",
            is_multimodal=True,
            processor_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        )
        # Check deprecation warning was emitted
        assert any("is_multimodal" in str(warning.message) for warning in w)
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)


def test_analyze_config_auto_detect_multimodal_from_processor():
    """Test that multimodality is auto-detected from processor_name presence."""
    # With processor_name: multimodal (no error)
    config_multimodal = AnalyzeConfig(
        dataset_path="/path/to/dataset.json",
        processor_name="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    assert config_multimodal.processor_name is not None

    # Without processor_name: text-only (no error)
    config_text = AnalyzeConfig(
        dataset_path="/path/to/dataset.json",
    )
    assert config_text.processor_name is None


def test_analyze_config_dataset_path_without_is_multimodal():
    """Test that dataset_path works without is_multimodal (deprecated field)."""
    # This should not raise an error anymore since is_multimodal is deprecated
    config = AnalyzeConfig(
        dataset_path="/path/to/dataset.json",
    )
    assert config.dataset_path == "/path/to/dataset.json"
    assert config.is_multimodal is None  # Not set


def test_analyze_config_deprecated_is_multimodal_still_accepted():
    """Test that is_multimodal is still accepted (for backwards compatibility)
    but emits a deprecation warning."""
    import warnings

    # Should work with is_multimodal=True (deprecated but accepted)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        config = AnalyzeConfig(
            dataset_path="/path/to/dataset.json",
            is_multimodal=True,
            processor_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        )
        assert config.is_multimodal is True

    # Should work with is_multimodal=False (deprecated but accepted)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        config = AnalyzeConfig(
            dataset_path="/path/to/dataset.json",
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
    config = AnalyzeConfig(dataset_name="test_dataset")

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
