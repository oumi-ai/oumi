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

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    GeneratedAttribute,
    TextMessage,
    TransformationStrategy,
    TransformationType,
    TransformedAttribute,
)
from oumi.core.configs.synthesis_config import SynthesisConfig
from oumi.core.synthesis.attribute_synthesizer import AttributeSynthesizer
from oumi.core.synthesis.attribute_transformation import AttributeTransformer
from oumi.core.synthesis.data_synthesizer import DataSynthesizer
from oumi.core.synthesis.dataset_planner import DatasetPlanner
from oumi.core.synthesis.synthesis_pipeline import SynthesisPipeline
from oumi.core.types.conversation import Role


@pytest.fixture
def mock_attribute_synthesizer():
    """Create a mock attribute synthesizer."""
    return Mock(spec=AttributeSynthesizer)


@pytest.fixture
def mock_attribute_transformer():
    """Create a mock attribute transformer."""
    return Mock(spec=AttributeTransformer)


@pytest.fixture
def mock_dataset_planner():
    """Create a mock dataset planner."""
    return Mock(spec=DatasetPlanner)


@pytest.fixture
def mock_data_synthesizer():
    """Create a mock data synthesizer."""
    return Mock(spec=DataSynthesizer)


@pytest.fixture
def basic_synthesis_config():
    """Create a basic synthesis configuration."""
    return SynthesisConfig(
        num_samples=5,
        strategy_params=GeneralSynthesisParams(),
        inference_config=InferenceConfig(),
    )


@pytest.fixture
def synthesis_config_with_generated_attributes():
    """Create a synthesis config with generated attributes."""
    generated_attr = GeneratedAttribute(
        id="test_generated_attr",
        instruction_messages=[
            TextMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
            TextMessage(role=Role.USER, content="Generate some content."),
        ],
    )
    strategy_params = GeneralSynthesisParams(generated_attributes=[generated_attr])
    return SynthesisConfig(
        num_samples=3,
        strategy_params=strategy_params,
        inference_config=InferenceConfig(),
    )


@pytest.fixture
def synthesis_config_with_transformed_attributes():
    """Create a synthesis config with transformed attributes."""
    transformed_attr = TransformedAttribute(
        id="test_transformed_attr",
        transformation_strategy=TransformationStrategy(
            type=TransformationType.STRING, string_transform="some_transformation"
        ),
    )
    strategy_params = GeneralSynthesisParams(transformed_attributes=[transformed_attr])
    return SynthesisConfig(
        num_samples=3,
        strategy_params=strategy_params,
        inference_config=InferenceConfig(),
    )


@pytest.fixture
def synthesis_config_with_passthrough_attributes():
    """Create a synthesis config with passthrough attributes."""
    strategy_params = GeneralSynthesisParams(passthrough_attributes=["attr1", "attr2"])
    return SynthesisConfig(
        num_samples=2,
        strategy_params=strategy_params,
        inference_config=InferenceConfig(),
    )


@pytest.fixture
def synthesis_config_with_output_path():
    """Create a synthesis config with output path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output.jsonl"
        strategy_params = GeneralSynthesisParams()
        config = SynthesisConfig(
            output_path=str(output_path),
            num_samples=2,
            strategy_params=strategy_params,
            inference_config=InferenceConfig(),
        )
        yield config


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return [
        {"attr1": "value1", "attr2": "value2", "attr3": "value3"},
        {"attr1": "value4", "attr2": "value5", "attr3": "value6"},
    ]


@patch("oumi.core.synthesis.synthesis_pipeline.DataSynthesizer")
@patch("oumi.core.synthesis.synthesis_pipeline.DatasetPlanner")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeTransformer")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeSynthesizer")
def test_synthesis_pipeline_initialization_with_generated_attributes(
    mock_attr_synth,
    mock_attr_transformer,
    mock_dataset_planner,
    mock_data_synth,
    synthesis_config_with_generated_attributes,
):
    """Test pipeline initialization when generated attributes are present."""
    pipeline = SynthesisPipeline(synthesis_config_with_generated_attributes)

    # Should create AttributeSynthesizer and DataSynthesizer
    mock_attr_synth.assert_called_once_with(
        synthesis_config_with_generated_attributes.strategy_params,
        synthesis_config_with_generated_attributes.inference_config,
    )
    mock_data_synth.assert_called_once()
    assert pipeline._data_synthesizer is not None


@patch("oumi.core.synthesis.synthesis_pipeline.DataSynthesizer")
@patch("oumi.core.synthesis.synthesis_pipeline.DatasetPlanner")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeTransformer")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeSynthesizer")
def test_synthesis_pipeline_initialization_without_generated_attributes(
    mock_attr_synth,
    mock_attr_transformer,
    mock_dataset_planner,
    mock_data_synth,
    basic_synthesis_config,
):
    """Test pipeline initialization when no generated attributes are present."""
    pipeline = SynthesisPipeline(basic_synthesis_config)

    # Should not create DataSynthesizer when no generated attributes
    mock_data_synth.assert_not_called()
    assert pipeline._data_synthesizer is None


@patch("oumi.core.synthesis.synthesis_pipeline.DatasetPlanner")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeTransformer")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeSynthesizer")
def test_synthesize_basic_flow(
    mock_attr_synth,
    mock_attr_transformer_class,
    mock_dataset_planner_class,
    basic_synthesis_config,
    sample_dataset,
    mock_dataset_planner,
    mock_attribute_transformer,
):
    """Test the basic synthesis flow without generated or transformed attributes."""
    mock_attr_transformer_class.return_value = mock_attribute_transformer
    mock_dataset_planner_class.return_value = mock_dataset_planner
    mock_dataset_planner.plan.return_value = sample_dataset

    pipeline = SynthesisPipeline(basic_synthesis_config)
    result = pipeline.synthesize()

    # Verify dataset planner was called correctly
    mock_dataset_planner.plan.assert_called_once_with(
        basic_synthesis_config.strategy_params, basic_synthesis_config.num_samples
    )

    # Should return the planned dataset
    assert result == sample_dataset


@patch("oumi.core.synthesis.synthesis_pipeline.DataSynthesizer")
@patch("oumi.core.synthesis.synthesis_pipeline.DatasetPlanner")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeTransformer")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeSynthesizer")
def test_synthesize_with_generated_attributes(
    mock_attr_synth,
    mock_attr_transformer_class,
    mock_dataset_planner_class,
    mock_data_synthesizer_class,
    synthesis_config_with_generated_attributes,
    sample_dataset,
    mock_dataset_planner,
    mock_data_synthesizer,
    mock_attribute_transformer,
):
    """Test synthesis flow with generated attributes."""
    synthesized_dataset = sample_dataset + [{"generated": "content"}]

    mock_attr_transformer_class.return_value = mock_attribute_transformer
    mock_dataset_planner_class.return_value = mock_dataset_planner
    mock_data_synthesizer_class.return_value = mock_data_synthesizer
    mock_dataset_planner.plan.return_value = sample_dataset
    mock_data_synthesizer.synthesize.return_value = synthesized_dataset

    pipeline = SynthesisPipeline(synthesis_config_with_generated_attributes)
    result = pipeline.synthesize()

    # Verify data synthesizer was called
    mock_data_synthesizer.synthesize.assert_called_once_with(sample_dataset)
    assert result == synthesized_dataset


@patch("oumi.core.synthesis.synthesis_pipeline.DatasetPlanner")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeTransformer")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeSynthesizer")
def test_synthesize_with_transformed_attributes(
    mock_attr_synth,
    mock_attr_transformer_class,
    mock_dataset_planner_class,
    synthesis_config_with_transformed_attributes,
    sample_dataset,
    mock_dataset_planner,
    mock_attribute_transformer,
):
    """Test synthesis flow with transformed attributes."""
    transformed_dataset = [{"transformed": "data"}]

    mock_attr_transformer_class.return_value = mock_attribute_transformer
    mock_dataset_planner_class.return_value = mock_dataset_planner
    mock_dataset_planner.plan.return_value = sample_dataset
    mock_attribute_transformer.transform.return_value = transformed_dataset

    pipeline = SynthesisPipeline(synthesis_config_with_transformed_attributes)
    result = pipeline.synthesize()

    # Verify attribute transformer was called
    mock_attribute_transformer.transform.assert_called_once_with(sample_dataset)
    assert result == transformed_dataset


@patch("oumi.core.synthesis.synthesis_pipeline.DatasetPlanner")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeTransformer")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeSynthesizer")
def test_synthesize_with_passthrough_attributes(
    mock_attr_synth,
    mock_attr_transformer_class,
    mock_dataset_planner_class,
    synthesis_config_with_passthrough_attributes,
    mock_dataset_planner,
    mock_attribute_transformer,
):
    """Test synthesis flow with passthrough attributes filtering."""
    full_dataset = [
        {"attr1": "value1", "attr2": "value2", "attr3": "value3"},
        {"attr1": "value4", "attr2": "value5", "attr3": "value6"},
    ]
    expected_filtered = [
        {"attr1": "value1", "attr2": "value2"},
        {"attr1": "value4", "attr2": "value5"},
    ]

    mock_attr_transformer_class.return_value = mock_attribute_transformer
    mock_dataset_planner_class.return_value = mock_dataset_planner
    mock_dataset_planner.plan.return_value = full_dataset

    pipeline = SynthesisPipeline(synthesis_config_with_passthrough_attributes)
    result = pipeline.synthesize()

    # Should filter to only passthrough attributes
    assert result == expected_filtered


@patch("oumi.core.synthesis.synthesis_pipeline.save_jsonlines")
@patch("oumi.core.synthesis.synthesis_pipeline.DatasetPlanner")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeTransformer")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeSynthesizer")
def test_synthesize_with_output_path(
    mock_attr_synth,
    mock_attr_transformer_class,
    mock_dataset_planner_class,
    mock_save,
    synthesis_config_with_output_path,
    sample_dataset,
    mock_dataset_planner,
    mock_attribute_transformer,
):
    """Test synthesis flow with output path saving."""
    mock_attr_transformer_class.return_value = mock_attribute_transformer
    mock_dataset_planner_class.return_value = mock_dataset_planner
    mock_dataset_planner.plan.return_value = sample_dataset

    pipeline = SynthesisPipeline(synthesis_config_with_output_path)
    result = pipeline.synthesize()

    # Verify save function was called
    mock_save.assert_called_once_with(
        Path(synthesis_config_with_output_path.output_path), sample_dataset
    )
    assert result == sample_dataset


@patch("oumi.core.synthesis.synthesis_pipeline.save_jsonlines")
@patch("oumi.core.synthesis.synthesis_pipeline.DataSynthesizer")
@patch("oumi.core.synthesis.synthesis_pipeline.DatasetPlanner")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeTransformer")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeSynthesizer")
def test_synthesize_full_pipeline(
    mock_attr_synth,
    mock_attr_transformer_class,
    mock_dataset_planner_class,
    mock_data_synthesizer_class,
    mock_save,
    mock_dataset_planner,
    mock_data_synthesizer,
    mock_attribute_transformer,
):
    """Test the complete synthesis pipeline with all features enabled."""
    # Create config with all features
    generated_attr = GeneratedAttribute(
        id="test_generated",
        instruction_messages=[
            TextMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
            TextMessage(role=Role.USER, content="Generate some content."),
        ],
    )
    transformed_attr = TransformedAttribute(
        id="test_transformed",
        transformation_strategy=TransformationStrategy(
            type=TransformationType.STRING, string_transform="some_transformation"
        ),
    )

    strategy_params = GeneralSynthesisParams(
        generated_attributes=[generated_attr],
        transformed_attributes=[transformed_attr],
        passthrough_attributes=["final_attr"],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output.jsonl"
        config = SynthesisConfig(
            output_path=str(output_path),
            num_samples=1,
            strategy_params=strategy_params,
            inference_config=InferenceConfig(),
        )

        # Mock data at each stage
        planned_data = [{"attr1": "value1", "attr2": "value2"}]
        synthesized_data = [
            {"attr1": "value1", "attr2": "value2", "generated": "content"}
        ]
        transformed_data = [
            {
                "attr1": "value1",
                "attr2": "value2",
                "generated": "content",
                "final_attr": "final",
            }
        ]
        expected_final = [{"final_attr": "final"}]

        # Setup mocks
        mock_attr_transformer_class.return_value = mock_attribute_transformer
        mock_dataset_planner_class.return_value = mock_dataset_planner
        mock_data_synthesizer_class.return_value = mock_data_synthesizer
        mock_dataset_planner.plan.return_value = planned_data
        mock_data_synthesizer.synthesize.return_value = synthesized_data
        mock_attribute_transformer.transform.return_value = transformed_data

        pipeline = SynthesisPipeline(config)
        result = pipeline.synthesize()

        # Verify all stages were called in order
        mock_dataset_planner.plan.assert_called_once()
        mock_data_synthesizer.synthesize.assert_called_once_with(planned_data)
        mock_attribute_transformer.transform.assert_called_once_with(synthesized_data)
        mock_save.assert_called_once_with(Path(output_path), expected_final)

        # Verify final result includes only passthrough attributes
        assert result == expected_final


@patch("oumi.core.synthesis.synthesis_pipeline.DatasetPlanner")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeTransformer")
@patch("oumi.core.synthesis.synthesis_pipeline.AttributeSynthesizer")
def test_synthesize_unsupported_output_extension(
    mock_attr_synth, mock_attr_transformer, mock_planner_class
):
    """Test that unsupported output file extensions raise an error."""
    strategy_params = GeneralSynthesisParams()

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output.txt"  # Unsupported extension
        # Create a config with a bad output path by bypassing validation
        config = SynthesisConfig(
            num_samples=1,
            strategy_params=strategy_params,
            inference_config=InferenceConfig(),
        )
        # Manually set the output path to bypass validation
        config.output_path = str(output_path)

        mock_planner = Mock()
        mock_planner_class.return_value = mock_planner
        mock_planner.plan.return_value = [{"test": "data"}]

        pipeline = SynthesisPipeline(config)

        with pytest.raises(ValueError, match="Unsupported output path"):
            pipeline.synthesize()
