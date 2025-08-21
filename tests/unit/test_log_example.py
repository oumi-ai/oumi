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

"""Tests for the log_examples configuration flow from config to collator."""

from unittest.mock import Mock, patch

import pytest

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainingConfig,
    TrainingParams,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer with required properties."""
    mock_tokenizer = Mock(spec=BaseTokenizer)
    mock_tokenizer.padding_side = "right"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.model_max_length = 2048
    return mock_tokenizer


@pytest.fixture
def base_training_config():
    """Base training config without debug flag set."""
    return TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                collator_name="text_with_padding",
                datasets=[DatasetParams(dataset_name="dummy", split="train")],
            )
        ),
        model=ModelParams(
            model_name="gpt2", tokenizer_name="gpt2", model_max_length=1024
        ),
        training=TrainingParams(
            log_examples=False,  # Default value
        ),
    )


@pytest.fixture
def debug_training_config(base_training_config):
    """Training config with debug flag enabled."""
    config = base_training_config
    config.training.log_examples = True
    return config


@patch("oumi.train.build_collator_from_config")
@patch("oumi.train.build_model")
@patch("oumi.train.build_dataset_mixture")
@patch("oumi.train.build_trainer")
@patch("oumi.train._load_checkpoint_and_recompute_checksum")
def test_train_function_passes_debug_flag_correctly(
    mock_load_checkpoint,
    mock_build_trainer,
    mock_build_dataset_mixture,
    mock_build_model,
    mock_build_collator,
    debug_training_config,
):
    """Test that train function passes log_examples to build_collator_from_config."""
    from oumi.train import train

    # Setup mocks to avoid actual training
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_model_tuple = (mock_model, mock_tokenizer)
    mock_build_model.return_value = mock_model_tuple

    mock_dataset = Mock()
    mock_build_dataset_mixture.return_value = mock_dataset

    mock_collator = Mock()
    mock_build_collator.return_value = mock_collator

    mock_trainer = Mock()
    mock_build_trainer.return_value = mock_trainer
    mock_trainer.train.return_value = Mock()

    mock_load_checkpoint.return_value = (None, None)

    # Call train function
    train(debug_training_config)

    # Verify that build_collator_from_config was called with the correct debug flag
    mock_build_collator.assert_called_once()
    call_args = mock_build_collator.call_args

    # The debug parameter should match config.training.log_examples
    assert call_args.args[0] is debug_training_config  # config parameter
    assert call_args.args[1] is mock_tokenizer  # tokenizer parameter
    assert call_args.kwargs["debug"] is True  # debug parameter should be True

    # Verify the debug flag came from the config
    assert debug_training_config.training.log_examples is True
