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

"""Tests for UlyssesSFTTrainer."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.oumi.core.trainers.ulysses_sft_trainer import (
    DEEPSPEED_ULYSSES_AVAILABLE,
    UlyssesSFTTrainer,
)


@pytest.fixture
def mock_model():
    """Mock model following Oumi patterns."""
    model = MagicMock(spec=torch.nn.Module)
    model.config = MagicMock()
    model.config._name_or_path = (
        "test-model"  # Provide a valid name to avoid None errors
    )
    model.config.vocab_size = 1000
    model.config._attn_implementation = "sdpa"
    return model


@pytest.fixture
def mock_dataset():
    """Mock dataset following Oumi patterns."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    sample_data = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
    ]
    dataset.__iter__ = MagicMock(return_value=iter(sample_data))
    # Ensure the first call to iter() returns the sample data
    dataset.__getitem__ = MagicMock(
        side_effect=lambda i: sample_data[i % len(sample_data)]
    )
    return dataset


@pytest.fixture
def mock_training_args():
    """Mock training arguments."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir="/tmp/test",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=10,
    )


@pytest.fixture
def sft_tokenizer():
    """SFT-compatible mock tokenizer with all required attributes."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 32001
    tokenizer.eos_token = "<eos>"
    tokenizer.eos_token_id = 32000
    tokenizer.model_max_length = 1024
    tokenizer.convert_tokens_to_ids = MagicMock(return_value=32000)
    return tokenizer


class TestUlyssesSFTTrainer:
    """Test suite for UlyssesSFTTrainer."""

    def test_trainer_creation_without_deepspeed(
        self, mock_model, sft_tokenizer, mock_dataset, mock_training_args
    ):
        """Test that trainer can be created without DeepSpeed."""
        with patch(
            "src.oumi.core.trainers.ulysses_sft_trainer.TILED_MLP_AVAILABLE", False
        ):
            trainer = UlyssesSFTTrainer(
                model=mock_model,
                args=mock_training_args,
                processing_class=sft_tokenizer,
                train_dataset=mock_dataset,
                sequence_parallel_size=1,
                model_name_or_path=None,
            )
        assert trainer.sequence_parallel_size == 1
        assert trainer.model_name_or_path is None
        assert trainer.sp_group is None
        assert trainer.sp_world_size == 1
        assert trainer.sp_rank == 0

    def test_trainer_creation_with_sp_fails_without_deepspeed(
        self, mock_model, mock_tokenizer, mock_dataset
    ):
        """Test trainer creation fails when SP enabled but DeepSpeed unavailable."""
        if not DEEPSPEED_ULYSSES_AVAILABLE:
            with pytest.raises(RuntimeError, match="DeepSpeed Ulysses SP is required"):
                UlyssesSFTTrainer(
                    model=mock_model,
                    processing_class=mock_tokenizer,
                    train_dataset=mock_dataset,
                    sequence_parallel_size=2,
                    model_name_or_path="test/model",
                )

    def test_shift_labels_conversion(self, mock_model, mock_tokenizer, mock_dataset):
        """Test shift_labels conversion from labels."""
        with patch(
            "src.oumi.core.trainers.ulysses_sft_trainer.TILED_MLP_AVAILABLE", False
        ):
            trainer = UlyssesSFTTrainer(
                model=mock_model,
                processing_class=mock_tokenizer,
                train_dataset=mock_dataset,
                sequence_parallel_size=1,
                model_name_or_path=None,
            )

        # Mock model for loss computation
        loss_model = MagicMock(spec=torch.nn.Module)

        # Create test labels [batch_size=2, seq_len=4]
        labels = torch.tensor([[1, 2, 3, -100], [4, 5, 6, 7]])
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]]),
            "labels": labels,
        }

        # Test shift_labels conversion
        with patch.object(
            trainer, "_compute_tiled_logits_loss", return_value=torch.tensor(1.0)
        ):
            trainer._compute_loss_ulysses_sp(loss_model, inputs)

        # Check that shift_labels was created correctly
        assert "shift_labels" in inputs
        shift_labels = inputs["shift_labels"]
        expected_shift_labels = torch.tensor([[2, 3, -100, -100], [5, 6, 7, -100]])
        assert torch.equal(shift_labels, expected_shift_labels)

    def test_tiled_mlp_setup(self, mock_model, mock_tokenizer, mock_dataset):
        """Test tiled MLP setup functionality."""
        with (
            patch(
                "src.oumi.core.trainers.ulysses_sft_trainer.enable_tiled_mlp_compute"
            ) as mock_enable,
            patch(
                "src.oumi.core.trainers.ulysses_sft_trainer.TILED_MLP_AVAILABLE", True
            ),
        ):
            UlyssesSFTTrainer(
                model=mock_model,
                processing_class=mock_tokenizer,
                train_dataset=mock_dataset,
                sequence_parallel_size=1,
                model_name_or_path="test/model",
                tiled_mlp_compute=True,
            )
            # Should have called enable_tiled_mlp_compute
            mock_enable.assert_called_once_with("test/model")

    def test_tiled_mlp_setup_failure_handling(
        self, mock_model, mock_tokenizer, mock_dataset
    ):
        """Test that tiled MLP setup failures are handled gracefully."""
        with (
            patch(
                "src.oumi.core.trainers.ulysses_sft_trainer.enable_tiled_mlp_compute"
            ) as mock_enable,
            patch(
                "src.oumi.core.trainers.ulysses_sft_trainer.TILED_MLP_AVAILABLE", True
            ),
        ):
            mock_enable.side_effect = ValueError("Unsupported model type")

            trainer = UlyssesSFTTrainer(
                model=mock_model,
                processing_class=mock_tokenizer,
                train_dataset=mock_dataset,
                sequence_parallel_size=1,
                model_name_or_path="test/model",
                tiled_mlp_compute=True,
            )
            # Should have disabled tiled MLP compute after failure
            assert trainer.tiled_mlp_compute is False

    def test_compute_loss_fallback_to_standard(
        self, mock_model, mock_tokenizer, mock_dataset
    ):
        """Test compute_loss falls back to standard implementation when SP disabled."""
        with patch(
            "src.oumi.core.trainers.ulysses_sft_trainer.TILED_MLP_AVAILABLE", False
        ):
            trainer = UlyssesSFTTrainer(
                model=mock_model,
                processing_class=mock_tokenizer,
                train_dataset=mock_dataset,
                sequence_parallel_size=1,
                model_name_or_path=None,
            )

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, 3]]),
        }

        with patch(
            "trl.SFTTrainer.compute_loss", return_value=torch.tensor(1.0)
        ) as mock_compute_loss:
            loss = trainer.compute_loss(mock_model, inputs)
            mock_compute_loss.assert_called_once_with(mock_model, inputs, False)
            assert loss == torch.tensor(1.0)

    def test_mpu_getter(self, mock_model, mock_tokenizer, mock_dataset):
        """Test MPU getter method."""
        with patch(
            "src.oumi.core.trainers.ulysses_sft_trainer.TILED_MLP_AVAILABLE", False
        ):
            trainer = UlyssesSFTTrainer(
                model=mock_model,
                processing_class=mock_tokenizer,
                train_dataset=mock_dataset,
                sequence_parallel_size=1,
                model_name_or_path=None,
            )

        # Initially None
        assert trainer.get_mpu() is None

        # Set mock MPU
        mock_mpu = MagicMock()
        trainer._mpu = mock_mpu
        assert trainer.get_mpu() is mock_mpu

    @pytest.mark.skipif(
        not DEEPSPEED_ULYSSES_AVAILABLE, reason="DeepSpeed Ulysses SP not available"
    )
    def test_ulysses_sp_setup(self, mock_model, mock_tokenizer, mock_dataset):
        """Test Ulysses SP setup when DeepSpeed is available."""
        with patch(
            "src.oumi.core.trainers.ulysses_sft_trainer.UlyssesSPAttentionHF"
        ) as mock_ulysses:
            mock_mpu = MagicMock()
            mock_ulysses.register_with_transformers.return_value = mock_mpu

            trainer = UlyssesSFTTrainer(
                model=mock_model,
                processing_class=mock_tokenizer,
                train_dataset=mock_dataset,
                sequence_parallel_size=2,
                model_name_or_path="test/model",
                attn_implementation="flash_attention_2",
                max_length=8192,
                micro_batch_size=1,
            )

            # Should have called register_with_transformers
            mock_ulysses.register_with_transformers.assert_called_once_with(
                model_name_or_path="test/model",
                core_attn_implementation="flash_attention_2",
                sequence_parallel_size=2,
                max_length=8192,
                micro_batch_size=1,
                seq_length_is_variable=True,
            )

            # Should have stored MPU
            assert trainer._mpu is mock_mpu

    def test_from_config_method(self, mock_model, mock_tokenizer, mock_dataset):
        """Test the from_config class method."""
        mock_processing_class = MagicMock()
        mock_args = MagicMock()

        with patch(
            "src.oumi.core.trainers.ulysses_sft_trainer.TILED_MLP_AVAILABLE", False
        ):
            trainer = UlyssesSFTTrainer.from_config(
                model=mock_model,
                processing_class=mock_processing_class,
                args=mock_args,
                sequence_parallel_size=1,  # Use 1 to avoid DeepSpeed requirement
                model_name_or_path=None,
                attn_implementation="sdpa",
                max_length=4096,
                micro_batch_size=1,
                tiled_mlp_compute=False,  # Set to False to avoid setup
                use_liger_kernel=False,
            )

        assert trainer.sequence_parallel_size == 1
        assert trainer.model_name_or_path is None
        assert trainer.attn_implementation == "sdpa"
        assert trainer.max_length == 4096
        assert trainer.micro_batch_size == 1
        assert trainer.tiled_mlp_compute is False
        assert trainer.use_liger_kernel is False


class TestUlyssesSFTTrainerConfiguration:
    """Test configuration-related functionality."""

    def test_trainer_parameter_extraction(
        self, mock_model, mock_tokenizer, mock_dataset
    ):
        """Test that all parameters are correctly extracted and stored."""
        with patch(
            "src.oumi.core.trainers.ulysses_sft_trainer.TILED_MLP_AVAILABLE", False
        ):
            trainer = UlyssesSFTTrainer(
                model=mock_model,
                processing_class=mock_tokenizer,
                train_dataset=mock_dataset,
                sequence_parallel_size=1,  # Use 1 to avoid DeepSpeed requirement
                model_name_or_path=None,
                attn_implementation="flash_attention_2",
                max_length=32768,
                micro_batch_size=2,
                tiled_mlp_compute=False,  # Set to False to avoid setup
                use_liger_kernel=False,
            )

        assert trainer.sequence_parallel_size == 1
        assert trainer.model_name_or_path is None
        assert trainer.attn_implementation == "flash_attention_2"
        assert trainer.max_length == 32768
        assert trainer.micro_batch_size == 2
        assert trainer.tiled_mlp_compute is False
        assert trainer.use_liger_kernel is False

    def test_default_parameter_values(self, mock_model, mock_tokenizer, mock_dataset):
        """Test that default parameter values are set correctly."""
        with patch(
            "src.oumi.core.trainers.ulysses_sft_trainer.TILED_MLP_AVAILABLE", False
        ):
            trainer = UlyssesSFTTrainer(
                model=mock_model,
                processing_class=mock_tokenizer,
                train_dataset=mock_dataset,
            )

        assert trainer.sequence_parallel_size == 1
        assert trainer.model_name_or_path is None
        assert trainer.attn_implementation == "sdpa"
        assert trainer.max_length == 4096
        assert trainer.micro_batch_size == 1
        assert trainer.tiled_mlp_compute is False
        assert trainer.use_liger_kernel is False
