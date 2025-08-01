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

"""Tests for UlyssesSFTTrainer V2 (Arctic-based implementation)."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import TrainingArguments

from src.oumi.core.trainers.arctic_base_trainer import TrainerRegistry
from src.oumi.core.trainers.components.memory_optimization import MemoryOptimizer
from src.oumi.core.trainers.components.sequence_parallel import SequenceParallelConfig
from src.oumi.core.trainers.ulysses_sft_trainer_v2 import UlyssesSFTTrainer


@pytest.fixture
def mock_model():
    """Mock model following Oumi patterns."""
    model = MagicMock(spec=torch.nn.Module)
    model.config = MagicMock()
    model.config._name_or_path = "test-model"
    model.config.vocab_size = 1000
    model.config._attn_implementation = "sdpa"
    model.model = MagicMock()  # For model.model access in loss computation
    model.lm_head = MagicMock()
    model.lm_head.weight = MagicMock()
    model.loss_function = MagicMock(return_value=torch.tensor(1.0))
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer with all required attributes."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 32001
    tokenizer.eos_token = "<eos>"
    tokenizer.eos_token_id = 32000
    tokenizer.model_max_length = 1024
    tokenizer.convert_tokens_to_ids = MagicMock(return_value=32000)
    tokenizer.apply_chat_template = MagicMock(return_value="formatted_text")
    return tokenizer


@pytest.fixture
def mock_dataset():
    """Mock dataset following Oumi patterns."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    sample_data = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([4, 5, 6])},
    ]
    dataset.__iter__ = MagicMock(return_value=iter(sample_data))
    dataset.__getitem__ = MagicMock(
        side_effect=lambda i: sample_data[i % len(sample_data)]
    )
    return dataset


@pytest.fixture
def training_args():
    """Mock training arguments."""
    with tempfile.TemporaryDirectory() as temp_dir:
        args = TrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            save_steps=100,
            logging_steps=10,
            learning_rate=5e-5,
            warmup_steps=0,
            optim="adamw_torch",
            lr_scheduler_type="linear",
        )
        yield args


class TestUlyssesSFTTrainerV2:
    """Test suite for UlyssesSFTTrainer V2."""

    def test_trainer_creation_without_deepspeed(
        self, mock_model, mock_tokenizer, mock_dataset, training_args
    ):
        """Test that trainer can be created without DeepSpeed."""
        trainer = UlyssesSFTTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
            processing_class=mock_tokenizer,
            sequence_parallel_size=1,
            model_name_or_path="test-model",
        )

        assert trainer.model == mock_model
        assert trainer.sequence_parallel_size == 1
        assert trainer.model_name_or_path == "test-model"
        assert trainer.sp_config.sequence_parallel_size == 1
        assert not trainer.sp_config.is_enabled()

    @patch(
        "src.oumi.core.trainers.components.sequence_parallel.DEEPSPEED_ULYSSES_AVAILABLE",
        False,
    )
    def test_trainer_creation_with_sp_no_deepspeed(
        self, mock_model, mock_tokenizer, mock_dataset, training_args
    ):
        """Test that trainer raises error when SP is requested but DeepSpeed is not available."""
        with pytest.raises(RuntimeError, match="DeepSpeed Ulysses SP is required"):
            UlyssesSFTTrainer(
                model=mock_model,
                args=training_args,
                train_dataset=mock_dataset,
                processing_class=mock_tokenizer,
                sequence_parallel_size=2,
                model_name_or_path="test-model",
            )

    @patch(
        "src.oumi.core.trainers.components.sequence_parallel.DEEPSPEED_ULYSSES_AVAILABLE",
        True,
    )
    @patch("src.oumi.core.trainers.components.sequence_parallel.UlyssesSPAttentionHF")
    def test_trainer_creation_with_sp(
        self, mock_sp_attention, mock_model, mock_tokenizer, mock_dataset, training_args
    ):
        """Test trainer creation with sequence parallelism enabled."""
        # Mock the SP setup
        mock_sp_attention.register_with_transformers.return_value = MagicMock()

        trainer = UlyssesSFTTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
            processing_class=mock_tokenizer,
            sequence_parallel_size=2,
            model_name_or_path="test-model",
        )

        assert trainer.sequence_parallel_size == 2
        assert trainer.sp_config.is_enabled()
        mock_sp_attention.register_with_transformers.assert_called_once()

    @patch(
        "src.oumi.core.trainers.components.memory_optimization.TILED_MLP_AVAILABLE",
        True,
    )
    @patch(
        "src.oumi.core.trainers.components.memory_optimization.enable_tiled_mlp_compute"
    )
    def test_tiled_mlp_setup(
        self,
        mock_enable_tiled_mlp,
        mock_model,
        mock_tokenizer,
        mock_dataset,
        training_args,
    ):
        """Test tiled MLP computation setup."""
        trainer = UlyssesSFTTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
            processing_class=mock_tokenizer,
            tiled_mlp_compute=True,
            model_name_or_path="test-model",
        )

        assert trainer.tiled_mlp_compute == True
        mock_enable_tiled_mlp.assert_called_once_with("test-model")

    @patch(
        "src.oumi.core.trainers.components.memory_optimization.LigerKernelOptimizer.is_available",
        return_value=True,
    )
    @patch(
        "src.oumi.core.trainers.components.memory_optimization.LigerKernelOptimizer.apply_liger_kernels"
    )
    def test_liger_kernel_setup(
        self,
        mock_apply_liger,
        mock_is_available,
        mock_model,
        mock_tokenizer,
        mock_dataset,
        training_args,
    ):
        """Test Liger kernel setup."""
        trainer = UlyssesSFTTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
            processing_class=mock_tokenizer,
            use_liger_kernel=True,
        )

        assert trainer.use_liger_kernel == True
        mock_apply_liger.assert_called_once_with(mock_model, True)

    def test_create_train_dataloader(
        self, mock_model, mock_tokenizer, mock_dataset, training_args
    ):
        """Test training dataloader creation."""
        trainer = UlyssesSFTTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
            processing_class=mock_tokenizer,
        )

        dataloader = trainer.create_train_dataloader()
        assert dataloader is not None
        assert hasattr(dataloader, "__iter__")

    def test_create_eval_dataloader(
        self, mock_model, mock_tokenizer, mock_dataset, training_args
    ):
        """Test evaluation dataloader creation."""
        trainer = UlyssesSFTTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            processing_class=mock_tokenizer,
        )

        dataloader = trainer.create_eval_dataloader()
        assert dataloader is not None
        assert hasattr(dataloader, "__iter__")

    def test_compute_loss_standard(
        self, mock_model, mock_tokenizer, mock_dataset, training_args
    ):
        """Test standard loss computation."""
        trainer = UlyssesSFTTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
            processing_class=mock_tokenizer,
        )

        # Mock model output
        mock_outputs = MagicMock()
        mock_outputs.loss = torch.tensor(1.0)
        mock_model.return_value = mock_outputs

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, 3]]),
        }
        loss = trainer.compute_loss(mock_model, inputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 1.0

    @patch(
        "src.oumi.core.trainers.components.sequence_parallel.DEEPSPEED_ULYSSES_AVAILABLE",
        True,
    )
    def test_compute_loss_with_sp(
        self, mock_model, mock_tokenizer, mock_dataset, training_args
    ):
        """Test loss computation with sequence parallelism."""
        trainer = UlyssesSFTTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
            processing_class=mock_tokenizer,
            sequence_parallel_size=2,
        )

        # Setup SP manager for testing
        trainer.sp_manager.sp_group = MagicMock()
        trainer.sp_manager._initialized = True

        # Mock SP batch format
        inputs = {"shift_labels": torch.tensor([[1, 2, 3]])}

        # Mock model output for SP loss
        mock_outputs = MagicMock()
        mock_outputs.loss = torch.tensor(1.0)
        mock_model.return_value = mock_outputs

        loss = trainer.compute_loss(mock_model, inputs)

        assert isinstance(loss, torch.Tensor)

    def test_deepspeed_config_preparation(
        self, mock_model, mock_tokenizer, mock_dataset, training_args
    ):
        """Test DeepSpeed configuration preparation."""
        # Create a temporary DeepSpeed config file
        ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ds_config, f)
            config_path = f.name

        try:
            training_args.deepspeed = config_path

            trainer = UlyssesSFTTrainer(
                model=mock_model,
                args=training_args,
                train_dataset=mock_dataset,
                processing_class=mock_tokenizer,
                sequence_parallel_size=2,
            )

            prepared_config = trainer._prepare_deepspeed_config()

            # Check that auto values are handled
            assert "ulysses_sequence_parallel_size" in prepared_config
            assert prepared_config["ulysses_sequence_parallel_size"] == 2

        finally:
            os.unlink(config_path)

    def test_registry_integration(self):
        """Test that trainer is properly registered."""
        trainer_cls = TrainerRegistry.get_trainer("ulysses_sft")
        assert trainer_cls == UlyssesSFTTrainer

    def test_from_config_method(self, mock_model, mock_tokenizer, training_args):
        """Test the from_config class method."""
        trainer = UlyssesSFTTrainer.from_config(
            model=mock_model,
            processing_class=mock_tokenizer,
            args=training_args,
            sequence_parallel_size=2,
            model_name_or_path="test-model",
            use_liger_kernel=True,
        )

        assert isinstance(trainer, UlyssesSFTTrainer)
        assert trainer.sequence_parallel_size == 2
        assert trainer.model_name_or_path == "test-model"
        assert trainer.use_liger_kernel == True


class TestSequenceParallelConfig:
    """Test suite for SequenceParallelConfig."""

    def test_config_creation(self):
        """Test config creation with default values."""
        config = SequenceParallelConfig()
        assert config.sequence_parallel_size == 1
        assert not config.is_enabled()

    def test_config_enabled(self):
        """Test config when SP is enabled."""
        config = SequenceParallelConfig(sequence_parallel_size=2)
        assert config.sequence_parallel_size == 2
        assert config.is_enabled()


class TestMemoryOptimizer:
    """Test suite for MemoryOptimizer."""

    def test_get_memory_usage_no_cuda(self):
        """Test memory usage when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            stats = MemoryOptimizer.get_memory_usage()
            assert stats == {"current_gb": 0.0, "max_gb": 0.0, "reserved_gb": 0.0}

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 1024)  # 1GB
    @patch(
        "torch.cuda.max_memory_allocated", return_value=2 * 1024 * 1024 * 1024
    )  # 2GB
    @patch("torch.cuda.memory_reserved", return_value=3 * 1024 * 1024 * 1024)  # 3GB
    def test_get_memory_usage_with_cuda(
        self, mock_reserved, mock_max, mock_current, mock_available
    ):
        """Test memory usage with CUDA available."""
        stats = MemoryOptimizer.get_memory_usage()
        assert stats["current_gb"] == 1.0
        assert stats["max_gb"] == 2.0
        assert stats["reserved_gb"] == 3.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.synchronize")
    def test_clear_memory(self, mock_sync, mock_clear, mock_available):
        """Test memory clearing."""
        MemoryOptimizer.clear_memory()
        mock_clear.assert_called_once()
        mock_sync.assert_called_once()


class TestComponentFactory:
    """Test suite for ComponentFactory (via ArcticBaseTrainer)."""

    def test_create_optimizer(self):
        """Test optimizer creation."""
        from src.oumi.core.trainers.arctic_base_trainer import ComponentFactory

        model = torch.nn.Linear(10, 5)
        optimizer = ComponentFactory.create_optimizer(
            model=model,
            optimizer_name="AdamW",
            learning_rate=1e-4,
            weight_decay=0.01,
        )

        assert optimizer is not None
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_create_data_loader(self):
        """Test data loader creation."""
        from src.oumi.core.trainers.arctic_base_trainer import ComponentFactory

        # Create a simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "labels": torch.tensor([1, 2, 3]),
                }

        dataset = SimpleDataset()
        dataloader = ComponentFactory.create_data_loader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
        )

        assert dataloader is not None
        assert dataloader.batch_size == 2
        assert len(dataloader) == 5  # 10 samples / 2 batch_size
