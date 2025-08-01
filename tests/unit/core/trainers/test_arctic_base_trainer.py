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

"""Tests for ArcticBaseTrainer and registry system."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import TrainingArguments

from src.oumi.core.trainers.arctic_base_trainer import (
    ArcticBaseTrainer,
    TrainerRegistry,
    TrainingState,
    TrainerCallback,
    CallbackMixin,
    ComponentFactory,
    LoggingCallback,
    CheckpointCallback,
)


class TestTrainerRegistry:
    """Test suite for TrainerRegistry."""

    def test_register_and_get_trainer(self):
        """Test trainer registration and retrieval."""
        @TrainerRegistry.register_trainer("test_trainer")
        class TestTrainer(ArcticBaseTrainer):
            def create_train_dataloader(self):
                return None
            
            def create_eval_dataloader(self):
                return None
            
            def compute_loss(self, model, inputs):
                return torch.tensor(1.0)

        # Test retrieval
        retrieved_cls = TrainerRegistry.get_trainer("test_trainer")
        assert retrieved_cls == TestTrainer

    def test_get_nonexistent_trainer(self):
        """Test error when getting non-existent trainer."""
        with pytest.raises(ValueError, match="Trainer 'nonexistent' not registered"):
            TrainerRegistry.get_trainer("nonexistent")

    def test_register_and_get_callback(self):
        """Test callback registration and retrieval."""
        @TrainerRegistry.register_callback("test_callback")
        class TestCallback(TrainerCallback):
            def on_train_begin(self, trainer, state):
                pass
            
            def on_train_end(self, trainer, state):
                pass
            
            def on_epoch_begin(self, trainer, state):
                pass
            
            def on_epoch_end(self, trainer, state):
                pass
            
            def on_step_begin(self, trainer, state):
                pass
            
            def on_step_end(self, trainer, state):
                pass

        # Test retrieval
        retrieved_cls = TrainerRegistry.get_callback("test_callback")
        assert retrieved_cls == TestCallback


class TestTrainingState:
    """Test suite for TrainingState."""

    def test_training_state_creation(self):
        """Test training state initialization."""
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.local_step == 0
        assert state.world_size == 1
        assert state.rank == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 1024)  # 1GB
    @patch("torch.cuda.max_memory_allocated", return_value=2 * 1024 * 1024 * 1024)  # 2GB
    def test_update_memory_stats(self, mock_max, mock_current, mock_available):
        """Test memory statistics update."""
        state = TrainingState()
        state.update_memory_stats()
        
        assert state.current_memory_gb == 1.0
        assert state.max_memory_gb == 2.0


class TestTrainerCallback:
    """Test suite for TrainerCallback."""

    def test_callback_creation(self):
        """Test that callbacks can be created."""
        class TestCallback(TrainerCallback):
            def on_train_begin(self, trainer, state):
                pass
            
            def on_train_end(self, trainer, state):
                pass
            
            def on_epoch_begin(self, trainer, state):
                pass
            
            def on_epoch_end(self, trainer, state):
                pass
            
            def on_step_begin(self, trainer, state):
                pass
            
            def on_step_end(self, trainer, state):
                pass

        callback = TestCallback()
        assert isinstance(callback, TrainerCallback)


class TestCallbackMixin:
    """Test suite for CallbackMixin."""

    def test_add_callback(self):
        """Test adding callbacks to mixin."""
        class TestMixin(CallbackMixin):
            def __init__(self):
                super().__init__()
        
        mixin = TestMixin()
        
        # Add callback instance
        callback = LoggingCallback()
        mixin.add_callback(callback)
        assert len(mixin.callbacks) == 1
        assert mixin.callbacks[0] == callback

    def test_call_callbacks(self):
        """Test calling methods on all callbacks."""
        class TestMixin(CallbackMixin):
            def __init__(self):
                super().__init__()
        
        mixin = TestMixin()
        
        # Add mock callback
        callback = MagicMock()
        mixin.add_callback(callback)
        
        # Call callbacks
        mixin._call_callbacks("test_method", "arg1", "arg2", kwarg="value")
        
        callback.test_method.assert_called_once_with("arg1", "arg2", kwarg="value")


class TestComponentFactory:
    """Test suite for ComponentFactory."""

    def test_create_optimizer(self):
        """Test optimizer creation."""
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

    def test_create_scheduler(self):
        """Test scheduler creation."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        scheduler = ComponentFactory.create_scheduler(
            optimizer=optimizer,
            scheduler_name="linear",
            num_training_steps=1000,
            num_warmup_steps=100,
        )
        
        assert scheduler is not None

    def test_create_data_loader(self):
        """Test data loader creation."""
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return {"input_ids": torch.tensor([1, 2, 3])}
        
        dataset = SimpleDataset()
        dataloader = ComponentFactory.create_data_loader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
        )
        
        assert dataloader is not None
        assert dataloader.batch_size == 2


class TestArcticBaseTrainer:
    """Test suite for ArcticBaseTrainer."""

    @pytest.fixture
    def mock_model(self):
        """Mock model for testing."""
        model = MagicMock(spec=torch.nn.Module)
        return model

    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for testing."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=10)
        return dataset

    @pytest.fixture
    def training_args(self):
        """Training arguments for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(
                output_dir=temp_dir,
                per_device_train_batch_size=1,
                num_train_epochs=1,
                save_steps=100,
                logging_steps=10,
            )
            yield args

    def test_trainer_creation(self, mock_model, mock_dataset, training_args):
        """Test basic trainer creation."""
        class ConcreteTrainer(ArcticBaseTrainer):
            def create_train_dataloader(self):
                return ComponentFactory.create_data_loader(
                    dataset=self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    shuffle=True,
                )
            
            def create_eval_dataloader(self):
                return None
            
            def compute_loss(self, model, inputs):
                return torch.tensor(1.0)

        trainer = ConcreteTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
        )
        
        assert trainer.model == mock_model
        assert trainer.args == training_args
        assert trainer.train_dataset == mock_dataset
        assert isinstance(trainer.state, TrainingState)

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=4)
    @patch("torch.distributed.get_rank", return_value=1)
    def test_distributed_initialization(self, mock_rank, mock_world_size, mock_init, mock_avail, 
                                       mock_model, mock_dataset, training_args):
        """Test distributed training initialization."""
        class ConcreteTrainer(ArcticBaseTrainer):
            def create_train_dataloader(self):
                return None
            
            def create_eval_dataloader(self):
                return None
            
            def compute_loss(self, model, inputs):
                return torch.tensor(1.0)

        trainer = ConcreteTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
        )
        
        assert trainer.state.world_size == 4
        assert trainer.state.rank == 1

    def test_default_callbacks_setup(self, mock_model, mock_dataset, training_args):
        """Test that default callbacks are set up."""
        class ConcreteTrainer(ArcticBaseTrainer):
            def create_train_dataloader(self):
                return None
            
            def create_eval_dataloader(self):
                return None
            
            def compute_loss(self, model, inputs):
                return torch.tensor(1.0)

        trainer = ConcreteTrainer(
            model=mock_model,
            args=training_args,
            train_dataset=mock_dataset,
        )
        
        # Should have at least logging and checkpoint callbacks
        assert len(trainer.callbacks) >= 2
        assert any(isinstance(cb, LoggingCallback) for cb in trainer.callbacks)
        assert any(isinstance(cb, CheckpointCallback) for cb in trainer.callbacks)


class TestDefaultCallbacks:
    """Test suite for default callbacks."""

    def test_logging_callback(self):
        """Test logging callback functionality."""
        callback = LoggingCallback(log_frequency=5)
        
        # Mock trainer and state
        trainer = MagicMock()
        state = TrainingState()
        state.global_step = 5
        state.train_loss = 1.5
        state.learning_rate = 1e-4
        state.current_memory_gb = 2.0
        
        # Test callback methods (should not raise errors)
        callback.on_train_begin(trainer, state)
        callback.on_train_end(trainer, state)
        callback.on_epoch_begin(trainer, state)
        callback.on_epoch_end(trainer, state)
        callback.on_step_begin(trainer, state)
        callback.on_step_end(trainer, state)

    def test_checkpoint_callback(self):
        """Test checkpoint callback functionality."""
        callback = CheckpointCallback()
        
        # Mock trainer and state
        trainer = MagicMock()
        trainer.args = MagicMock()
        trainer.args.output_dir = "/tmp/test"
        trainer.args.save_strategy = "steps"
        trainer.args.save_steps = 10
        trainer.model = MagicMock()
        trainer.processing_class = None
        trainer.optimizer = None
        trainer.lr_scheduler = None
        
        state = TrainingState()
        state.global_step = 10
        
        # Test callback methods
        with patch("os.makedirs"), \
             patch("torch.save"), \
             patch("builtins.open", create=True):
            callback.on_train_begin(trainer, state)
            callback.on_train_end(trainer, state)
            callback.on_epoch_begin(trainer, state)
            callback.on_epoch_end(trainer, state)
            callback.on_step_begin(trainer, state)
            callback.on_step_end(trainer, state)