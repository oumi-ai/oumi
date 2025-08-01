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

"""Arctic-style Base Trainer with Registry System.

This module provides the foundation for custom trainers following ArcticTraining
patterns, with full control over the training loop and advanced parallelism support.
"""

import abc
import math
import os
import time
from typing import Any, Dict, List, Optional, Type, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from oumi.utils.logging import logger


class TrainerRegistry:
    """Registry for trainer components following ArcticTraining patterns."""
    
    _trainers: Dict[str, Type["ArcticBaseTrainer"]] = {}
    _callbacks: Dict[str, Type["TrainerCallback"]] = {}
    _optimizers: Dict[str, Type] = {}
    _data_factories: Dict[str, Type] = {}
    
    @classmethod
    def register_trainer(cls, name: str):
        """Register a trainer class."""
        def decorator(trainer_cls):
            cls._trainers[name] = trainer_cls
            return trainer_cls
        return decorator
    
    @classmethod
    def register_callback(cls, name: str):
        """Register a callback class."""
        def decorator(callback_cls):
            cls._callbacks[name] = callback_cls
            return callback_cls
        return decorator
    
    @classmethod
    def register_optimizer(cls, name: str):
        """Register an optimizer class."""
        def decorator(optimizer_cls):
            cls._optimizers[name] = optimizer_cls
            return optimizer_cls
        return decorator
    
    @classmethod
    def register_data_factory(cls, name: str):
        """Register a data factory class."""
        def decorator(factory_cls):
            cls._data_factories[name] = factory_cls
            return factory_cls
        return decorator
    
    @classmethod
    def get_trainer(cls, name: str) -> Type["ArcticBaseTrainer"]:
        """Get registered trainer class."""
        if name not in cls._trainers:
            raise ValueError(f"Trainer '{name}' not registered. Available: {list(cls._trainers.keys())}")
        return cls._trainers[name]
    
    @classmethod
    def get_callback(cls, name: str) -> Type["TrainerCallback"]:
        """Get registered callback class."""
        if name not in cls._callbacks:
            raise ValueError(f"Callback '{name}' not registered. Available: {list(cls._callbacks.keys())}")
        return cls._callbacks[name]
    
    @classmethod
    def get_optimizer(cls, name: str) -> Type:
        """Get registered optimizer class."""
        if name not in cls._optimizers:
            raise ValueError(f"Optimizer '{name}' not registered. Available: {list(cls._optimizers.keys())}")
        return cls._optimizers[name]
    
    @classmethod
    def get_data_factory(cls, name: str) -> Type:
        """Get registered data factory class."""
        if name not in cls._data_factories:
            raise ValueError(f"Data factory '{name}' not registered. Available: {list(cls._data_factories.keys())}")
        return cls._data_factories[name]


@dataclass
class TrainingState:
    """Training state tracking following ArcticTraining patterns."""
    
    # Core state
    epoch: int = 0
    global_step: int = 0
    local_step: int = 0
    
    # Metrics
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    
    # Timing
    epoch_start_time: float = field(default_factory=time.time)
    step_start_time: float = field(default_factory=time.time)
    
    # Memory tracking
    max_memory_gb: float = 0.0
    current_memory_gb: float = 0.0
    
    # Distributed info
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Sequence parallelism
    sp_world_size: int = 1
    sp_rank: int = 0
    sp_group: Optional[Any] = None
    
    def update_memory_stats(self):
        """Update memory statistics."""
        if torch.cuda.is_available():
            current_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            max_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.current_memory_gb = current_mb / 1024
            self.max_memory_gb = max_mb / 1024


class TrainerCallback(abc.ABC):
    """Abstract base class for trainer callbacks."""
    
    @abc.abstractmethod
    def on_train_begin(self, trainer: "ArcticBaseTrainer", state: TrainingState) -> None:
        """Called at the beginning of training."""
        pass
    
    @abc.abstractmethod
    def on_train_end(self, trainer: "ArcticBaseTrainer", state: TrainingState) -> None:
        """Called at the end of training."""
        pass
    
    @abc.abstractmethod
    def on_epoch_begin(self, trainer: "ArcticBaseTrainer", state: TrainingState) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    @abc.abstractmethod
    def on_epoch_end(self, trainer: "ArcticBaseTrainer", state: TrainingState) -> None:
        """Called at the end of each epoch."""
        pass
    
    @abc.abstractmethod
    def on_step_begin(self, trainer: "ArcticBaseTrainer", state: TrainingState) -> None:
        """Called at the beginning of each training step."""
        pass
    
    @abc.abstractmethod
    def on_step_end(self, trainer: "ArcticBaseTrainer", state: TrainingState) -> None:
        """Called at the end of each training step."""
        pass
    
    def on_loss_computed(self, trainer: "ArcticBaseTrainer", state: TrainingState, loss: torch.Tensor) -> None:
        """Called after loss computation."""
        pass


class CallbackMixin:
    """Mixin for callback functionality."""
    
    def __init__(self):
        self.callbacks: List[TrainerCallback] = []
    
    def add_callback(self, callback: Union[TrainerCallback, str, Type[TrainerCallback]]):
        """Add a callback to the trainer."""
        if isinstance(callback, str):
            callback_cls = TrainerRegistry.get_callback(callback)
            callback = callback_cls()
        elif isinstance(callback, type):
            callback = callback()
        
        self.callbacks.append(callback)
    
    def _call_callbacks(self, method_name: str, *args, **kwargs):
        """Call a method on all callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                getattr(callback, method_name)(*args, **kwargs)


class ComponentFactory:
    """Factory for creating trainer components."""
    
    @staticmethod
    def create_optimizer(
        model: torch.nn.Module,
        optimizer_name: str,
        learning_rate: float,
        weight_decay: float = 0.01,
        **kwargs
    ) -> torch.optim.Optimizer:
        """Create optimizer instance."""
        if optimizer_name in TrainerRegistry._optimizers:
            optimizer_cls = TrainerRegistry.get_optimizer(optimizer_name)
        else:
            # Fallback to torch optimizers
            if hasattr(torch.optim, optimizer_name):
                optimizer_cls = getattr(torch.optim, optimizer_name)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer_cls(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_name: str,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        **kwargs
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        from transformers import get_scheduler
        
        return get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    
    @staticmethod
    def create_data_loader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        collate_fn: Optional[Callable] = None,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """Create data loader instance."""
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            **kwargs
        )


class ArcticBaseTrainer(CallbackMixin, abc.ABC):
    """Abstract base trainer following ArcticTraining patterns.
    
    This provides the foundation for custom trainers with:
    - Full control over training loop
    - Registry-based component system
    - Callback system for extensibility
    - Advanced parallelism support
    - Memory optimization capabilities
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        data_collator: Optional[Callable] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        **kwargs
    ):
        """Initialize the base trainer.
        
        Args:
            model: Model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator function
            processing_class: Tokenizer or processor
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.processing_class = processing_class
        
        # Training components (initialized later)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.eval_dataloader: Optional[DataLoader] = None
        
        # Training state
        self.state = TrainingState()
        
        # Initialize distributed training
        self._init_distributed()
        
        # Setup callbacks
        self._setup_default_callbacks()
    
    def _init_distributed(self):
        """Initialize distributed training state."""
        if dist.is_available() and dist.is_initialized():
            self.state.world_size = dist.get_world_size()
            self.state.rank = dist.get_rank()
            if torch.cuda.is_available():
                self.state.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.state.world_size = 1
            self.state.rank = 0
            self.state.local_rank = 0
    
    def _setup_default_callbacks(self):
        """Setup default callbacks."""
        # Add default logging callback
        self.add_callback(LoggingCallback())
        
        # Add checkpoint callback if output_dir is specified
        if self.args.output_dir:
            self.add_callback(CheckpointCallback())
    
    @abc.abstractmethod
    def create_train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        pass
    
    @abc.abstractmethod
    def create_eval_dataloader(self) -> DataLoader:
        """Create evaluation data loader."""
        pass
    
    @abc.abstractmethod
    def compute_loss(self, model: torch.nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for a batch."""
        pass
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer and learning rate scheduler."""
        # Create optimizer
        self.optimizer = ComponentFactory.create_optimizer(
            model=self.model,
            optimizer_name=self.args.optim,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        
        # Create scheduler
        warmup_steps = self.args.warmup_steps
        if self.args.warmup_ratio > 0:
            warmup_steps = int(self.args.warmup_ratio * num_training_steps)
        
        self.lr_scheduler = ComponentFactory.create_scheduler(
            optimizer=self.optimizer,
            scheduler_name=self.args.lr_scheduler_type,
            num_training_steps=num_training_steps,
            num_warmup_steps=warmup_steps,
        )
        
        return self.optimizer, self.lr_scheduler
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup training components
        self._setup_training()
        
        # Call training begin callbacks
        self._call_callbacks("on_train_begin", self, self.state)
        
        try:
            # Main training loop
            for epoch in range(int(self.args.num_train_epochs)):
                self.state.epoch = epoch
                self.state.epoch_start_time = time.time()
                
                # Epoch begin callbacks
                self._call_callbacks("on_epoch_begin", self, self.state)
                
                # Train one epoch
                self._train_epoch()
                
                # Epoch end callbacks
                self._call_callbacks("on_epoch_end", self, self.state)
                
                # Evaluation
                if self.eval_dataloader is not None:
                    self._evaluate()
        
        finally:
            # Call training end callbacks
            self._call_callbacks("on_train_end", self, self.state)
        
        logger.info("Training completed!")
    
    def _setup_training(self):
        """Setup training components."""
        # Create data loaders
        self.train_dataloader = self.create_train_dataloader()
        if self.eval_dataset is not None:
            self.eval_dataloader = self.create_eval_dataloader()
        
        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.args.gradient_accumulation_steps
        num_training_steps = int(self.args.num_train_epochs * num_update_steps_per_epoch)
        
        # Create optimizer and scheduler
        self.create_optimizer_and_scheduler(num_training_steps)
        
        logger.info(f"Training setup complete:")
        logger.info(f"  Num examples: {len(self.train_dataset)}")
        logger.info(f"  Num epochs: {self.args.num_train_epochs}")
        logger.info(f"  Batch size per device: {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size: {self.args.per_device_train_batch_size * self.state.world_size}")
        logger.info(f"  Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps: {num_training_steps}")
    
    def _train_epoch(self):
        """Train one epoch."""
        self.model.train()
        
        for step, batch in enumerate(self.train_dataloader):
            self.state.local_step = step
            self.state.step_start_time = time.time()
            
            # Step begin callbacks
            self._call_callbacks("on_step_begin", self, self.state)
            
            # Training step
            loss = self._training_step(batch)
            
            # Update state
            self.state.train_loss = loss.item()
            self.state.global_step += 1
            
            # Step end callbacks
            self._call_callbacks("on_step_end", self, self.state)
            
            # Update memory stats
            self.state.update_memory_stats()
    
    def _training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Execute one training step."""
        # Move batch to device
        batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        loss = self.compute_loss(self.model, batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation check
        if (self.state.local_step + 1) % self.args.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Update learning rate in state
            self.state.learning_rate = self.optimizer.param_groups[0]['lr']
        
        # Call loss computed callback
        self._call_callbacks("on_loss_computed", self, self.state, loss)
        
        return loss * self.args.gradient_accumulation_steps  # Return unscaled loss
    
    def _evaluate(self):
        """Run evaluation."""
        logger.info("Running evaluation...")
        self.model.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                loss = self.compute_loss(self.model, batch)
                total_loss += loss.item()
                num_samples += 1
        
        self.state.eval_loss = total_loss / num_samples
        logger.info(f"Evaluation loss: {self.state.eval_loss:.4f}")


# Default callbacks

@TrainerRegistry.register_callback("logging")
class LoggingCallback(TrainerCallback):
    """Default logging callback."""
    
    def __init__(self, log_frequency: int = 10):
        self.log_frequency = log_frequency
    
    def on_train_begin(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        logger.info("Training started")
    
    def on_train_end(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        logger.info(f"Training completed in {state.global_step} steps")
    
    def on_epoch_begin(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        logger.info(f"Starting epoch {state.epoch}")
    
    def on_epoch_end(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        epoch_time = time.time() - state.epoch_start_time
        logger.info(f"Completed epoch {state.epoch} in {epoch_time:.2f}s")
    
    def on_step_begin(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        pass
    
    def on_step_end(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        if state.global_step % self.log_frequency == 0:
            step_time = time.time() - state.step_start_time
            logger.info(
                f"Step {state.global_step}: loss={state.train_loss:.4f}, "
                f"lr={state.learning_rate:.2e}, time={step_time:.2f}s, "
                f"memory={state.current_memory_gb:.2f}GB"
            )


@TrainerRegistry.register_callback("checkpoint")
class CheckpointCallback(TrainerCallback):
    """Default checkpoint callback."""
    
    def on_train_begin(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        pass
    
    def on_train_end(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        # Save final checkpoint
        if trainer.args.output_dir:
            self._save_checkpoint(trainer, state, "final")
    
    def on_epoch_begin(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        pass
    
    def on_epoch_end(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        # Save epoch checkpoint
        if trainer.args.output_dir and trainer.args.save_strategy == "epoch":
            self._save_checkpoint(trainer, state, f"epoch-{state.epoch}")
    
    def on_step_begin(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        pass
    
    def on_step_end(self, trainer: ArcticBaseTrainer, state: TrainingState) -> None:
        # Save step checkpoint
        if (trainer.args.output_dir and 
            trainer.args.save_strategy == "steps" and 
            trainer.args.save_steps > 0 and 
            state.global_step % trainer.args.save_steps == 0):
            self._save_checkpoint(trainer, state, f"step-{state.global_step}")
    
    def _save_checkpoint(self, trainer: ArcticBaseTrainer, state: TrainingState, suffix: str):
        """Save model checkpoint."""
        import os
        
        checkpoint_dir = os.path.join(trainer.args.output_dir, f"checkpoint-{suffix}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if hasattr(trainer.model, 'save_pretrained'):
            trainer.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(trainer.model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
        
        # Save tokenizer
        if trainer.processing_class and hasattr(trainer.processing_class, 'save_pretrained'):
            trainer.processing_class.save_pretrained(checkpoint_dir)
        
        # Save optimizer state
        if trainer.optimizer:
            torch.save(trainer.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        
        # Save scheduler state
        if trainer.lr_scheduler:
            torch.save(trainer.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        
        # Save training state
        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")