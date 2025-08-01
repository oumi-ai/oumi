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

"""Ulysses Sequence Parallelism enabled SFT Trainer - Version 2.

This module provides a custom SFT trainer based on ArcticTraining patterns
that integrates Ulysses sequence parallelism for training on extremely long
sequences using DeepSpeed's native implementation.
"""

import json
from typing import Any, Callable, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
)

from oumi.utils.logging import logger

from .arctic_base_trainer import ArcticBaseTrainer, ComponentFactory, TrainerRegistry
from .components.memory_optimization import LigerKernelOptimizer, MemoryOptimizer
from .components.sequence_parallel import (
    SequenceParallelConfig,
    SequenceParallelLossComputer,
    SequenceParallelManager,
)


@TrainerRegistry.register_trainer("ulysses_sft")
class UlyssesSFTTrainer(ArcticBaseTrainer):
    """SFT Trainer with Ulysses sequence parallelism support.

    This trainer extends ArcticBaseTrainer to support Ulysses sequence parallelism,
    enabling training on extremely long sequences by sharding them across multiple GPUs.
    Uses DeepSpeed's native Ulysses SP implementation for robustness and performance.
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, torch.nn.Module]] = None,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        data_collator: Optional[Callable] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list] = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Optional[Callable] = None,
        # Ulysses SP specific parameters
        sequence_parallel_size: int = 1,
        model_name_or_path: Optional[str] = None,
        attn_implementation: str = "sdpa",
        max_length: int = 4096,
        micro_batch_size: int = 1,
        tiled_mlp_compute: bool = False,
        use_liger_kernel: bool = False,
        **kwargs,
    ):
        """Initialize UlyssesSFTTrainer.

        Args:
            model: Model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator function
            processing_class: Tokenizer or processor
            compute_metrics: Metrics computation function
            callbacks: Training callbacks
            optimizers: Optimizer and scheduler tuple
            preprocess_logits_for_metrics: Logits preprocessing function
            sequence_parallel_size: Number of GPUs for sequence parallelism
            model_name_or_path: Model name or path for configuration
            attn_implementation: Attention implementation (flash_attention_2, etc.)
            max_length: Maximum sequence length
            micro_batch_size: Micro batch size
            tiled_mlp_compute: Whether to enable tiled MLP computation
            use_liger_kernel: Whether to use Liger kernel optimizations
            **kwargs: Additional arguments
        """
        # Initialize parent first
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
            **kwargs,
        )

        # Store SP-specific parameters
        self.sequence_parallel_size = sequence_parallel_size
        self.model_name_or_path = model_name_or_path
        self.tiled_mlp_compute = tiled_mlp_compute
        self.use_liger_kernel = use_liger_kernel

        # Setup sequence parallelism
        self.sp_config = SequenceParallelConfig(
            sequence_parallel_size=sequence_parallel_size,
            model_name_or_path=model_name_or_path,
            attn_implementation=attn_implementation,
            max_length=max_length,
            micro_batch_size=micro_batch_size,
        )

        self.sp_manager = SequenceParallelManager(self.sp_config)
        self.loss_computer = SequenceParallelLossComputer(
            self.sp_manager, use_liger_kernel=use_liger_kernel
        )

        # Setup memory optimizations
        self._setup_optimizations()

        # Setup SP before training
        if self.sp_config.is_enabled():
            self.sp_manager.setup()

        logger.info("UlyssesSFTTrainer V2 initialized successfully:")
        logger.info(f"  - Sequence parallel size: {self.sequence_parallel_size}")
        logger.info(f"  - Model name/path: {self.model_name_or_path}")
        logger.info(f"  - Tiled MLP compute: {self.tiled_mlp_compute}")
        logger.info(f"  - Liger kernel: {self.use_liger_kernel}")
        logger.info(f"  - SP enabled: {self.sp_config.is_enabled()}")

    def _setup_optimizations(self):
        """Setup memory and kernel optimizations."""
        # Setup tiled MLP compute
        if self.tiled_mlp_compute and self.model_name_or_path:
            success = MemoryOptimizer.setup_tiled_mlp_compute(self.model_name_or_path)
            if not success:
                self.tiled_mlp_compute = False

        # Setup Liger kernel optimizations
        if self.use_liger_kernel:
            if LigerKernelOptimizer.is_available():
                LigerKernelOptimizer.apply_liger_kernels(
                    self.model, self.use_liger_kernel
                )
            else:
                logger.warning("Liger kernel not available, disabling")
                self.use_liger_kernel = False

    def create_train_dataloader(self) -> DataLoader:
        """Create training data loader with SP support."""
        logger.info("Creating training dataloader...")

        # Create base dataloader
        dataloader = ComponentFactory.create_data_loader(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

        # Wrap with SP support if enabled and groups are initialized
        if self.sp_manager.is_initialized:
            dataloader = self.sp_manager.wrap_dataloader(dataloader, self.args.device)

        logger.info("Training dataloader created successfully")
        return dataloader

    def create_eval_dataloader(self) -> DataLoader:
        """Create evaluation data loader."""
        if self.eval_dataset is None:
            return None

        logger.info("Creating evaluation dataloader...")

        # For evaluation, use standard dataloader (no SP needed)
        dataloader = ComponentFactory.create_data_loader(
            dataset=self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

        logger.info("Evaluation dataloader created successfully")
        return dataloader

    def compute_loss(
        self, model: torch.nn.Module, inputs: dict[str, Any]
    ) -> torch.Tensor:
        """Compute loss with Ulysses SP support."""
        return self.loss_computer.compute_loss(model, inputs, return_outputs=False)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer and scheduler with DeepSpeed integration."""
        if self.is_deepspeed_enabled and self.sp_manager.get_mpu() is not None:
            return self._create_optimizer_with_deepspeed(num_training_steps)
        else:
            return super().create_optimizer_and_scheduler(num_training_steps)

    def _create_optimizer_with_deepspeed(self, num_training_steps: int):
        """Create optimizer with DeepSpeed and Ulysses SP integration."""
        logger.info("Initializing DeepSpeed with Ulysses SP MPU")

        try:
            import deepspeed

            # Get DeepSpeed config
            ds_config = self._prepare_deepspeed_config()

            # Create optimizer and scheduler first
            if self.optimizer is None:
                from transformers import Trainer

                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                    self.args
                )
                self.optimizer = optimizer_cls(
                    self.model.parameters(), **optimizer_kwargs
                )

            if self.lr_scheduler is None:
                warmup_steps = self.get_warmup_steps(num_training_steps)
                if warmup_steps is None:
                    warmup_steps = 0

                self.lr_scheduler = self.get_scheduler(
                    name=self.args.lr_scheduler_type,
                    optimizer=self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps,
                )

            logger.info(
                f"Initializing DeepSpeed with ulysses_sequence_parallel_size="
                f"{ds_config.get('ulysses_sequence_parallel_size', 'NOT SET')}"
            )

            # Save original config before DeepSpeed wrapping
            original_config = self.model.config

            # Initialize DeepSpeed with MPU - this creates SP groups
            engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                config=ds_config,
                mpu=self.sp_manager.get_mpu(),  # Pass our MPU
            )

            # Update trainer components
            self.model = engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

            # Restore original config if DeepSpeed replaced it
            if hasattr(self.model, "config") and isinstance(self.model.config, dict):
                self.model.config = original_config
                logger.info(
                    "Restored original model config after DeepSpeed initialization"
                )

            logger.info("DeepSpeed initialized successfully with Ulysses SP MPU")

            # Initialize SP groups now that DeepSpeed is ready
            self.sp_manager.initialize_groups()

            # Recreate training dataloader with SP support if groups are available
            if self.sp_manager.is_initialized:
                logger.info("Recreating training dataloader with SP support...")
                self.train_dataloader = self.create_train_dataloader()

            return optimizer, lr_scheduler

        except Exception as e:
            import traceback

            logger.error(f"Failed to initialize DeepSpeed with MPU: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.warning("Falling back to standard initialization")
            return super().create_optimizer_and_scheduler(num_training_steps)

    def _prepare_deepspeed_config(self) -> dict[str, Any]:
        """Prepare DeepSpeed configuration with SP settings."""
        ds_config = self.args.deepspeed
        if isinstance(ds_config, str):
            with open(ds_config) as f:
                ds_config = json.load(f)
        else:
            ds_config = ds_config.copy()

        # Ensure ulysses_sequence_parallel_size is in the config
        if "ulysses_sequence_parallel_size" not in ds_config:
            logger.warning(
                f"Adding ulysses_sequence_parallel_size={self.sequence_parallel_size} "
                "to DeepSpeed config"
            )
            ds_config["ulysses_sequence_parallel_size"] = self.sequence_parallel_size

        # Handle 'auto' values in config
        self._fix_deepspeed_auto_values(ds_config)

        return ds_config

    def _fix_deepspeed_auto_values(self, ds_config: dict[str, Any]):
        """Fix 'auto' values in DeepSpeed config."""
        # Handle train_batch_size
        if ds_config.get("train_batch_size") == "auto":
            del ds_config["train_batch_size"]
            logger.info("Removed train_batch_size='auto' to let DeepSpeed calculate it")

        # Handle train_micro_batch_size_per_gpu
        if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
            ds_config["train_micro_batch_size_per_gpu"] = (
                self.args.per_device_train_batch_size
            )
            logger.info(
                f"Set train_micro_batch_size_per_gpu={ds_config['train_micro_batch_size_per_gpu']}"
            )

        # Handle gradient_accumulation_steps
        if ds_config.get("gradient_accumulation_steps") == "auto":
            ds_config["gradient_accumulation_steps"] = (
                self.args.gradient_accumulation_steps
            )
            logger.info(
                f"Set gradient_accumulation_steps={ds_config['gradient_accumulation_steps']}"
            )

        # Convert string numbers to integers
        for key in [
            "train_batch_size",
            "train_micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
            "micro_batch_per_gpu",
        ]:
            if key in ds_config and isinstance(ds_config[key], str):
                try:
                    ds_config[key] = int(ds_config[key])
                    logger.info(f"Converted {key} from string to int: {ds_config[key]}")
                except ValueError:
                    logger.error(f"Failed to convert {key}='{ds_config[key]}' to int")

    @property
    def is_deepspeed_enabled(self) -> bool:
        """Check if DeepSpeed is enabled."""
        return self.args.deepspeed is not None

    def get_warmup_steps(self, num_training_steps: int) -> int:
        """Get number of warmup steps."""
        warmup_steps = self.args.warmup_steps
        if self.args.warmup_ratio > 0:
            warmup_steps = int(self.args.warmup_ratio * num_training_steps)
        return warmup_steps

    def get_scheduler(
        self, name: str, optimizer, num_warmup_steps: int, num_training_steps: int
    ):
        """Get learning rate scheduler."""
        from transformers import get_scheduler

        return get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def initialize_after_deepspeed(self):
        """Initialize SP-related components after DeepSpeed initialization.

        This method should be called after DeepSpeed has been initialized
        to properly set up sequence parallel groups. Based on ArcticTraining pattern.
        """
        if self.sp_manager.is_enabled and not self.sp_manager.is_initialized:
            logger.info("Initializing SP groups after DeepSpeed initialization")
            self.sp_manager.initialize_groups()

            # Recreate training dataloader with SP support if groups are available
            if self.sp_manager.is_initialized and hasattr(self, "train_dataloader"):
                logger.info("Recreating training dataloader with SP support...")
                self.train_dataloader = self.create_train_dataloader()
        else:
            logger.info("SP already initialized or not enabled")

    @classmethod
    def from_config(
        cls,
        model: PreTrainedModel,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        args: Optional[TrainingArguments] = None,
        sequence_parallel_size: int = 1,
        model_name_or_path: Optional[str] = None,
        attn_implementation: str = "sdpa",
        max_length: int = 4096,
        micro_batch_size: int = 1,
        tiled_mlp_compute: bool = False,
        use_liger_kernel: bool = False,
        **kwargs,
    ) -> "UlyssesSFTTrainer":
        """Create UlyssesSFTTrainer from configuration.

        Args:
            model: Model to train
            processing_class: Tokenizer or processor
            args: Training arguments
            sequence_parallel_size: Number of GPUs for sequence parallelism
            model_name_or_path: Model name or path for configuration
            attn_implementation: Attention implementation
            max_length: Maximum sequence length
            micro_batch_size: Micro batch size
            tiled_mlp_compute: Whether to enable tiled MLP computation
            use_liger_kernel: Whether to use Liger kernel optimizations
            **kwargs: Additional arguments

        Returns:
            Configured UlyssesSFTTrainer instance
        """
        return cls(
            model=model,
            processing_class=processing_class,
            args=args,
            sequence_parallel_size=sequence_parallel_size,
            model_name_or_path=model_name_or_path,
            attn_implementation=attn_implementation,
            max_length=max_length,
            micro_batch_size=micro_batch_size,
            tiled_mlp_compute=tiled_mlp_compute,
            use_liger_kernel=use_liger_kernel,
            **kwargs,
        )
