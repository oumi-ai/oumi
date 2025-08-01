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

"""Sequence parallelism components for trainers."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed.nn.functional
from torch.utils.data import DataLoader

from oumi.utils.logging import logger

try:
    from deepspeed.runtime.sequence_parallel.ulysses_sp import (
        UlyssesSPAttentionHF,
        UlyssesSPDataLoaderAdapter,
    )
    from deepspeed.utils import groups

    DEEPSPEED_ULYSSES_AVAILABLE = True
except ImportError:
    logger.warning("DeepSpeed Ulysses SP not available")
    DEEPSPEED_ULYSSES_AVAILABLE = False


@dataclass
class SequenceParallelConfig:
    """Configuration for sequence parallelism."""

    # Core SP settings
    sequence_parallel_size: int = 1
    model_name_or_path: Optional[str] = None
    attn_implementation: str = "sdpa"
    max_length: int = 4096
    micro_batch_size: int = 1
    seq_length_is_variable: bool = True

    def is_enabled(self) -> bool:
        """Check if sequence parallelism is enabled."""
        return self.sequence_parallel_size > 1


class SequenceParallelManager:
    """Manager for Ulysses sequence parallelism."""

    def __init__(self, config: SequenceParallelConfig):
        """Initialize the SP manager.

        Args:
            config: Sequence parallel configuration
        """
        self.config = config
        self.sp_group = None
        self.sp_world_size = 1
        self.sp_rank = 0
        self._mpu = None
        self._initialized = False

    def setup(self) -> bool:
        """Setup Ulysses sequence parallelism.

        Returns:
            True if setup successful, False otherwise
        """
        if not self.config.is_enabled():
            logger.info("Sequence parallelism not enabled")
            return True

        if not DEEPSPEED_ULYSSES_AVAILABLE:
            raise RuntimeError(
                "DeepSpeed Ulysses SP is required for sequence_parallel_size > 1. "
                "Please install DeepSpeed with Ulysses SP support."
            )

        logger.info(
            f"Setting up Ulysses SP with sequence_parallel_size={self.config.sequence_parallel_size}"
        )

        try:
            # Register Ulysses SP with transformers using DeepSpeed's implementation
            mpu_result = UlyssesSPAttentionHF.register_with_transformers(
                model_name_or_path=self.config.model_name_or_path,
                core_attn_implementation=self.config.attn_implementation,
                sequence_parallel_size=self.config.sequence_parallel_size,
                max_length=self.config.max_length,
                micro_batch_size=self.config.micro_batch_size,
                seq_length_is_variable=self.config.seq_length_is_variable,
            )

            # Store the MPU for DeepSpeed initialization
            self._mpu = mpu_result

            logger.info(
                "Ulysses SP attention patches applied successfully using DeepSpeed"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to setup Ulysses SP: {e}")
            raise

    def initialize_groups(self) -> bool:
        """Initialize sequence parallel process groups after DeepSpeed init.

        SP groups are only available after deepspeed.initialize() has been called.
        Based on ArcticTraining reference implementation.

        Returns:
            True if initialization successful, False otherwise
        """
        if not self.config.is_enabled() or not DEEPSPEED_ULYSSES_AVAILABLE:
            return True

        if self._initialized:
            logger.info("SP groups already initialized")
            return True

        try:
            logger.info("Attempting to get SP groups...")

            # SP groups are available after deepspeed.initialize()
            self.sp_group = groups._get_sequence_parallel_group()
            self.sp_world_size = groups._get_sequence_parallel_world_size()
            self.sp_rank = groups._get_sequence_parallel_rank()

            logger.info(
                f"Initialized SP groups: rank={self.sp_rank}, "
                f"world_size={self.sp_world_size}"
            )

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SP groups: {e}")
            logger.warning(
                "SP groups initialization failed. This usually means DeepSpeed "
                "hasn't been initialized yet or SP groups haven't been created. "
                "This will be retried later."
            )
            return False

    def get_mpu(self):
        """Get the model parallel unit (MPU) for DeepSpeed initialization.

        Returns:
            MPU object for DeepSpeed, or None if not using SP
        """
        return self._mpu

    def wrap_dataloader(
        self, dataloader: DataLoader, device: Optional[torch.device] = None
    ) -> DataLoader:
        """Wrap dataloader with SP support if enabled.

        Args:
            dataloader: Original dataloader
            device: Target device

        Returns:
            Wrapped dataloader or original if SP not enabled
        """
        if not self.config.is_enabled() or not DEEPSPEED_ULYSSES_AVAILABLE:
            return dataloader

        if not self._initialized:
            logger.warning("SP groups not initialized, cannot wrap dataloader")
            return dataloader

        if self.sp_group is None:
            logger.warning("SP group not available, cannot wrap dataloader")
            return dataloader

        logger.info(
            f"Wrapping dataloader with UlyssesSPDataLoaderAdapter, "
            f"sp_rank={self.sp_rank}, sp_world_size={self.sp_world_size}"
        )

        # Debug: Check what keys are in the first batch before wrapping
        try:
            first_batch = next(iter(dataloader))
            logger.info("First batch keys before SP wrapping:")
            for key, value in first_batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.info(f"  {key}: {type(value)}")
                    
            # Check if labels field is present
            if "labels" not in first_batch:
                logger.error(
                    "CRITICAL: 'labels' field missing from batch! "
                    "This will cause UlyssesSPDataLoaderAdapter to fail. "
                    f"Available keys: {list(first_batch.keys())}"
                )
                logger.error(
                    "This suggests the dataset or data collator is not properly configured for SFT training."
                )
            
            # Check sequence length divisibility for SP
            if self.config.is_enabled():
                for key, value in first_batch.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                        seq_len = value.shape[1]  # Assume [batch, seq_len, ...]
                        if seq_len % self.config.sequence_parallel_size != 0:
                            logger.warning(
                                f"Sequence length {seq_len} for tensor '{key}' is not divisible by "
                                f"sequence_parallel_size={self.config.sequence_parallel_size}. "
                                f"This will cause SP errors. Consider setting model_max_length to a value "
                                f"divisible by {self.config.sequence_parallel_size}."
                            )
                            
                            # Suggest compatible lengths
                            suggested_lengths = []
                            for i in range(1, 5):  # Suggest a few options
                                lower = (seq_len // self.config.sequence_parallel_size) * self.config.sequence_parallel_size
                                upper = lower + self.config.sequence_parallel_size
                                if lower > 0:
                                    suggested_lengths.append(lower)
                                suggested_lengths.append(upper)
                            
                            unique_suggestions = sorted(set(suggested_lengths))[:3]  # Top 3 suggestions
                            logger.warning(f"Suggested model_max_length values: {unique_suggestions}")
                
        except Exception as e:
            logger.error(f"Failed to inspect first batch: {e}")

        try:
            wrapped_dataloader = UlyssesSPDataLoaderAdapter(
                dataloader,
                sp_rank=self.sp_rank,
                sp_group=self.sp_group,
                sp_world_size=self.sp_world_size,
                device=device,
            )

            logger.info(
                "Successfully wrapped dataloader with UlyssesSPDataLoaderAdapter"
            )
            return wrapped_dataloader

        except Exception as e:
            logger.error(f"Failed to wrap dataloader with SP: {e}")
            logger.warning("Falling back to standard dataloader")
            return dataloader

    def compute_loss_with_sp_aggregation(
        self, loss: torch.Tensor, shift_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with SP aggregation across ranks.

        Args:
            loss: Local loss tensor
            shift_labels: Shifted labels for token counting

        Returns:
            Aggregated loss across SP ranks
        """
        if not self.config.is_enabled() or self.sp_group is None:
            return loss

        # Differentiable weighted per-shard-loss aggregation across SP ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(
            loss, group=self.sp_group
        )
        good_tokens = sum((shift_labels != -100).view(-1))
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(
            good_tokens, group=self.sp_group
        )

        total_loss = sum(
            losses_per_rank[rank] * good_tokens_per_rank[rank]
            for rank in range(self.sp_world_size)
        )
        total_good_tokens = sum(good_tokens_per_rank)
        aggregated_loss = total_loss / total_good_tokens

        return aggregated_loss

    def validate_sp_batch(self, inputs: dict[str, Any]) -> bool:
        """Validate that batch has proper SP format.

        Args:
            inputs: Input batch

        Returns:
            True if batch is properly formatted for SP
        """
        if not self.config.is_enabled():
            return True

        # For SP, we expect 'shift_labels' instead of 'labels'
        if "labels" in inputs:
            logger.error(
                "Found 'labels' in batch - they shouldn't be there for Ulysses SP, "
                "instead 'shift_labels' should be there. Check that "
                "UlyssesSPDataLoaderAdapter has been applied to the original DataLoader object."
            )
            return False

        if "shift_labels" not in inputs:
            logger.error(
                "shift_labels are missing from the batch - check that "
                "UlyssesSPDataLoaderAdapter has been applied to the original DataLoader object."
            )
            return False

        return True

    @property
    def is_initialized(self) -> bool:
        """Check if SP groups are initialized."""
        return self._initialized

    @property
    def is_enabled(self) -> bool:
        """Check if SP is enabled."""
        return self.config.is_enabled()


class SequenceParallelLossComputer:
    """Compute loss with sequence parallelism support."""

    def __init__(
        self, sp_manager: SequenceParallelManager, use_liger_kernel: bool = False
    ):
        """Initialize loss computer.

        Args:
            sp_manager: Sequence parallel manager
            use_liger_kernel: Whether to use Liger kernel for loss computation
        """
        self.sp_manager = sp_manager
        self.use_liger_kernel = use_liger_kernel

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
    ) -> torch.Tensor:
        """Compute loss with SP support.

        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, optionally with model outputs
        """
        # Check if we're using SP
        has_sp_data = self.sp_manager.validate_sp_batch(inputs)

        if self.sp_manager.is_enabled and has_sp_data:
            return self._compute_loss_with_sp(model, inputs, return_outputs)
        else:
            return self._compute_loss_standard(model, inputs, return_outputs)

    def _compute_loss_with_sp(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
    ) -> torch.Tensor:
        """Compute loss for Ulysses sequence parallel training."""
        logger.debug("Using Ulysses SP loss computation")

        shift_labels = inputs["shift_labels"]

        if self.use_liger_kernel:
            logger.debug("Using Liger fused cross-entropy...")
            outputs = model(**inputs, use_cache=False)
            loss = outputs.loss
        else:
            logger.debug("Using tiled logits+loss computation...")
            from .memory_optimization import MemoryOptimizer

            loss = MemoryOptimizer.compute_tiled_logits_loss(
                model, inputs, shift_labels
            )

        # Aggregate loss across SP ranks
        loss = self.sp_manager.compute_loss_with_sp_aggregation(loss, shift_labels)

        if return_outputs:
            # For SP, return None for outputs to avoid issues with sharded outputs
            return (loss, None)
        else:
            return loss

    def _compute_loss_standard(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
    ) -> torch.Tensor:
        """Compute loss using standard method."""
        logger.info("Using standard loss computation")
        logger.info(f"Input keys: {list(inputs.keys())}")
        
        # Debug inputs
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        outputs = model(**inputs, use_cache=False)
        
        logger.info(f"Model outputs type: {type(outputs)}")
        logger.info(f"Model outputs has loss: {hasattr(outputs, 'loss')}")
        
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
            logger.info(f"Loss from model: {loss}")
        else:
            logger.error("Model outputs do not contain 'loss' attribute!")
            logger.error(f"Available attributes: {dir(outputs)}")
            raise RuntimeError("Model forward pass did not return loss")

        if return_outputs:
            return (loss, outputs)
        else:
            return loss
