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

"""Ulysses Sequence Parallelism enabled SFT Trainer.

This module provides a custom SFTTrainer that integrates Ulysses sequence
parallelism for training on extremely long sequences using DeepSpeed's
native implementation for robustness and performance.
"""

import math
from typing import Callable, Optional, Union

import torch
import torch.distributed.nn.functional
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
)
from trl import SFTTrainer

from oumi.utils.logging import logger

try:
    from deepspeed.runtime.sequence_parallel.ulysses_sp import (
        TiledFusedLogitsLoss,
        UlyssesSPAttentionHF,
        UlyssesSPDataLoaderAdapter,
    )
    from deepspeed.utils import groups

    DEEPSPEED_ULYSSES_AVAILABLE = True
except ImportError:
    logger.warning(
        "DeepSpeed Ulysses SP not available. Please install DeepSpeed with "
        "Ulysses SP support."
    )
    DEEPSPEED_ULYSSES_AVAILABLE = False

try:
    from oumi.core.parallel.tiled_compute import enable_tiled_mlp_compute

    TILED_MLP_AVAILABLE = True
except ImportError:
    logger.warning("Tiled MLP compute not available")
    TILED_MLP_AVAILABLE = False


class UlyssesSFTTrainer(SFTTrainer):
    """SFT Trainer with Ulysses sequence parallelism support.

    This trainer extends TRL's SFTTrainer to support Ulysses sequence parallelism,
    enabling training on extremely long sequences by sharding them across multiple GPUs.
    Uses DeepSpeed's native Ulysses SP implementation for robustness and performance.
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, torch.nn.Module]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list] = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Optional[Callable] = None,
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
            data_collator: Data collator function
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processing_class: Tokenizer or processor
            compute_loss_func: Custom loss computation function
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
            **kwargs: Additional arguments passed to SFTTrainer
        """
        if not DEEPSPEED_ULYSSES_AVAILABLE and sequence_parallel_size > 1:
            raise RuntimeError(
                "DeepSpeed Ulysses SP is required for sequence_parallel_size > 1. "
                "Please install DeepSpeed with Ulysses SP support."
            )

        self.sequence_parallel_size = sequence_parallel_size
        self.model_name_or_path = model_name_or_path
        self.attn_implementation = attn_implementation
        self.max_length = max_length
        self.micro_batch_size = micro_batch_size
        self.tiled_mlp_compute = tiled_mlp_compute
        self.use_liger_kernel = use_liger_kernel

        # Initialize SP-related attributes
        self.sp_group = None
        self.sp_world_size = 1
        self.sp_rank = 0
        self._mpu = None

        # Setup tiled MLP compute before model initialization if enabled
        if self.tiled_mlp_compute and self.model_name_or_path:
            self._setup_tiled_mlp_compute()

        # Setup Ulysses SP before parent initialization if needed
        if self.sequence_parallel_size > 1:
            self._setup_ulysses_sp()

        # Initialize parent SFTTrainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs,
        )

    def _setup_tiled_mlp_compute(self):
        """Setup tiled MLP computation for memory efficiency."""
        if not TILED_MLP_AVAILABLE:
            logger.warning("Tiled MLP compute not available, skipping setup")
            return

        if not self.model_name_or_path:
            logger.warning(
                "Model name/path not provided, cannot setup tiled MLP compute"
            )
            return

        logger.info("Setting up tiled MLP compute for memory efficiency")
        try:
            enable_tiled_mlp_compute(self.model_name_or_path)
            logger.info("Tiled MLP compute enabled successfully")
        except Exception as e:
            logger.error(f"Failed to setup tiled MLP compute: {e}")
            # Don't raise, just continue without tiled MLP
            self.tiled_mlp_compute = False

    def _setup_ulysses_sp(self):
        """Setup Ulysses sequence parallelism using DeepSpeed's implementation."""
        if not DEEPSPEED_ULYSSES_AVAILABLE:
            raise RuntimeError("DeepSpeed Ulysses SP not available")

        logger.info(
            f"Setting up Ulysses SP with "
            f"sequence_parallel_size={self.sequence_parallel_size}"
        )

        try:
            # Register Ulysses SP with transformers using DeepSpeed's implementation
            self._mpu = UlyssesSPAttentionHF.register_with_transformers(
                model_name_or_path=self.model_name_or_path,
                core_attn_implementation=self.attn_implementation,
                sequence_parallel_size=self.sequence_parallel_size,
                max_length=self.max_length,
                micro_batch_size=self.micro_batch_size,
                seq_length_is_variable=True,
            )
            logger.info(
                "Ulysses SP attention patches applied successfully using DeepSpeed"
            )
        except Exception as e:
            logger.error(f"Failed to setup Ulysses SP: {e}")
            raise

    def _initialize_sp_groups(self):
        """Initialize sequence parallel process groups after DeepSpeed init."""
        if self.sequence_parallel_size > 1 and DEEPSPEED_ULYSSES_AVAILABLE:
            try:
                # Get SP process groups from DeepSpeed
                self.sp_group = groups._get_sequence_parallel_group()
                self.sp_world_size = groups._get_sequence_parallel_world_size()
                self.sp_rank = groups._get_sequence_parallel_rank()

                logger.info(
                    f"Initialized SP groups: rank={self.sp_rank}, "
                    f"world_size={self.sp_world_size}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize SP groups: {e}")
                # Fallback to single GPU mode
                self.sequence_parallel_size = 1
                self.sp_group = None
                self.sp_world_size = 1
                self.sp_rank = 0

    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader, adapted for Ulysses SP if enabled.

        Returns:
            Training DataLoader, wrapped with UlyssesSPDataLoaderAdapter if SP enabled
        """
        # Get the standard training dataloader from parent
        dataloader = super().get_train_dataloader()

        # Wrap with Ulysses SP data loader adapter if sequence parallelism is enabled
        if self.sequence_parallel_size > 1 and DEEPSPEED_ULYSSES_AVAILABLE:
            # Initialize SP groups if not already done
            if self.sp_group is None:
                self._initialize_sp_groups()

            if self.sp_group is not None:
                logger.info(
                    f"Wrapping dataloader with UlyssesSPDataLoaderAdapter, "
                    f"sp_rank={self.sp_rank}, sp_world_size={self.sp_world_size}"
                )

                dataloader = UlyssesSPDataLoaderAdapter(
                    dataloader,
                    sp_rank=self.sp_rank,
                    sp_group=self.sp_group,
                    sp_world_size=self.sp_world_size,
                    device=self.args.device if self.args else None,
                )
            else:
                logger.warning(
                    "SP groups not available, falling back to standard dataloader"
                )

        return dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Get evaluation dataloader.

        Args:
            eval_dataset: Optional evaluation dataset

        Returns:
            Evaluation DataLoader
        """
        # For evaluation, we use the standard dataloader since Ulysses SP
        # is primarily for training very long sequences
        return super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with support for Ulysses SP.

        This method implements the sophisticated loss computation from ArcticTraining
        that handles sequence parallel training correctly.

        Args:
            model: The model to compute loss for
            inputs: Input batch
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, optionally with model outputs
        """
        # Handle SP-specific loss computation
        if self.sequence_parallel_size > 1 and DEEPSPEED_ULYSSES_AVAILABLE:
            return self._compute_loss_ulysses_sp(model, inputs, return_outputs)
        else:
            # Standard loss computation for non-SP case
            return super().compute_loss(model, inputs, return_outputs)

    def _compute_loss_ulysses_sp(self, model, inputs, return_outputs=False):
        """Compute loss for Ulysses sequence parallel training.

        Based on ArcticTraining's sophisticated loss computation that handles:
        1. shift_labels preprocessing
        2. Tiled logits+loss computation for memory efficiency
        3. Weighted loss aggregation across SP ranks

        Args:
            model: The model to compute loss for
            inputs: Input batch (should contain 'shift_labels' if from SP data loader)
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, optionally with model outputs
        """
        # Handle both 'labels' and 'shift_labels' formats for flexibility
        if "shift_labels" in inputs:
            shift_labels = inputs["shift_labels"]
        elif "labels" in inputs:
            # Convert labels to shift_labels if needed
            labels = inputs["labels"]
            # Shift labels for causal LM: shift right by one position
            shift_labels = labels[..., 1:].contiguous()
            # Pad with -100 (ignore index) at the end
            shift_labels = torch.cat(
                [
                    shift_labels,
                    torch.full(
                        (shift_labels.shape[0], 1),
                        -100,
                        dtype=shift_labels.dtype,
                        device=shift_labels.device,
                    ),
                ],
                dim=1,
            )
            # Update inputs to use shift_labels
            inputs = {k: v for k, v in inputs.items() if k != "labels"}
            inputs["shift_labels"] = shift_labels
        else:
            raise ValueError(
                "Missing 'labels' or 'shift_labels' in batch for loss computation."
            )

        shift_labels = inputs["shift_labels"]

        # Choose loss computation strategy
        if self.use_liger_kernel:
            # Use Liger fused cross-entropy for maximum efficiency
            outputs = model(**inputs, use_cache=False)
            loss = outputs.loss
        else:
            # Use tiled logits+loss computation for memory efficiency
            loss = self._compute_tiled_logits_loss(model, inputs, shift_labels)

        # Differentiable weighted per-shard-loss aggregation across SP ranks
        if self.sp_group is not None:
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
            loss = total_loss / total_good_tokens

        if return_outputs:
            # For SP, we need to be careful about returning outputs
            # since they may be sharded. For now, return None for outputs
            # in SP mode to avoid issues.
            return (loss, None)
        else:
            return loss

    def _compute_tiled_logits_loss(self, model, inputs, shift_labels):
        """Compute loss using tiled logits computation for memory efficiency.

        Args:
            model: The model
            inputs: Input batch
            shift_labels: Pre-shifted labels for loss computation

        Returns:
            Computed loss tensor
        """
        # Automatically determine number of shards based on memory target
        # Target ~1GB of fp32 logits per shard
        slice_size_in_gb = 1.0
        bs, seqlen = shift_labels.shape
        vocab_size = model.config.vocab_size
        logits_numel = bs * seqlen * vocab_size
        size_in_gb = logits_numel * 4 / (2**30)  # fp32
        num_shards = max(1, math.ceil(size_in_gb / slice_size_in_gb))

        logger.debug(f"Using {num_shards} shards for tiled logits computation")

        # Get model outputs (hidden states)
        outputs = model.model(**inputs, use_cache=False)
        hidden_states = outputs.last_hidden_state
        compute_params = [model.lm_head.weight]
        mask = None
        output_reduction = "sum"

        def fused_logits_loss_fn(model_arg, hidden_states_arg, shift_labels_arg):
            """Fused logits and loss computation function."""
            vocab_size = model_arg.config.vocab_size
            logits = model_arg.lm_head(hidden_states_arg)

            if all((shift_labels_arg == -100).squeeze()):
                # Handle case where all labels are -100 (padding)
                loss_sum = (logits.sum() * 0.0).float()
            else:
                good_items = sum((shift_labels_arg != -100).squeeze())
                loss = model_arg.loss_function(
                    logits=logits,
                    labels=None,
                    vocab_size=vocab_size,
                    shift_labels=shift_labels_arg,
                )
                loss_sum = loss * good_items
            return loss_sum

        # Apply tiled computation
        total_loss_sum = TiledFusedLogitsLoss.apply(
            fused_logits_loss_fn,
            model,
            hidden_states,
            shift_labels,
            mask,
            num_shards,
            compute_params,
            output_reduction,
        )

        total_good_items = sum((shift_labels != -100).squeeze())
        loss = total_loss_sum / total_good_items

        return loss

    def get_mpu(self):
        """Get the model parallel unit (MPU) for DeepSpeed initialization.

        Returns:
            MPU object for DeepSpeed, or None if not using SP
        """
        return self._mpu

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
