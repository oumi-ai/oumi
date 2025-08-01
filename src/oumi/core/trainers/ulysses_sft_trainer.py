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
    Trainer,
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
        self.original_sequence_parallel_size = sequence_parallel_size  # Preserve original value
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
        self._deepspeed_initialized_with_mpu = False

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
            mpu_result = UlyssesSPAttentionHF.register_with_transformers(
                model_name_or_path=self.model_name_or_path,
                core_attn_implementation=self.attn_implementation,
                sequence_parallel_size=self.sequence_parallel_size,
                max_length=self.max_length,
                micro_batch_size=self.micro_batch_size,
                seq_length_is_variable=True,
            )
            # The DeepSpeed implementation returns the parallel_state_sp module itself
            # which contains the SP groups functionality, not a separate MPU object
            self._mpu = mpu_result
            
            logger.info(
                "Ulysses SP attention patches applied successfully using DeepSpeed"
            )
        except Exception as e:
            logger.error(f"Failed to setup Ulysses SP: {e}")
            raise

    def _initialize_sp_groups(self):
        """Initialize sequence parallel process groups after DeepSpeed init.

        Based on ArcticTraining reference implementation, SP groups are only
        available after deepspeed.initialize() has been called.
        """
        if self.sequence_parallel_size > 1 and DEEPSPEED_ULYSSES_AVAILABLE:
            try:
                logger.info("Attempting to get SP group...")
                # SP groups are available after deepspeed.initialize()
                # This follows the ArcticTraining pattern from trainer.py:236-243
                self.sp_group = groups._get_sequence_parallel_group()
                logger.info("Got SP group, getting world size...")
                
                self.sp_world_size = groups._get_sequence_parallel_world_size()
                logger.info("Got SP world size, getting rank...")
                
                self.sp_rank = groups._get_sequence_parallel_rank()
                logger.info("Got SP rank")

                logger.info(
                    f"Initialized SP groups: rank={self.sp_rank}, "
                    f"world_size={self.sp_world_size}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize SP groups: {e}")
                logger.warning(
                    "SP groups initialization failed. This usually means DeepSpeed "
                    "hasn't been initialized yet or SP groups haven't been created. "
                    "Falling back to standard training mode."
                )
                # Don't disable SP here - we might initialize it later via custom DeepSpeed init
                # Just clear the SP group info for now
                self.sp_group = None
                self.sp_world_size = 1
                self.sp_rank = 0

    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader, adapted for Ulysses SP if enabled.

        Returns:
            Training DataLoader, wrapped with UlyssesSPDataLoaderAdapter if SP enabled
        """
        logger.info("Getting train dataloader...")
        
        # Get the standard training dataloader from parent
        dataloader = super().get_train_dataloader()
        logger.info("Got standard dataloader from parent")

        # Wrap with Ulysses SP data loader adapter if sequence parallelism is enabled
        # Use original_sequence_parallel_size in case sequence_parallel_size was reset
        if self.original_sequence_parallel_size > 1 and DEEPSPEED_ULYSSES_AVAILABLE:
            logger.info("Sequence parallelism enabled, checking SP groups...")
            
            # Initialize SP groups if not already done
            # Called post-DeepSpeed initialization, so groups should be available
            if self.sp_group is None:
                logger.info("SP groups not initialized, attempting initialization...")
                self._initialize_sp_groups()

            if self.sp_group is not None:
                logger.info(
                    f"Wrapping dataloader with UlyssesSPDataLoaderAdapter, "
                    f"sp_rank={self.sp_rank}, sp_world_size={self.sp_world_size}"
                )

                logger.info("About to create UlyssesSPDataLoaderAdapter...")
                dataloader = UlyssesSPDataLoaderAdapter(
                    dataloader,
                    sp_rank=self.sp_rank,
                    sp_group=self.sp_group,
                    sp_world_size=self.sp_world_size,
                    device=self.args.device if self.args else None,
                )
                logger.info("Created UlyssesSPDataLoaderAdapter successfully")

                logger.info(
                    "Successfully wrapped dataloader w/UlyssesSPDataLoaderAdapter"
                )
            else:
                logger.warning(
                    "SP groups still not available after initialization attempt. "
                    "This may mean DeepSpeed hasn't been properly initialized w/SP. "
                    "Falling back to standard dataloader."
                )
        else:
            logger.info("Sequence parallelism not enabled or not available")

        logger.info("Returning dataloader")
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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute loss with support for Ulysses SP.

        This method implements the sophisticated loss computation from ArcticTraining
        that handles sequence parallel training correctly.

        Args:
            model: The model to compute loss for
            inputs: Input batch
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in the batch (for compatibility)

        Returns:
            Loss tensor, optionally with model outputs
        """
        # Handle SP-specific loss computation
        if self.sequence_parallel_size > 1 and DEEPSPEED_ULYSSES_AVAILABLE:
            return self._compute_loss_ulysses_sp(model, inputs, return_outputs)
        else:
            # Standard loss computation for non-SP case
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )

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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Override to pass MPU to DeepSpeed initialization for Ulysses SP.

        This ensures the MPU created during Ulysses SP setup is properly passed
        to DeepSpeed initialization so sequence parallel groups are created.
        Based on ArcticTraining pattern where MPU is passed to deepspeed.initialize().
        """
        if self.is_deepspeed_enabled and self._mpu is not None:
            logger.info("Initializing DeepSpeed with Ulysses SP MPU")

            try:
                # Initialize DeepSpeed with our MPU
                # This is where the sequence parallel groups get created
                import deepspeed

                # Get DeepSpeed config
                ds_config = self.args.deepspeed
                if isinstance(ds_config, str):
                    import json

                    with open(ds_config) as f:
                        ds_config = json.load(f)
                
                # Ensure ulysses_sequence_parallel_size is in the config
                # Use original value, not potentially modified self.sequence_parallel_size
                if 'ulysses_sequence_parallel_size' not in ds_config:
                    logger.warning(
                        f"Adding ulysses_sequence_parallel_size={self.original_sequence_parallel_size} to DeepSpeed config"
                    )
                    ds_config['ulysses_sequence_parallel_size'] = self.original_sequence_parallel_size
                
                # Handle 'auto' values and fix batch size types in config
                # When values are 'auto', DeepSpeed will calculate them based on other parameters
                # We need to remove them or set appropriate defaults
                if 'train_batch_size' in ds_config and ds_config['train_batch_size'] == 'auto':
                    # Let DeepSpeed calculate from micro_batch_size * gradient_accumulation * world_size
                    del ds_config['train_batch_size']
                    logger.info("Removed train_batch_size='auto' to let DeepSpeed calculate it")
                
                if 'train_micro_batch_size_per_gpu' in ds_config and ds_config['train_micro_batch_size_per_gpu'] == 'auto':
                    # Use the micro_batch_size from training args
                    ds_config['train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
                    logger.info(f"Set train_micro_batch_size_per_gpu={ds_config['train_micro_batch_size_per_gpu']}")
                
                if 'gradient_accumulation_steps' in ds_config and ds_config['gradient_accumulation_steps'] == 'auto':
                    # Use gradient accumulation from training args
                    ds_config['gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
                    logger.info(f"Set gradient_accumulation_steps={ds_config['gradient_accumulation_steps']}")
                
                # Now convert any remaining string numbers to integers
                for key in ['train_batch_size', 'train_micro_batch_size_per_gpu', 
                           'gradient_accumulation_steps', 'micro_batch_per_gpu']:
                    if key in ds_config and isinstance(ds_config[key], str):
                        try:
                            ds_config[key] = int(ds_config[key])
                            logger.info(f"Converted {key} from string to int: {ds_config[key]}")
                        except ValueError:
                            logger.error(f"Failed to convert {key}='{ds_config[key]}' to int")

                # Create optimizer if needed (similar to HF Trainer pattern)
                if self.optimizer is None:
                    optimizer_cls, optimizer_kwargs = (
                        Trainer.get_optimizer_cls_and_kwargs(self.args)
                    )
                    self.optimizer = optimizer_cls(
                        self.model.parameters(), **optimizer_kwargs
                    )

                # Create scheduler if needed
                if self.lr_scheduler is None:
                    # Calculate warmup steps safely
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
                    f"Initializing DeepSpeed with ulysses_sequence_parallel_size={ds_config.get('ulysses_sequence_parallel_size', 'NOT SET')}"
                )

                # Save the original model config before DeepSpeed wrapping
                original_config = self.model.config
                
                # Initialize DeepSpeed with MPU - this creates SP groups
                engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    config=ds_config,
                    mpu=self._mpu,  # This is the key - pass our MPU
                )

                # Update trainer's components
                self.model = engine
                self.optimizer = optimizer
                self.lr_scheduler = lr_scheduler
                
                # Restore original config if DeepSpeed replaced it with a dict
                if hasattr(self.model, 'config') and isinstance(self.model.config, dict):
                    self.model.config = original_config
                    logger.info("Restored original model config after DeepSpeed initialization")

                logger.info("DeepSpeed initialized successfully with Ulysses SP MPU")
                
                # Mark that we've initialized with MPU
                self._deepspeed_initialized_with_mpu = True

                # SP groups should now be available
                logger.info("About to initialize SP groups...")
                
                # Now initialize SP groups since DeepSpeed is ready
                self._initialize_sp_groups()
                
                logger.info("SP groups initialization completed")
                return optimizer, lr_scheduler

            except Exception as e:
                import traceback
                logger.error(f"Failed to initialize DeepSpeed with MPU: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                logger.warning("Falling back to standard DeepSpeed initialization")
                # Fall back to standard initialization
                return super().create_optimizer_and_scheduler(num_training_steps)
        else:
            # Standard initialization for non-SP case
            return super().create_optimizer_and_scheduler(num_training_steps)

    def get_mpu(self):
        """Get the model parallel unit (MPU) for DeepSpeed initialization.

        Returns:
            MPU object for DeepSpeed, or None if not using SP
        """
        return self._mpu

    def initialize_after_deepspeed(self):
        """Initialize SP-related components after DeepSpeed initialization.

        This method should be called after DeepSpeed has been initialized
        to properly set up sequence parallel groups. Based on ArcticTraining pattern.
        """
        if self.sequence_parallel_size > 1 and DEEPSPEED_ULYSSES_AVAILABLE:
            # Skip this if we've already initialized with our custom DeepSpeed init
            if not self._deepspeed_initialized_with_mpu:
                logger.info("Initializing SP groups after DeepSpeed initialization")
                self._initialize_sp_groups()
            else:
                logger.info("SP groups already initialized via custom DeepSpeed init")

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
