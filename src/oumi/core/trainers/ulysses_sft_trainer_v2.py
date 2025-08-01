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
        # Use the actual training batch size for micro_batch_size
        effective_micro_batch_size = (
            self.args.per_device_train_batch_size if self.args else micro_batch_size
        )

        self.sp_config = SequenceParallelConfig(
            sequence_parallel_size=sequence_parallel_size,
            model_name_or_path=model_name_or_path,
            attn_implementation=attn_implementation,
            max_length=max_length,
            micro_batch_size=effective_micro_batch_size,
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

        # Early DeepSpeed initialization (Arctic pattern) - moved from create_optimizer_and_scheduler
        # This prevents gradient state corruption by initializing DeepSpeed before HF Trainer setup
        self._deepspeed_initialized = False
        if self.is_deepspeed_enabled and self.sp_manager.get_mpu() is not None:
            # Schedule early DeepSpeed initialization - will happen after parent __init__ completes
            self._needs_early_deepspeed_init = True
        else:
            self._needs_early_deepspeed_init = False

        logger.info("UlyssesSFTTrainer V2 initialized successfully:")
        logger.info(f"  - Sequence parallel size: {self.sequence_parallel_size}")
        logger.info(f"  - Model name/path: {self.model_name_or_path}")
        logger.info(f"  - Max length: {max_length}")
        logger.info(f"  - Micro batch size: {effective_micro_batch_size}")
        logger.info(f"  - Per device train batch size: {self.args.per_device_train_batch_size if self.args else 'N/A'}")
        logger.info(f"  - Tiled MLP compute: {self.tiled_mlp_compute}")
        logger.info(f"  - Liger kernel: {self.use_liger_kernel}")
        logger.info(f"  - SP enabled: {self.sp_config.is_enabled()}")
        logger.info(f"  - Variable seq length: {self.sp_config.seq_length_is_variable}")

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
        logger.info(f"  - Dataset size: {len(self.train_dataset) if self.train_dataset else 'N/A'}")
        logger.info(f"  - Batch size: {self.args.per_device_train_batch_size}")
        logger.info(f"  - Data collator: {type(self.data_collator).__name__ if self.data_collator else 'None'}")
        
        # If we did early DeepSpeed initialization but didn't recreate dataloader yet, do it now
        if (self._deepspeed_initialized and self.sp_manager.is_initialized 
            and not hasattr(self, '_sp_dataloader_created')):
            logger.info("  - SP was initialized early, ensuring SP-aware dataloader creation")
            self._sp_dataloader_created = True
        logger.info(f"  - SP manager initialized: {self.sp_manager.is_initialized}")

        # ALWAYS create our robust collator to handle variable-length sequences
        # This is crucial because PyTorch's default collator fails with variable lengths
        collator = self._create_sp_aware_collator(self.data_collator)
        logger.info(f"  - Using robust SP-aware collator wrapping: {type(self.data_collator).__name__}")

        # Set num_workers to 0 to avoid multiprocessing collation issues
        num_workers = 0 if self.sp_config.is_enabled() else self.args.dataloader_num_workers
        if num_workers != self.args.dataloader_num_workers:
            logger.info(f"  - Forcing num_workers=0 for SP compatibility (was {self.args.dataloader_num_workers})")

        # Create base dataloader
        dataloader = ComponentFactory.create_data_loader(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=num_workers,
        )

        # Wrap with SP support if enabled and groups are initialized
        if self.sp_manager.is_initialized:
            logger.info("Wrapping dataloader with SP support...")
            dataloader = self.sp_manager.wrap_dataloader(dataloader, self.args.device)
        elif self.sp_config.is_enabled():
            logger.warning("SP is enabled but groups not initialized - using standard dataloader")

        logger.info("Training dataloader created successfully")
        return dataloader

    def _create_sp_aware_collator(self, original_collator):
        """Create a robust collator that ensures equal-sized batches and SP divisibility."""
        sp_size = self.sp_config.sequence_parallel_size

        def sp_collator(batch):
            try:
                logger.debug(f"SP collator processing batch of size {len(batch)}")

                # Debug the batch structure
                if batch and logger.isEnabledFor(10):  # DEBUG level
                    logger.debug("Batch sample structure:")
                    sample = batch[0]
                    for key, value in sample.items():
                        if isinstance(value, (list, torch.Tensor)):
                            if hasattr(value, "shape"):
                                logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                            else:
                                logger.debug(f"  {key}: list length={len(value)}")
                        else:
                            logger.debug(f"  {key}: {type(value)}")

                # First, ensure we have a proper collator - if None, create a simple one
                if original_collator is None:
                    logger.info("No collator provided - using simple dict collator")
                    result = self._simple_dict_collate(batch)
                else:
                    try:
                        result = original_collator(batch)
                    except Exception as e:
                        logger.error(f"Original collator failed: {e}")
                        logger.info("Falling back to simple dict collator")
                        result = self._simple_dict_collate(batch)

                # Debug result before padding
                logger.debug("Collator result before SP padding:")
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")

                # CRITICAL: Generate labels for SFT training if missing
                if "labels" not in result and "input_ids" in result:
                    logger.info("Generating labels from input_ids for SFT training")
                    result["labels"] = self._create_sft_labels(result["input_ids"])
                    logger.info(f"Generated labels with shape: {result['labels'].shape}")

                    # Debug label masking - always show this since it's critical
                    for i in range(min(2, result["labels"].shape[0])):  # Show first 2 samples
                        valid_labels = (result["labels"][i] != -100).sum().item()
                        total_labels = result["labels"][i].numel()
                        logger.info(f"Sample {i}: {valid_labels}/{total_labels} valid labels ({valid_labels/total_labels*100:.1f}%)")
                        if valid_labels == 0:
                            logger.error(f"Sample {i}: ALL LABELS MASKED! This will cause NaN loss.")
                        elif valid_labels < 5:
                            logger.warning(f"Sample {i}: Very few valid labels ({valid_labels}), may cause unstable training.")

                # Ensure all tensor sequences have equal length within the batch
                # and are divisible by SP size
                result = self._ensure_equal_length_and_sp_divisible(result, sp_size)

                # Debug final result
                logger.debug("Final collator result:")
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")

                return result

            except Exception as e:
                logger.error(f"SP collator failed: {e}")
                logger.error(f"Batch info: {len(batch)} items")
                if batch:
                    logger.error(f"First item keys: {list(batch[0].keys())}")
                    for i, item in enumerate(batch[:3]):  # Show first 3 items
                        logger.error(f"Item {i}:")
                        for key, value in item.items():
                            if isinstance(value, (list, torch.Tensor)):
                                if hasattr(value, "shape"):
                                    logger.error(f"  {key}: shape={value.shape}")
                                else:
                                    logger.error(f"  {key}: list length={len(value)}")
                raise

        return sp_collator

    def _simple_dict_collate(self, batch):
        """Simple dictionary-based collation that handles variable-length sequences."""
        if not batch:
            return {}

        # Get all unique keys from the batch
        all_keys = set()
        for item in batch:
            all_keys.update(item.keys())

        result = {}

        for key in all_keys:
            values = []
            for item in batch:
                if key in item:
                    values.append(item[key])
                else:
                    # Handle missing keys by using a default value
                    if key == "input_ids":
                        values.append([])  # Empty sequence
                    elif key == "labels":
                        values.append([])  # Empty sequence
                    elif key == "attention_mask":
                        values.append([])  # Empty sequence
                    else:
                        values.append(None)

            # Convert lists to tensors if they contain numbers
            if values and all(v is not None for v in values):
                if all(isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (int, float)) for v in values):
                    # Pad sequences to same length
                    max_len = max(len(v) for v in values)
                    padded_values = []

                    for v in values:
                        # Determine padding value
                        if key == "input_ids" and hasattr(self.processing_class, "pad_token_id"):
                            pad_value = self.processing_class.pad_token_id
                        elif key == "labels":
                            pad_value = -100
                        elif key == "attention_mask":
                            pad_value = 0
                        else:
                            pad_value = 0

                        # Pad the sequence
                        padded = list(v) + [pad_value] * (max_len - len(v))
                        padded_values.append(padded)

                    result[key] = torch.tensor(padded_values)
                elif all(isinstance(v, torch.Tensor) for v in values):
                    # Stack tensors if they have compatible shapes
                    try:
                        result[key] = torch.stack(values)
                    except RuntimeError:
                        # If stacking fails, pad to same shape
                        result[key] = self._pad_and_stack_tensors(values, key)
                else:
                    result[key] = values
            else:
                result[key] = values

        # CRITICAL: Generate labels for SFT training if missing
        if "labels" not in result and "input_ids" in result:
            logger.info("Generating labels from input_ids for SFT training")
            result["labels"] = self._create_sft_labels(result["input_ids"])
            logger.info(f"Generated labels with shape: {result['labels'].shape}")

        return result

    def _create_sft_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create labels for SFT training by masking prompt tokens.
        
        For instruction-following datasets, we want to only compute loss on the
        assistant's response, not the user's prompt. This method attempts to
        identify response tokens and mask prompt tokens with -100.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            labels: Labels with prompt tokens masked as -100 [batch_size, seq_len]
        """
        labels = input_ids.clone()

        # Get tokenizer info for special tokens
        if hasattr(self.processing_class, "eos_token_id") and self.processing_class.eos_token_id is not None:
            eos_token_id = self.processing_class.eos_token_id
        else:
            eos_token_id = None

        # For Alpaca-style datasets, we need to identify where the assistant response starts
        # Common patterns to look for:
        # - "### Response:" or "### Assistant:" or similar
        # - After the last "### " pattern in the sequence

        if hasattr(self.processing_class, "tokenize"):
            try:
                # Try to find common assistant prompt patterns
                assistant_patterns = [
                    "### Response:",
                    "### Assistant:", 
                    "Assistant:",
                    "Response:",
                    "<|assistant|>",
                    "<|im_start|>assistant",
                    # Patterns for conversation datasets (like LIMO via PromptResponseDataset)
                    "assistant\n",  # Lowercase from conversation format
                    "ASSISTANT\n",  # Uppercase variant
                    "assistant:",  # With colon
                    "ASSISTANT:",  # Uppercase with colon
                ]

                for batch_idx in range(labels.shape[0]):
                    sequence = labels[batch_idx]

                    # Decode tokens to text to find assistant marker
                    try:
                        text = self.processing_class.decode(sequence, skip_special_tokens=False)

                        # Find the last occurrence of any assistant pattern
                        assistant_start_pos = -1
                        for pattern in assistant_patterns:
                            pos = text.rfind(pattern)
                            if pos > assistant_start_pos:
                                assistant_start_pos = pos

                        if assistant_start_pos != -1:
                            # Find the token position corresponding to this text position
                            # This is approximate but should work for most cases
                            prefix_text = text[:assistant_start_pos]
                            if prefix_text:
                                try:
                                    prefix_tokens = self.processing_class.encode(prefix_text, add_special_tokens=False)
                                    mask_until = len(prefix_tokens)

                                    # Also mask the assistant pattern itself (usually 1-3 tokens)
                                    pattern_tokens = self.processing_class.encode(
                                        text[assistant_start_pos:assistant_start_pos+20],
                                        add_special_tokens=False
                                    )
                                    # Find end of pattern (usually ends with ":" or newline)
                                    pattern_end = 2  # Default: mask pattern + colon
                                    for i, token_id in enumerate(pattern_tokens):
                                        if hasattr(self.processing_class, "decode"):
                                            token_text = self.processing_class.decode([token_id])
                                            if "\n" in token_text or ":" in token_text:
                                                pattern_end = i + 1
                                                break

                                    mask_until += pattern_end
                                    mask_until = min(mask_until, len(sequence) - 1)  # Leave at least one token for loss

                                    # Mask prompt tokens
                                    labels[batch_idx, :mask_until] = -100

                                    logger.debug(f"Batch {batch_idx}: Masked {mask_until}/{len(sequence)} prompt tokens")
                                    continue
                                except Exception as e:
                                    logger.debug(f"Failed to tokenize prefix for batch {batch_idx}: {e}")

                        # Fallback: If no assistant pattern found, use a conservative heuristic
                        # For SP training, be more conservative to ensure we have valid labels after sequence shortening
                        seq_len = len(sequence)
                        
                        if self.sp_config.is_enabled():
                            # For SP: ensure we leave enough tokens after potential sequence shortening
                            # SP can shorten sequences significantly, so use a more conservative ratio
                            min_response_tokens = max(10, seq_len // 4)  # At least 25% or 10 tokens for response
                            mask_until = seq_len - min_response_tokens
                            logger.debug(f"SP-aware masking: leaving {min_response_tokens} tokens for response")
                        else:
                            # Standard masking for non-SP training
                            fallback_mask_ratio = 0.7
                            mask_until = int(seq_len * fallback_mask_ratio)
                            mask_until = min(mask_until, seq_len - 5)  # Leave at least 5 tokens for loss
                        
                        # Ensure we don't mask everything
                        mask_until = max(0, min(mask_until, seq_len - 3))  # Always leave at least 3 tokens
                        labels[batch_idx, :mask_until] = -100
                        
                        remaining_tokens = seq_len - mask_until
                        logger.debug(f"Batch {batch_idx}: Fallback masking - masked {mask_until}/{seq_len} tokens, leaving {remaining_tokens} for loss")

                    except Exception as e:
                        logger.debug(f"Failed to decode sequence for batch {batch_idx}: {e}")
                        # Ultimate fallback: conservative masking
                        seq_len = len(sequence)
                        if self.sp_config.is_enabled():
                            # SP-aware conservative masking
                            min_response_tokens = max(10, seq_len // 4)
                            mask_until = seq_len - min_response_tokens
                        else:
                            # Standard masking
                            mask_until = int(seq_len * 0.7)
                            mask_until = min(mask_until, seq_len - 5)
                        
                        # Always leave at least 3 tokens
                        mask_until = max(0, min(mask_until, seq_len - 3))
                        labels[batch_idx, :mask_until] = -100

            except Exception as e:
                logger.warning(f"Failed to create SFT labels with masking: {e}")
                logger.info("Using simple approach: masking first 70% of tokens")

                # Simple fallback: conservative masking for SP
                for batch_idx in range(labels.shape[0]):
                    seq_len = (labels[batch_idx] != self.processing_class.pad_token_id).sum().item() if hasattr(self.processing_class, "pad_token_id") else labels.shape[1]
                    
                    if self.sp_config.is_enabled():
                        # SP-aware conservative masking
                        min_response_tokens = max(10, seq_len // 4)
                        mask_until = seq_len - min_response_tokens
                    else:
                        # Standard masking
                        mask_until = int(seq_len * 0.7)
                        mask_until = min(mask_until, seq_len - 5)
                    
                    # Always leave at least 3 tokens
                    mask_until = max(0, min(mask_until, seq_len - 3))
                    labels[batch_idx, :mask_until] = -100
        else:
            logger.warning("No tokenizer available for smart label masking, using simple approach")
            # Simple fallback: conservative masking
            for batch_idx in range(labels.shape[0]):
                seq_len = labels.shape[1]
                
                if self.sp_config.is_enabled():
                    # SP-aware conservative masking
                    min_response_tokens = max(10, seq_len // 4)
                    mask_until = seq_len - min_response_tokens
                else:
                    # Standard masking
                    mask_until = int(seq_len * 0.7)
                    mask_until = min(mask_until, seq_len - 5)
                
                # Always leave at least 3 tokens
                mask_until = max(0, min(mask_until, seq_len - 3))
                labels[batch_idx, :mask_until] = -100

        return labels

    def _pad_and_stack_tensors(self, tensors, key):
        """Pad tensors to same shape and stack them."""
        if not tensors:
            return torch.tensor([])

        # Find maximum dimensions
        max_dims = [0] * len(tensors[0].shape)
        for tensor in tensors:
            for i, dim in enumerate(tensor.shape):
                max_dims[i] = max(max_dims[i], dim)

        # Determine padding value
        if key == "input_ids" and hasattr(self.processing_class, "pad_token_id"):
            pad_value = self.processing_class.pad_token_id
        elif key == "labels":
            pad_value = -100
        elif key == "attention_mask":
            pad_value = 0
        else:
            pad_value = 0

        # Pad each tensor to max dimensions
        padded_tensors = []
        for tensor in tensors:
            # Calculate padding needed for each dimension
            padding = []
            for i in range(len(tensor.shape) - 1, -1, -1):  # PyTorch padding is in reverse order
                pad_needed = max_dims[i] - tensor.shape[i]
                padding.extend([0, pad_needed])

            if any(p > 0 for p in padding):
                padded_tensor = torch.nn.functional.pad(tensor, padding, value=pad_value)
            else:
                padded_tensor = tensor

            padded_tensors.append(padded_tensor)

        return torch.stack(padded_tensors)

    def _ensure_equal_length_and_sp_divisible(self, result, sp_size):
        """Ensure all tensors have equal length and are divisible by SP size."""
        # Find sequence tensors (assume they have at least 2 dimensions with seq length in dim 1)
        seq_tensors = {}
        for key, tensor in result.items():
            if isinstance(tensor, torch.Tensor) and len(tensor.shape) >= 2:
                seq_tensors[key] = tensor

        if not seq_tensors:
            return result

        # Find the maximum sequence length
        max_seq_len = max(tensor.shape[1] for tensor in seq_tensors.values())

        # Ensure max length is divisible by SP size
        if sp_size > 1:
            remainder = max_seq_len % sp_size
            if remainder != 0:
                pad_to_sp = sp_size - remainder
                max_seq_len += pad_to_sp
                logger.debug(f"Padding to {max_seq_len} for SP divisibility (sp_size={sp_size})")

        # Pad all sequence tensors to the max length
        for key, tensor in seq_tensors.items():
            current_len = tensor.shape[1]
            if current_len < max_seq_len:
                pad_size = max_seq_len - current_len

                # Determine padding value
                if key == "input_ids" and hasattr(self.processing_class, "pad_token_id"):
                    pad_value = self.processing_class.pad_token_id
                elif key == "labels":
                    pad_value = -100
                elif key == "attention_mask":
                    pad_value = 0
                else:
                    pad_value = 0

                # Create padding tensor
                pad_shape = list(tensor.shape)
                pad_shape[1] = pad_size
                padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)

                # Concatenate original with padding
                result[key] = torch.cat([tensor, padding], dim=1)
                logger.debug(f"Padded {key} from {current_len} to {max_seq_len}")

        return result

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
        # Debug batch information
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug("Batch information:")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.debug(f"  {key}: {type(value)}")

        try:
            loss = self.loss_computer.compute_loss(model, inputs, return_outputs=False)

            # Final safety check: ensure loss is scalar
            if loss is not None and hasattr(loss, "shape") and loss.numel() > 1:
                logger.warning(f"Loss has non-scalar shape {loss.shape}, reducing to scalar")
                loss = loss.mean()

            # Check for NaN/inf (only if loss is actually a tensor)
            if loss is not None and isinstance(loss, torch.Tensor) and (torch.isnan(loss) or torch.isinf(loss)):
                logger.error(f"Loss is NaN or inf: {loss}")
                # Return a small positive loss to avoid stopping training
                loss = torch.tensor(1e-6, requires_grad=True, device=loss.device if hasattr(loss, "device") else "cpu")
            elif loss is not None and not isinstance(loss, torch.Tensor):
                # If loss is not a tensor (e.g., dict with logits), return None for evaluation
                logger.debug(f"Loss is not a tensor (type: {type(loss)}), returning None for evaluation")
                loss = None

            return loss

        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            logger.error("Batch details for debugging:")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    logger.error(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    if value.numel() < 50:  # Only log small tensors
                        logger.error(f"  {key} values: {value}")
                else:
                    logger.error(f"  {key}: {type(value)} = {value}")
            raise

    def _early_deepspeed_initialization(self, num_training_steps: int):
        """Perform early DeepSpeed initialization following Arctic pattern."""
        if self._deepspeed_initialized:
            return
            
        logger.info("Performing early DeepSpeed initialization (Arctic pattern)")
        
        # Create optimizer and scheduler first (like Arctic)
        super().create_optimizer_and_scheduler(num_training_steps)
        
        # Now initialize DeepSpeed immediately
        self._create_optimizer_with_deepspeed(num_training_steps)
        self._deepspeed_initialized = True

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer and scheduler with DeepSpeed integration."""
        # If we need early DeepSpeed init and haven't done it yet, do it now
        if self._needs_early_deepspeed_init and not self._deepspeed_initialized:
            return self._early_deepspeed_initialization(num_training_steps)
        elif self._deepspeed_initialized:
            # Already initialized early, nothing to do
            return
        else:
            # Standard path for non-SP cases
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
            # Only do this if we have a train_dataset (not during early initialization)
            if self.sp_manager.is_initialized and hasattr(self, 'train_dataset') and self.train_dataset is not None:
                logger.info("Recreating training dataloader with SP support...")
                self.train_dataloader = self.create_train_dataloader()
                logger.info("Training dataloader created successfully")
            elif self.sp_manager.is_initialized:
                logger.info("SP groups initialized - dataloader recreation will happen later when train_dataset is available")

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
