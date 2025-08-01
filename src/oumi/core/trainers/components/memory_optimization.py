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

"""Memory optimization components for trainers."""

import math
from typing import Any, Dict

import torch

from oumi.utils.logging import logger

try:
    from deepspeed.runtime.sequence_parallel.ulysses_sp import (
        TiledFusedLogitsLoss,
    )

    DEEPSPEED_ULYSSES_AVAILABLE = True
except ImportError:
    logger.warning("DeepSpeed Ulysses SP not available for tiled loss computation")
    DEEPSPEED_ULYSSES_AVAILABLE = False

try:
    from oumi.core.parallel.tiled_compute import enable_tiled_mlp_compute

    TILED_MLP_AVAILABLE = True
except ImportError:
    logger.warning("Tiled MLP compute not available")
    TILED_MLP_AVAILABLE = False


class MemoryOptimizer:
    """Memory optimization utilities for training."""

    @staticmethod
    def setup_tiled_mlp_compute(model_name_or_path: str) -> bool:
        """Setup tiled MLP computation for memory efficiency.

        Args:
            model_name_or_path: Model name or path for configuration

        Returns:
            True if successfully enabled, False otherwise
        """
        if not TILED_MLP_AVAILABLE:
            logger.warning("Tiled MLP compute not available, skipping setup")
            return False

        if not model_name_or_path:
            logger.warning(
                "Model name/path not provided, cannot setup tiled MLP compute"
            )
            return False

        logger.info("Setting up tiled MLP compute for memory efficiency")
        try:
            enable_tiled_mlp_compute(model_name_or_path)
            logger.info("Tiled MLP compute enabled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to setup tiled MLP compute: {e}")
            return False

    @staticmethod
    def compute_tiled_logits_loss(
        model: torch.nn.Module,
        inputs: dict[str, Any],
        shift_labels: torch.Tensor,
        slice_size_gb: float = 1.0,
    ) -> torch.Tensor:
        """Compute loss using tiled logits computation for memory efficiency.

        Args:
            model: The model
            inputs: Input batch
            shift_labels: Pre-shifted labels for loss computation
            slice_size_gb: Target memory size per shard in GB

        Returns:
            Computed loss tensor
        """
        if not DEEPSPEED_ULYSSES_AVAILABLE:
            raise RuntimeError("DeepSpeed Ulysses SP required for tiled logits loss")

        logger.info("Starting tiled logits loss computation...")

        # Automatically determine number of shards based on memory target
        bs, seqlen = shift_labels.shape
        vocab_size = model.config.vocab_size
        logits_numel = bs * seqlen * vocab_size
        size_in_gb = logits_numel * 4 / (2**30)  # fp32
        num_shards = max(1, math.ceil(size_in_gb / slice_size_gb))

        logger.info(f"Using {num_shards} shards for tiled logits computation")
        logger.info(
            f"Batch size: {bs}, Sequence length: {seqlen}, Vocab size: {vocab_size}"
        )

        # Get model outputs (hidden states)
        logger.info("Getting model outputs (forward pass)...")
        outputs = model.model(**inputs, use_cache=False)
        logger.info("Model forward pass completed")

        hidden_states = outputs.last_hidden_state
        logger.info(f"Hidden states shape: {hidden_states.shape}")

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

    @staticmethod
    def get_memory_usage() -> dict[str, float]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {"current_gb": 0.0, "max_gb": 0.0, "reserved_gb": 0.0}

        current_mb = torch.cuda.memory_allocated() / (1024**2)
        max_mb = torch.cuda.max_memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.memory_reserved() / (1024**2)

        return {
            "current_gb": current_mb / 1024,
            "max_gb": max_mb / 1024,
            "reserved_gb": reserved_mb / 1024,
        }

    @staticmethod
    def clear_memory():
        """Clear GPU memory caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class LigerKernelOptimizer:
    """Liger kernel optimization utilities."""

    @staticmethod
    def is_available() -> bool:
        """Check if Liger kernels are available."""
        try:
            import liger_kernel  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def apply_liger_kernels(model: torch.nn.Module, use_liger_kernel: bool = True):
        """Apply Liger kernel optimizations to model.

        Args:
            model: Model to optimize
            use_liger_kernel: Whether to apply optimizations
        """
        if not use_liger_kernel or not LigerKernelOptimizer.is_available():
            return

        try:
            from liger_kernel.transformers import apply_liger_kernel_to_llama

            apply_liger_kernel_to_llama()
            logger.info("Liger kernel optimizations applied successfully")
        except Exception as e:
            logger.warning(f"Failed to apply Liger kernel optimizations: {e}")
