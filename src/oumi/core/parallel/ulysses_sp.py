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

"""Ulysses Sequence Parallelism implementation for long sequence training.

This module provides sequence parallelism capabilities by sharding sequences
across multiple GPUs, enabling training on extremely long sequences (500K+ tokens).

Based on the implementation from Snowflake's ArcticTraining repository:
https://github.com/snowflakedb/ArcticTraining/pull/45
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Union
from transformers import PreTrainedModel
import importlib.util

from oumi.utils.logging import logger


class UlyssesSPDataLoaderAdapter:
    """Data loader adapter to shard batches for Ulysses sequence parallelism.
    
    Shards the normal data loader batches to be used by UlyssesSPAttentionHF.
    Each GPU receives a shard of the sequence dimension from the batch.
    """
    
    def __init__(
        self, 
        dataloader: DataLoader,
        sequence_parallel_size: int,
        rank: Optional[int] = None
    ):
        """Initialize the Ulysses SP data loader adapter.
        
        Args:
            dataloader: Original data loader to wrap
            sequence_parallel_size: Number of GPUs to shard sequences across
            rank: Current rank in the sequence parallel group (auto-detected if None)
        """
        self.dataloader = dataloader
        self.sequence_parallel_size = sequence_parallel_size
        self.rank = rank if rank is not None else dist.get_rank()
        
        if not dist.is_initialized():
            raise RuntimeError("Distributed training must be initialized for Ulysses SP")
            
        if sequence_parallel_size > dist.get_world_size():
            raise ValueError(
                f"Sequence parallel size ({sequence_parallel_size}) cannot exceed "
                f"world size ({dist.get_world_size()})"
            )
    
    def __iter__(self):
        """Iterate over sharded batches."""
        for batch in self.dataloader:
            yield self._shard_batch(batch)
    
    def __len__(self):
        """Return length of original dataloader."""
        return len(self.dataloader)
    
    def _shard_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shard batch along sequence dimension.
        
        Args:
            batch: Input batch with tensors to shard
            
        Returns:
            Sharded batch where sequence dimension is divided across GPUs
        """
        sharded_batch = {}
        
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and len(tensor.shape) >= 2:
                # Assume sequence dimension is dimension 1 (after batch dimension)
                seq_len = tensor.shape[1]
                
                # Calculate shard size and ensure even division
                shard_size = seq_len // self.sequence_parallel_size
                if seq_len % self.sequence_parallel_size != 0:
                    logger.warning(
                        f"Sequence length {seq_len} not evenly divisible by "
                        f"sequence_parallel_size {self.sequence_parallel_size}. "
                        f"Truncating to {shard_size * self.sequence_parallel_size}"
                    )
                
                # Extract shard for current rank
                start_idx = self.rank * shard_size
                end_idx = start_idx + shard_size
                
                sharded_batch[key] = tensor[:, start_idx:end_idx, ...]
            else:
                # Non-tensor or 1D tensors pass through unchanged
                sharded_batch[key] = tensor
                
        return sharded_batch


class UlyssesSPAttentionHF:
    """Ulysses Sequence Parallelism attention implementation for HuggingFace models.
    
    Port of UlyssesAttention from Megatron-DeepSpeed with modern MHA variations.
    Enables sequence parallelism across distributed training by tiling computation
    across sequence dimensions.
    """
    
    def __init__(self, sequence_parallel_size: int):
        """Initialize Ulysses SP attention.
        
        Args:
            sequence_parallel_size: Number of GPUs to distribute sequence across
        """
        self.sequence_parallel_size = sequence_parallel_size
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if not dist.is_initialized():
            raise RuntimeError("Distributed training must be initialized for Ulysses SP")
    
    @staticmethod
    def register_with_transformers(model: PreTrainedModel, sequence_parallel_size: int):
        """Register Ulysses SP attention with a transformers model.
        
        This method patches the attention layers in the model to use sequence
        parallelism when forward passes are computed.
        
        Args:
            model: HuggingFace model to patch
            sequence_parallel_size: Size of sequence parallel group
        """
        if not dist.is_initialized():
            logger.warning("Distributed training not initialized, skipping Ulysses SP registration")
            return
            
        if sequence_parallel_size <= 1:
            logger.info("Sequence parallel size <= 1, skipping Ulysses SP registration")
            return
            
        # Get model type to determine attention layer names
        model_type = model.config.model_type.lower()
        
        # Common attention layer names for different model architectures
        attention_layer_names = {
            'llama': ['self_attn'],
            'qwen2': ['self_attn'],
            'deepseek': ['self_attn'],
            'phi': ['self_attn'],
            'mistral': ['self_attn'],
        }
        
        layer_names = attention_layer_names.get(model_type, ['self_attn'])
        
        # Patch attention layers
        patched_count = 0
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                if hasattr(module, 'forward'):
                    # Store original forward method
                    original_forward = module.forward
                    
                    # Create patched forward method
                    def create_patched_forward(orig_forward, sp_size):
                        def patched_forward(self, *args, **kwargs):
                            return _ulysses_sp_attention_forward(
                                orig_forward, self, sp_size, *args, **kwargs
                            )
                        return patched_forward
                    
                    # Apply patch
                    module.forward = create_patched_forward(original_forward, sequence_parallel_size).__get__(
                        module, module.__class__
                    )
                    patched_count += 1
        
        logger.info(f"Patched {patched_count} attention layers for Ulysses SP")


def _ulysses_sp_attention_forward(
    original_forward,
    attention_module,
    sequence_parallel_size: int,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.Tensor] = None,
    **kwargs
):
    """Ulysses sequence parallel attention forward pass.
    
    Implements sequence parallelism by:
    1. All-gathering sequence shards from all GPUs
    2. Computing attention on full sequences
    3. Scattering results back to individual GPUs
    
    Args:
        original_forward: Original attention forward method
        attention_module: The attention module being patched
        sequence_parallel_size: Size of sequence parallel group
        hidden_states: Input hidden states (sharded)
        attention_mask: Attention mask (if provided)
        position_ids: Position IDs (if provided)
        past_key_value: Past key-value cache
        output_attentions: Whether to output attention weights
        use_cache: Whether to use caching
        cache_position: Cache position tensor
        **kwargs: Additional keyword arguments
        
    Returns:
        Attention output, optionally with attention weights and past key-value
    """
    if sequence_parallel_size <= 1 or not dist.is_initialized():
        # No sequence parallelism, use original forward
        return original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
    
    batch_size, seq_len, hidden_size = hidden_states.shape
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # All-gather sequence shards
    gathered_hidden_states = torch.zeros(
        batch_size, seq_len * sequence_parallel_size, hidden_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # Gather hidden states from all sequence parallel ranks
    dist.all_gather_into_tensor(
        gathered_hidden_states.view(-1),
        hidden_states.view(-1)
    )
    
    # Adjust attention mask and position IDs if provided
    gathered_attention_mask = None
    gathered_position_ids = None
    
    if attention_mask is not None:
        gathered_attention_mask = torch.zeros(
            batch_size, seq_len * sequence_parallel_size,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        dist.all_gather_into_tensor(
            gathered_attention_mask.view(-1),
            attention_mask.view(-1)
        )
    
    if position_ids is not None:
        gathered_position_ids = torch.zeros(
            batch_size, seq_len * sequence_parallel_size,
            dtype=position_ids.dtype,
            device=position_ids.device
        )
        dist.all_gather_into_tensor(
            gathered_position_ids.view(-1),
            position_ids.view(-1)
        )
    
    # Compute attention on full sequence
    attention_output = original_forward(
        hidden_states=gathered_hidden_states,
        attention_mask=gathered_attention_mask,
        position_ids=gathered_position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs
    )
    
    # Extract outputs
    if isinstance(attention_output, tuple):
        full_hidden_states = attention_output[0]
        additional_outputs = attention_output[1:]
    else:
        full_hidden_states = attention_output
        additional_outputs = ()
    
    # Scatter results back to individual ranks
    start_idx = rank * seq_len
    end_idx = start_idx + seq_len
    
    scattered_hidden_states = full_hidden_states[:, start_idx:end_idx, :]
    
    # Return in same format as original
    if additional_outputs:
        return (scattered_hidden_states,) + additional_outputs
    else:
        return scattered_hidden_states


def is_ulysses_sp_available() -> bool:
    """Check if Ulysses sequence parallelism is available.
    
    Returns:
        True if distributed training is available, False otherwise
    """
    return dist.is_available() and importlib.util.find_spec("torch.distributed") is not None


def setup_ulysses_sp(
    model: PreTrainedModel,
    dataloader: DataLoader,
    sequence_parallel_size: int
) -> tuple[PreTrainedModel, DataLoader]:
    """Setup Ulysses sequence parallelism for model and dataloader.
    
    Args:
        model: HuggingFace model to enable sequence parallelism for
        dataloader: DataLoader to shard sequences
        sequence_parallel_size: Number of GPUs to distribute sequences across
        
    Returns:
        Tuple of (patched_model, adapted_dataloader)
    """
    if not is_ulysses_sp_available():
        logger.warning("Ulysses SP not available, returning original model and dataloader")
        return model, dataloader
    
    if sequence_parallel_size <= 1:
        logger.info("Sequence parallel size <= 1, no setup needed")
        return model, dataloader
    
    # Register attention patches
    UlyssesSPAttentionHF.register_with_transformers(model, sequence_parallel_size)
    
    # Adapt dataloader
    adapted_dataloader = UlyssesSPDataLoaderAdapter(
        dataloader=dataloader,
        sequence_parallel_size=sequence_parallel_size
    )
    
    logger.info(f"Ulysses SP setup complete with sequence_parallel_size={sequence_parallel_size}")
    
    return model, adapted_dataloader