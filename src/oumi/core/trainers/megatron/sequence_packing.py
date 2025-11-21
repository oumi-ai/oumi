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

"""Sequence packing utilities for efficient batch processing in GRPO training."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PackedBatch:
    """Container for a packed batch with metadata.

    Attributes:
        input_ids: Packed input token IDs, shape [1, bin_size]
        attention_mask: Attention mask for packed sequences, shape [1, bin_size]
        labels: Packed labels (if applicable), shape [1, bin_size]
        seq_starts: Start position of each sequence in the bin
        seq_lengths: Original length of each sequence
        num_sequences: Number of sequences packed in this batch
        original_indices: Original indices of sequences in unpacked batch
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    seq_starts: list[int]
    seq_lengths: list[int]
    num_sequences: int
    original_indices: list[int]


def pack_sequences(
    batch: dict,
    max_bin_size: int,
    pad_token_id: int = 0,
) -> list[PackedBatch]:
    """Pack variable-length sequences into fixed-size bins using first-fit algorithm.

    This implements efficient sequence packing for RL training where sequences
    have variable lengths. Multiple short sequences are concatenated into a single
    bin to maximize GPU utilization.

    Args:
        batch: Dictionary containing 'input_ids', 'attention_mask', and optionally 'labels'
        max_bin_size: Maximum size of each packed bin (typically max_seq_length)
        pad_token_id: Token ID used for padding (default: 0)

    Returns:
        List of PackedBatch objects, one per bin

    Example:
        >>> batch = {
        ...     "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]]),
        ...     "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
        ... }
        >>> packed_batches = pack_sequences(batch, max_bin_size=10)
        >>> len(packed_batches)  # May be 1 if all sequences fit
        1
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch.get("labels", None)

    batch_size = input_ids.shape[0]
    device = input_ids.device

    # Compute actual sequence lengths (excluding padding)
    seq_lengths = attention_mask.sum(dim=1).cpu().tolist()

    # Sort sequences by length (descending) for better packing efficiency
    sorted_indices = sorted(range(batch_size), key=lambda i: seq_lengths[i], reverse=True)

    # First-fit bin packing algorithm
    bins: list[PackedBatch] = []

    for idx in sorted_indices:
        seq_len = seq_lengths[idx]

        if seq_len == 0:
            continue  # Skip empty sequences

        if seq_len > max_bin_size:
            raise ValueError(
                f"Sequence {idx} has length {seq_len} which exceeds max_bin_size {max_bin_size}. "
                "Consider increasing max_seq_length or filtering longer sequences."
            )

        # Try to fit sequence into existing bin
        placed = False
        for bin_batch in bins:
            current_bin_size = sum(bin_batch.seq_lengths)
            if current_bin_size + seq_len <= max_bin_size:
                # Sequence fits in this bin
                _append_to_bin(bin_batch, input_ids[idx], attention_mask[idx],
                              labels[idx] if labels is not None else None,
                              seq_len, idx, device)
                placed = True
                break

        # If doesn't fit in any bin, create new bin
        if not placed:
            new_bin = _create_new_bin(
                input_ids[idx], attention_mask[idx],
                labels[idx] if labels is not None else None,
                seq_len, idx, max_bin_size, pad_token_id, device
            )
            bins.append(new_bin)

    return bins


def _append_to_bin(
    bin_batch: PackedBatch,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor],
    seq_len: int,
    original_idx: int,
    device: torch.device,
) -> None:
    """Append a sequence to an existing bin (in-place modification)."""
    current_size = sum(bin_batch.seq_lengths)

    # Append input_ids
    bin_batch.input_ids[0, current_size:current_size + seq_len] = input_ids[:seq_len]

    # Append attention_mask
    bin_batch.attention_mask[0, current_size:current_size + seq_len] = attention_mask[:seq_len]

    # Append labels if present
    if labels is not None and bin_batch.labels is not None:
        bin_batch.labels[0, current_size:current_size + seq_len] = labels[:seq_len]

    # Update metadata
    bin_batch.seq_starts.append(current_size)
    bin_batch.seq_lengths.append(seq_len)
    bin_batch.num_sequences += 1
    bin_batch.original_indices.append(original_idx)


def _create_new_bin(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor],
    seq_len: int,
    original_idx: int,
    max_bin_size: int,
    pad_token_id: int,
    device: torch.device,
) -> PackedBatch:
    """Create a new bin with the first sequence."""
    # Initialize empty bin
    packed_input_ids = torch.full((1, max_bin_size), pad_token_id, dtype=torch.long, device=device)
    packed_attention_mask = torch.zeros((1, max_bin_size), dtype=torch.long, device=device)
    packed_labels = None

    if labels is not None:
        packed_labels = torch.full((1, max_bin_size), -100, dtype=torch.long, device=device)
        packed_labels[0, :seq_len] = labels[:seq_len]

    # Place first sequence
    packed_input_ids[0, :seq_len] = input_ids[:seq_len]
    packed_attention_mask[0, :seq_len] = attention_mask[:seq_len]

    return PackedBatch(
        input_ids=packed_input_ids,
        attention_mask=packed_attention_mask,
        labels=packed_labels,
        seq_starts=[0],
        seq_lengths=[seq_len],
        num_sequences=1,
        original_indices=[original_idx],
    )


def create_causal_mask_for_packed_sequences(
    seq_starts: list[int],
    seq_lengths: list[int],
    bin_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create causal attention mask for packed sequences.

    Each sequence should only attend to tokens within its own sequence boundary.
    This prevents cross-contamination between different sequences in the same bin.

    Args:
        seq_starts: Start position of each sequence
        seq_lengths: Length of each sequence
        bin_size: Total bin size
        device: Device for tensor creation
        dtype: Data type for mask (default: float32)

    Returns:
        Attention mask of shape [1, 1, bin_size, bin_size]
        Values are 0.0 for allowed attention, -inf for masked positions
    """
    # Initialize mask (all masked)
    mask = torch.full((bin_size, bin_size), float('-inf'), dtype=dtype, device=device)

    # Fill in causal attention for each sequence
    for start, length in zip(seq_starts, seq_lengths):
        end = start + length
        # Create causal mask for this sequence
        for i in range(start, end):
            # Token at position i can attend to all previous tokens in its sequence
            mask[i, start:i+1] = 0.0

    # Add batch and head dimensions: [1, 1, bin_size, bin_size]
    return mask.unsqueeze(0).unsqueeze(0)


def unpack_logprobs(
    packed_logprobs: torch.Tensor,
    seq_starts: list[int],
    seq_lengths: list[int],
    original_indices: list[int],
    batch_size: int,
    max_seq_length: int,
) -> torch.Tensor:
    """Unpack log probabilities from packed format back to original batch format.

    Args:
        packed_logprobs: Packed logprobs, shape [1, bin_size]
        seq_starts: Start position of each sequence in the bin
        seq_lengths: Original length of each sequence
        original_indices: Original indices in the unpacked batch
        batch_size: Size of the original unpacked batch
        max_seq_length: Maximum sequence length in original batch

    Returns:
        Unpacked logprobs, shape [batch_size, max_seq_length]
    """
    device = packed_logprobs.device
    unpacked = torch.full(
        (batch_size, max_seq_length),
        -100.0,  # Use -100 as padding value
        dtype=packed_logprobs.dtype,
        device=device
    )

    for start, length, orig_idx in zip(seq_starts, seq_lengths, original_indices):
        # Extract sequence logprobs from packed tensor
        # Note: logprobs are 1 token shorter than sequences (no logprob for first token)
        logprob_length = length - 1
        if logprob_length > 0:
            unpacked[orig_idx, :logprob_length] = packed_logprobs[0, start:start + logprob_length]

    return unpacked


def unpack_rewards(
    packed_rewards: torch.Tensor,
    original_indices: list[int],
    batch_size: int,
) -> torch.Tensor:
    """Unpack rewards from packed format back to original batch format.

    Args:
        packed_rewards: Packed rewards, shape [num_sequences_in_bin]
        original_indices: Original indices in the unpacked batch
        batch_size: Size of the original unpacked batch

    Returns:
        Unpacked rewards, shape [batch_size]
    """
    device = packed_rewards.device
    unpacked = torch.zeros(batch_size, dtype=packed_rewards.dtype, device=device)

    for i, orig_idx in enumerate(original_indices):
        unpacked[orig_idx] = packed_rewards[i]

    return unpacked


def compute_packing_efficiency(packed_batches: list[PackedBatch], max_bin_size: int) -> dict:
    """Compute efficiency metrics for sequence packing.

    Args:
        packed_batches: List of packed batches
        max_bin_size: Maximum bin size

    Returns:
        Dictionary with efficiency metrics
    """
    if not packed_batches:
        return {
            "num_bins": 0,
            "total_sequences": 0,
            "total_tokens": 0,
            "total_capacity": 0,
            "utilization": 0.0,
            "avg_sequences_per_bin": 0.0,
        }

    total_sequences = sum(batch.num_sequences for batch in packed_batches)
    total_tokens = sum(sum(batch.seq_lengths) for batch in packed_batches)
    total_capacity = len(packed_batches) * max_bin_size

    return {
        "num_bins": len(packed_batches),
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "total_capacity": total_capacity,
        "utilization": total_tokens / total_capacity if total_capacity > 0 else 0.0,
        "avg_sequences_per_bin": total_sequences / len(packed_batches),
    }
