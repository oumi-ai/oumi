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

"""GRPO loss functions for Megatron-based training.

Adapted from Megatron-LM: https://github.com/NVIDIA/Megatron-LM
"""

from typing import Optional

import torch


def calculate_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clamp_eps_lower: float = 0.2,
    clamp_eps_upper: float = 0.2,
    kl_beta: float = 0.001,
    entropy_weight: float = 0.0,
    inference_logprobs: Optional[torch.Tensor] = None,
    is_truncation_coef: Optional[float] = None,
    seq_starts: Optional[list] = None,
    seq_lengths: Optional[list] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Calculate GRPO loss with KL divergence penalty and entropy regularization.

    This implements the Group Relative Policy Optimization (GRPO) algorithm from
    "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
    (https://arxiv.org/pdf/2402.03300).

    Args:
        current_logprobs: π logprobs from current policy, shape [batch, seq] for
            unpacked or [1, bin_size] for packed sequences.
        old_logprobs: π_old logprobs from frozen policy, same shape as current_logprobs.
        ref_logprobs: π_ref logprobs from reference model, same shape as current_logprobs.
        advantages: Advantage values, shape [batch,] for unpacked or
            [num_sequences_in_bin,] for packed.
        clamp_eps_lower: Lower bound for ratio clipping (default: 0.2).
        clamp_eps_upper: Upper bound for ratio clipping. For vanilla GRPO, this should
            equal clamp_eps_lower (default: 0.2).
        kl_beta: Weight for KL penalty term measuring distance between π and π_ref
            (default: 0.001).
        entropy_weight: Weight for entropy regularization term (default: 0.0).
        inference_logprobs: Optional π_old logprobs from inference engine. If provided,
            importance sampling correction will be applied.
        is_truncation_coef: Importance sampling truncation coefficient. Applied only
            if inference_logprobs is provided.
        seq_starts: For packed sequences: start positions of each sequence in the bin.
        seq_lengths: For packed sequences: original lengths of each sequence.

    Returns:
        A tuple containing:
        - loss: Per-token GRPO loss, shape [batch, seq] or [1, bin_size]
        - kl_term: KL divergence term, same shape as loss
        - ratios: π/π_old importance sampling ratios, same shape as loss
        - entropy_term: Entropy regularization term, same shape as loss
        - truncated_from_above: Boolean mask for upper-clipped ratios, same shape as loss
        - truncated_from_below: Boolean mask for lower-clipped ratios, same shape as loss

    Example:
        >>> # Simple unpacked case
        >>> loss, kl, ratios, entropy, trunc_above, trunc_below = calculate_grpo_loss(
        ...     current_logprobs=policy_logprobs,
        ...     old_logprobs=frozen_logprobs,
        ...     ref_logprobs=reference_logprobs,
        ...     advantages=computed_advantages,
        ...     clamp_eps_lower=0.2,
        ...     clamp_eps_upper=0.2,
        ...     kl_beta=0.001,
        ... )

    References:
        DeepSeekMath paper: https://arxiv.org/pdf/2402.03300
        Megatron-LM implementation: https://github.com/NVIDIA/Megatron-LM
    """
    # Validate shape compatibility
    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"Shape mismatch: current_logprobs {current_logprobs.shape} vs "
            f"old_logprobs {old_logprobs.shape}"
        )

    if current_logprobs.shape != ref_logprobs.shape:
        raise ValueError(
            f"Shape mismatch: current_logprobs {current_logprobs.shape} vs "
            f"ref_logprobs {ref_logprobs.shape}"
        )

    # Calculate importance sampling ratios: exp(log π - log π_old)
    ratios = (current_logprobs - old_logprobs).exp()

    # Clip ratios to [1 - ε_lower, 1 + ε_upper] for stability
    clamped_ratios = ratios.clamp(1 - clamp_eps_lower, 1 + clamp_eps_upper)

    # Track which ratios were clipped for logging/analysis
    truncated_from_above = torch.gt(ratios, 1 + clamp_eps_upper)
    truncated_from_below = torch.lt(ratios, 1 - clamp_eps_lower)

    # Handle advantages based on whether sequences are packed or unpacked
    if seq_starts is not None and seq_lengths is not None:
        # Packed sequences: map each sequence's advantage to its tokens
        bin_size = current_logprobs.shape[1]
        packed_advantages = torch.zeros(
            (1, bin_size),
            device=current_logprobs.device,
            dtype=current_logprobs.dtype,
        )

        for seq_idx, (start, seq_len) in enumerate(zip(seq_starts, seq_lengths)):
            # Logprobs are 1 token shorter than sequences (no logprob for first token)
            end = min(start + seq_len - 1, bin_size)
            if end > start:
                packed_advantages[0, start:end] = advantages[seq_idx].item()

        advantages = packed_advantages
    else:
        # Unpacked sequences: broadcast single advantage per sequence
        # Reshape from [batch,] to [batch, 1] to match logprobs shape [batch, seq]
        advantages = advantages.view(-1, 1)

    # Calculate KL divergence penalty: KL(π_ref || π)
    # Using the identity: KL = E[exp(log π_ref - log π) - (log π_ref - log π) - 1]
    ref_diff = ref_logprobs - current_logprobs
    kl_term = ref_diff.exp() - ref_diff - 1

    # Calculate entropy regularization: H = -E[π log π]
    # Entropy encourages exploration by penalizing overconfident predictions
    entropy_term = current_logprobs.exp() * current_logprobs

    # Importance sampling correction (optional)
    is_weights = torch.tensor(1.0, dtype=old_logprobs.dtype, device=old_logprobs.device)

    if inference_logprobs is not None:
        # Correct for mismatch between training and inference distributions
        is_weights = (old_logprobs - inference_logprobs).exp()

        # Optionally truncate IS weights to prevent extreme corrections
        if is_truncation_coef is not None:
            is_weights = torch.min(
                is_weights,
                torch.tensor(
                    is_truncation_coef, dtype=old_logprobs.dtype, device=old_logprobs.device
                ),
            )

    # Final GRPO loss:
    # L = -IS_weight * min(r * A, clip(r) * A) + β * KL + λ * H
    # where:
    #   - First term: Clipped policy gradient objective
    #   - Second term: KL penalty to stay close to reference model
    #   - Third term: Entropy bonus for exploration
    loss = (
        -is_weights * torch.min(ratios * advantages, clamped_ratios * advantages)
        + kl_beta * kl_term
        + entropy_weight * entropy_term
    )

    return loss, kl_term, ratios, entropy_term, truncated_from_above, truncated_from_below


def compute_advantages(
    rewards: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute advantages from rewards using group normalization.

    In GRPO, advantages are computed by comparing rewards within each group
    and normalizing by the group statistics.

    Args:
        rewards: Reward values, shape [batch,] or [num_sequences,]
        normalize: Whether to normalize advantages by standard deviation (default: True)
        eps: Small constant for numerical stability (default: 1e-8)

    Returns:
        Advantages tensor with same shape as rewards

    Example:
        >>> rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> advantages = compute_advantages(rewards, normalize=True)
    """
    # Center advantages by subtracting mean
    advantages = rewards - rewards.mean()

    # Optionally normalize by standard deviation
    if normalize:
        std = rewards.std()
        if std > eps:
            advantages = advantages / (std + eps)

    return advantages


def gather_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Gather log probabilities for the given labels from logits.

    Args:
        logits: Model output logits, shape [batch, seq, vocab_size]
        labels: Target token IDs, shape [batch, seq]
        ignore_index: Label value to ignore when computing log probs (default: -100)

    Returns:
        Log probabilities for each token, shape [batch, seq]

    Example:
        >>> logits = model(input_ids).logits
        >>> logprobs = gather_log_probs(logits, labels)
    """
    # Convert logits to log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Replace ignore_index with 0 for gathering (will be masked out later)
    labels_for_gather = labels.clone()
    labels_for_gather[labels == ignore_index] = 0

    # Gather log probs for target tokens
    # Shape: [batch, seq, vocab_size] -> [batch, seq]
    gathered_log_probs = torch.gather(
        log_probs, dim=-1, index=labels_for_gather.unsqueeze(-1)
    ).squeeze(-1)

    # Mask out ignored tokens
    mask = (labels != ignore_index).float()
    gathered_log_probs = gathered_log_probs * mask

    return gathered_log_probs
