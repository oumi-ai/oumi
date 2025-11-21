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

"""Unit tests for Megatron GRPO loss functions."""

import pytest
import torch

from oumi.core.trainers.megatron.loss_functions import (
    calculate_grpo_loss,
    compute_advantages,
    gather_log_probs,
)


class TestCalculateGRPOLoss:
    """Tests for calculate_grpo_loss function."""

    def test_basic_loss_computation(self):
        """Test basic GRPO loss computation with simple inputs."""
        batch_size, seq_len = 2, 4

        # Create simple logprobs
        current_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        old_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        ref_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        advantages = torch.tensor([1.0, -1.0])

        loss, kl, ratios, entropy, trunc_above, trunc_below = calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
        )

        # Check shapes
        assert loss.shape == (batch_size, seq_len)
        assert kl.shape == (batch_size, seq_len)
        assert ratios.shape == (batch_size, seq_len)
        assert entropy.shape == (batch_size, seq_len)
        assert trunc_above.shape == (batch_size, seq_len)
        assert trunc_below.shape == (batch_size, seq_len)

        # Check that loss is finite
        assert torch.isfinite(loss).all()

    def test_ratio_clipping(self):
        """Test that importance sampling ratios are clipped correctly."""
        batch_size, seq_len = 2, 4

        # Create logprobs where policy has changed significantly
        current_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        old_logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]])  # Much lower
        ref_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        advantages = torch.tensor([1.0, 1.0])

        loss, kl, ratios, entropy, trunc_above, trunc_below = calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clamp_eps_lower=0.2,
            clamp_eps_upper=0.2,
        )

        # Ratios should be large (current >> old)
        assert (ratios > 1.0).any(), "Some ratios should be > 1.0 when current > old"

        # Some ratios should be clipped
        assert trunc_above.any() or trunc_below.any(), "Some ratios should be clipped"

    def test_kl_penalty(self):
        """Test that KL penalty is computed correctly."""
        batch_size, seq_len = 2, 4

        # Create logprobs where current differs from reference
        current_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        old_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        ref_logprobs = torch.tensor([[-0.5, -0.5, -0.5, -0.5], [-0.5, -0.5, -0.5, -0.5]])  # Different
        advantages = torch.tensor([1.0, 1.0])

        loss, kl, ratios, entropy, trunc_above, trunc_below = calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            kl_beta=0.1,  # Non-zero KL coefficient
        )

        # KL term should be non-zero when distributions differ
        assert (kl != 0.0).any(), "KL term should be non-zero when distributions differ"

    def test_entropy_regularization(self):
        """Test that entropy regularization is computed correctly."""
        batch_size, seq_len = 2, 4

        current_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        old_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        ref_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        advantages = torch.tensor([1.0, 1.0])

        loss, kl, ratios, entropy, trunc_above, trunc_below = calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            entropy_weight=0.1,  # Non-zero entropy weight
        )

        # Entropy term should be negative (due to negative sign in H = -Σ p log p)
        # After fixing the sign, entropy_term = -(p * log p), which is positive
        assert (entropy >= 0.0).all(), "Entropy term should be non-negative after sign fix"

    def test_gradient_masking(self):
        """Test that gradient masking excludes padding tokens."""
        batch_size, seq_len = 2, 4

        current_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        old_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        ref_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        advantages = torch.tensor([1.0, 1.0])

        # Create mask: first sequence has 3 valid tokens, second has 2
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])

        loss, kl, ratios, entropy, trunc_above, trunc_below = calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            mask=mask,
        )

        # Masked positions should have zero loss
        assert loss[0, 3] == 0.0, "Masked position should have zero loss"
        assert loss[1, 2] == 0.0, "Masked position should have zero loss"
        assert loss[1, 3] == 0.0, "Masked position should have zero loss"

    def test_importance_sampling_correction(self):
        """Test importance sampling correction when inference logprobs differ."""
        batch_size, seq_len = 2, 4

        current_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        old_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        ref_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4], [-0.15, -0.25, -0.35, -0.45]])
        inference_logprobs = torch.tensor([[-0.2, -0.3, -0.4, -0.5], [-0.25, -0.35, -0.45, -0.55]])
        advantages = torch.tensor([1.0, 1.0])

        loss_without_is, *_ = calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
        )

        loss_with_is, *_ = calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            inference_logprobs=inference_logprobs,
        )

        # Losses should differ when IS correction is applied
        assert not torch.allclose(loss_without_is, loss_with_is), "Losses should differ with IS correction"

    def test_shape_mismatch_raises_error(self):
        """Test that shape mismatches raise appropriate errors."""
        current_logprobs = torch.randn(2, 4)
        old_logprobs = torch.randn(2, 5)  # Wrong shape
        ref_logprobs = torch.randn(2, 4)
        advantages = torch.randn(2)

        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_grpo_loss(current_logprobs, old_logprobs, ref_logprobs, advantages)


class TestComputeAdvantages:
    """Tests for compute_advantages function."""

    def test_basic_advantage_computation(self):
        """Test basic advantage computation with normalization."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        advantages = compute_advantages(rewards, normalize=True)

        # Advantages should be centered (mean ≈ 0)
        assert torch.abs(advantages.mean()) < 1e-6, "Advantages should be centered"

        # Advantages should be normalized (std ≈ 1)
        assert torch.abs(advantages.std() - 1.0) < 1e-6, "Advantages should be normalized"

    def test_advantage_without_normalization(self):
        """Test advantage computation without normalization."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        advantages = compute_advantages(rewards, normalize=False)

        # Advantages should still be centered
        assert torch.abs(advantages.mean()) < 1e-6, "Advantages should be centered"

        # But not normalized
        assert advantages.std() != 1.0, "Advantages should not be normalized"

    def test_advantage_with_constant_rewards(self):
        """Test advantage computation with all identical rewards."""
        rewards = torch.tensor([2.0, 2.0, 2.0, 2.0])

        advantages = compute_advantages(rewards, normalize=True)

        # All advantages should be zero (or very close) when all rewards are equal
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-6)

    def test_advantage_preserves_order(self):
        """Test that advantage computation preserves reward ordering."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

        advantages = compute_advantages(rewards, normalize=False)

        # Higher rewards should have higher advantages
        assert advantages[0] < advantages[1] < advantages[2] < advantages[3]


class TestGatherLogProbs:
    """Tests for gather_log_probs function."""

    def test_basic_logprob_gathering(self):
        """Test basic log probability gathering."""
        batch_size, seq_len, vocab_size = 2, 4, 10

        # Create simple logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        logprobs = gather_log_probs(logits, labels)

        # Check shape
        assert logprobs.shape == (batch_size, seq_len)

        # Check that logprobs are valid (negative or zero)
        assert (logprobs <= 0.0).all(), "Log probabilities should be non-positive"

    def test_ignore_index_masking(self):
        """Test that ignore_index tokens are masked out."""
        batch_size, seq_len, vocab_size = 2, 4, 10
        ignore_index = -100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Set some labels to ignore_index
        labels[0, 2] = ignore_index
        labels[1, 3] = ignore_index

        logprobs = gather_log_probs(logits, labels, ignore_index=ignore_index)

        # Ignored positions should have zero logprob
        assert logprobs[0, 2] == 0.0
        assert logprobs[1, 3] == 0.0

    def test_return_mask(self):
        """Test that mask is returned correctly when requested."""
        batch_size, seq_len, vocab_size = 2, 4, 10
        ignore_index = -100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[0, 2] = ignore_index

        logprobs, mask = gather_log_probs(logits, labels, ignore_index=ignore_index, return_mask=True)

        # Check mask shape
        assert mask.shape == (batch_size, seq_len)

        # Check mask values
        assert mask[0, 2] == 0.0, "Ignored position should have mask=0"
        assert mask[0, 0] == 1.0, "Valid position should have mask=1"

    def test_logprobs_sum_to_valid_values(self):
        """Test that gathered logprobs are numerically valid."""
        batch_size, seq_len, vocab_size = 2, 4, 10

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        logprobs = gather_log_probs(logits, labels)

        # Check that all logprobs are finite
        assert torch.isfinite(logprobs).all(), "All logprobs should be finite"

        # Check that logprobs are in valid range (typically between -20 and 0)
        assert (logprobs >= -50.0).all(), "Logprobs should not be too negative"
        assert (logprobs <= 0.0).all(), "Logprobs should be non-positive"


class TestGRPOLossIntegration:
    """Integration tests combining multiple loss function components."""

    def test_full_grpo_pipeline(self):
        """Test complete GRPO loss computation pipeline."""
        batch_size, seq_len, vocab_size = 2, 8, 100

        # 1. Create mock model outputs
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # 2. Gather logprobs
        current_logprobs, mask = gather_log_probs(logits, labels, return_mask=True)

        # 3. Create old/ref logprobs (detached)
        old_logprobs = current_logprobs.detach()
        ref_logprobs = current_logprobs.detach()

        # 4. Compute rewards and advantages
        rewards = torch.tensor([1.5, -0.5])
        advantages = compute_advantages(rewards, normalize=True)

        # 5. Compute GRPO loss
        loss, kl, ratios, entropy, trunc_above, trunc_below = calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            mask=mask,
            kl_beta=0.01,
            entropy_weight=0.01,
        )

        # 6. Compute total loss
        total_loss = loss.sum() / mask.sum()

        # Test that we can backpropagate
        total_loss.backward()

        # Check that gradients exist
        assert logits.grad is not None, "Gradients should exist after backward"
        assert torch.isfinite(logits.grad).all(), "Gradients should be finite"

    def test_zero_advantages(self):
        """Test behavior when all advantages are zero."""
        batch_size, seq_len = 2, 4

        current_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = current_logprobs.clone()
        ref_logprobs = current_logprobs.clone()
        advantages = torch.zeros(batch_size)

        loss, *_ = calculate_grpo_loss(
            current_logprobs, old_logprobs, ref_logprobs, advantages
        )

        # With zero advantages, policy gradient term should be near zero
        # (only KL and entropy terms remain)
        assert torch.isfinite(loss).all()
