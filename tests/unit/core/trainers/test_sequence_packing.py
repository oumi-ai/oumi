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

"""Unit tests for sequence packing utilities."""

import pytest
import torch

from oumi.core.trainers.megatron.sequence_packing import (
    PackedBatch,
    compute_packing_efficiency,
    create_causal_mask_for_packed_sequences,
    pack_sequences,
    unpack_logprobs,
    unpack_rewards,
)


class TestPackSequences:
    """Test sequence packing functionality."""

    def test_pack_simple_sequences(self):
        """Test packing of simple sequences."""
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 0, 0],
                [4, 5, 6, 7, 0],
                [8, 9, 0, 0, 0],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 0, 0, 0],
            ]),
        }

        packed_batches = pack_sequences(batch, max_bin_size=10, pad_token_id=0)

        # Should fit all sequences in one or two bins
        assert len(packed_batches) <= 2
        assert all(isinstance(pb, PackedBatch) for pb in packed_batches)

        # Verify all sequences are accounted for
        total_sequences = sum(pb.num_sequences for pb in packed_batches)
        assert total_sequences == 3

    def test_pack_sequences_respects_max_bin_size(self):
        """Test that packing respects maximum bin size."""
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]),
        }

        # Max bin size too small to fit both, should create 2 bins
        packed_batches = pack_sequences(batch, max_bin_size=10, pad_token_id=0)
        assert len(packed_batches) == 2

    def test_pack_sequences_with_labels(self):
        """Test packing with labels included."""
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 0],
                [4, 5, 6, 7],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ]),
            "labels": torch.tensor([
                [1, 2, 3, -100],
                [4, 5, 6, 7],
            ]),
        }

        packed_batches = pack_sequences(batch, max_bin_size=10, pad_token_id=0)

        # Verify labels are packed
        for pb in packed_batches:
            assert pb.labels is not None
            assert pb.labels.shape[1] == 10  # max_bin_size

    def test_pack_empty_sequences_skipped(self):
        """Test that empty sequences are skipped."""
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 0],
                [0, 0, 0, 0],  # Empty sequence
                [4, 5, 0, 0],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 0],
                [0, 0, 0, 0],  # Empty
                [1, 1, 0, 0],
            ]),
        }

        packed_batches = pack_sequences(batch, max_bin_size=10, pad_token_id=0)

        # Should only pack 2 non-empty sequences
        total_sequences = sum(pb.num_sequences for pb in packed_batches)
        assert total_sequences == 2

    def test_pack_sequence_exceeds_max_size_raises(self):
        """Test that sequences exceeding max_bin_size raise an error."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # Max bin size too small
        with pytest.raises(ValueError, match="exceeds max_bin_size"):
            pack_sequences(batch, max_bin_size=3, pad_token_id=0)

    def test_packed_batch_metadata(self):
        """Test that PackedBatch metadata is correct."""
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 0],
                [4, 5, 0, 0],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ]),
        }

        packed_batches = pack_sequences(batch, max_bin_size=10, pad_token_id=0)
        pb = packed_batches[0]

        # Check metadata
        assert len(pb.seq_starts) == pb.num_sequences
        assert len(pb.seq_lengths) == pb.num_sequences
        assert len(pb.original_indices) == pb.num_sequences

        # Verify sequence boundaries
        assert pb.seq_starts[0] == 0
        if pb.num_sequences > 1:
            assert pb.seq_starts[1] == pb.seq_lengths[0]


class TestCausalMaskForPackedSequences:
    """Test causal mask creation for packed sequences."""

    def test_causal_mask_single_sequence(self):
        """Test causal mask for a single sequence."""
        seq_starts = [0]
        seq_lengths = [4]
        bin_size = 8

        mask = create_causal_mask_for_packed_sequences(
            seq_starts, seq_lengths, bin_size, device=torch.device("cpu")
        )

        assert mask.shape == (1, 1, bin_size, bin_size)

        # First 4 positions should have causal attention
        # Position 0 can only see itself
        assert mask[0, 0, 0, 0] == 0.0
        assert torch.isinf(mask[0, 0, 0, 1])

        # Position 3 can see positions 0-3
        assert mask[0, 0, 3, 0] == 0.0
        assert mask[0, 0, 3, 3] == 0.0
        assert torch.isinf(mask[0, 0, 3, 4])

    def test_causal_mask_multiple_sequences(self):
        """Test causal mask for multiple packed sequences."""
        seq_starts = [0, 3]
        seq_lengths = [3, 4]
        bin_size = 8

        mask = create_causal_mask_for_packed_sequences(
            seq_starts, seq_lengths, bin_size, device=torch.device("cpu")
        )

        # First sequence (positions 0-2)
        # Position 0 sees only itself
        assert mask[0, 0, 0, 0] == 0.0
        assert torch.isinf(mask[0, 0, 0, 1])

        # Position 2 sees 0-2, not position 3 (different sequence)
        assert mask[0, 0, 2, 2] == 0.0
        assert torch.isinf(mask[0, 0, 2, 3])

        # Second sequence (positions 3-6)
        # Position 3 sees only itself (start of new sequence)
        assert mask[0, 0, 3, 3] == 0.0
        assert torch.isinf(mask[0, 0, 3, 0])  # Can't see previous sequence
        assert torch.isinf(mask[0, 0, 3, 4])

        # Position 6 sees 3-6
        assert mask[0, 0, 6, 3] == 0.0
        assert mask[0, 0, 6, 6] == 0.0
        assert torch.isinf(mask[0, 0, 6, 0])  # Can't see previous sequence


class TestUnpackLogprobs:
    """Test unpacking of log probabilities."""

    def test_unpack_simple_logprobs(self):
        """Test unpacking of logprobs from packed format."""
        packed_logprobs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        seq_starts = [0, 2]
        seq_lengths = [2, 3]
        original_indices = [0, 1]
        batch_size = 2
        max_seq_length = 3

        unpacked = unpack_logprobs(
            packed_logprobs, seq_starts, seq_lengths,
            original_indices, batch_size, max_seq_length
        )

        assert unpacked.shape == (batch_size, max_seq_length)

        # First sequence: logprobs at positions 0 (length-1 = 1 logprob)
        # Note: logprobs are 1 shorter than sequence length
        assert unpacked[0, 0] == 0.1

        # Second sequence: logprobs at positions 2-3 (length-1 = 2 logprobs)
        assert unpacked[1, 0] == 0.3
        assert unpacked[1, 1] == 0.4

    def test_unpack_with_padding(self):
        """Test that unpacking handles padding correctly."""
        packed_logprobs = torch.tensor([[0.1, 0.2, 0.3]])
        seq_starts = [0]
        seq_lengths = [3]
        original_indices = [0]
        batch_size = 2
        max_seq_length = 5

        unpacked = unpack_logprobs(
            packed_logprobs, seq_starts, seq_lengths,
            original_indices, batch_size, max_seq_length
        )

        # First sequence has 2 logprobs (length-1)
        assert unpacked[0, 0] == 0.1
        assert unpacked[0, 1] == 0.2

        # Second sequence (not in packed batch) should be -100.0
        assert unpacked[1, 0] == -100.0


class TestUnpackRewards:
    """Test unpacking of rewards."""

    def test_unpack_rewards(self):
        """Test unpacking rewards from packed format."""
        packed_rewards = torch.tensor([1.0, 2.0, 3.0])
        original_indices = [0, 2, 1]  # Out of order
        batch_size = 3

        unpacked = unpack_rewards(packed_rewards, original_indices, batch_size)

        assert unpacked.shape == (batch_size,)
        assert unpacked[0] == 1.0
        assert unpacked[1] == 3.0
        assert unpacked[2] == 2.0

    def test_unpack_rewards_partial_batch(self):
        """Test unpacking when not all batch items are packed."""
        packed_rewards = torch.tensor([1.0, 2.0])
        original_indices = [1, 3]
        batch_size = 5

        unpacked = unpack_rewards(packed_rewards, original_indices, batch_size)

        assert unpacked.shape == (batch_size,)
        assert unpacked[1] == 1.0
        assert unpacked[3] == 2.0
        assert unpacked[0] == 0.0  # Not in packed batch
        assert unpacked[2] == 0.0


class TestPackingEfficiency:
    """Test packing efficiency computation."""

    def test_compute_efficiency_single_bin(self):
        """Test efficiency computation for single bin."""
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 0],
                [4, 5, 0, 0],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ]),
        }

        packed_batches = pack_sequences(batch, max_bin_size=10, pad_token_id=0)
        efficiency = compute_packing_efficiency(packed_batches, max_bin_size=10)

        assert efficiency["num_bins"] == 1
        assert efficiency["total_sequences"] == 2
        assert efficiency["total_tokens"] == 5  # 3 + 2
        assert efficiency["total_capacity"] == 10
        assert efficiency["utilization"] == 0.5  # 5/10
        assert efficiency["avg_sequences_per_bin"] == 2.0

    def test_compute_efficiency_multiple_bins(self):
        """Test efficiency computation for multiple bins."""
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
            ]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]),
        }

        packed_batches = pack_sequences(batch, max_bin_size=10, pad_token_id=0)
        efficiency = compute_packing_efficiency(packed_batches, max_bin_size=10)

        assert efficiency["num_bins"] == 2
        assert efficiency["total_sequences"] == 2
        assert efficiency["total_tokens"] == 16
        assert efficiency["total_capacity"] == 20
        assert efficiency["utilization"] == 0.8  # 16/20

    def test_compute_efficiency_empty(self):
        """Test efficiency computation with no bins."""
        efficiency = compute_packing_efficiency([], max_bin_size=10)

        assert efficiency["num_bins"] == 0
        assert efficiency["total_sequences"] == 0
        assert efficiency["utilization"] == 0.0
