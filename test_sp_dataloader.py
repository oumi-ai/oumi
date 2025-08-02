#!/usr/bin/env python3
"""Test script to debug UlyssesSPDataLoaderAdapter behavior locally.
This helps us understand why 'labels' aren't being converted to 'shift_labels'.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def test_sp_dataloader_adapter():
    """Test UlyssesSPDataLoaderAdapter behavior locally."""
    print("Testing UlyssesSPDataLoaderAdapter...")

    try:
        from deepspeed.runtime.sequence_parallel.ulysses_sp import (
            UlyssesSPDataLoaderAdapter,
        )

        print("✓ Successfully imported UlyssesSPDataLoaderAdapter")
    except ImportError as e:
        print(f"✗ Failed to import UlyssesSPDataLoaderAdapter: {e}")
        return False

    # Create a simple test dataset
    batch_size = 2
    seq_len = 12  # Divisible by 3 for SP size 3
    vocab_size = 1000

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(-100, vocab_size, (batch_size, seq_len))

    print("Created test data:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")
    print(f"  labels shape: {labels.shape}")

    # Create a simple dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Test the first batch
    first_batch = next(iter(dataloader))
    print("\nOriginal batch structure:")
    print(f"  Batch length: {len(first_batch)}")
    print(f"  Tensor shapes: {[t.shape for t in first_batch]}")

    # Convert to dictionary format (as expected by UlyssesSPDataLoaderAdapter)
    def collate_to_dict(batch):
        input_ids, attention_mask, labels = batch[0]
        return {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "labels": labels.unsqueeze(0),
        }

    dict_dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_to_dict
    )
    dict_batch = next(iter(dict_dataloader))
    print("\nDict batch structure:")
    for key, value in dict_batch.items():
        print(f"  {key}: {value.shape} dtype={value.dtype}")

    # Test UlyssesSPDataLoaderAdapter with mock SP parameters
    print("\nTesting UlyssesSPDataLoaderAdapter...")

    try:
        # Mock SP parameters
        sp_rank = 0
        sp_world_size = 3
        device = torch.device("cpu")

        # Create mock SP group (this might not work without actual distributed setup)
        # For testing, we'll try to see what happens
        wrapped_dataloader = UlyssesSPDataLoaderAdapter(
            dict_dataloader,
            sp_rank=sp_rank,
            sp_group=None,  # This might cause issues, but let's see
            sp_world_size=sp_world_size,
            device=device,
        )

        print("✓ Successfully created UlyssesSPDataLoaderAdapter")

        # Try to get a batch
        wrapped_batch = next(iter(wrapped_dataloader))
        print("\nWrapped batch structure:")
        for key, value in wrapped_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")

        # Check if 'shift_labels' exists
        if "shift_labels" in wrapped_batch:
            print("✓ Found 'shift_labels' in wrapped batch!")
        else:
            print("✗ 'shift_labels' not found in wrapped batch")

        if "labels" in wrapped_batch:
            print("⚠ 'labels' still present in wrapped batch")
        else:
            print("✓ 'labels' properly removed from wrapped batch")

        return True

    except Exception as e:
        print(f"✗ UlyssesSPDataLoaderAdapter test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sp_groups():
    """Test SP group creation without full distributed setup."""
    print("\nTesting SP group availability...")

    try:
        from deepspeed.utils import groups

        print("✓ Successfully imported deepspeed.utils.groups")

        # Try to get SP groups (this will likely fail without distributed setup)
        try:
            sp_group = groups._get_sequence_parallel_group()
            print(f"✓ SP group found: {sp_group}")
        except Exception as e:
            print(f"✗ No SP group found (expected without distributed setup): {e}")

        try:
            sp_world_size = groups._get_sequence_parallel_world_size()
            print(f"✓ SP world size: {sp_world_size}")
        except Exception as e:
            print(f"✗ SP world size not available: {e}")

    except ImportError as e:
        print(f"✗ Failed to import deepspeed groups: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Local SP Component Testing")
    print("=" * 60)

    test_sp_groups()
    print("\n" + "=" * 60)
    test_sp_dataloader_adapter()
    print("=" * 60)
