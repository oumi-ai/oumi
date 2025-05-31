#!/usr/bin/env python3
"""Test Molmo collator with real vision dataset data.

This script loads actual images from a vision dataset and tests
the collation process with different batch sizes.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from oumi.builders import build_tokenizer
from oumi.core.collators.vision_language_sft_collator import VisionLanguageSftCollator
from oumi.core.configs import ModelParams
from oumi.datasets.vision_language import PixmoCapDataset
from oumi.utils.logging import logger


def main():
    """Test Molmo collator with a real vision-language dataset."""
    print("Testing Molmo Collator with Real Vision Dataset")
    print("=" * 80)

    # 1. Setup model parameters
    print("\n1. Setting up Molmo model parameters...")
    model_params = ModelParams(
        model_name="allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype_str="float32",
        model_max_length=2048,
    )
    print(f"   Model: {model_params.model_name}")

    # 2. Build tokenizer
    print("\n2. Building tokenizer...")
    tokenizer = build_tokenizer(model_params)
    print(f"   ✓ Tokenizer: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {len(tokenizer)}")

    # 3. Create collator
    print("\n3. Creating VisionLanguageSftCollator...")
    collator = VisionLanguageSftCollator(
        tokenizer=tokenizer,
        processor_name=model_params.model_name,
        processor_kwargs={"trust_remote_code": True},
        max_length=512,
        truncation=True,
        label_ignore_index=-100,
        allow_multi_image_inputs=False,  # Molmo single image
        trust_remote_code=True,
        process_individually=True,
    )
    print("   ✓ Collator created")

    # 4. Load dataset
    print("\n4. Loading PixmoCapDataset (image captioning)...")
    try:
        dataset = PixmoCapDataset(
            split="train",
            limit=10,  # Only load 10 examples
            processor_name=model_params.model_name,
            trust_remote_code=True,
            return_conversations=True,
            tokenizer=tokenizer,
        )
        print(f"   ✓ Dataset loaded: {len(dataset)} examples")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        print("   Trying alternative dataset...")
        raise e

    # 5. Test single example
    print("\n5. Testing single example...")
    example = dataset[0]
    print(f"   Example keys: {list(example.keys())}")

    if "conversation_json" in example:
        from oumi.core.types import Conversation

        conv = Conversation.from_json(example["conversation_json"])
        print(f"   Messages: {len(conv.messages)}")
        # print(
        #     f"   Images: {len([m for m in conv.messages if m.contains_images()])} if conv.images else 0}"
        # )

        # Show first message
        if conv.messages:
            print(
                f"   First message: {conv.messages[0].role} - {conv.messages[0].content[:50]}..."
            )

    # 6. Test batch size 1
    # print("\n6. Testing batch size 1...")
    # batch_1 = [dataset[0]]
    # try:
    #     result_1 = collator(batch_1)
    #     print(f"   ✓ Collation successful")
    #     print(f"   Output keys: {list(result_1.keys())}")

    #     # Print shapes
    #     for key, value in result_1.items():
    #         if isinstance(value, torch.Tensor):
    #             print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")

    #     # Check for Molmo features
    #     molmo_features = ["images", "image_masks", "image_input_idx"]
    #     found_features = [f for f in molmo_features if f in result_1]
    #     if found_features:
    #         print(f"   ✓ Found Molmo features: {found_features}")

    # except Exception as e:
    #     print(f"   ✗ Error: {e}")
    #     logger.exception("Collation error:")
    #     raise

    # 7. Test batch size 4
    print("\n7. Testing batch size 4...")
    batch_4 = [dataset[i] for i in range(min(4, len(dataset)))]
    try:
        result_4 = collator(batch_4)
        print("   ✓ Collation successful")
        print(f"   Output keys: {list(result_4.keys())}")

        # Print shapes
        for key, value in result_4.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")

        # Verify batch size
        batch_size = len(batch_4)
        if "input_ids" in result_4:
            actual_batch_size = result_4["input_ids"].shape[0]
            assert actual_batch_size == batch_size, (
                f"Expected batch size {batch_size}, got {actual_batch_size}"
            )
            print(f"   ✓ Batch size verified: {actual_batch_size}")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        logger.exception("Collation error:")
        raise

    # 8. Test memory efficiency
    print("\n8. Checking tensor properties...")
    if "images" in result_4:
        images = result_4["images"]
        print("   Images tensor:")
        print(f"   - Shape: {images.shape}")
        print(
            f"   - Memory: {images.element_size() * images.nelement() / 1024 / 1024:.2f} MB"
        )
        print(f"   - Device: {images.device}")
        print(f"   - Requires grad: {images.requires_grad}")

    print("\n" + "=" * 80)
    print("✅ All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
