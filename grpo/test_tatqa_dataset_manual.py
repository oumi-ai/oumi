"""Manual test script for TatQA dataset loading.

Run this to verify the dataset loads correctly before training.

Usage:
    python test_tatqa_dataset_manual.py
"""

import sys
sys.path.insert(0, "/data/shanghong/oumi")

import pandas as pd
from grpo.tatqa_dataset import TatqaDataset
from oumi.core.types.conversation import Conversation


def test_dataset_loading():
    """Test dataset loads successfully."""
    print("\n1. Loading dataset...")
    dataset = TatqaDataset(
        dataset_name="tatqa_data",
        dataset_path="/data/shanghong/oumi/gold/tatqa_data/train_final_with_groundtruth_max2048.jsonl",
        split="train",
    )
    print(f"   ✓ Dataset loaded: {len(dataset)} samples")
    return dataset


def test_single_sample(dataset):
    """Test single sample structure."""
    print("\n2. Testing single sample structure...")
    sample = dataset[0]
    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   Ground truth: {sample.get('ground_truth')}")
    print(f"   Prompt type: {type(sample['prompt'])}")
    print(f"   Prompt length: {len(sample['prompt'])} messages")

    assert "prompt" in sample, "Sample should have 'prompt' field"
    assert "ground_truth" in sample, "Sample should have 'ground_truth' field"
    print("   ✓ Sample structure correct")
    return sample


def test_prompt_format(sample):
    """Test prompt format (no assistant message)."""
    print("\n3. Testing prompt format...")
    prompt = sample["prompt"]
    assert isinstance(prompt, list), "Prompt should be a list"
    assert all("role" in msg and "content" in msg for msg in prompt), "Messages should have role and content"

    roles = [msg["role"] for msg in prompt]
    print(f"   Prompt roles: {roles}")
    assert "assistant" not in roles, "Prompt should NOT contain assistant message (GRPO will generate)"
    assert "user" in roles, "Prompt should contain user message"
    print("   ✓ Prompt format correct (no assistant message)")


def test_user_message_content(sample):
    """Test user message contains table and question."""
    print("\n4. Testing user message content...")
    prompt = sample["prompt"]
    user_message = next(msg for msg in prompt if msg["role"] == "user")
    user_content = user_message["content"]

    assert "Table:" in user_content, "User message should contain table"
    assert "Question:" in user_content, "User message should contain question"
    print("   ✓ User message contains table and question")
    print(f"\n   User message preview:\n   {user_content[:250]}...")


def test_ground_truth_extraction(sample):
    """Test ground truth extracted correctly."""
    print("\n5. Testing ground truth extraction...")
    assert "ground_truth" in sample, "Sample should have ground_truth field"
    assert sample["ground_truth"] is not None, "Ground truth should not be None"
    assert len(str(sample["ground_truth"])) > 0, "Ground truth should not be empty"
    print(f"   ✓ Ground truth extracted: '{sample['ground_truth']}'")


def test_multiple_samples(dataset):
    """Test first 10 samples."""
    print("\n6. Testing multiple samples...")
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        assert "prompt" in sample, f"Sample {i} missing prompt"
        assert "ground_truth" in sample, f"Sample {i} missing ground_truth"
        assert len(sample["prompt"]) > 0, f"Sample {i} has empty prompt"
        gt_preview = str(sample["ground_truth"])[:30]
        print(f"   Sample {i}: {len(sample['prompt'])} messages, gt='{gt_preview}...'")
    print("   ✓ All samples have correct structure")


def test_conversation_format(dataset):
    """Test conversational format conversion."""
    print("\n7. Testing conversation format conversion...")
    conversation = dataset.transform_conversation(pd.Series(dataset._data.iloc[0]))
    assert isinstance(conversation, Conversation), "Should return Conversation object"
    print(f"   ✓ Conversation format works: {len(conversation.messages)} messages")


def test_statistics(dataset):
    """Display dataset statistics."""
    print("\n8. Dataset Statistics:")
    print(f"   Total samples: {len(dataset)}")

    n_samples = min(100, len(dataset))
    avg_prompt_len = sum(len(dataset[i]["prompt"]) for i in range(n_samples)) / n_samples
    print(f"   Average prompt length: {avg_prompt_len:.1f} messages")

    ground_truths = [str(dataset[i].get("ground_truth", "")) for i in range(n_samples)]
    avg_gt_length = sum(len(gt) for gt in ground_truths) / len(ground_truths)
    print(f"   Average ground truth length: {avg_gt_length:.1f} characters")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing TatQA Dataset")
    print("=" * 70)

    dataset = test_dataset_loading()
    sample = test_single_sample(dataset)
    test_prompt_format(sample)
    test_user_message_content(sample)
    test_ground_truth_extraction(sample)
    test_multiple_samples(dataset)
    test_conversation_format(dataset)
    test_statistics(dataset)

    print("\n" + "=" * 70)
    print("✅ All dataset tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
