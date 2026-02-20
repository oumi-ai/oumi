"""Manual test script for TatQA reward function.

Run this to verify the reward function works correctly before training.

Usage:
    python test_tatqa_reward_manual.py
"""

import sys
sys.path.insert(0, "/data/shanghong/oumi")

from grpo.tatqa_reward import tatqa_reward

# Sample completions
correct_sample_1 = """<think>
Looking at the table, the net debt receipts in 2019 is shown in the Statement of cash flows row.
The value is 129,454 (in thousands of $).
</think>

<answer>129,454</answer>"""

correct_sample_2 = """The answer is clearly visible in the table.

<answer>129454</answer>"""  # Without comma

correct_sample_3 = """<think>
The net debt receipts for 2019 can be found in the table.
</think>

<answer>$129,454 thousand</answer>"""  # With units

wrong_answer = """<think>
Looking at the 2018 column...
</think>

<answer>127,625</answer>"""  # Wrong row

no_answer_tag = """The net debt receipts in 2019 were 129,454."""  # Missing tags

malformed = """<answer>some text without numbers</answer>"""


def test_exact_match():
    """Test exact match."""
    print("\n1. Testing exact match...")
    test_rewards = tatqa_reward(
        completions=[correct_sample_1],
        ground_truth=["129,454"],
        use_judge=False,  # Test without judge first
    )
    assert test_rewards == [1.0], f"Expected [1.0], got {test_rewards}"
    print("   ✓ Exact match test passed")


def test_normalized_match():
    """Test normalized match (no comma)."""
    print("\n2. Testing normalized match...")
    test_rewards = tatqa_reward(
        completions=[correct_sample_2],
        ground_truth=["129,454"],
        use_judge=False,
    )
    assert test_rewards == [1.0], f"Expected [1.0], got {test_rewards}"
    print("   ✓ Normalized match test passed")


def test_wrong_answer():
    """Test wrong answer."""
    print("\n3. Testing wrong answer...")
    test_rewards = tatqa_reward(
        completions=[wrong_answer],
        ground_truth=["129,454"],
        use_judge=False,
    )
    assert test_rewards == [0.0], f"Expected [0.0], got {test_rewards}"
    print("   ✓ Wrong answer test passed")


def test_format_errors():
    """Test format errors."""
    print("\n4. Testing format errors...")
    test_rewards = tatqa_reward(
        completions=[no_answer_tag, malformed],
        ground_truth=["129,454", "129,454"],
        use_judge=False,
    )
    assert test_rewards == [0.0, 0.0], f"Expected [0.0, 0.0], got {test_rewards}"
    print("   ✓ Format error test passed")


def test_batch_processing():
    """Test batch processing."""
    print("\n5. Testing batch processing...")
    test_rewards = tatqa_reward(
        completions=[correct_sample_1, correct_sample_2, wrong_answer, no_answer_tag],
        ground_truth=["129,454", "129,454", "129,454", "129,454"],
        use_judge=False,
    )
    assert test_rewards == [1.0, 1.0, 0.0, 0.0], f"Expected [1.0, 1.0, 0.0, 0.0], got {test_rewards}"
    print("   ✓ Batch processing test passed")


def test_with_judge():
    """Test reward function with LLM judge (requires judge model)."""
    print("\n6. Testing with LLM judge...")
    print("   This test requires the judge model to be available.")
    print("   Attempting to call judge model...")

    import os
    judge_config = "configs/tatqa_judge_config.yaml"
    if not os.path.exists(judge_config):
        print(f"   ⚠ Judge config not found at {judge_config}")
        print("   Skipping judge test")
        return False

    try:
        # Test case where judge should be needed (units difference)
        # Ground truth: "129,454"
        # Prediction: "$129,454 thousand" (with units - needs judge)
        test_rewards = tatqa_reward(
            completions=[correct_sample_3],  # Has "$" and "thousand"
            ground_truth=["129,454"],
            use_judge=True,
            judge_config_file=judge_config,
        )
        print(f"   Judge returned reward: {test_rewards}")

        # Judge should determine this is correct (same value, different formatting)
        # But we don't assert because judge behavior can vary
        print(f"   ✓ Judge model called successfully")
        print(f"   Note: Judge decided this is {'correct' if test_rewards[0] > 0.5 else 'incorrect'}")
        return True

    except Exception as e:
        print(f"   ⚠ Judge test failed with error: {e}")
        print("   This is expected if:")
        print("     - Judge model is not loaded/available")
        print("     - VLLM is not running")
        print("     - GPU memory is insufficient")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing TatQA Reward Function")
    print("=" * 70)

    # Fast-path tests (no judge)
    test_exact_match()
    test_normalized_match()
    test_wrong_answer()
    test_format_errors()
    test_batch_processing()

    print("\n" + "=" * 70)
    print("✅ All fast-path tests passed!")
    print("=" * 70)

    # Judge test (optional)
    print("\n" + "=" * 70)
    print("Testing with LLM Judge (Optional)")
    print("=" * 70)
    judge_success = test_with_judge()

    if judge_success:
        print("\n✅ Judge test passed! Full reward function is working.")
    else:
        print("\n⚠ Judge test skipped or failed.")
        print("Fast-path matching works, but judge needs to be set up for full functionality.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
