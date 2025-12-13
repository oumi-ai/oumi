#!/usr/bin/env python3
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

"""RLVR (RL from Verifiable Rewards) Demo Script.

This script demonstrates the rubric-based reward function without running
full GRPO training. Useful for testing the judge integration.

Usage:
    python scripts/rlvr_demo.py

Requirements:
    - OPENAI_API_KEY environment variable set
"""

import json
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def demo_rubric_reward():
    """Demonstrate the rubric-based reward function."""
    from oumi.datasets.grpo.rewards.rubric_reward import (
        _get_or_create_judge,
        compute_rubric_reward,
    )

    print("=" * 60)
    print("RLVR Demo: Rubric-Based Reward Function")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY not set. Using mock rewards.")
        print("Set OPENAI_API_KEY to use actual LLM judge.\n")
        use_mock = True
    else:
        use_mock = False
        print("\nUsing OpenAI API for judge evaluation.\n")

    # Sample prompts and rubrics
    examples = [
        {
            "prompt": "Write a short product description for wireless Bluetooth headphones.",
            "rubrics": [
                "Mentions key features (wireless, Bluetooth)",
                "Highlights benefits to the user",
                "Is concise (under 100 words)",
                "Uses professional tone",
            ],
            "good_completion": (
                "Experience ultimate freedom with our premium wireless Bluetooth "
                "headphones. Featuring advanced noise cancellation and 30-hour battery "
                "life, these lightweight headphones deliver crystal-clear audio for "
                "music, calls, and gaming. The ergonomic design ensures all-day "
                "comfort. Connect seamlessly to any device and enjoy your sound, "
                "your way. Order now and elevate your listening experience."
            ),
            "bad_completion": "Headphones are good. Buy them.",
        },
        {
            "prompt": "Explain photosynthesis in simple terms for a 10-year-old.",
            "rubrics": [
                "Uses simple vocabulary appropriate for children",
                "Explains the basic process accurately",
                "Mentions plants, sunlight, and water",
                "Is engaging and easy to follow",
            ],
            "good_completion": (
                "Photosynthesis is like a superpower that plants have! Here's how "
                "it works: Plants use their leaves to catch sunlight, kind of like "
                "solar panels. They also drink water through their roots and breathe "
                "in air through tiny holes in their leaves. When they mix the "
                "sunlight, water, and air together, they make their own food - "
                "it's like cooking without a stove! And the best part? They give "
                "us fresh oxygen to breathe as a thank-you gift."
            ),
            "bad_completion": (
                "Photosynthesis involves the conversion of electromagnetic radiation "
                "into chemical energy via chlorophyll-mediated electron transport "
                "chains in the thylakoid membrane."
            ),
        },
    ]

    if use_mock:
        # Mock rewards for demo without API
        for i, example in enumerate(examples):
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {example['prompt'][:50]}...")
            print(f"\nRubrics: {example['rubrics']}")
            print(f"\nGood completion reward: 0.85 (mock)")
            print(f"Bad completion reward: 0.25 (mock)")
    else:
        # Use actual judge
        judge = _get_or_create_judge(judge_model="gpt-4o-mini")

        for i, example in enumerate(examples):
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {example['prompt'][:50]}...")
            print(f"\nRubrics: {example['rubrics']}")

            # Evaluate good completion
            good_reward = compute_rubric_reward(
                prompt=example["prompt"],
                completion=example["good_completion"],
                rubrics=example["rubrics"],
                judge=judge,
            )
            print(f"\nGood completion reward: {good_reward:.2f}")
            print(f"  Completion: {example['good_completion'][:100]}...")

            # Evaluate bad completion
            bad_reward = compute_rubric_reward(
                prompt=example["prompt"],
                completion=example["bad_completion"],
                rubrics=example["rubrics"],
                judge=judge,
            )
            print(f"\nBad completion reward: {bad_reward:.2f}")
            print(f"  Completion: {example['bad_completion'][:100]}...")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def demo_dataset_loading():
    """Demonstrate loading the RLVR dataset."""
    print("\n" + "=" * 60)
    print("Loading RLVR Dataset")
    print("=" * 60)

    # Import dataset
    from oumi.datasets.grpo.rlvr_rubric import RlvrRubricDataset

    # Load sample data
    sample_data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "examples",
        "grpo_rlvr",
        "sample_data.jsonl",
    )

    if os.path.exists(sample_data_path):
        dataset = RlvrRubricDataset(dataset_path=sample_data_path)
        print(f"\nLoaded {len(dataset)} examples from {sample_data_path}")

        # Show first few examples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {sample['prompt'][:80]}...")
            print(f"Rubrics: {sample['rubrics']}")
    else:
        print(f"\nSample data not found at: {sample_data_path}")
        print("Run from the oumi root directory.")


def main():
    """Run the RLVR demo."""
    print("\n" + "#" * 60)
    print("# RLVR (RL from Verifiable Rewards) Demo")
    print("#" * 60)

    # Demo 1: Dataset loading
    demo_dataset_loading()

    # Demo 2: Reward function
    demo_rubric_reward()

    print("\n" + "#" * 60)
    print("# Next Steps")
    print("#" * 60)
    print("""
To run full GRPO training with rubric-based rewards:

1. Set your OpenAI API key:
   export OPENAI_API_KEY='your-key-here'

2. Run training:
   oumi train -c configs/examples/grpo_rlvr/train.yaml

3. Monitor training in WandB (if enabled)

For more information, see the proposal document:
   docs/proposals/surge-partnership-proposal.md
""")


if __name__ == "__main__":
    main()
