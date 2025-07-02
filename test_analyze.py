#!/usr/bin/env python3
"""Simple test script to verify DatasetAnalyzer functionality."""

from oumi.analyze import DatasetAnalyzer


def test_analyzer():
    """Test the DatasetAnalyzer class."""
    try:
        # Test Alpaca dataset
        print("=" * 60)
        print("Testing Alpaca Dataset")
        print("=" * 60)

        # Create an analyzer instance
        print("Creating DatasetAnalyzer for 'alpaca'...")
        analyzer = DatasetAnalyzer("alpaca")

        # Test getting dataset size
        print(f"\nDataset size: {analyzer.get_dataset_size()} conversations")

        # Test getting a conversation
        print("\nGetting conversation 0...")
        conversation = analyzer.get_conversation(0)
        print(f"Conversation type: {type(conversation)}")

        # Test getting conversation length
        print(f"Conversation 0 length: {analyzer.get_conversation_length(0)} messages")

        # Test printing a conversation
        print("\nPrinting conversation 0...")
        analyzer.print_conversation(0)

        # Test getting another conversation
        print("\nGetting conversation 5...")
        conversation5 = analyzer.get_conversation(5)
        print(f"Conversation 5 type: {type(conversation5)}")
        print(f"Conversation 5 length: {analyzer.get_conversation_length(5)} messages")

        print("\nAlpaca test completed successfully!")

        # Test Ultrachat dataset
        print("\n" + "=" * 60)
        print("Testing Ultrachat Dataset")
        print("=" * 60)

        # Create an analyzer instance for ultrachat with split
        print("Creating DatasetAnalyzer for 'ultrachat' with split='train_sft'...")
        ultrachat_analyzer = DatasetAnalyzer("ultrachat", split="train_sft")

        # Test getting dataset size
        print(f"\nDataset size: {ultrachat_analyzer.get_dataset_size()} conversations")

        # Test getting a conversation
        print("\nGetting conversation 0...")
        ultrachat_conversation = ultrachat_analyzer.get_conversation(0)
        print(f"Conversation type: {type(ultrachat_conversation)}")

        # Test getting conversation length
        print(
            "Conversation 0 length: "
            + str(ultrachat_analyzer.get_conversation_length(0))
            + " messages"
        )

        # Test printing a conversation
        print("\nPrinting conversation 0...")
        ultrachat_analyzer.print_conversation(0)

        # Test getting another conversation
        print("\nGetting conversation 3...")
        ultrachat_conversation3 = ultrachat_analyzer.get_conversation(3)
        print(f"Conversation 3 type: {type(ultrachat_conversation3)}")
        print(
            "Conversation 3 length: "
            + str(ultrachat_analyzer.get_conversation_length(3))
            + " messages"
        )

        print("\nUltrachat test completed successfully!")

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_analyzer()
