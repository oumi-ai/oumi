#!/usr/bin/env python3
"""Simplified test script for the Plugin-Style Analyzer System."""

from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import (
    AnalyzerConfig,
    AnalyzerPluginConfig,
    DatasetSchema,
    InputConfig,
    OutputConfig,
)


def test_basic_functionality():
    """Test basic analyzer functionality."""
    print("=" * 50)
    print("Testing Basic Analyzer Functionality")
    print("=" * 50)

    # Create basic configuration
    config = AnalyzerConfig(
        input=InputConfig(
            name="tatsu-lab/alpaca",
            split="train",
            max_conversations=5,  # Small subset for testing
            schema=DatasetSchema(type="conversation"),
        ),
    )

    analyzer = DatasetAnalyzer(config)

    # Test dataset access
    print(f"Dataset size: {len(analyzer.dataset)} conversations")

    # Test conversation access
    conversation = analyzer.dataset.conversation(0)
    print(f"Conversation 0 length: {len(conversation.messages)} messages")

    print("✅ Basic functionality test completed")


def test_plugin_analysis():
    """Test plugin-style analysis with length analyzer."""
    print("\n" + "=" * 50)
    print("Testing Plugin Analysis")
    print("=" * 50)

    # Create length analyzer
    length_analyzer = AnalyzerPluginConfig(
        id="length",
        enabled=True,
        config={
            "char_count": True,
            "word_count": True,
            "sentence_count": True,
            "token_count": False,
        },
    )

    # Create configuration
    config = AnalyzerConfig(
        input=InputConfig(
            name="tatsu-lab/alpaca",
            split="train",
            max_conversations=3,  # Very small subset for quick testing
            schema=DatasetSchema(type="conversation"),
        ),
        outputs=OutputConfig(
            path="./test_results",
        ),
        analyzers=[length_analyzer],
    )

    analyzer = DatasetAnalyzer(config)

    # Run analysis
    print("Running analysis...")
    results = analyzer.analyze_dataset()

    # Check results
    if "sample_level_results" in results:
        sample_results = results["sample_level_results"]
        print(f"Dataset: {results.get('dataset_name', 'unknown')}")
        print(f"Total conversations: {sample_results.get('total_conversations', 0)}")
        print(
            f"Analyzed conversations: {sample_results.get('conversations_analyzed', 0)}"
        )
        print(f"Total messages: {sample_results.get('total_messages', 0)}")

        # Check first message metrics
        messages = sample_results.get("messages", [])
        if messages:
            first_message = messages[0]
            print("\nFirst message metrics:")
            print(f"  Role: {first_message.get('role', 'N/A')}")
            print(f"  Characters: {first_message.get('length_char_count', 'N/A')}")
            print(f"  Words: {first_message.get('length_word_count', 'N/A')}")
            print(f"  Sentences: {first_message.get('length_sentence_count', 'N/A')}")

    print("✅ Plugin analysis test completed")


if __name__ == "__main__":
    test_basic_functionality()
    test_plugin_analysis()

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)
