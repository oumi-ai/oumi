#!/usr/bin/env python3
"""Simplified test script for the Plugin-Style Analyzer System."""

from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import (
    DatasetAnalyzeConfig,
    SampleAnalyzeConfig,
)


def test_sample_analysis():
    """Test sample-level analysis with length analyzer."""
    print("\n" + "=" * 50)
    print("Testing Sample Analysis")
    print("=" * 50)

    # Create length analyzer
    length_analyzer = SampleAnalyzeConfig(
        id="length",
        enabled=True,
        config={
            "char_count": True,
            "word_count": True,
            "sentence_count": True,
            "token_count": False,
        },
    )

    # Create configuration using simple fields
    config = DatasetAnalyzeConfig(
        dataset_name="tatsu-lab/alpaca",
        split="train",
        sample_count=3,  # Limit analysis to 3 conversations
        output_path="./test_results",
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

    print("✅ Sample analysis test completed")


def test_yaml_config_loading():
    """Test loading analyzer configuration from YAML string (in-memory)."""
    print("\n" + "=" * 50)
    print("Testing YAML Config Loading (In-Memory)")
    print("=" * 50)

    # Define YAML configuration as a string
    yaml_string = """
    dataset_name: "tatsu-lab/alpaca"
    split: "train"
    sample_count: 3
    output_path: "./analysis_results"
    analyzers:
      - id: "length"
        enabled: true
        config:
          char_count: true
          word_count: true
          sentence_count: true
          token_count: false
    """

    # Load configuration from YAML string
    config = DatasetAnalyzeConfig.from_str(yaml_string)

    # Verify configuration was loaded correctly
    print(f"Loaded dataset: {config.dataset_name}")
    print(f"Split: {config.split}")
    print(f"Sample count: {config.sample_count}")
    print(f"Output path: {config.output_path}")
    print(f"Number of analyzers: {len(config.analyzers)}")

    # Verify analyzer configuration
    if config.analyzers:
        analyzer = config.analyzers[0]
        print(f"Analyzer ID: {analyzer.id}")
        print(f"Analyzer enabled: {analyzer.enabled}")
        print(f"Analyzer config: {analyzer.config}")

    # Test that the configuration works with the analyzer
    analyzer = DatasetAnalyzer(config)

    # Test dataset access
    print(f"Dataset size: {len(analyzer.dataset)} conversations")

    # Test conversation access
    conversation = analyzer.dataset.conversation(0)
    print(f"Conversation 0 length: {len(conversation.messages)} messages")

    print("✅ YAML config loading (in-memory) test completed")


if __name__ == "__main__":
    test_sample_analysis()
    test_yaml_config_loading()

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)
