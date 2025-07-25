#!/usr/bin/env python3
"""Demo script for DatasetAnalyzer with length analyzer.

This script demonstrates how to use the DatasetAnalyzer with a length analyzer
to analyze conversation datasets. It's similar to the notebook example but
runs as a standalone script.
"""

# Import the length analyzer to register it
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.analyze.length_analyzer import LengthAnalyzer  # noqa: F401
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams


def main():
    """Run the analyzer demo."""
    print("🚀 Starting DatasetAnalyzer Demo")
    print("=" * 50)

    # Create length analyzer configuration
    length_analyzer = SampleAnalyzerParams(
        id="length",
        config={
            "char_count": True,
            "word_count": True,
            "sentence_count": True,
            "token_count": False,
        },
    )

    # Create configuration using simple fields
    config = AnalyzeConfig(
        dataset_name="tatsu-lab/alpaca",
        split="train",
        sample_count=50,
        output_path="./demo_results",
        analyzers=[length_analyzer],
    )

    # Create analyzer - analysis runs automatically
    print("📊 Creating analyzer (analysis runs automatically)...")
    analyzer = DatasetAnalyzer(config)

    # Display results
    print("\n📈 Analysis Results:")
    print(f"Dataset: {analyzer.dataset_name}")
    print(f"Has analysis results: {analyzer.has_analysis_results()}")

    if analyzer.has_analysis_results():
        results = analyzer._analysis_results
        assert results is not None  # Type assertion for linter
        print(f"Total conversations: {results.total_conversations}")
        print(f"Conversations analyzed: {results.conversations_analyzed}")
        print(f"Total messages: {results.total_messages}")

        print("\n📝 Sample Message Analysis:")
        for i, message in enumerate(results.messages[:3]):  # Show first 3 messages
            print(f"\nMessage {i + 1} ({message.role}):")
            print(
                f"  Content: '{message.text_content[:100]}...'"
            )  # Truncate long content
            print(f"  Metrics: {message.analyzer_metrics}")

        # Demo the new query functionality
        print("\n🔍 Query Demo:")

        # Query for short messages
        short_messages = analyzer.query("length_word_count < 10")
        print(f"Short messages (< 10 words): {len(short_messages)} found")

        # Query for assistant messages
        assistant_messages = analyzer.query("role == 'assistant'")
        print(f"Assistant messages: {len(assistant_messages)} found")

        # Query for long user messages
        long_user = analyzer.query("role == 'user' and length_word_count > 20")
        print(f"Long user messages (> 20 words): {len(long_user)} found")

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
