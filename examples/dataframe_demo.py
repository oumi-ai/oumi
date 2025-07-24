#!/usr/bin/env python3
"""Demo script for DatasetAnalyzer DataFrame functionality.

This script demonstrates how to use the new to_dataframe() method
to easily convert analysis results to pandas DataFrames for querying.
"""

# Import the length analyzer to register it

from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.analyze.length_analyzer import LengthAnalyzer  # noqa: F401
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams


def main():
    """Run the DataFrame demo."""
    print("🚀 Starting DatasetAnalyzer DataFrame Demo")
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
        sample_count=10,  # Small sample for demo
        output_path="./test_output",
        analyzers=[length_analyzer],
    )

    # Create and run analyzer
    print("📊 Running analysis...")
    analyzer = DatasetAnalyzer(config)
    results = analyzer.analyze_dataset()

    print("\n📈 Analysis Results:")
    print(f"Dataset: {results.dataset_name}")
    print(f"Total conversations: {results.total_conversations}")
    print(f"Conversations analyzed: {results.conversations_analyzed}")
    print(f"Total messages: {results.total_messages}")

    # Convert to DataFrame
    print("\n🔄 Converting to DataFrame...")
    results_df = results.to_dataframe()

    print("\n📊 DataFrame Info:")
    print(f"Shape: {results_df.shape}")
    print(f"Columns: {list(results_df.columns)}")

    print("\n📋 First 5 rows:")
    print(results_df.head())

    # Demonstrate querying capabilities
    print("\n🔍 Query Examples:")

    # Filter by role
    user_messages = results_df[results_df["role"] == "user"]
    assistant_messages = results_df[results_df["role"] == "assistant"]
    system_messages = results_df[results_df["role"] == "system"]

    print(f"User messages: {len(user_messages)}")
    print(f"Assistant messages: {len(assistant_messages)}")
    print(f"System messages: {len(system_messages)}")

    # Filter by message length
    long_messages = results_df[results_df["length_char_count"] > 200]
    short_messages = results_df[results_df["length_char_count"] < 50]

    print(f"Long messages (>200 chars): {len(long_messages)}")
    print(f"Short messages (<50 chars): {len(short_messages)}")

    # Complex queries
    long_assistant_messages = results_df[
        (results_df["role"] == "assistant") & (results_df["length_word_count"] > 50)
    ]

    print(f"Long assistant messages (>50 words): {len(long_assistant_messages)}")

    # Statistics
    print("\n📊 Statistics:")
    print(f"Average character count: {results_df['length_char_count'].mean():.1f}")
    print(f"Average word count: {results_df['length_word_count'].mean():.1f}")
    print(f"Average sentence count: {results_df['length_sentence_count'].mean():.1f}")

    # Group by role
    role_stats = (
        results_df.groupby("role")
        .agg(
            {
                "length_char_count": ["mean", "std", "min", "max"],
                "length_word_count": ["mean", "std", "min", "max"],
                "length_sentence_count": ["mean", "std", "min", "max"],
            }
        )
        .round(1)
    )

    print("\n📊 Statistics by Role:")
    print(role_stats)

    print("\n✅ DataFrame demo completed successfully!")


if __name__ == "__main__":
    main()
