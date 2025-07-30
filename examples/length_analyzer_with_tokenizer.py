#!/usr/bin/env python3
"""Example: Using LengthAnalyzer with tokenizer for accurate token counting.

This example demonstrates how to use the LengthAnalyzer with a tokenizer
to get accurate token counts instead of just word counts.
"""

from oumi.builders import build_tokenizer
from oumi.core.analyze import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, ModelParams
from oumi.core.configs.analyze_config import SampleAnalyzerParams


def example_length_analyzer_with_tokenizer():
    """Example using LengthAnalyzer with tokenizer for accurate token counting."""
    print("=== LengthAnalyzer with Tokenizer Example ===\n")

    # Build a tokenizer
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_kwargs={"pad_token": "<|endoftext|>"},
    )
    tokenizer = build_tokenizer(model_params)

    # Create config with length analyzer and tokenizer
    config = AnalyzeConfig(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        split="train_sft",
        sample_count=10,  # Small sample for demonstration
        analyzers=[
            SampleAnalyzerParams(
                id="length",
                params={
                    "char_count": True,
                    "word_count": True,
                    "sentence_count": True,
                    "token_count": True,  # Enable token counting
                },
            )
        ],
    )

    # Create analyzer
    analyzer = DatasetAnalyzer(config)

    print(f"Dataset: {analyzer.dataset_name}")
    print(f"Tokenizer: {type(analyzer.tokenizer).__name__}")
    print(f"Sample count: {config.sample_count}")

    # Run analysis
    print("\nRunning analysis...")
    analyzer.analyze_dataset()

    # Get results
    results = analyzer.analysis_results
    if results and results.messages:
        print(f"Analyzed {len(results.messages)} messages")

        # Show statistics
        results_df = results.to_dataframe()

        # Length metrics columns
        length_cols = [col for col in results_df.columns if col.startswith("length_")]
        print(f"\nLength metrics available: {length_cols}")

        # Show statistics for each metric
        for col in length_cols:
            if col in results_df.columns:
                print(f"\n{col}:")
                print(f"  Mean: {results_df[col].mean():.2f}")
                print(f"  Median: {results_df[col].median():.2f}")
                print(f"  Min: {results_df[col].min()}")
                print(f"  Max: {results_df[col].max()}")

        # Show sample messages with their metrics
        print("\nSample messages with metrics:")
        for i, message in enumerate(results.messages[:3]):
            print(f"\nMessage {i + 1} ({message.role}):")
            print(f"  Text: {message.text_content[:100]}...")

            # Show length metrics
            length_metrics = {
                k: v for k, v in message.to_dict().items() if k.startswith("length_")
            }
            for metric, value in length_metrics.items():
                print(f"  {metric}: {value}")

    return analyzer


def example_query_by_token_count():
    """Example of querying messages by token count."""
    print("\n=== Querying by Token Count ===\n")

    # Build tokenizer
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_kwargs={"pad_token": "<|endoftext|>"},
    )
    tokenizer = build_tokenizer(model_params)

    # Create config
    config = AnalyzeConfig(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        split="train_sft",
        sample_count=50,  # Larger sample for better queries
        analyzers=[
            SampleAnalyzerParams(
                id="length",
                params={
                    "char_count": True,
                    "word_count": True,
                    "sentence_count": True,
                    "token_count": True,
                },
            )
        ],
    )

    # Create analyzer
    analyzer = DatasetAnalyzer(config)

    # Run analysis
    print("Running analysis...")
    analyzer.analyze_dataset()

    # Query examples
    queries = [
        ("Short messages (< 20 tokens)", "length_token_count < 20"),
        (
            "Medium messages (20-100 tokens)",
            "length_token_count >= 20 and length_token_count <= 100",
        ),
        ("Long messages (> 100 tokens)", "length_token_count > 100"),
        (
            "User messages with high token count",
            "role == 'user' and length_token_count > 50",
        ),
        (
            "Assistant messages with low token count",
            "role == 'assistant' and length_token_count < 30",
        ),
    ]

    for description, query in queries:
        try:
            results = analyzer.query(query)
            print(f"\n{description}:")
            print(f"  Query: {query}")
            print(f"  Results: {len(results)} messages")

            if len(results) > 0:
                # Show statistics
                if "length_token_count" in results.columns:
                    print("  Token count stats:")
                    print(f"    Mean: {results['length_token_count'].mean():.2f}")
                    print(f"    Median: {results['length_token_count'].median():.2f}")
                    print(f"    Min: {results['length_token_count'].min()}")
                    print(f"    Max: {results['length_token_count'].max()}")

                # Show sample message
                sample = results.iloc[0]
                print("  Sample message:")
                print(f"    Role: {sample['role']}")
                print(f"    Text: {sample['text_content'][:80]}...")
                if "length_token_count" in sample:
                    print(f"    Token count: {sample['length_token_count']}")

        except Exception as e:
            print(f"\n{description}:")
            print(f"  Query failed: {e}")

    return analyzer


def example_token_vs_word_count_comparison():
    """Example comparing token count vs word count."""
    print("\n=== Token Count vs Word Count Comparison ===\n")

    # Build tokenizer
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_kwargs={"pad_token": "<|endoftext|>"},
    )
    tokenizer = build_tokenizer(model_params)

    # Create config
    config = AnalyzeConfig(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        split="train_sft",
        sample_count=20,
        analyzers=[
            SampleAnalyzerParams(
                id="length",
                params={
                    "word_count": True,
                    "token_count": True,
                },
            )
        ],
    )

    # Create analyzer
    analyzer = DatasetAnalyzer(config)

    # Run analysis
    print("Running analysis...")
    analyzer.analyze_dataset()

    # Get results
    results = analyzer.analysis_results
    if results and results.messages:
        results_df = results.to_dataframe()

        # Calculate token-to-word ratios
        if (
            "length_token_count" in results_df.columns
            and "length_word_count" in results_df.columns
        ):
            results_df["token_word_ratio"] = (
                results_df["length_token_count"] / results_df["length_word_count"]
            )

            print("Token vs Word Count Analysis:")
            print(f"  Total messages: {len(results_df)}")
            print(
                f"  Average token count: {results_df['length_token_count'].mean():.2f}"
            )
            print(f"  Average word count: {results_df['length_word_count'].mean():.2f}")
            print(
                f"  Average token/word ratio: "
                f"{results_df['token_word_ratio'].mean():.2f}"
            )

            # Show messages with high token/word ratios
            high_ratio = results_df[results_df["token_word_ratio"] > 2.0]
            print(f"\nMessages with high token/word ratio (>2.0): {len(high_ratio)}")

            if len(high_ratio) > 0:
                print("Sample high-ratio messages:")
                for i, (_, row) in enumerate(high_ratio.head(3).iterrows()):
                    print(f"  {i + 1}. Ratio: {row['token_word_ratio']:.2f}")
                    print(
                        f"     Tokens: {row['length_token_count']}, "
                        f"Words: {row['length_word_count']}"
                    )
                    print(f"     Text: {row['text_content'][:60]}...")

            # Show messages with low token/word ratios
            low_ratio = results_df[results_df["token_word_ratio"] < 1.2]
            print(f"\nMessages with low token/word ratio (<1.2): {len(low_ratio)}")

            if len(low_ratio) > 0:
                print("Sample low-ratio messages:")
                for i, (_, row) in enumerate(low_ratio.head(3).iterrows()):
                    print(f"  {i + 1}. Ratio: {row['token_word_ratio']:.2f}")
                    print(
                        f"     Tokens: {row['length_token_count']}, "
                        f"Words: {row['length_word_count']}"
                    )
                    print(f"     Text: {row['text_content'][:60]}...")

    return analyzer


if __name__ == "__main__":
    # Run examples
    analyzer1 = example_length_analyzer_with_tokenizer()
    analyzer2 = example_query_by_token_count()
    analyzer3 = example_token_vs_word_count_comparison()

    print("\n" + "=" * 60)
    print("LengthAnalyzer with tokenizer examples completed!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print("- Token counting is now available in LengthAnalyzer")
    print("- Tokenizer is automatically passed to analyzers")
    print("- Accurate token counts vs approximate word counts")
    print("- Query capabilities based on token counts")
    print("- Backward compatibility maintained")
