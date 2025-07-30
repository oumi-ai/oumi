#!/usr/bin/env python3
"""Example using DatasetAnalyzer with a tokenizer.

This example demonstrates how to use the DatasetAnalyzer with a tokenizer
to enable token counting and other tokenization-based analysis.
"""

from oumi.builders import build_tokenizer
from oumi.core.analyze import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, ModelParams


def example_with_tokenizer():
    """Example using DatasetAnalyzer with a tokenizer."""
    print("=== Example: DatasetAnalyzer with Tokenizer ===\n")

    # Build a tokenizer
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_kwargs={"pad_token": "<|endoftext|>"},
    )
    tokenizer = build_tokenizer(model_params)

    # Create config with tokenizer
    config = AnalyzeConfig(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        split="train_sft",
        sample_count=5,  # Small sample for demonstration
    )

    # Create analyzer
    analyzer = DatasetAnalyzer(config)

    print(f"Dataset: {analyzer.dataset_name}")
    print(f"Dataset size: {len(analyzer.dataset)}")
    print(f"Tokenizer: {type(analyzer.tokenizer).__name__}")
    print(f"Sample count: {config.sample_count}")

    # Demonstrate dataset functionality with tokenizer
    if len(analyzer.dataset) > 0:
        print("\n--- Dataset Access Examples ---")

        # Raw data access
        raw_example = analyzer.dataset.raw(0)
        print(f"Raw example keys: {list(raw_example.keys())}")

        # Conversation access
        conversation = analyzer.dataset.conversation(0)
        print(f"Conversation has {len(conversation.messages)} messages")

        # Tokenized data access (only available with tokenizer)
        if analyzer.tokenizer is not None:
            tokenized = analyzer.dataset[0]
            print(f"Tokenized data keys: {list(tokenized.keys())}")
            print(f"Input IDs length: {len(tokenized['input_ids'])}")

            # Decode to see the formatted prompt
            decoded = analyzer.tokenizer.decode(tokenized["input_ids"])
            print(f"Decoded prompt (first 200 chars): {decoded[:200]}...")

    return analyzer


def example_without_tokenizer():
    """Example using DatasetAnalyzer without a tokenizer (backward compatibility)."""
    print("\n=== Example: DatasetAnalyzer without Tokenizer ===\n")

    # Create config without tokenizer
    config = AnalyzeConfig(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        sample_count=5,  # Small sample for demonstration
    )

    # Create analyzer
    analyzer = DatasetAnalyzer(config)

    print(f"Dataset: {analyzer.dataset_name}")
    print(f"Dataset size: {len(analyzer.dataset)}")
    print(f"Tokenizer: {analyzer.tokenizer}")
    print(f"Sample count: {config.sample_count}")

    # Demonstrate basic dataset functionality
    if len(analyzer.dataset) > 0:
        print("\n--- Dataset Access Examples ---")

        # Raw data access
        raw_example = analyzer.dataset.raw(0)
        print(f"Raw example keys: {list(raw_example.keys())}")

        # Conversation access
        conversation = analyzer.dataset.conversation(0)
        print(f"Conversation has {len(conversation.messages)} messages")

        # Note: Tokenized access is not available without tokenizer
        print("Note: Tokenized data access requires a tokenizer")

    return analyzer


def example_analysis_with_tokenizer():
    """Example of running analysis with a tokenizer-enabled analyzer."""
    print("\n=== Example: Analysis with Tokenizer ===\n")

    # Build a tokenizer
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_kwargs={"pad_token": "<|endoftext|>"},
    )
    tokenizer = build_tokenizer(model_params)

    # Create config with tokenizer and analyzers
    config = AnalyzeConfig(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        split="train_sft",
        sample_count=10,  # Small sample for demonstration
        analyzers=[
            # Add sample analyzers here if available
            # SampleAnalyzerParams(id="length_analyzer", params={}),
            # SampleAnalyzerParams(id="sentiment_analyzer", params={}),
        ],
    )

    # Create analyzer
    analyzer = DatasetAnalyzer(config)

    print(f"Analyzer created with {len(analyzer.sample_analyzers)} analyzers")
    print(f"Dataset size: {len(analyzer.dataset)}")

    # Run analysis (if analyzers are configured)
    if analyzer.sample_analyzers:
        print("\nRunning analysis...")
        analyzer.analyze_dataset()

        # Query results
        results = analyzer.query("role == 'user'")
        print(f"Found {len(results)} user messages")

        if len(results) > 0:
            print("Sample user message:")
            print(f"  Content: {results.iloc[0]['text_content'][:100]}...")
    else:
        print("\nNo analyzers configured. Add analyzers to run analysis.")

    return analyzer


if __name__ == "__main__":
    # Run examples
    analyzer_with_tokenizer = example_with_tokenizer()
    analyzer_without_tokenizer = example_without_tokenizer()
    analyzer_for_analysis = example_analysis_with_tokenizer()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
