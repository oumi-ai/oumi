#!/usr/bin/env python3
"""Example: Computing true conversation-level token length.

This example demonstrates how to get the actual token length of a conversation
by tokenizing the entire conversation (with prompts, formatting, special tokens)
rather than just aggregating individual message token counts.
"""

import pandas as pd

from oumi.builders import build_tokenizer
from oumi.core.analyze import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, ModelParams
from oumi.core.configs.analyze_config import SampleAnalyzerParams


def example_conversation_tokenization():
    """Example of computing true conversation-level token length."""
    print("=== Conversation Tokenization Analysis ===\n")

    # Build a tokenizer
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
        sample_count=10,  # Small sample for demonstration
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

    print(f"Dataset: {analyzer.dataset_name}")
    print(f"Tokenizer: {type(analyzer.tokenizer).__name__}")
    print(f"Sample count: {config.sample_count}")

    # Run message-level analysis
    print("\nRunning message-level analysis...")
    analyzer.analyze_dataset()

    # Get results
    results = analyzer.analysis_results
    if results and results.messages:
        print(f"Analyzed {len(results.messages)} messages")

        # Convert to DataFrame for easier analysis
        results_df = results.to_dataframe()

        # Compute conversation-level metrics including true tokenization
        conversation_metrics = compute_conversation_metrics_with_tokenization(
            results_df, analyzer.dataset, analyzer.tokenizer
        )

        print(f"\nComputed metrics for {len(conversation_metrics)} conversations")

        # Display comparison between aggregated vs true tokenization
        print_conversation_tokenization_comparison(conversation_metrics)

        # Show detailed examples
        print_detailed_conversation_examples(
            conversation_metrics, analyzer.dataset, analyzer.tokenizer
        )

    return analyzer, conversation_metrics


def compute_conversation_metrics_with_tokenization(
    results_df: pd.DataFrame, dataset, tokenizer
) -> pd.DataFrame:
    """Compute conversation-level metrics including true conversation tokenization.

    Args:
        results_df: DataFrame with message-level analysis results
        dataset: The dataset object for accessing conversations
        tokenizer: The tokenizer for computing true conversation token counts

    Returns:
        DataFrame with conversation-level metrics including true tokenization
    """
    # Group by conversation_index
    conversation_groups = results_df.groupby("conversation_index")

    conversation_metrics = []

    for conv_idx, group in conversation_groups:
        # Basic conversation info
        conv_metrics = {
            "conversation_index": conv_idx,
            "conversation_id": group["conversation_id"].iloc[0],
            "message_count": len(group),
            "user_message_count": len(group[group["role"] == "user"]),
            "assistant_message_count": len(group[group["role"] == "assistant"]),
        }

        # Aggregated token metrics (from individual messages)
        if "length_token_count" in group.columns:
            conv_metrics.update(
                {
                    "aggregated_total_tokens": group["length_token_count"].sum(),
                    "aggregated_avg_tokens_per_message": group[
                        "length_token_count"
                    ].mean(),
                    "aggregated_max_tokens_in_message": group[
                        "length_token_count"
                    ].max(),
                    "aggregated_min_tokens_in_message": group[
                        "length_token_count"
                    ].min(),
                }
            )

        # True conversation tokenization
        try:
            # Get the conversation from the dataset
            conversation = dataset.conversation(conv_idx)

            # Tokenize the entire conversation
            tokenized = dataset.tokenize(conversation, tokenize=True)

            # Get the true token count
            if "input_ids" in tokenized and tokenized["input_ids"]:
                # input_ids is a list of integers representing token IDs
                true_token_count = len(tokenized["input_ids"])
            else:
                true_token_count = 0

            conv_metrics.update(
                {
                    "true_conversation_tokens": true_token_count,
                    "tokenization_difference": true_token_count
                    - conv_metrics.get("aggregated_total_tokens", 0),
                }
            )

        except Exception as e:
            print(f"Warning: Failed to tokenize conversation {conv_idx}: {e}")
            conv_metrics.update(
                {
                    "true_conversation_tokens": 0,
                    "tokenization_difference": 0,
                }
            )

        # Word and character metrics
        if "length_word_count" in group.columns:
            conv_metrics.update(
                {
                    "total_words": group["length_word_count"].sum(),
                    "avg_words_per_message": group["length_word_count"].mean(),
                }
            )

        if "length_char_count" in group.columns:
            conv_metrics.update(
                {
                    "total_chars": group["length_char_count"].sum(),
                    "avg_chars_per_message": group["length_char_count"].mean(),
                }
            )

        conversation_metrics.append(conv_metrics)

    return pd.DataFrame(conversation_metrics)


def print_conversation_tokenization_comparison(conversation_metrics: pd.DataFrame):
    """Print comparison between aggregated and true conversation tokenization."""
    print("\n=== Conversation Tokenization Comparison ===")

    if "true_conversation_tokens" in conversation_metrics.columns:
        print("\nToken Count Statistics:")
        print(
            f"  Total true conversation tokens: "
            f"{conversation_metrics['true_conversation_tokens'].sum():,}"
        )
        print(
            f"  Average true tokens per conversation: "
            f"{conversation_metrics['true_conversation_tokens'].mean():.1f}"
        )
        print(
            f"  Median true tokens per conversation: "
            f"{conversation_metrics['true_conversation_tokens'].median():.1f}"
        )
        print(
            f"  Min true tokens per conversation: "
            f"{conversation_metrics['true_conversation_tokens'].min()}"
        )
        print(
            f"  Max true tokens per conversation: "
            f"{conversation_metrics['true_conversation_tokens'].max()}"
        )

        if "aggregated_total_tokens" in conversation_metrics.columns:
            print("\nComparison (True vs Aggregated):")
            print(
                f"  Average difference: "
                f"{conversation_metrics['tokenization_difference'].mean():.1f} tokens"
            )
            print(
                f"  Median difference: "
                f"{conversation_metrics['tokenization_difference'].median():.1f} tokens"
            )
            print(
                f"  Min difference: "
                f"{conversation_metrics['tokenization_difference'].min()} tokens"
            )
            print(
                f"  Max difference: "
                f"{conversation_metrics['tokenization_difference'].max()} tokens"
            )

            # Show conversations with largest differences
            largest_diff = conversation_metrics.nlargest(3, "tokenization_difference")
            print("\nConversations with largest tokenization differences:")
            for i, (_, conv) in enumerate(largest_diff.iterrows()):
                print(f"  {i + 1}. Conversation {conv['conversation_id']}:")
                print(f"     True tokens: {conv['true_conversation_tokens']}")
                print(f"     Aggregated tokens: {conv['aggregated_total_tokens']}")
                print(f"     Difference: {conv['tokenization_difference']} tokens")


def print_detailed_conversation_examples(
    conversation_metrics: pd.DataFrame, dataset, tokenizer
):
    """Print detailed examples showing conversation tokenization."""
    print("\n=== Detailed Conversation Examples ===")

    # Show a few examples with different characteristics
    examples = conversation_metrics.head(3)

    for i, (_, conv) in enumerate(examples.iterrows()):
        conv_idx = conv["conversation_index"]
        print(f"\nExample {i + 1} - Conversation {conv['conversation_id']}:")
        print(f"  Messages: {conv['message_count']} total")
        print(f"    - User: {conv['user_message_count']}")
        print(f"    - Assistant: {conv['assistant_message_count']}")

        if "true_conversation_tokens" in conv:
            print("  Token Counts:")
            print(f"    - True conversation tokens: {conv['true_conversation_tokens']}")
            if "aggregated_total_tokens" in conv:
                print(
                    f"    - Aggregated message tokens: "
                    f"{conv['aggregated_total_tokens']}"
                )
                print(f"    - Difference: {conv['tokenization_difference']} tokens")

        # Show the actual conversation
        try:
            conversation = dataset.conversation(conv_idx)
            print("  Conversation preview:")

            # Show first few messages
            for j, message in enumerate(conversation.messages[:3]):
                role = message.role.value
                content = message.compute_flattened_text_content()
                print(f"    {j + 1}. {role}: {content[:100]}...")

            if len(conversation.messages) > 3:
                print(f"    ... and {len(conversation.messages) - 3} more messages")

        except Exception as e:
            print(f"    Error accessing conversation: {e}")


def example_tokenization_analysis():
    """Example analyzing tokenization patterns across conversations."""
    print("\n=== Tokenization Pattern Analysis ===\n")

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
        sample_count=30,
        analyzers=[
            SampleAnalyzerParams(
                id="length",
                params={
                    "token_count": True,
                    "word_count": True,
                },
            )
        ],
    )

    # Create analyzer
    analyzer = DatasetAnalyzer(config)

    # Run analysis
    print("Running analysis...")
    analyzer.analyze_dataset()

    # Get results and compute conversation metrics
    results = analyzer.analysis_results
    if results and results.messages:
        results_df = results.to_dataframe()
        conversation_metrics = compute_conversation_metrics_with_tokenization(
            results_df, analyzer.dataset, analyzer.tokenizer
        )

        # Analyze patterns
        print_tokenization_patterns(conversation_metrics)

    return analyzer, conversation_metrics


def print_tokenization_patterns(conversation_metrics: pd.DataFrame):
    """Print analysis of tokenization patterns."""
    print("\n=== Tokenization Pattern Analysis ===")

    if "true_conversation_tokens" not in conversation_metrics.columns:
        return

    # Analyze conversations by token count ranges
    ranges = [
        ("Short conversations (< 500 tokens)", "true_conversation_tokens < 500"),
        (
            "Medium conversations (500-1500 tokens)",
            "true_conversation_tokens >= 500 and true_conversation_tokens <= 1500",
        ),
        ("Long conversations (> 1500 tokens)", "true_conversation_tokens > 1500"),
    ]

    for description, query in ranges:
        try:
            filtered = conversation_metrics.query(query)
            print(f"\n{description}:")
            print(f"  Query: {query}")
            print(f"  Results: {len(filtered)} conversations")

            if len(filtered) > 0:
                print("  Token count stats:")
                print(f"    Mean: {filtered['true_conversation_tokens'].mean():.1f}")
                print(
                    f"    Median: {filtered['true_conversation_tokens'].median():.1f}"
                )
                print(f"    Min: {filtered['true_conversation_tokens'].min()}")
                print(f"    Max: {filtered['true_conversation_tokens'].max()}")

                if "tokenization_difference" in filtered.columns:
                    print("  Tokenization difference stats:")
                    print(
                        f"    Mean difference: "
                        f"{filtered['tokenization_difference'].mean():.1f}"
                    )
                    print(
                        f"    Median difference: "
                        f"{filtered['tokenization_difference'].median():.1f}"
                    )

        except Exception as e:
            print(f"\n{description}:")
            print(f"  Query failed: {e}")

    # Analyze conversations with large tokenization differences
    if "tokenization_difference" in conversation_metrics.columns:
        large_diff = conversation_metrics[
            conversation_metrics["tokenization_difference"] > 50
        ]
        print(
            f"\nConversations with large tokenization differences (> 50 tokens): "
            f"{len(large_diff)}"
        )

        if len(large_diff) > 0:
            print("Sample conversations with large differences:")
            for i, (_, conv) in enumerate(large_diff.head(3).iterrows()):
                print(f"  {i + 1}. Conversation {conv['conversation_id']}:")
                print(f"     True tokens: {conv['true_conversation_tokens']}")
                print(f"     Aggregated tokens: {conv['aggregated_total_tokens']}")
                print(f"     Difference: {conv['tokenization_difference']} tokens")
                print(f"     Messages: {conv['message_count']}")


if __name__ == "__main__":
    # Run examples
    analyzer1, conv_metrics1 = example_conversation_tokenization()
    analyzer2, conv_metrics2 = example_tokenization_analysis()

    print("\n" + "=" * 60)
    print("Conversation tokenization analysis completed!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print(
        "- True conversation tokenization includes prompts, formatting, "
        "and special tokens"
    )
    print("- Significant differences between aggregated and true token counts")
    print("- Important for model training and inference context windows")
    print("- Conversation-level analysis should use true tokenization")
    print("- Message-level analysis still useful for detailed breakdowns")
