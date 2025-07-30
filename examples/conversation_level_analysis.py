#!/usr/bin/env python3
"""Example: Computing conversation-level metrics with DatasetAnalyzer.

This example demonstrates how to extend the DatasetAnalyzer to compute
conversation-level metrics, such as total token count per conversation,
average tokens per message, etc.
"""

import pandas as pd

from oumi.builders import build_tokenizer
from oumi.core.analyze import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, ModelParams
from oumi.core.configs.analyze_config import SampleAnalyzerParams


def example_conversation_level_analysis():
    """Example of computing conversation-level metrics from message-level results."""
    print("=== Conversation-Level Analysis Example ===\n")

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
        sample_count=20,  # Small sample for demonstration
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

    # Run analysis
    print("\nRunning analysis...")
    analyzer.analyze_dataset()

    # Get results
    results = analyzer.analysis_results
    if results and results.messages:
        print(f"Analyzed {len(results.messages)} messages")

        # Convert to DataFrame for easier analysis
        results_df = results.to_dataframe()

        # Group by conversation to compute conversation-level metrics
        conversation_metrics = compute_conversation_metrics(results_df)

        print(f"\nComputed metrics for {len(conversation_metrics)} conversations")

        # Display conversation-level statistics
        print_conversation_statistics(conversation_metrics)

        # Show sample conversations with their metrics
        print_sample_conversations(conversation_metrics, results_df)

    return analyzer, conversation_metrics


def compute_conversation_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute conversation-level metrics from message-level results.

    Args:
        results_df: DataFrame with message-level analysis results

    Returns:
        DataFrame with conversation-level metrics
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

        # Token metrics
        if "length_token_count" in group.columns:
            conv_metrics.update(
                {
                    "total_tokens": group["length_token_count"].sum(),
                    "avg_tokens_per_message": group["length_token_count"].mean(),
                    "max_tokens_in_message": group["length_token_count"].max(),
                    "min_tokens_in_message": group["length_token_count"].min(),
                    "token_std": group["length_token_count"].std(),
                }
            )

            # Token metrics by role
            user_messages = group[group["role"] == "user"]
            assistant_messages = group[group["role"] == "assistant"]

            if len(user_messages) > 0:
                conv_metrics.update(
                    {
                        "user_total_tokens": user_messages["length_token_count"].sum(),
                        "user_avg_tokens": user_messages["length_token_count"].mean(),
                        "user_max_tokens": user_messages["length_token_count"].max(),
                    }
                )
            else:
                conv_metrics.update(
                    {
                        "user_total_tokens": 0,
                        "user_avg_tokens": 0,
                        "user_max_tokens": 0,
                    }
                )

            if len(assistant_messages) > 0:
                conv_metrics.update(
                    {
                        "assistant_total_tokens": assistant_messages[
                            "length_token_count"
                        ].sum(),
                        "assistant_avg_tokens": assistant_messages[
                            "length_token_count"
                        ].mean(),
                        "assistant_max_tokens": assistant_messages[
                            "length_token_count"
                        ].max(),
                    }
                )
            else:
                conv_metrics.update(
                    {
                        "assistant_total_tokens": 0,
                        "assistant_avg_tokens": 0,
                        "assistant_max_tokens": 0,
                    }
                )

        # Word metrics
        if "length_word_count" in group.columns:
            conv_metrics.update(
                {
                    "total_words": group["length_word_count"].sum(),
                    "avg_words_per_message": group["length_word_count"].mean(),
                    "max_words_in_message": group["length_word_count"].max(),
                    "min_words_in_message": group["length_word_count"].min(),
                }
            )

        # Character metrics
        if "length_char_count" in group.columns:
            conv_metrics.update(
                {
                    "total_chars": group["length_char_count"].sum(),
                    "avg_chars_per_message": group["length_char_count"].mean(),
                    "max_chars_in_message": group["length_char_count"].max(),
                    "min_chars_in_message": group["length_char_count"].min(),
                }
            )

        conversation_metrics.append(conv_metrics)

    return pd.DataFrame(conversation_metrics)


def print_conversation_statistics(conversation_metrics: pd.DataFrame):
    """Print statistics for conversation-level metrics."""
    print("\n=== Conversation-Level Statistics ===")

    if "total_tokens" in conversation_metrics.columns:
        print("\nToken Statistics:")
        print(
            f"  Total tokens across all conversations: "
            f"{conversation_metrics['total_tokens'].sum():,}"
        )
        print(
            f"  Average tokens per conversation: "
            f"{conversation_metrics['total_tokens'].mean():.1f}"
        )
        print(
            f"  Median tokens per conversation: "
            f"{conversation_metrics['total_tokens'].median():.1f}"
        )
        print(
            f"  Min tokens per conversation: "
            f"{conversation_metrics['total_tokens'].min()}"
        )
        print(
            f"  Max tokens per conversation: "
            f"{conversation_metrics['total_tokens'].max()}"
        )

        # Average tokens per message type
        print(f"  User messages: {conversation_metrics['user_avg_tokens'].mean():.1f}")
        print(
            f"  Assistant messages: "
            f"{conversation_metrics['assistant_avg_tokens'].mean():.1f}"
        )

        # Word statistics
        print("\nWord Statistics:")
        print(
            f"  Total words across all conversations: "
            f"{conversation_metrics['total_words'].sum():,}"
        )
        print(
            f"  Average words per conversation: "
            f"{conversation_metrics['total_words'].mean():.1f}"
        )
        print(
            f"  Average words per message: "
            f"{conversation_metrics['avg_words_per_message'].mean():.1f}"
        )

    print("\nMessage Count Statistics:")
    print(
        f"  Average messages per conversation: "
        f"{conversation_metrics['message_count'].mean():.1f}"
    )
    print(
        f"  Average user messages per conversation: "
        f"{conversation_metrics['user_message_count'].mean():.1f}"
    )
    print(
        f"  Average assistant messages per conversation: "
        f"{conversation_metrics['assistant_message_count'].mean():.1f}"
    )


def print_sample_conversations(
    conversation_metrics: pd.DataFrame, results_df: pd.DataFrame
):
    """Print sample conversations with their metrics."""
    print("\n=== Sample Conversations ===")

    # Sort by total tokens to show variety
    sorted_convs = conversation_metrics.sort_values("total_tokens", ascending=False)

    for i, (_, conv) in enumerate(sorted_convs.head(5).iterrows()):
        print(f"\nConversation {i + 1} (ID: {conv['conversation_id']}):")
        print(f"  Messages: {conv['message_count']} total")
        print(f"    - User: {conv['user_message_count']}")
        print(f"    - Assistant: {conv['assistant_message_count']}")

        if "total_tokens" in conv:
            print(f"  Tokens: {conv['total_tokens']} total")
            print(f"    - Average per message: {conv['avg_tokens_per_message']:.1f}")
            print(f"    - User total: {conv['user_total_tokens']}")
            print(f"    - Assistant total: {conv['assistant_total_tokens']}")
            print(
                f"    - Range: {conv['min_tokens_in_message']} - "
                f"{conv['max_tokens_in_message']}"
            )

        if "total_words" in conv:
            print(f"  Words: {conv['total_words']} total")
            print(f"    - Average per message: {conv['avg_words_per_message']:.1f}")

        # Show first message as preview
        conv_messages = results_df[
            results_df["conversation_index"] == conv["conversation_index"]
        ]
        if len(conv_messages) > 0:
            first_msg = conv_messages.iloc[0]
            print(
                f"  First message ({first_msg['role']}): "
                f"{first_msg['text_content'][:80]}..."
            )


def example_query_conversations_by_token_count():
    """Example of querying conversations based on token count."""
    print("\n=== Querying Conversations by Token Count ===\n")

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
        sample_count=50,
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
        conversation_metrics = compute_conversation_metrics(results_df)

        # Query examples for conversations
        queries = [
            ("Short conversations (< 100 total tokens)", "total_tokens < 100"),
            (
                "Medium conversations (100-500 tokens)",
                "total_tokens >= 100 and total_tokens <= 500",
            ),
            ("Long conversations (> 500 tokens)", "total_tokens > 500"),
            ("Conversations with many messages (> 5 messages)", "message_count > 5"),
            (
                "Conversations with long user messages (avg > 50 tokens)",
                "user_avg_tokens > 50",
            ),
            (
                "Conversations with short assistant messages (avg < 20 tokens)",
                "assistant_avg_tokens < 20",
            ),
        ]

        for description, query in queries:
            try:
                filtered_convs = conversation_metrics.query(query)
                print(f"\n{description}:")
                print(f"  Query: {query}")
                print(f"  Results: {len(filtered_convs)} conversations")

                if len(filtered_convs) > 0:
                    # Show statistics
                    if "total_tokens" in filtered_convs.columns:
                        print("  Token count stats:")
                        print(f"    Mean: {filtered_convs['total_tokens'].mean():.1f}")
                        print(
                            f"    Median: {filtered_convs['total_tokens'].median():.1f}"
                        )
                        print(f"    Min: {filtered_convs['total_tokens'].min()}")
                        print(f"    Max: {filtered_convs['total_tokens'].max()}")

                    # Show sample conversation
                    sample = filtered_convs.iloc[0]
                    print("  Sample conversation:")
                    print(f"    ID: {sample['conversation_id']}")
                    print(f"    Messages: {sample['message_count']}")
                    if "total_tokens" in sample:
                        print(f"    Total tokens: {sample['total_tokens']}")

            except Exception as e:
                print(f"\n{description}:")
                print(f"  Query failed: {e}")

    return analyzer, conversation_metrics


if __name__ == "__main__":
    # Run examples
    analyzer1, conv_metrics1 = example_conversation_level_analysis()
    analyzer2, conv_metrics2 = example_query_conversations_by_token_count()

    print("\n" + "=" * 60)
    print("Conversation-level analysis examples completed!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print("- Conversation-level metrics computed from message-level results")
    print("- Total tokens, average tokens per message, role-specific metrics")
    print("- Query capabilities for conversation-level analysis")
    print("- Maintains all existing message-level functionality")
    print("- Easy to extend for additional conversation-level metrics")
