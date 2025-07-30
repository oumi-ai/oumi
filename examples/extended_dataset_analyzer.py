#!/usr/bin/env python3
"""Example: Extending DatasetAnalyzer with conversation-level metrics.

This example demonstrates how to extend the DatasetAnalyzer class to include
conversation-level metrics as a built-in feature, rather than computing them
separately from message-level results.
"""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from oumi.builders import build_tokenizer
from oumi.core.analyze import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, ModelParams
from oumi.core.configs.analyze_config import SampleAnalyzerParams


@dataclass
class ConversationAnalysisResult:
    """Result of analyzing a single conversation.

    Attributes:
        conversation_id: Unique identifier for the conversation
        conversation_index: Index of the conversation in the dataset
        message_count: Total number of messages in the conversation
        user_message_count: Number of user messages
        assistant_message_count: Number of assistant messages
        analyzer_metrics: Dictionary of conversation-level metrics computed by analyzers
    """

    ANALYZER_METRICS_FIELD = "analyzer_metrics"

    conversation_id: str
    conversation_index: int
    message_count: int
    user_message_count: int
    assistant_message_count: int
    analyzer_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with flattened analyzer metrics."""
        base_dict = {
            "conversation_id": self.conversation_id,
            "conversation_index": self.conversation_index,
            "message_count": self.message_count,
            "user_message_count": self.user_message_count,
            "assistant_message_count": self.assistant_message_count,
        }
        # Flatten analyzer_metrics into the main dict
        analyzer_metrics = getattr(self, self.ANALYZER_METRICS_FIELD, {})
        base_dict.update(analyzer_metrics)
        return base_dict


class ExtendedDatasetAnalyzer(DatasetAnalyzer):
    """Extended DatasetAnalyzer that includes conversation-level metrics."""

    def __init__(self, config: AnalyzeConfig):
        """Initialize the extended dataset analyzer."""
        super().__init__(config)
        self._conversation_results: Optional[list[ConversationAnalysisResult]] = None

    def analyze_dataset(self) -> None:
        """Analyze the dataset and compute both message and conversation metrics."""
        # First run the original message-level analysis
        super().analyze_dataset()

        # Then compute conversation-level metrics
        self._compute_conversation_metrics()

    def _compute_conversation_metrics(self) -> None:
        """Compute conversation-level metrics from message-level results."""
        if self._analysis_results is None:
            raise ValueError("Message-level analysis must be run first")

        logger.info("Step 2: Computing conversation-level metrics...")

        # Convert message results to DataFrame for easier grouping
        message_df = self._analysis_results.to_dataframe()

        # Group by conversation
        conversation_groups = message_df.groupby("conversation_index")

        conversation_results = []

        for conv_idx, group in conversation_groups:
            # Basic conversation info
            conversation_id = group["conversation_id"].iloc[0]
            message_count = len(group)
            user_message_count = len(group[group["role"] == "user"])
            assistant_message_count = len(group[group["role"] == "assistant"])

            # Ensure conv_idx is an integer
            conv_idx_int = int(conv_idx)

            # Compute conversation-level metrics using analyzers
            analyzer_metrics = {}
            for analyzer_id, analyzer in self.sample_analyzers.items():
                try:
                    # Get all messages for this conversation
                    conversation_texts = group["text_content"].tolist()
                    conversation_roles = group["role"].tolist()

                    # Compute conversation-level metrics
                    conv_metrics = self._compute_conversation_analyzer_metrics(
                        analyzer, conversation_texts, conversation_roles, analyzer_id
                    )
                    analyzer_metrics.update(conv_metrics)
                except Exception as e:
                    logger.warning(
                        f"Conversation analyzer {analyzer_id} failed for conversation "
                        f"{conv_idx}: {e}"
                    )

            conversation_result = ConversationAnalysisResult(
                conversation_id=conversation_id,
                conversation_index=conv_idx_int,
                message_count=message_count,
                user_message_count=user_message_count,
                assistant_message_count=assistant_message_count,
                **{ConversationAnalysisResult.ANALYZER_METRICS_FIELD: analyzer_metrics},
            )

            conversation_results.append(conversation_result)

        self._conversation_results = conversation_results
        logger.info(
            f"Computed conversation-level metrics for "
            f"{len(conversation_results)} conversations"
        )

    def _compute_conversation_analyzer_metrics(
        self,
        analyzer,
        conversation_texts: list[str],
        conversation_roles: list[str],
        analyzer_id: str,
    ) -> dict[str, Any]:
        """Compute conversation-level metrics for a specific analyzer."""
        metrics = {}

        # For length analyzer, compute conversation-level metrics
        if analyzer_id == "length":
            # Get token counts if available
            token_counts = []
            word_counts = []
            char_counts = []

            for text in conversation_texts:
                # Use the analyzer to get metrics for each message
                message_metrics = analyzer.analyze_message(text)

                if "token_count" in message_metrics:
                    token_counts.append(message_metrics["token_count"])
                if "word_count" in message_metrics:
                    word_counts.append(message_metrics["word_count"])
                if "char_count" in message_metrics:
                    char_counts.append(message_metrics["char_count"])

            # Compute conversation-level statistics
            if token_counts:
                metrics.update(
                    {
                        "total_tokens": sum(token_counts),
                        "avg_tokens_per_message": sum(token_counts) / len(token_counts),
                        "max_tokens_in_message": max(token_counts),
                        "min_tokens_in_message": min(token_counts),
                        "token_std": pd.Series(token_counts).std(),
                    }
                )

                # Role-specific token metrics
                user_tokens = []
                assistant_tokens = []
                for i, role in enumerate(conversation_roles):
                    if role == "user":
                        user_tokens.append(token_counts[i])
                    elif role == "assistant":
                        assistant_tokens.append(token_counts[i])

                if user_tokens:
                    metrics.update(
                        {
                            "user_total_tokens": sum(user_tokens),
                            "user_avg_tokens": sum(user_tokens) / len(user_tokens),
                            "user_max_tokens": max(user_tokens),
                        }
                    )
                else:
                    metrics.update(
                        {
                            "user_total_tokens": 0,
                            "user_avg_tokens": 0,
                            "user_max_tokens": 0,
                        }
                    )

                if assistant_tokens:
                    metrics.update(
                        {
                            "assistant_total_tokens": sum(assistant_tokens),
                            "assistant_avg_tokens": sum(assistant_tokens)
                            / len(assistant_tokens),
                            "assistant_max_tokens": max(assistant_tokens),
                        }
                    )
                else:
                    metrics.update(
                        {
                            "assistant_total_tokens": 0,
                            "assistant_avg_tokens": 0,
                            "assistant_max_tokens": 0,
                        }
                    )

            if word_counts:
                metrics.update(
                    {
                        "total_words": sum(word_counts),
                        "avg_words_per_message": sum(word_counts) / len(word_counts),
                        "max_words_in_message": max(word_counts),
                        "min_words_in_message": min(word_counts),
                    }
                )

            if char_counts:
                metrics.update(
                    {
                        "total_chars": sum(char_counts),
                        "avg_chars_per_message": sum(char_counts) / len(char_counts),
                        "max_chars_in_message": max(char_counts),
                        "min_chars_in_message": min(char_counts),
                    }
                )

        return metrics

    @property
    def conversation_results(self) -> Optional[list[ConversationAnalysisResult]]:
        """Get the conversation-level analysis results."""
        return self._conversation_results

    def conversation_results_to_dataframe(self) -> pd.DataFrame:
        """Convert conversation results to a pandas DataFrame."""
        if self._conversation_results is None:
            raise ValueError("Conversation analysis not yet run")

        conversation_dicts = [conv.to_dict() for conv in self._conversation_results]
        return pd.DataFrame(conversation_dicts)

    def query_conversations(self, query_expression: str) -> pd.DataFrame:
        """Query conversation-level results using pandas query expression."""
        if self._conversation_results is None:
            logger.info("Conversation analysis not yet run, starting analysis...")
            self.analyze_dataset()

        conversation_df = self.conversation_results_to_dataframe()

        try:
            filtered_df = conversation_df.query(query_expression)
            logger.info(
                f"Conversation query '{query_expression}' returned "
                f"{len(filtered_df)} rows"
            )
        except Exception as e:
            logger.error(f"Conversation query failed: {e}")
            raise ValueError(
                f"Invalid conversation query expression '{query_expression}': {e}"
            )

        return filtered_df


def example_extended_analyzer():
    """Example using the extended DatasetAnalyzer with conversation-level metrics."""
    print("=== Extended DatasetAnalyzer Example ===\n")

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
        sample_count=15,
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

    # Create extended analyzer
    analyzer = ExtendedDatasetAnalyzer(config)

    print(f"Dataset: {analyzer.dataset_name}")
    print(f"Tokenizer: {type(analyzer.tokenizer).__name__}")
    print(f"Sample count: {config.sample_count}")

    # Run analysis (both message and conversation level)
    print("\nRunning analysis...")
    analyzer.analyze_dataset()

    # Get both message and conversation results
    message_results = analyzer.analysis_results
    conversation_results = analyzer.conversation_results

    if message_results and conversation_results:
        print(f"Analyzed {len(message_results.messages)} messages")
        print(f"Analyzed {len(conversation_results)} conversations")

        # Convert to DataFrames
        message_df = message_results.to_dataframe()
        conversation_df = analyzer.conversation_results_to_dataframe()

        print("\n=== Message-Level Results ===")
        print(f"Message DataFrame shape: {message_df.shape}")
        print(f"Columns: {list(message_df.columns)}")

        print("\n=== Conversation-Level Results ===")
        print(f"Conversation DataFrame shape: {conversation_df.shape}")
        print(f"Columns: {list(conversation_df.columns)}")

        # Show conversation statistics
        if "total_tokens" in conversation_df.columns:
            print("\nConversation Token Statistics:")
            print(
                f"  Total tokens across all conversations: "
                f"{conversation_df['total_tokens'].sum():,}"
            )
            print(
                f"  Average tokens per conversation: "
                f"{conversation_df['total_tokens'].mean():.1f}"
            )
            print(
                f"  Median tokens per conversation: "
                f"{conversation_df['total_tokens'].median():.1f}"
            )
            print(
                f"  Min tokens per conversation: "
                f"{conversation_df['total_tokens'].min()}"
            )
            print(
                f"  Max tokens per conversation: "
                f"{conversation_df['total_tokens'].max()}"
            )

        # Query examples
        print("\n=== Query Examples ===")

        # Message-level queries
        short_messages = analyzer.query("length_token_count < 20")
        print(f"Short messages (< 20 tokens): {len(short_messages)} messages")

        # Conversation-level queries
        long_conversations = analyzer.query_conversations("total_tokens > 1000")
        print(
            f"Long conversations (> 1000 tokens): "
            f"{len(long_conversations)} conversations"
        )

        many_messages = analyzer.query_conversations("message_count > 5")
        print(
            f"Conversations with many messages (> 5): "
            f"{len(many_messages)} conversations"
        )

    return analyzer


def example_conversation_queries():
    """Example of various conversation-level queries."""
    print("\n=== Conversation Query Examples ===\n")

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

    # Create extended analyzer
    analyzer = ExtendedDatasetAnalyzer(config)

    # Run analysis
    print("Running analysis...")
    analyzer.analyze_dataset()

    # Query examples
    queries = [
        ("Short conversations (< 500 tokens)", "total_tokens < 500"),
        (
            "Medium conversations (500-1500 tokens)",
            "total_tokens >= 500 and total_tokens <= 1500",
        ),
        ("Long conversations (> 1500 tokens)", "total_tokens > 1500"),
        ("Conversations with many messages (> 6)", "message_count > 6"),
        ("Conversations with few messages (≤ 3)", "message_count <= 3"),
        (
            "Conversations with long user messages (avg > 100 tokens)",
            "user_avg_tokens > 100",
        ),
        (
            "Conversations with short assistant messages (avg < 50 tokens)",
            "assistant_avg_tokens < 50",
        ),
        (
            "Balanced conversations (similar user/assistant counts)",
            "abs(user_message_count - assistant_message_count) <= 1",
        ),
    ]

    for description, query in queries:
        try:
            results = analyzer.query_conversations(query)
            print(f"\n{description}:")
            print(f"  Query: {query}")
            print(f"  Results: {len(results)} conversations")

            if len(results) > 0:
                if "total_tokens" in results.columns:
                    print("  Token count stats:")
                    print(f"    Mean: {results['total_tokens'].mean():.1f}")
                    print(f"    Median: {results['total_tokens'].median():.1f}")
                    print(f"    Min: {results['total_tokens'].min()}")
                    print(f"    Max: {results['total_tokens'].max()}")

                # Show sample conversation
                sample = results.iloc[0]
                print("  Sample conversation:")
                print(f"    ID: {sample['conversation_id']}")
                print(f"    Messages: {sample['message_count']}")
                if "total_tokens" in sample:
                    print(f"    Total tokens: {sample['total_tokens']}")

        except Exception as e:
            print(f"\n{description}:")
            print(f"  Query failed: {e}")

    return analyzer


if __name__ == "__main__":
    # Import logger for the extended analyzer
    from oumi.utils.logging import logger

    # Run examples
    analyzer1 = example_extended_analyzer()
    analyzer2 = example_conversation_queries()

    print("\n" + "=" * 60)
    print("Extended DatasetAnalyzer examples completed!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print("- Extended DatasetAnalyzer with built-in conversation-level metrics")
    print("- Automatic computation of conversation statistics")
    print("- Query capabilities for both message and conversation levels")
    print("- Maintains all existing functionality")
    print("- Easy to extend with additional conversation-level analyzers")
