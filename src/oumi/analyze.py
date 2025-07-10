# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from oumi.core.configs import AnalyzerConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.logging import logger


def _generate_timestamped_filename(prefix: str, save_format: str) -> str:
    """Generate a timestamped filename with the specified format.

    Args:
        prefix: The filename prefix
        save_format: The file format (json, yaml, csv, parquet)

    Returns:
        Timestamped filename with extension
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{save_format}"


def _load_dataset_from_config(config: AnalyzerConfig) -> BaseMapDataset:
    """Load dataset based on configuration.

    Currently only supports datasets registered in the REGISTRY.
    TODO: Add support for loading datasets from HuggingFace Hub.
    TODO: Add support for loading custom datasets from local file paths.
    """
    input_config = config.input
    dataset_name = input_config.name

    if not dataset_name:
        raise ValueError("Dataset name is required")

    try:
        # Load dataset from the REGISTRY
        from oumi.core.registry import REGISTRY

        dataset_class = REGISTRY.get_dataset(dataset_name)

        if dataset_class is not None:
            # Load registered dataset
            return dataset_class(split=input_config.split)
        else:
            # TODO: Implement HuggingFace Hub loading
            raise NotImplementedError(
                f"Dataset '{dataset_name}' is not registered in the REGISTRY. "
                "Loading from HuggingFace Hub is not yet implemented."
            )

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


class Analyzer:
    """Base class for dataset analysis functionality."""

    def __init__(self, config: AnalyzerConfig):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzerConfig object containing all analysis parameters
        """
        self.config = config
        self.dataset_name = config.input.name
        self.split = config.input.split
        self.dataset = _load_dataset_from_config(config)

    def get_conversation(self, index: int = 0):
        """Get a conversation from the dataset.

        Args:
            index: Index of the conversation to retrieve.

        Returns:
            The conversation at the specified index.
        """
        return self.dataset.conversation(index)

    def get_conversation_length(self, index: int = 0) -> int:
        """Get the length (number of messages) of a conversation.

        Args:
            index: Index of the conversation to check.

        Returns:
            int: Number of messages in the conversation.
        """
        conversation = self.get_conversation(index)
        return len(conversation.messages)

    def get_dataset_size(self) -> int:
        """Get the total number of conversations in the dataset.

        Returns:
            int: Total number of conversations.
        """
        return len(self.dataset)

    def print_conversation(self, index: int = 0):
        """Print a conversation from the dataset.

        Args:
            index: Index of the conversation to print.
        """
        conversation = self.get_conversation(index)
        print(f"Conversation {index} from {self.dataset_name} dataset:")
        print("=" * 50)
        print(repr(conversation))
        print("=" * 50)
        return conversation

    def _save_results(
        self, results: dict[str, Any], output_path: str, save_format: str
    ):
        """Save analysis results to file based on the specified format.

        Args:
            results: Analysis results dictionary to save
            output_path: Path where to save the results
            save_format: Format to save the results (json, yaml, csv, parquet)
        """
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if save_format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif save_format == "yaml":
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        elif save_format == "csv":
            # For CSV, we'll save basic stats in a tabular format
            import csv

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerow(
                    ["dataset_name", results.get("dataset_name", "Unknown")]
                )

                # Handle different result types
                if (
                    "sample_level_results" in results
                    and "aggregation_results" in results
                ):
                    # Combined results from two-step analysis
                    sample_results = results["sample_level_results"]
                    aggregation_results = results["aggregation_results"]

                    writer.writerow(
                        [
                            "total_conversations",
                            sample_results.get("total_conversations", 0),
                        ]
                    )
                    writer.writerow(
                        ["total_messages", sample_results.get("total_messages", 0)]
                    )

                    # Message level stats
                    message_stats = sample_results.get("message_level_stats", {})
                    writer.writerow(
                        [
                            "avg_message_length",
                            f"{message_stats.get('avg_message_length', 0):.2f}",
                        ]
                    )

                    # Aggregation stats
                    if "conversation_stats" in aggregation_results:
                        stats = aggregation_results["conversation_stats"]
                        writer.writerow(["min_turns", stats.get("min_turns", 0)])
                        writer.writerow(["max_turns", stats.get("max_turns", 0)])
                        writer.writerow(
                            ["mean_turns", f"{stats.get('mean_turns', 0):.2f}"]
                        )
                        writer.writerow(["median_turns", stats.get("median_turns", 0)])
                else:
                    # Single result type (either sample or aggregation)
                    writer.writerow(
                        ["total_conversations", results.get("total_conversations", 0)]
                    )
                    writer.writerow(
                        ["total_messages", results.get("total_messages", 0)]
                    )

                    if "conversation_stats" in results:
                        stats = results["conversation_stats"]
                        writer.writerow(["min_turns", stats.get("min_turns", 0)])
                        writer.writerow(["max_turns", stats.get("max_turns", 0)])
                        writer.writerow(
                            [
                                "mean_turns",
                                f"{stats.get('mean_turns', 0):.2f}",
                            ]
                        )
                        writer.writerow(["median_turns", stats.get("median_turns", 0)])
        elif save_format == "parquet":
            # For parquet, we'll save the results as a structured format
            import pandas as pd

            # Handle different result types
            if "sample_level_results" in results and "aggregation_results" in results:
                # Combined results from two-step analysis
                sample_results = results["sample_level_results"]
                aggregation_results = results["aggregation_results"]

                data = {
                    "dataset_name": [results.get("dataset_name", "Unknown")],
                    "total_conversations": [
                        sample_results.get("total_conversations", 0)
                    ],
                    "total_messages": [sample_results.get("total_messages", 0)],
                }

                # Message level stats
                message_stats = sample_results.get("message_level_stats", {})
                data.update(
                    {
                        "avg_message_length": [
                            message_stats.get("avg_message_length", 0)
                        ],
                    }
                )

                # Aggregation stats
                if "conversation_stats" in aggregation_results:
                    stats = aggregation_results["conversation_stats"]
                    data.update(
                        {
                            "min_turns": [stats.get("min_turns", 0)],
                            "max_turns": [stats.get("max_turns", 0)],
                            "mean_turns": [stats.get("mean_turns", 0)],
                            "median_turns": [stats.get("median_turns", 0)],
                        }
                    )

            else:
                # Single result type (either sample or aggregation)
                data = {
                    "dataset_name": [results.get("dataset_name", "Unknown")],
                    "total_conversations": [results.get("total_conversations", 0)],
                    "total_messages": [results.get("total_messages", 0)],
                }

                if "conversation_stats" in results:
                    stats = results["conversation_stats"]
                    data.update(
                        {
                            "min_turns": [stats.get("min_turns", 0)],
                            "max_turns": [stats.get("max_turns", 0)],
                            "mean_turns": [stats.get("mean_turns", 0)],
                            "median_turns": [stats.get("median_turns", 0)],
                        }
                    )

            dataframe = pd.DataFrame(data)
            dataframe.to_parquet(output_file, index=False)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

        if self.config.verbose:
            logger.info(f"Results saved to: {output_file}")

    def analyze_dataset(self) -> dict[str, Any]:
        """Analyze the dataset and return analysis results.

        This method performs a two-step analysis:
        1. Per-sample (message) level analysis
        2. Aggregation at conversation and global levels

        Returns:
            Dict[str, Any]: Analysis results containing various metrics and insights.
        """
        verbose = self.config.verbose

        if verbose:
            logger.info(f"Starting analysis of dataset: {self.dataset_name}")

        # Step 1: Per-sample (message) level analysis
        if verbose:
            logger.info("Step 1: Computing per-sample (message) level analysis...")

        sample_results = self._compute_sample_level_analysis()

        # Save sample-level results
        if hasattr(self.config, "outputs") and self.config.outputs.analysis_output:
            sample_output_path = _generate_timestamped_filename(
                f"{self.config.outputs.analysis_output}_sample_level",
                self.config.outputs.save_format,
            )

            # Combine path with filename
            full_sample_path = Path(self.config.outputs.path) / sample_output_path

            self._save_results(
                sample_results,
                str(full_sample_path),
                self.config.outputs.save_format,
            )

            if verbose:
                logger.info(f"Sample-level results saved to: {sample_output_path}")

        # Step 2: Aggregation at conversation and global levels
        if verbose:
            logger.info("Step 2: Computing aggregation analysis...")

        aggregation_results = self._compute_aggregation_analysis(sample_results)

        # Save aggregation results
        if hasattr(self.config, "outputs") and self.config.outputs.aggregation_output:
            # Combine path with filename
            aggregation_filename = _generate_timestamped_filename(
                self.config.outputs.aggregation_output, self.config.outputs.save_format
            )
            full_aggregation_path = (
                Path(self.config.outputs.path) / aggregation_filename
            )

            self._save_results(
                aggregation_results,
                str(full_aggregation_path),
                self.config.outputs.save_format,
            )

            if verbose:
                logger.info(f"Aggregation results saved to: {aggregation_filename}")

        # Combine results for return
        final_results = {
            "dataset_name": self.dataset_name,
            "sample_level_results": sample_results,
            "aggregation_results": aggregation_results,
        }

        return final_results

    def _compute_sample_level_analysis(self) -> dict[str, Any]:
        """Perform per-sample (message) level analysis."""
        verbose = self.config.verbose
        total_conversations = len(self.dataset)

        # Apply conversation limit if specified
        max_conversations = self.config.input.max_conversations
        if max_conversations is not None and max_conversations > 0:
            conversations_to_analyze = min(total_conversations, max_conversations)
            if verbose:
                logger.info(
                    f"Limiting analysis to first {max_conversations} "
                    f"conversations (dataset has {total_conversations} total)"
                )
        else:
            conversations_to_analyze = total_conversations

        if verbose:
            logger.info(
                "Analyzing %d conversations for sample-level metrics",
                conversations_to_analyze,
            )

        # Collect all messages with their metadata
        messages_data = []

        for conv_idx in range(conversations_to_analyze):
            conversation = self.get_conversation(conv_idx)

            for msg_idx, message in enumerate(conversation.messages):
                # Get text content (simplified for text-only)
                if isinstance(message.content, str):
                    text_content = message.content
                else:
                    # For multimodal content, extract text only
                    text_content = message.compute_flattened_text_content()

                # Basic message information
                message_data = {
                    "conversation_id": conversation.conversation_id
                    or f"conv_{conv_idx}",
                    "conversation_index": conv_idx,
                    "message_index": msg_idx,
                    "message_id": message.id or f"msg_{conv_idx}_{msg_idx}",
                    "role": message.role.value,
                    "text_content": text_content,
                }

                # Compute configured metrics
                message_data.update(self._compute_message_metrics(text_content))

                messages_data.append(message_data)

        sample_results = {
            "dataset_name": self.dataset_name,
            "total_conversations": total_conversations,
            "conversations_analyzed": conversations_to_analyze,
            "total_messages": len(messages_data),
            "messages": messages_data,
            "message_level_stats": self._compute_message_level_stats(messages_data),
        }

        return sample_results

    def _compute_message_metrics(self, text_content: str) -> dict[str, Any]:
        """Compute metrics for a single message based on configuration."""
        metrics = {}

        # Process each metric category
        sample_metrics = self.config.sample_level_metrics

        # Language metrics
        if sample_metrics.language.enabled:
            try:
                language_metrics = self._compute_language_metrics(
                    text_content, sample_metrics.language
                )
                metrics.update(language_metrics)
            except Exception as e:
                logger.warning(f"Failed to compute language metrics: {e}")

        # Length metrics
        if sample_metrics.length.enabled:
            try:
                length_metrics = self._compute_length_metrics(
                    text_content, sample_metrics.length
                )
                metrics.update(length_metrics)
            except Exception as e:
                logger.warning(f"Failed to compute length metrics: {e}")

        # Safety metrics
        if sample_metrics.safety.enabled:
            try:
                safety_metrics = self._compute_safety_metrics(
                    text_content, sample_metrics.safety
                )
                metrics.update(safety_metrics)
            except Exception as e:
                logger.warning(f"Failed to compute safety metrics: {e}")

        return metrics

    def _compute_language_metrics(
        self, text_content: str, language_config
    ) -> dict[str, Any]:
        """Compute language-related metrics for the given text content."""
        # Placeholder for language detection - will be implemented later
        return {
            "primary_language": "unknown",
            "confidence": 0.0,
            "top_languages": [],
            "is_multilingual": False,
        }

    def _compute_length_metrics(
        self, text_content: str, length_config
    ) -> dict[str, Any]:
        """Compute length-related metrics for the given text content."""
        metrics = {}

        if length_config.char_count:
            metrics["length_char_count"] = len(text_content)

        if length_config.word_count:
            metrics["length_word_count"] = len(text_content.split())

        if length_config.sentence_count:
            # Simple sentence counting - can be improved later
            sentences = text_content.replace("!", ".").replace("?", ".").split(".")
            metrics["length_sentence_count"] = len([s for s in sentences if s.strip()])

        if length_config.token_count:
            # Simple token counting - split by whitespace
            # TODO: Implement proper tokenization using a tokenizer
            tokens = text_content.split()
            metrics["length_token_count"] = len(tokens)

        return metrics

    def _compute_safety_metrics(
        self, text_content: str, safety_config
    ) -> dict[str, Any]:
        """Compute safety-related metrics for the given text content."""
        metrics = {}

        # Profanity detection
        if hasattr(safety_config, "profanity") and safety_config.profanity.enabled:
            # TODO: Implement profanity detection logic
            metrics["safety_profanity"] = {"flagged": False, "matches": []}

        # Slur detection
        if hasattr(safety_config, "slurs") and safety_config.slurs.enabled:
            # TODO: Implement slur detection logic
            metrics["safety_slurs"] = {"flagged": False, "matches": []}

        # Explicit content detection
        if hasattr(safety_config, "explicit") and safety_config.explicit.enabled:
            # TODO: Implement explicit content detection logic
            metrics["safety_explicit"] = {"flagged": False, "matches": []}

        # Hate speech detection
        if hasattr(safety_config, "hate_speech") and safety_config.hate_speech.enabled:
            # TODO: Implement hate speech detection logic
            metrics["safety_hate_speech"] = {"flagged": False, "matches": []}

        # PII (Personally Identifiable Information) detection
        if hasattr(safety_config, "pii") and safety_config.pii.enabled:
            # TODO: Implement PII detection logic
            metrics["safety_pii"] = {"flagged": False, "matches": []}

        return metrics

    def _compute_message_level_stats(
        self, messages_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute aggregate statistics across all messages."""
        if not messages_data:
            return {}

        stats = {}

        # Get all metric names from the first message (excluding basic fields)
        basic_fields = {
            "conversation_id",
            "conversation_index",
            "message_index",
            "message_id",
            "role",
            "text_content",
        }
        metric_names = set(messages_data[0].keys()) - basic_fields

        for metric_name in metric_names:
            # Filter out None values and ensure we have numeric values
            values = [
                msg.get(metric_name)
                for msg in messages_data
                if msg.get(metric_name) is not None
            ]

            if not values:
                continue

            # Compute statistics for numeric metrics
            if all(isinstance(v, (int, float)) for v in values):
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    stats[f"avg_{metric_name}"] = sum(numeric_values) / len(
                        numeric_values
                    )
                    stats[f"min_{metric_name}"] = min(numeric_values)
                    stats[f"max_{metric_name}"] = max(numeric_values)
                    stats[f"total_{metric_name}"] = sum(numeric_values)

        return stats

    def _compute_aggregation_analysis(
        self, sample_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform aggregation analysis using the intermediate results."""
        verbose = self.config.verbose

        if verbose:
            logger.info(
                f"Starting aggregation analysis of dataset: {self.dataset_name}"
            )

        # Extract data from sample results
        messages_data = sample_results.get("messages", [])
        total_conversations = sample_results.get("total_conversations", 0)
        conversations_analyzed = sample_results.get(
            "conversations_analyzed", total_conversations
        )
        total_messages = sample_results.get("total_messages", 0)

        # Aggregate conversation-level statistics
        conversation_stats = {}

        # Get aggregation metrics configuration
        agg_metrics = self.config.aggregation_metrics
        conv_agg_metrics = self.config.conversation_aggregation_metrics

        # Build conversation stats if needed
        if conv_agg_metrics.enabled and conv_agg_metrics.turn_count:
            for message_data in messages_data:
                conv_id = message_data["conversation_id"]
                if conv_id not in conversation_stats:
                    conversation_stats[conv_id] = {
                        "conversation_id": conv_id,
                        "turn_count": 0,
                    }
                conversation_stats[conv_id]["turn_count"] += 1
        elif conv_agg_metrics.enabled:
            for message_data in messages_data:
                conv_id = message_data["conversation_id"]
                if conv_id not in conversation_stats:
                    conversation_stats[conv_id] = {
                        "conversation_id": conv_id,
                    }
        else:
            for message_data in messages_data:
                conv_id = message_data["conversation_id"]
                if conv_id not in conversation_stats:
                    conversation_stats[conv_id] = {"conversation_id": conv_id}

        # Calculate conversation turn statistics
        conversation_stats_consolidated = {}
        if conv_agg_metrics.enabled and conv_agg_metrics.turn_count:
            conversation_turns = [
                conv_stat["turn_count"] for conv_stat in conversation_stats.values()
            ]
            conversation_stats_consolidated = {
                "min_turns": min(conversation_turns) if conversation_turns else 0,
                "max_turns": max(conversation_turns) if conversation_turns else 0,
                "mean_turns": sum(conversation_turns) / len(conversation_turns)
                if conversation_turns
                else 0,
                "median_turns": sorted(conversation_turns)[len(conversation_turns) // 2]
                if conversation_turns
                else 0,
            }

        # Prepare aggregation metrics
        aggregation_metrics = {
            "total_conversations": total_conversations,
            "conversations_analyzed": conversations_analyzed,
            "total_messages": total_messages,
        }

        # Build aggregation results based on configuration
        aggregation_results = {}

        # Add basic stats if enabled
        if agg_metrics.basic_stats:
            aggregation_results["basic_stats"] = aggregation_metrics

        # Add conversation stats if enabled and turn_count is enabled
        if (
            agg_metrics.conversation_stats
            and conv_agg_metrics.enabled
            and conv_agg_metrics.turn_count
        ):
            aggregation_results["conversation_stats"] = conversation_stats_consolidated

        # Save conversation-level data to separate file if enabled
        if (
            conv_agg_metrics.enabled
            and hasattr(self.config, "outputs")
            and self.config.outputs.conversation_level_output
        ):
            conversation_level_data = list(conversation_stats.values())
            conversation_level_filename = _generate_timestamped_filename(
                self.config.outputs.conversation_level_output,
                self.config.outputs.save_format,
            )
            conversation_level_path = (
                Path(self.config.outputs.path) / conversation_level_filename
            )

            self._save_results(
                {"conversation_level_data": conversation_level_data},
                str(conversation_level_path),
                self.config.outputs.save_format,
            )

            if verbose:
                logger.info(
                    f"Conversation-level data saved to: {conversation_level_filename}"
                )

        return aggregation_results
