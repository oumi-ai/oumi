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
from pathlib import Path
from typing import Any

import yaml

from oumi.core.configs import AnalyzerConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.logging import logger


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
                    if "conversation_length_stats" in aggregation_results:
                        stats = aggregation_results["conversation_length_stats"]
                        writer.writerow(
                            ["min_conversation_length", stats.get("min", 0)]
                        )
                        writer.writerow(
                            ["max_conversation_length", stats.get("max", 0)]
                        )
                        writer.writerow(
                            ["mean_conversation_length", f"{stats.get('mean', 0):.2f}"]
                        )
                        writer.writerow(
                            ["median_conversation_length", stats.get("median", 0)]
                        )

                    if "role_analysis" in aggregation_results:
                        role_analysis = aggregation_results["role_analysis"]
                        avg_messages = role_analysis.get(
                            "avg_messages_per_conversation", 0
                        )
                        writer.writerow(
                            [
                                "avg_messages_per_conversation",
                                f"{avg_messages:.2f}",
                            ]
                        )
                        writer.writerow(
                            [
                                "conversations_with_system",
                                role_analysis.get(
                                    "conversations_with_system_messages", 0
                                ),
                            ]
                        )
                        writer.writerow(
                            [
                                "conversations_with_tool",
                                role_analysis.get(
                                    "conversations_with_tool_messages", 0
                                ),
                            ]
                        )

                        # Add role counts
                        role_counts = role_analysis.get("role_counts", {})
                        for role, count in role_counts.items():
                            writer.writerow([f"role_count_{role}", count])
                else:
                    # Single result type (either sample or aggregation)
                    writer.writerow(
                        ["total_conversations", results.get("total_conversations", 0)]
                    )
                    writer.writerow(
                        ["total_messages", results.get("total_messages", 0)]
                    )

                    if "conversation_length_stats" in results:
                        stats = results["conversation_length_stats"]
                        writer.writerow(["min_length", stats.get("min", 0)])
                        writer.writerow(["max_length", stats.get("max", 0)])
                        writer.writerow(
                            [
                                "mean_length",
                                f"{stats.get('mean', 0):.2f}",
                            ]
                        )
                        writer.writerow(["median_length", stats.get("median", 0)])

                    if "role_analysis" in results:
                        role_analysis = results["role_analysis"]
                        avg_messages = role_analysis.get(
                            "avg_messages_per_conversation", 0
                        )
                        writer.writerow(
                            [
                                "avg_messages_per_conversation",
                                f"{avg_messages:.2f}",
                            ]
                        )
                        writer.writerow(
                            [
                                "conversations_with_system",
                                role_analysis.get(
                                    "conversations_with_system_messages", 0
                                ),
                            ]
                        )
                        writer.writerow(
                            [
                                "conversations_with_tool",
                                role_analysis.get(
                                    "conversations_with_tool_messages", 0
                                ),
                            ]
                        )

                        # Add role counts
                        role_counts = role_analysis.get("role_counts", {})
                        for role, count in role_counts.items():
                            writer.writerow([f"role_count_{role}", count])
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
                if "conversation_length_stats" in aggregation_results:
                    stats = aggregation_results["conversation_length_stats"]
                    data.update(
                        {
                            "min_conversation_length": [stats.get("min", 0)],
                            "max_conversation_length": [stats.get("max", 0)],
                            "mean_conversation_length": [stats.get("mean", 0)],
                            "median_conversation_length": [stats.get("median", 0)],
                        }
                    )

                if "role_analysis" in aggregation_results:
                    role_analysis = aggregation_results["role_analysis"]
                    data.update(
                        {
                            "avg_messages_per_conversation": [
                                role_analysis.get("avg_messages_per_conversation", 0)
                            ],
                            "conversations_with_system": [
                                role_analysis.get(
                                    "conversations_with_system_messages", 0
                                )
                            ],
                            "conversations_with_tool": [
                                role_analysis.get("conversations_with_tool_messages", 0)
                            ],
                        }
                    )

                    # Add role counts
                    role_counts = role_analysis.get("role_counts", {})
                    for role, count in role_counts.items():
                        data[f"role_count_{role}"] = [count]
            else:
                # Single result type (either sample or aggregation)
                data = {
                    "dataset_name": [results.get("dataset_name", "Unknown")],
                    "total_conversations": [results.get("total_conversations", 0)],
                    "total_messages": [results.get("total_messages", 0)],
                }

                if "conversation_length_stats" in results:
                    stats = results["conversation_length_stats"]
                    data.update(
                        {
                            "min_length": [stats.get("min", 0)],
                            "max_length": [stats.get("max", 0)],
                            "mean_length": [stats.get("mean", 0)],
                            "median_length": [stats.get("median", 0)],
                        }
                    )

                if "role_analysis" in results:
                    role_analysis = results["role_analysis"]
                    data.update(
                        {
                            "avg_messages_per_conversation": [
                                role_analysis.get("avg_messages_per_conversation", 0)
                            ],
                            "conversations_with_system": [
                                role_analysis.get(
                                    "conversations_with_system_messages", 0
                                )
                            ],
                            "conversations_with_tool": [
                                role_analysis.get("conversations_with_tool_messages", 0)
                            ],
                        }
                    )

                    # Add role counts
                    role_counts = role_analysis.get("role_counts", {})
                    for role, count in role_counts.items():
                        data[f"role_count_{role}"] = [count]

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
            sample_output_path = (
                self.config.outputs.analysis_output.replace(
                    ".json", "_sample_level.json"
                )
                .replace(".yaml", "_sample_level.yaml")
                .replace(".csv", "_sample_level.csv")
                .replace(".parquet", "_sample_level.parquet")
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
            full_aggregation_path = (
                Path(self.config.outputs.path) / self.config.outputs.aggregation_output
            )

            self._save_results(
                aggregation_results,
                str(full_aggregation_path),
                self.config.outputs.save_format,
            )

            if verbose:
                logger.info(
                    "Aggregation results saved to: "
                    f"{self.config.outputs.aggregation_output}"
                )

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
        total_messages = sample_results.get("total_messages", 0)

        # Aggregate conversation-level statistics
        conversation_stats = {}
        role_counts = {"system": 0, "user": 0, "assistant": 0, "tool": 0}
        conversations_with_system = set()
        conversations_with_tool = set()

        for message_data in messages_data:
            conv_id = message_data["conversation_id"]
            role = message_data["role"]

            # Count roles
            role_counts[role] += 1

            # Track conversations with system/tool messages
            if role == "system":
                conversations_with_system.add(conv_id)
            elif role == "tool":
                conversations_with_tool.add(conv_id)

            # Aggregate conversation-level stats
            if conv_id not in conversation_stats:
                conversation_stats[conv_id] = {
                    "conversation_id": conv_id,
                    "message_count": 0,
                    "roles": set(),
                    "has_system": False,
                    "has_tool": False,
                }

            conv_stat = conversation_stats[conv_id]
            conv_stat["message_count"] += 1
            conv_stat["roles"].add(role)
            conv_stat["has_system"] = conv_stat["has_system"] or role == "system"
            conv_stat["has_tool"] = conv_stat["has_tool"] or role == "tool"

        # Convert sets to lists for JSON serialization
        for conv_stat in conversation_stats.values():
            conv_stat["roles"] = list(conv_stat["roles"])

        # Calculate conversation length statistics
        conversation_lengths = [
            conv_stat["message_count"] for conv_stat in conversation_stats.values()
        ]

        # Calculate statistics
        avg_messages_per_conversation = (
            total_messages / len(conversation_lengths) if conversation_lengths else 0
        )
        role_distribution = (
            {role: count / total_messages * 100 for role, count in role_counts.items()}
            if total_messages > 0
            else role_counts
        )

        # Get the number of conversations that were actually analyzed
        conversations_analyzed = sample_results.get(
            "conversations_analyzed", total_conversations
        )

        aggregation_results = {
            "dataset_name": self.dataset_name,
            "total_conversations": total_conversations,
            "conversations_analyzed": conversations_analyzed,
            "total_messages": total_messages,
            "conversation_length_stats": {
                "min": min(conversation_lengths) if conversation_lengths else 0,
                "max": max(conversation_lengths) if conversation_lengths else 0,
                "mean": sum(conversation_lengths) / len(conversation_lengths)
                if conversation_lengths
                else 0,
                "median": sorted(conversation_lengths)[len(conversation_lengths) // 2]
                if conversation_lengths
                else 0,
            },
            "role_analysis": {
                "total_messages": total_messages,
                "avg_messages_per_conversation": round(
                    avg_messages_per_conversation, 2
                ),
                "role_counts": role_counts,
                "role_distribution_percent": role_distribution,
                "conversations_with_system_messages": len(conversations_with_system),
                "conversations_with_tool_messages": len(conversations_with_tool),
                "conversations_with_system_percent": round(
                    len(conversations_with_system) / len(conversation_lengths) * 100, 2
                )
                if conversation_lengths
                else 0,
                "conversations_with_tool_percent": round(
                    len(conversations_with_tool) / len(conversation_lengths) * 100, 2
                )
                if conversation_lengths
                else 0,
            },
            "conversation_level_data": list(conversation_stats.values()),
        }

        return aggregation_results
