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
from typing import Any, Optional

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
                writer.writerow(
                    ["total_conversations", results.get("total_conversations", 0)]
                )
                writer.writerow(
                    ["conversations_analyzed", results.get("conversations_analyzed", 0)]
                )

                if "conversation_length_stats" in results:
                    stats = results["conversation_length_stats"]
                    writer.writerow(["min_length", stats.get("min", 0)])
                    writer.writerow(["max_length", stats.get("max", 0)])
                    writer.writerow(["mean_length", f"{stats.get('mean', 0):.2f}"])
                    writer.writerow(["median_length", stats.get("median", 0)])
        elif save_format == "parquet":
            # For parquet, we'll save the results as a structured format
            import pandas as pd

            # Create a DataFrame from the results
            data = {
                "dataset_name": [results.get("dataset_name", "Unknown")],
                "total_conversations": [results.get("total_conversations", 0)],
                "conversations_analyzed": [results.get("conversations_analyzed", 0)],
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

            dataframe = pd.DataFrame(data)
            dataframe.to_parquet(output_file, index=False)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

        if self.config.verbose:
            logger.info(f"Results saved to: {output_file}")

    def analyze_dataset(self) -> dict[str, Any]:
        """Analyze the dataset and return analysis results.

        Returns:
            Dict[str, Any]: Analysis results containing various metrics and insights.
        """
        # Use config parameters to determine analysis scope
        verbose = self.config.verbose

        if verbose:
            logger.info(f"Starting analysis of dataset: {self.dataset_name}")

        # Determine how many conversations to analyze
        total_conversations = len(self.dataset)
        conversations_to_analyze = total_conversations

        if verbose:
            logger.info(
                f"Analyzing {conversations_to_analyze} out of "
                f"{total_conversations} conversations"
            )

        # Perform basic analysis
        results = self._basic_analysis(conversations_to_analyze)

        # Add metadata
        results["config"] = {
            "conversations_analyzed": conversations_to_analyze,
            "total_conversations": total_conversations,
        }

        # Save results if output configuration is provided
        if hasattr(self.config, "outputs") and self.config.outputs.analysis_output:
            self._save_results(
                results,
                self.config.outputs.analysis_output,
                self.config.outputs.save_format,
            )

        return results

    def _basic_analysis(
        self,
        conversations_to_analyze: Optional[int] = None,
    ) -> dict[str, Any]:
        """Perform basic analysis of the dataset."""
        if conversations_to_analyze is None:
            conversations_to_analyze = len(self.dataset)

        # Basic statistics
        total_conversations = len(self.dataset)
        conversation_lengths = []

        for i in range(min(conversations_to_analyze, total_conversations)):
            conversation = self.get_conversation(i)
            conversation_lengths.append(len(conversation.messages))

        results = {
            "dataset_name": self.dataset_name,
            "total_conversations": total_conversations,
            "conversations_analyzed": min(
                conversations_to_analyze, total_conversations
            ),
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
        }

        return results
