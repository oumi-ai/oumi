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


class ConversationHelper:
    """Helper class for conversation-related operations and data access."""

    def __init__(self, dataset: BaseMapDataset, dataset_name: str):
        """Initialize the conversation helper.

        Args:
            dataset: The dataset to work with
            dataset_name: Name of the dataset for display purposes
        """
        self.dataset = dataset
        self.dataset_name = dataset_name

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


def generate_filename(prefix: str, save_format: str) -> str:
    """Generate a filename with the specified format.

    Args:
        prefix: The filename prefix
        save_format: The file format (json, yaml)

    Returns:
        Filename with extension
    """
    return f"{prefix}.{save_format}"


def load_dataset_from_config(config: AnalyzerConfig) -> BaseMapDataset:
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


def save_results(results: dict[str, Any], output_path: str, save_format: str):
    """Save analysis results to file based on the specified format.

    Args:
        results: Analysis results dictionary to save
        output_path: Path where to save the results
        save_format: Format to save the results (json, yaml)
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
    else:
        raise ValueError(f"Unsupported save format: {save_format}")

    logger.info(f"Results saved to: {output_file}")


def compute_sample_level_analysis(
    dataset: BaseMapDataset, config: AnalyzerConfig, analyzers: dict
) -> dict[str, Any]:
    """Perform per-sample (message) level analysis using plugin analyzers."""
    total_conversations = len(dataset)

    # Apply conversation limit if specified
    max_conversations = config.input.max_conversations
    if max_conversations is not None and max_conversations > 0:
        conversations_to_analyze = min(total_conversations, max_conversations)
        logger.info(
            f"Limiting analysis to first {max_conversations} "
            f"conversations (dataset has {total_conversations} total)"
        )
    else:
        conversations_to_analyze = total_conversations

    logger.info(
        "Analyzing %d conversations for sample-level metrics",
        conversations_to_analyze,
    )

    # Collect all messages with their metadata
    messages_data = []

    for conv_idx in range(conversations_to_analyze):
        conversation = dataset.conversation(conv_idx)

        for msg_idx, message in enumerate(conversation.messages):
            # Get text content
            if isinstance(message.content, str):
                text_content = message.content
            else:
                # For multimodal content, extract text only
                text_content = message.compute_flattened_text_content()

            # Basic message information
            message_data = {
                "conversation_id": conversation.conversation_id or f"conv_{conv_idx}",
                "conversation_index": conv_idx,
                "message_index": msg_idx,
                "message_id": message.id or f"msg_{conv_idx}_{msg_idx}",
                "role": message.role.value,
                "text_content": text_content,
            }

            # Compute metrics using all configured analyzers
            message_metadata = {
                "conversation_id": message_data["conversation_id"],
                "conversation_index": conv_idx,
                "message_index": msg_idx,
                "role": message.role.value,
            }

            for analyzer_id, analyzer in analyzers.items():
                try:
                    analyzer_metrics = analyzer.analyze_message(
                        text_content, message_metadata
                    )
                    # Prefix metrics with analyzer ID to avoid conflicts
                    for key, value in analyzer_metrics.items():
                        message_data[f"{analyzer_id}_{key}"] = value
                except Exception as e:
                    logger.warning(
                        f"Analyzer {analyzer_id} failed for message "
                        f"{conv_idx}_{msg_idx}: {e}"
                    )

            messages_data.append(message_data)

    sample_results = {
        "dataset_name": config.input.name,
        "total_conversations": total_conversations,
        "conversations_analyzed": conversations_to_analyze,
        "total_messages": len(messages_data),
        "messages": messages_data,
    }

    return sample_results
