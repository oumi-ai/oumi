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

from oumi.core.configs import DatasetAnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.logging import logger


def load_dataset_from_config(config: DatasetAnalyzeConfig) -> BaseMapDataset:
    """Load dataset based on configuration.

    This function loads datasets directly from the registry for analysis purposes,
    avoiding the need for tokenizers and other training infrastructure.
    """
    from oumi.core.registry import REGISTRY

    dataset_name = config.dataset_name
    split = config.split

    if not dataset_name:
        raise ValueError("Dataset name is required")

    try:
        # Load dataset from the REGISTRY
        dataset_class = REGISTRY.get_dataset(dataset_name, subset=None)

        if dataset_class is not None:
            # Load registered dataset with basic parameters
            dataset = dataset_class(
                dataset_name=dataset_name,
                dataset_path=None,
                split=split,
                subset=None,
                trust_remote_code=False,
            )

            # Ensure we return a BaseMapDataset
            if isinstance(dataset, BaseMapDataset):
                return dataset
            else:
                raise NotImplementedError(
                    f"Dataset type {type(dataset)} is not supported for analysis. "
                    "Please use a dataset that inherits from BaseMapDataset."
                )
        else:
            # TODO: Implement HuggingFace Hub loading
            raise NotImplementedError(
                f"Dataset '{dataset_name}' is not registered in the REGISTRY. "
                "Loading from HuggingFace Hub is not yet implemented."
            )

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def save_results(results: dict[str, Any], output_path: str):
    """Save analysis results to file as JSON only.

    Args:
        results: Analysis results dictionary to save
        output_path: Path where to save the results
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_file}")


def compute_sample_level_analysis(
    dataset: BaseMapDataset, config: DatasetAnalyzeConfig, analyzers: dict
) -> dict[str, Any]:
    """Perform per-sample (message) level analysis using plugin analyzers."""
    total_conversations = len(dataset)

    # Apply conversation limit if specified
    max_conversations = config.sample_count

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

    # Get dataset name from the config
    dataset_name = config.dataset_name

    sample_results = {
        "dataset_name": dataset_name,
        "total_conversations": total_conversations,
        "conversations_analyzed": conversations_to_analyze,
        "total_messages": len(messages_data),
        "messages": messages_data,
    }

    return sample_results
