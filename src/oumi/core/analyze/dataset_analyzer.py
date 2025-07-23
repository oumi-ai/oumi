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

from typing import Any

from oumi.core.configs import AnalyzeConfig
from oumi.core.registry.registry import REGISTRY
from oumi.utils.analysis_utils import load_dataset_from_config
from oumi.utils.logging import logger


class DatasetAnalyzer:
    """Orchestrates dataset analysis by creating and managing sample analyzers."""

    def __init__(self, config: AnalyzeConfig):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzeConfig object containing all analysis parameters
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.split = config.split

        self.dataset = load_dataset_from_config(config)
        self.sample_analyzers = self._initialize_sample_analyzers()

    def _initialize_sample_analyzers(self):
        """Initialize sample analyzer plugins from configuration."""
        sample_analyzers = {}
        for analyzer_config in self.config.analyzers:
            try:
                # Get the analyzer class from the registry
                analyzer_class = REGISTRY.get_sample_analyzer(analyzer_config.id)
                if analyzer_class is None:
                    logger.error(
                        f"Sample analyzer '{analyzer_config.id}' not found in registry"
                    )
                    continue

                # Create analyzer instance with configuration
                config_dict = {
                    "id": analyzer_config.id,
                    **analyzer_config.config,
                }
                sample_analyzer = analyzer_class(config_dict)
                sample_analyzers[analyzer_config.id] = sample_analyzer
                logger.info(f"Initialized sample analyzer: {analyzer_config.id}")
            except Exception as e:
                logger.error(
                    f"Failed to initialize sample analyzer {analyzer_config.id}: {e}"
                )
                logger.error(f"Analyzer configuration: {analyzer_config}")
        return sample_analyzers

    def analyze_dataset(self) -> dict[str, Any]:
        """Analyze the dataset and return analysis results.

        This method performs sample-level analysis using the configured sample
        analyzers. Each sample analyzer processes individual messages and returns
        metrics for each message.

        Returns:
            Dict[str, Any]: Analysis results containing sample-level metrics and
            insights.

        Raises:
            ValueError: If no analyzers are configured for analysis.
        """
        if not self.sample_analyzers:
            raise ValueError(
                "No analyzers configured for analysis. Please add at least one "
                "analyzer to the configuration before calling analyze_dataset()."
            )

        logger.info(f"Starting analysis of dataset: {self.dataset_name}")
        logger.info(
            f"Using {len(self.sample_analyzers)} sample analyzers: "
            f"{list(self.sample_analyzers.keys())}"
        )

        total_conversations = len(self.dataset)
        conversations_to_analyze = min(
            total_conversations, self.config.sample_count or total_conversations
        )

        logger.info(f"Analyzing {conversations_to_analyze} conversations")

        # Step 1: Per-message level analysis
        logger.info("Step 1: Computing message metrics...")

        sample_results = self._compute_message_metrics()

        final_results = {
            "dataset_name": self.dataset_name,
            "sample_level_results": sample_results,
        }
        return final_results

    def _compute_message_metrics(self) -> dict[str, Any]:
        """Compute metrics for all messages in the dataset."""
        total_conversations = len(self.dataset)

        # Apply conversation limit if specified
        max_conversations = self.config.sample_count

        if max_conversations is not None and max_conversations > 0:
            conversations_to_analyze = min(total_conversations, max_conversations)
            logger.info(
                f"Limiting analysis to first {max_conversations} "
                f"conversations (dataset has {total_conversations} total)"
            )
        elif max_conversations == 0:
            conversations_to_analyze = 0
            logger.info("sample_count=0 specified, analyzing no conversations")
        else:
            conversations_to_analyze = total_conversations

        logger.info(
            "Analyzing %d conversations for message-level metrics",
            conversations_to_analyze,
        )

        # Collect all messages with their metadata
        messages_data = []

        for conv_idx in range(conversations_to_analyze):
            conversation = self.dataset.conversation(conv_idx)

            for msg_idx, message in enumerate(conversation.messages):
                message_data = self._compute_per_message_metrics(
                    message, conv_idx, msg_idx, conversation
                )
                messages_data.append(message_data)

        sample_results = {
            "dataset_name": self.dataset_name,
            "total_conversations": total_conversations,
            "conversations_analyzed": conversations_to_analyze,
            "total_messages": len(messages_data),
            "messages": messages_data,
        }

        return sample_results

    def _compute_per_message_metrics(
        self, message, conv_idx: int, msg_idx: int, conversation
    ) -> dict[str, Any]:
        """Compute metrics for a single message."""
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

        for analyzer_id, analyzer in self.sample_analyzers.items():
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

        return message_data

    # TODO: Add save_to_file method to save analysis results to JSONL file
    # def save_to_file(self, output_path: str) -> None:
    #     """Save analysis results to JSONL file."""
    #     pass

    # TODO: Add load_from_file method to load analysis results from JSONL file
    # def load_from_file(self) -> dict[str, Any]:
    #     """Load analysis results from JSONL file into dict format."""
    #     pass

    # TODO: Add query method to query analysis results using pandas
    # def query(self, query_expression: str) -> pd.DataFrame:
    #     """Query analysis results using pandas query expression."""
    #     pass
