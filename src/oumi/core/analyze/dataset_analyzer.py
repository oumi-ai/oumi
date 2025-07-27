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

from dataclasses import asdict, dataclass
from typing import Any

from tqdm import tqdm

from oumi.core.configs import AnalyzeConfig
from oumi.core.registry.registry import REGISTRY
from oumi.utils.analysis_utils import load_dataset_from_config
from oumi.utils.logging import logger


@dataclass
class MessageAnalysisResult:
    """Result of analyzing a single message in a conversation.

    Attributes:
        conversation_id: Unique identifier for the conversation
        conversation_index: Index of the conversation in the dataset
        message_index: Index of the message within the conversation
        role: Role of the message sender (e.g., 'user', 'assistant')
        message_id: Unique identifier for the message
        text_content: The text content of the message
        analyzer_metrics: Dictionary of metrics computed by sample analyzers,
            with keys prefixed by analyzer ID to avoid conflicts
    """

    conversation_id: str
    conversation_index: int
    message_index: int
    role: str
    message_id: str
    text_content: str
    analyzer_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


@dataclass
class DatasetAnalysisResult:
    """Complete result of dataset analysis.

    Attributes:
        dataset_name: Name of the analyzed dataset
        total_conversations: Total number of conversations in the dataset
        conversations_analyzed: Number of conversations actually analyzed
        total_messages: Total number of messages analyzed
        messages: List of analysis results for each individual message
    """

    dataset_name: str
    total_conversations: int
    conversations_analyzed: int
    total_messages: int
    messages: list[MessageAnalysisResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


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
        for analyzer_params in self.config.analyzers:
            try:
                # Get the analyzer class from the registry
                analyzer_class = REGISTRY.get_sample_analyzer(analyzer_params.id)
                if analyzer_class is None:
                    raise ValueError(
                        f"Sample analyzer '{analyzer_params.id}' not found in registry"
                    )

                # Create analyzer instance with configuration
                config_dict = {
                    "id": analyzer_params.id,
                    **analyzer_params.config,
                }
                sample_analyzer = analyzer_class(config_dict)
                sample_analyzers[analyzer_params.id] = sample_analyzer
                logger.info(f"Initialized sample analyzer: {analyzer_params.id}")
            except Exception as e:
                logger.error(
                    f"Failed to initialize sample analyzer {analyzer_params.id}: {e}"
                )
                logger.error(f"Analyzer configuration: {analyzer_params}")
        return sample_analyzers

    def analyze_dataset(self) -> DatasetAnalysisResult:
        """Analyze the dataset and return analysis results.

        This method performs sample-level analysis using the configured sample
        analyzers. Each sample analyzer processes individual messages and returns
        metrics for each message.

        Returns:
            DatasetAnalysisResult: Analysis results containing sample-level metrics and
            insights, with strongly typed structure for better documentation and IDE
            support.

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

        dataset_analysis_result = self._compute_message_metrics()

        return dataset_analysis_result

    def _compute_message_metrics(self) -> DatasetAnalysisResult:
        """Compute metrics for all messages in the dataset.

        Returns:
            DatasetAnalysisResult: Structured results containing message-level analysis
            with metadata about the analysis scope and individual message results.
        """
        total_conversations = len(self.dataset)

        # Apply conversation limit if specified
        max_conversations = self.config.sample_count

        if max_conversations is not None:
            if max_conversations <= 0:
                raise ValueError(
                    f"sample_count must be positive, got {max_conversations}. "
                    "Use None to analyze all conversations."
                )
            conversations_to_analyze = min(total_conversations, max_conversations)
            logger.info(
                f"Limiting analysis to first {max_conversations} "
                f"conversations (dataset has {total_conversations} total)"
            )
        else:
            conversations_to_analyze = total_conversations

        logger.info(
            "Analyzing %d conversations for message-level metrics",
            conversations_to_analyze,
        )

        # Collect all message analysis results
        message_results = []

        # Use tqdm for progress monitoring
        for conv_idx in tqdm(
            range(conversations_to_analyze),
            desc=f"Analyzing {self.dataset_name}",
            unit="conv",
        ):
            conversation = self.dataset.conversation(conv_idx)
            for msg_idx, message in enumerate(conversation.messages):
                message_result = self._compute_per_message_metrics(
                    message, conv_idx, msg_idx, conversation
                )
                message_results.append(message_result)

        dataset_analysis_result = DatasetAnalysisResult(
            dataset_name=self.dataset_name
            or "",  # Config validation ensures this is not None
            total_conversations=total_conversations,
            conversations_analyzed=conversations_to_analyze,
            total_messages=len(message_results),
            messages=message_results,
        )

        return dataset_analysis_result

    def _compute_per_message_metrics(
        self, message, conv_idx: int, msg_idx: int, conversation
    ) -> MessageAnalysisResult:
        """Compute metrics for a single message.

        Args:
            message: The message object to analyze
            conv_idx: Index of the conversation in the dataset
            msg_idx: Index of the message within the conversation
            conversation: The conversation object containing the message

        Returns:
            MessageAnalysisResult: Structured result containing message metadata
            and analyzer metrics for the individual message.
        """
        # Get text content
        if isinstance(message.content, str):
            text_content = message.content
        else:
            # For multimodal content, extract text only
            text_content = message.compute_flattened_text_content()

        # Extract basic message information
        conversation_id = conversation.conversation_id or f"conv_{conv_idx}"
        message_id = message.id or f"msg_{conv_idx}_{msg_idx}"
        role = message.role.value

        # Compute metrics using all configured analyzers
        analyzer_metrics: dict[str, Any] = {}
        for analyzer_id, analyzer in self.sample_analyzers.items():
            try:
                analyzer_metrics_raw = analyzer.analyze_message(text_content)
                # Prefix metrics with analyzer ID to avoid conflicts
                for key, value in analyzer_metrics_raw.items():
                    analyzer_metrics[f"{analyzer_id}_{key}"] = value
            except Exception as e:
                logger.warning(
                    f"Analyzer {analyzer_id} failed for message "
                    f"{conv_idx}_{msg_idx}: {e}"
                )

        return MessageAnalysisResult(
            conversation_id=conversation_id,
            conversation_index=conv_idx,
            message_index=msg_idx,
            role=role,
            message_id=message_id,
            text_content=text_content,
            analyzer_metrics=analyzer_metrics,
        )
