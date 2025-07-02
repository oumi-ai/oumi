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

from typing import Any, Optional

from oumi.core.configs import AnalyzerConfig, InputConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.logging import logger


def _get_analyzer(config: AnalyzerConfig):
    """Returns the analyzer based on the provided config."""
    # TODO: Implement analyzer builder similar to inference engine builder
    return None


def _load_dataset_from_config(config: AnalyzerConfig) -> BaseMapDataset:
    """Load dataset based on configuration."""
    return _load_dataset_from_v1_config(config)


def _load_dataset_from_v1_config(config: AnalyzerConfig) -> BaseMapDataset:
    """Load dataset using v1.0.0 configuration structure."""
    input_config = config.input

    try:
        if input_config.source == "oumi":
            return _load_oumi_dataset(input_config)
        elif input_config.source == "huggingface":
            return _load_huggingface_dataset(input_config)
        elif input_config.source == "custom":
            return _load_custom_dataset(input_config)
        else:
            raise ValueError(f"Unsupported source: {input_config.source}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {input_config.source}: {e}")
        raise


def _load_oumi_dataset(input_config: InputConfig) -> BaseMapDataset:
    """Load dataset from Oumi registry."""
    dataset_name = input_config.name
    if not dataset_name:
        raise ValueError("Dataset name is required for oumi source")

    try:
        if dataset_name.lower() == "alpaca":
            from oumi.datasets.sft.alpaca import AlpacaDataset

            return AlpacaDataset(split=input_config.split)
        elif dataset_name.lower() == "dolly":
            from oumi.datasets.sft.dolly import ArgillaDollyDataset

            return ArgillaDollyDataset(split=input_config.split)
        elif dataset_name.lower() == "ultrachat":
            from oumi.datasets.sft.ultrachat import UltrachatH4Dataset

            return UltrachatH4Dataset(split=input_config.split)
        elif dataset_name.lower() == "magpie":
            from oumi.datasets.sft.magpie import ArgillaMagpieUltraDataset

            return ArgillaMagpieUltraDataset(split=input_config.split)
        else:
            # Try to load from registry for other datasets
            from oumi.core.registry import REGISTRY

            dataset_class = REGISTRY.get_dataset(dataset_name)
            if dataset_class is None:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            return dataset_class(split=input_config.split)
    except ImportError as e:
        logger.error(f"Failed to import dataset {dataset_name}: {e}")
        raise


def _load_huggingface_dataset(input_config: InputConfig) -> BaseMapDataset:
    """Load dataset from HuggingFace Hub."""
    dataset_name = input_config.name
    if not dataset_name:
        raise ValueError("Dataset name is required for huggingface source")

    try:
        import datasets

        from oumi.datasets.sft.sft_jsonlines import TextSftJsonLinesDataset

        # Load from HuggingFace Hub
        hf_dataset = datasets.load_dataset(
            path=dataset_name, split=input_config.split, trust_remote_code=True
        )

        # Convert to list format
        data = list(hf_dataset)

        # Convert to Oumi format
        return TextSftJsonLinesDataset(data=data, split=input_config.split)
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset {dataset_name}: {e}")
        raise


def _load_custom_dataset(input_config: InputConfig) -> BaseMapDataset:
    """Load dataset from custom file path."""
    dataset_path = input_config.path
    if not dataset_path:
        raise ValueError("Dataset path is required for custom source")

    try:
        from oumi.datasets.sft.sft_jsonlines import TextSftJsonLinesDataset

        # Load based on file format
        if input_config.format == "jsonl":
            return TextSftJsonLinesDataset(
                dataset_path=dataset_path, split=input_config.split
            )
        else:
            # Default to JSONL format
            return TextSftJsonLinesDataset(
                dataset_path=dataset_path, split=input_config.split
            )
    except Exception as e:
        logger.error(f"Failed to load custom dataset from {dataset_path}: {e}")
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
        self.analyzer = _get_analyzer(self.config)

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

    def analyze_dataset(self) -> dict[str, Any]:
        """Analyze the dataset and return analysis results.

        Returns:
            Dict[str, Any]: Analysis results containing various metrics and insights.
        """
        # Use config parameters to determine analysis type and scope
        verbose = self.config.verbose

        # For now, use basic analysis as default since we removed legacy fields
        analysis_type = "basic"
        sample_count = None
        max_conversations = None
        include_examples = True
        example_count = 3

        if verbose:
            logger.info(
                f"Starting {analysis_type} analysis of dataset: {self.dataset_name}"
            )

        # Determine how many conversations to analyze
        total_conversations = len(self.dataset)
        if sample_count is not None:
            conversations_to_analyze = min(sample_count, total_conversations)
        elif max_conversations is not None:
            conversations_to_analyze = min(max_conversations, total_conversations)
        else:
            conversations_to_analyze = total_conversations

        if verbose:
            logger.info(
                f"Analyzing {conversations_to_analyze} out of "
                f"{total_conversations} conversations"
            )

        # Perform analysis based on type
        if analysis_type == "basic":
            results = self._basic_analysis(
                conversations_to_analyze, include_examples, example_count
            )
        elif analysis_type == "conversation":
            results = self._conversation_analysis(
                conversations_to_analyze, include_examples, example_count
            )
        elif analysis_type == "content":
            results = self._content_analysis(
                conversations_to_analyze, include_examples, example_count
            )
        elif analysis_type == "quality":
            results = self._quality_analysis(
                conversations_to_analyze, include_examples, example_count
            )
        elif analysis_type == "full":
            results = self._full_analysis(
                conversations_to_analyze, include_examples, example_count
            )
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        # Add metadata
        results["config"] = {
            "analysis_type": analysis_type,
            "sample_count": sample_count,
            "max_conversations": max_conversations,
            "conversations_analyzed": conversations_to_analyze,
            "total_conversations": total_conversations,
        }

        return results

    def _basic_analysis(
        self,
        conversations_to_analyze: Optional[int] = None,
        include_examples: bool = True,
        example_count: int = 3,
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

        if include_examples:
            results["examples"] = []
            for i in range(min(example_count, total_conversations)):
                conversation = self.get_conversation(i)
                results["examples"].append(
                    {
                        "index": i,
                        "length": len(conversation.messages),
                        "conversation": str(conversation)[:500] + "..."
                        if len(str(conversation)) > 500
                        else str(conversation),
                    }
                )

        return results

    def _conversation_analysis(
        self, conversations_to_analyze: int, include_examples: bool, example_count: int
    ) -> dict[str, Any]:
        """Perform detailed conversation analysis."""
        # TODO: Implement conversation analysis
        return {"status": "not_implemented", "type": "conversation"}

    def _content_analysis(
        self, conversations_to_analyze: int, include_examples: bool, example_count: int
    ) -> dict[str, Any]:
        """Perform content analysis."""
        # TODO: Implement content analysis
        return {"status": "not_implemented", "type": "content"}

    def _quality_analysis(
        self, conversations_to_analyze: int, include_examples: bool, example_count: int
    ) -> dict[str, Any]:
        """Perform quality analysis."""
        # TODO: Implement quality analysis
        return {"status": "not_implemented", "type": "quality"}

    def _full_analysis(
        self, conversations_to_analyze: int, include_examples: bool, example_count: int
    ) -> dict[str, Any]:
        """Perform comprehensive analysis."""
        # TODO: Implement full analysis
        return {"status": "not_implemented", "type": "full"}
