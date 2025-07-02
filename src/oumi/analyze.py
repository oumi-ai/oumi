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

from pathlib import Path
from typing import Any, Optional, Union

from oumi.core.configs import BaseConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.logging import logger


def _load_oumi_dataset(
    dataset_name: str, dataset_params: dict[str, Any]
) -> BaseMapDataset:
    """Load an Oumi dataset using the dataset registry.

    Args:
        dataset_name: Name of the dataset to load.
        dataset_params: Parameters for dataset loading.

    Returns:
        BaseMapDataset: Loaded dataset.
    """
    # TODO: Implement Oumi dataset loading using the registry
    # from oumi.builders.data import build_dataset
    # return build_dataset(dataset_name, **dataset_params)
    logger.info(f"Loading Oumi dataset: {dataset_name}")
    raise NotImplementedError("Oumi dataset loading not yet implemented")


def _get_analyzer(config: BaseConfig):
    """Returns the analyzer based on the provided config."""
    # TODO: Implement analyzer builder similar to inference engine builder
    return None


def _load_dataset_from_config(config: BaseConfig) -> BaseMapDataset:
    """Load dataset based on configuration."""
    # Placeholder implementation
    raise NotImplementedError("Dataset loading from config not yet implemented")


class DatasetAnalyzer:
    """Base class for dataset analysis functionality."""

    def __init__(
        self,
        dataset_name: str,
        config: Optional[BaseConfig] = None,
        split: Optional[str] = None,
    ):
        """Initialize the dataset analyzer with dataset name and optional configuration.

        Args:
            dataset_name: Name of the dataset to analyze (e.g., 'alpaca', 'dolly', etc.)
            config: Optional configuration object.
            split: Optional split name (e.g., 'train', 'validation', 'test')
        """
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.dataset = self._load_dataset(dataset_name, split)
        self.analyzer = _get_analyzer(config) if config else None

    def _load_dataset(
        self, dataset_name: str, split: Optional[str] = None
    ) -> BaseMapDataset:
        """Load the specified dataset.

        Args:
            dataset_name: Name of the dataset to load.
            split: Optional split name for the dataset.

        Returns:
            BaseMapDataset: Loaded dataset.
        """
        try:
            if dataset_name.lower() == "alpaca":
                from oumi.datasets.sft.alpaca import AlpacaDataset

                return AlpacaDataset(split=split)
            elif dataset_name.lower() == "dolly":
                from oumi.datasets.sft.dolly import ArgillaDollyDataset

                return ArgillaDollyDataset(split=split)
            elif dataset_name.lower() == "ultrachat":
                from oumi.datasets.sft.ultrachat import UltrachatH4Dataset

                return UltrachatH4Dataset(split=split)
            elif dataset_name.lower() == "magpie":
                from oumi.datasets.sft.magpie import ArgillaMagpieUltraDataset

                return ArgillaMagpieUltraDataset(split=split)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except ImportError as e:
            logger.error(f"Failed to import dataset {dataset_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

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
        # TODO: Implement dataset analysis logic
        return {"status": "not_implemented", "dataset_name": self.dataset_name}

    def get_statistics(self) -> dict[str, Any]:
        """Get basic statistics about the dataset.

        Returns:
            Dict[str, Any]: Basic statistics like size, distribution, etc.
        """
        # TODO: Implement statistics calculation
        return {"size": 0, "distribution": {}, "dataset_name": self.dataset_name}

    def check_quality(self) -> dict[str, Any]:
        """Check the quality of the dataset.

        Returns:
            Dict[str, Any]: Quality metrics and issues found.
        """
        # TODO: Implement quality checking logic
        return {"quality_score": 0.0, "issues": [], "dataset_name": self.dataset_name}

    def find_patterns(self) -> dict[str, Any]:
        """Find patterns in the dataset.

        Returns:
            Dict[str, Any]: Pattern analysis results.
        """
        # TODO: Implement pattern detection logic
        return {"patterns": [], "insights": [], "dataset_name": self.dataset_name}


# Convenience functions for easy usage
def analyze_dataset(
    dataset_name: str, config: Optional[BaseConfig] = None, split: Optional[str] = None
) -> dict[str, Any]:
    """Analyze a dataset and return analysis results.

    Args:
        dataset_name: Name of the dataset to analyze (e.g., 'alpaca', 'dolly', etc.)
        config: Optional configuration object.
        split: Optional split name (e.g., 'train', 'validation', 'test')

    Returns:
        Dict[str, Any]: Analysis results containing various metrics and insights.
    """
    analyzer = DatasetAnalyzer(dataset_name, config, split)
    return analyzer.analyze_dataset()


def get_statistics(
    dataset_name: str, config: Optional[BaseConfig] = None, split: Optional[str] = None
) -> dict[str, Any]:
    """Get basic statistics about the dataset.

    Args:
        dataset_name: Name of the dataset to analyze (e.g., 'alpaca', 'dolly', etc.)
        config: Optional configuration object.
        split: Optional split name (e.g., 'train', 'validation', 'test')

    Returns:
        Dict[str, Any]: Basic statistics like size, distribution, etc.
    """
    analyzer = DatasetAnalyzer(dataset_name, config, split)
    return analyzer.get_statistics()


def check_quality(
    dataset_name: str, config: Optional[BaseConfig] = None, split: Optional[str] = None
) -> dict[str, Any]:
    """Check the quality of the dataset.

    Args:
        dataset_name: Name of the dataset to analyze (e.g., 'alpaca', 'dolly', etc.)
        config: Optional configuration object.
        split: Optional split name (e.g., 'train', 'validation', 'test')

    Returns:
        Dict[str, Any]: Quality metrics and issues found.
    """
    analyzer = DatasetAnalyzer(dataset_name, config, split)
    return analyzer.check_quality()


def find_patterns(
    dataset_name: str, config: Optional[BaseConfig] = None, split: Optional[str] = None
) -> dict[str, Any]:
    """Find patterns in the dataset.

    Args:
        dataset_name: Name of the dataset to analyze (e.g., 'alpaca', 'dolly', etc.)
        config: Optional configuration object.
        split: Optional split name (e.g., 'train', 'validation', 'test')

    Returns:
        Dict[str, Any]: Pattern analysis results.
    """
    analyzer = DatasetAnalyzer(dataset_name, config, split)
    return analyzer.find_patterns()


def get_conversation_length(
    dataset_name: str,
    index: int = 0,
    config: Optional[BaseConfig] = None,
    split: Optional[str] = None,
) -> int:
    """Get the length (number of messages) of a conversation.

    Args:
        dataset_name: Name of the dataset to analyze (e.g., 'alpaca', 'dolly', etc.)
        index: Index of the conversation to check.
        config: Optional configuration object.
        split: Optional split name (e.g., 'train', 'validation', 'test')

    Returns:
        int: Number of messages in the conversation.
    """
    analyzer = DatasetAnalyzer(dataset_name, config, split)
    return analyzer.get_conversation_length(index)


def get_dataset_size(
    dataset_name: str, config: Optional[BaseConfig] = None, split: Optional[str] = None
) -> int:
    """Get the total number of conversations in the dataset.

    Args:
        dataset_name: Name of the dataset to analyze (e.g., 'alpaca', 'dolly', etc.)
        config: Optional configuration object.
        split: Optional split name (e.g., 'train', 'validation', 'test')

    Returns:
        int: Total number of conversations.
    """
    analyzer = DatasetAnalyzer(dataset_name, config, split)
    return analyzer.get_dataset_size()


def analyze(
    config: BaseConfig,
    dataset: Optional[BaseMapDataset] = None,
    analyzer=None,
    *,
    analysis_type: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> list[dict[str, Any]]:
    """Runs dataset analysis using the provided configuration.

    Args:
        config: The configuration to use for analysis.
        dataset: The dataset to analyze. If None, will be loaded from config.
        analyzer: The analyzer to use for analysis. If unspecified, the analyzer
            will be inferred from `config`.
        analysis_type: Type of analysis to perform (e.g., 'stats', 'quality',
            'patterns').
        output_dir: Directory to store output files.

    Returns:
        List[Dict[str, Any]]: A list of analysis results.
    """
    if not analyzer:
        analyzer = _get_analyzer(config)

    # Load dataset if not provided
    if dataset is None:
        try:
            dataset = _load_dataset_from_config(config)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return [{"error": f"Failed to load dataset: {e}"}]

    result = {"analysis_type": analysis_type, "status": "not_implemented"}
    return [result]


def simple_dataset_demo():
    """Simple demonstration of reading an AlpacaDataset.

    Prints the first conversation in the dataset.
    """
    try:
        from oumi.datasets.sft.alpaca import AlpacaDataset

        # Create a small AlpacaDataset instance
        small_alpaca = AlpacaDataset()

        # Get the first conversation
        first_conversation = small_alpaca.conversation(0)

        # Print the first conversation
        print("First conversation from AlpacaDataset:")
        print("=" * 50)
        print(repr(first_conversation))
        print("=" * 50)

        return first_conversation

    except ImportError as e:
        logger.error(f"Failed to import AlpacaDataset: {e}")
        print("Error: Could not import AlpacaDataset")
        return None
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        print(f"Error: {e}")
        return None
