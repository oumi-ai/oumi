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

from dataclasses import dataclass
from typing import Any, Optional

from oumi.core.analyze.column_types import ColumnType, ContentType
from oumi.core.configs import AnalyzeConfig, DatasetSource
from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import REGISTRY
from oumi.utils.analysis_utils import (
    build_tokenizer_from_config,
    load_dataset_from_config,
)
from oumi.utils.logging import logger


@dataclass
class AnalysisComponents:
    """Container for all components needed for dataset analysis.

    Attributes:
        config: The analysis configuration
        dataset: The loaded dataset (if using conversation format)
        schema: Schema dict for explicit field types
        sample_analyzers: Dictionary of initialized sample analyzers
        tokenizer: The tokenizer instance (if configured)
        dataset_name: Name of the dataset
    """

    config: AnalyzeConfig
    dataset: Optional[BaseMapDataset]
    schema: dict[str, Any]
    sample_analyzers: dict[str, Any]
    tokenizer: Optional[Any]
    dataset_name: str


class ConfigReader:
    """Reads configuration and initializes all components needed for dataset analysis.

    This class handles:
    - Reading and validating configuration
    - Building tokenizer from config
    - Loading dataset from config or using provided data
    - Initializing sample analyzers
    - Setting up column configurations
    """

    def __init__(self):
        """Initialize the ConfigReader."""
        pass

    def read_config(
        self,
        config: AnalyzeConfig,
        dataset: Optional[BaseMapDataset] = None,
        schema: Optional[dict] = None,
    ) -> AnalysisComponents:
        """Read configuration and initialize all analysis components.

        Args:
            config: AnalyzeConfig object containing analysis parameters
            dataset: Optional pre-loaded dataset for conversation data
            schema: Optional schema dict for explicit field types

        Returns:
            AnalysisComponents containing all initialized components

        Raises:
            ValueError: If configuration is invalid or required components are missing
        """
        # Build tokenizer from config if provided
        tokenizer = build_tokenizer_from_config(config.tokenizer_config)

        # Initialize dataset based on source
        if config.dataset_source == DatasetSource.DIRECT:
            dataset, schema = self._handle_direct_source(config, dataset, schema)
        elif config.dataset_source == DatasetSource.CONFIG:
            dataset, schema = self._handle_config_source(config, dataset, tokenizer)
        else:
            raise ValueError(f"Invalid dataset_source: {config.dataset_source}")

        # Get dataset name
        dataset_name = self._get_dataset_name(config, dataset)

        # Validate schema
        self._validate_schema(schema)

        # Initialize sample analyzers
        sample_analyzers = self._initialize_sample_analyzers(config, tokenizer)

        return AnalysisComponents(
            config=config,
            dataset=dataset,
            schema=schema,
            sample_analyzers=sample_analyzers,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
        )

    def _handle_direct_source(
        self,
        config: AnalyzeConfig,
        dataset: Optional[BaseMapDataset],
        schema: Optional[dict],
    ) -> tuple[Optional[BaseMapDataset], dict]:
        """Handle DatasetSource.DIRECT configuration.

        Args:
            config: The analysis configuration
            dataset: Optional pre-loaded dataset
            schema: Optional schema configuration

        Returns:
            Tuple of (dataset, schema)

        Raises:
            ValueError: If required components are missing for direct mode
        """
        if dataset is not None:
            # Use provided dataset
            logger.info(f"Using provided dataset with {len(dataset)} conversations")
            # Setup schema for conversation format
            if schema is None:
                schema = self._get_conversation_schema()
            return dataset, schema
        else:
            raise ValueError(
                "Config specifies dataset_source=DatasetSource.DIRECT but no "
                "dataset was provided. Please provide a dataset."
            )

    def _handle_config_source(
        self,
        config: AnalyzeConfig,
        dataset: Optional[BaseMapDataset],
        tokenizer: Optional[Any],
    ) -> tuple[Optional[BaseMapDataset], dict]:
        """Handle DatasetSource.CONFIG configuration.

        Args:
            config: The analysis configuration
            dataset: Should be None for config source
            tokenizer: Optional tokenizer instance

        Returns:
            Tuple of (dataset, column_config)

        Raises:
            ValueError: If dataset is provided when using config source
        """
        if dataset is not None:
            raise ValueError(
                f"Dataset provided but config.dataset_source is "
                f"'{config.dataset_source.value}'. When using "
                f"DatasetSource.CONFIG, do not pass a dataset to the "
                f"constructor. Set dataset_source=DatasetSource.DIRECT "
                f"if you want to use the provided dataset."
            )

        dataset = load_dataset_from_config(config, tokenizer)
        logger.info(f"Loaded dataset from config: {config.dataset_name}")

        # Setup schema for conversation format
        schema = self._get_conversation_schema()

        return dataset, schema

    def _get_dataset_name(
        self, config: AnalyzeConfig, dataset: Optional[BaseMapDataset]
    ) -> str:
        """Get the dataset name from config or dataset.

        Args:
            config: The analysis configuration
            dataset: Optional dataset instance

        Returns:
            The dataset name
        """
        if config.dataset_name:
            return config.dataset_name
        elif dataset is not None:
            return getattr(dataset, "dataset_name", "Custom Dataset")
        else:
            return "Custom Dataset"

    def _get_conversation_schema(self) -> dict:
        """Get column configuration for conversation format based on known structure.

        Returns:
            Dictionary mapping column names to their configuration.
        """
        return {
            # Items DataFrame columns (conversation-level)
            "item_id": {
                "type": ColumnType.STRING,
                "content_type": ContentType.METADATA,
                "description": "Conversation identifier",
            },
            "item_type": {
                "type": ColumnType.STRING,
                "content_type": ContentType.METADATA,
                "description": "Type of item (conversation)",
            },
            "rendered_item": {
                "type": ColumnType.STRING,
                "content_type": ContentType.TEXT,
                "description": "Rendered conversation for token counting and display",
            },
            # Rows DataFrame columns (message-level)
            "row_index": {
                "type": ColumnType.INT,
                "content_type": ContentType.METADATA,
                "description": "Message index within conversation",
            },
            "role": {
                "type": ColumnType.STRING,
                "content_type": ContentType.METADATA,
                "description": "Message role (user/assistant/system)",
            },
            "content": {
                "type": ColumnType.STRING,
                "content_type": ContentType.TEXT,
                "description": "Message text content",
            },
            # Additional fields that might be present in conversations
            "timestamp": {
                "type": ColumnType.TIMESTAMP,
                "content_type": ContentType.METADATA,
                "description": "Message timestamp",
            },
            "processing_time": {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.NUMERIC,
                "description": "AI processing time in seconds",
            },
            "model": {
                "type": ColumnType.STRING,
                "content_type": ContentType.METADATA,
                "description": "Model used for generation",
            },
            "temperature": {
                "type": ColumnType.FLOAT,
                "content_type": ContentType.METADATA,
                "description": "Sampling temperature",
            },
            "max_tokens": {
                "type": ColumnType.INT,
                "content_type": ContentType.METADATA,
                "description": "Maximum tokens to generate",
            },
        }

    def _validate_schema(self, schema: dict) -> None:
        """Validate that schema is properly formatted.

        Args:
            schema: Schema configuration dictionary

        Raises:
            AssertionError: If column configuration is invalid
        """
        for col_name, config in schema.items():
            assert "type" in config, f"Column {col_name} must have 'type'"
            assert "content_type" in config, (
                f"Column {col_name} must have 'content_type'"
            )
            # content_type can be "metadata" (not analyzed) or a valid content type

    def _initialize_sample_analyzers(
        self, config: AnalyzeConfig, tokenizer: Optional[Any]
    ) -> dict[str, Any]:
        """Initialize sample analyzer plugins from configuration.

        Args:
            config: The analysis configuration
            tokenizer: Optional tokenizer instance

        Returns:
            Dictionary mapping analyzer IDs to analyzer instances
        """
        sample_analyzers = {}
        if config.analyzers is None:
            return sample_analyzers

        for analyzer_params in config.analyzers:
            try:
                # Get the analyzer class from the registry
                analyzer_class = REGISTRY.get_sample_analyzer(analyzer_params.id)
                if analyzer_class is None:
                    raise ValueError(
                        f"Sample analyzer '{analyzer_params.id}' not found in registry"
                    )

                # Prepare parameters for analyzer constructor
                analyzer_kwargs = dict(analyzer_params.params)

                if tokenizer is not None:
                    analyzer_kwargs["tokenizer"] = tokenizer

                # Create analyzer instance with keyword arguments
                sample_analyzer = analyzer_class(**analyzer_kwargs)
                sample_analyzers[analyzer_params.id] = sample_analyzer
                logger.info(f"Initialized sample analyzer: {analyzer_params.id}")
            except Exception as e:
                logger.error(
                    f"Failed to initialize sample analyzer {analyzer_params.id}: {e}"
                )
                logger.error(f"Analyzer configuration: {analyzer_params}")
        return sample_analyzers
