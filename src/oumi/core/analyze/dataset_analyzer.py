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

import copy
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union, cast

import pandas as pd

from oumi.core.analyze.dataframe_analyzer import DataFrameAnalyzer, DataFrameWithSchema
from oumi.core.configs import AnalyzeConfig, DatasetSource
from oumi.core.datasets import BaseMapDataset
from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset
from oumi.core.registry import REGISTRY
from oumi.utils.analysis_utils import (
    build_tokenizer_from_config,
    compute_statistics,
    convert_dataset_to_dataframes,
    get_schema_for_format,
    load_dataset_from_config,
)
from oumi.utils.logging import logger


@dataclass
class MessageAnalysisResult:
    """Result of analyzing a single message in a conversation.

    Attributes:
        message_index: Index of the message within the conversation
        role: Role of the message sender (e.g., 'user', 'assistant')
        message_id: Unique identifier for the message
        text_content: The text content of the message
        analyzer_metrics: Dictionary containing analyzer metrics for this message
    """

    message_index: int
    role: str
    message_id: str
    text_content: str
    analyzer_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary with flattened analyzer metrics.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


@dataclass
class ConversationAnalysisResult:
    """Result of analyzing a conversation as a whole.

    Attributes:
        analyzer_metrics: Dictionary containing analyzer metrics for the conversation
    """

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
    """

    dataset_name: str
    total_conversations: int
    conversations_analyzed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


class DatasetAnalyzer:
    """Orchestrates the analysis of datasets using multiple sample analyzers."""

    def __init__(self, config: AnalyzeConfig, dataset: Optional[BaseMapDataset] = None):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzeConfig object containing all analysis parameters
            dataset: Optional pre-loaded dataset. If provided, this dataset will be used
                    instead of loading from the config.
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.split = config.split

        # Build tokenizer from config if provided
        self.tokenizer = build_tokenizer_from_config(config.tokenizer_config)

        # Use provided dataset or load from config based on dataset_source
        if config.dataset_source == DatasetSource.DIRECT:
            # Direct mode: must provide dataset
            if dataset is None:
                raise ValueError(
                    "Config specifies dataset_source=DatasetSource.DIRECT but no "
                    "dataset was provided. Either pass a dataset to "
                    "DatasetAnalyzer.__init__() or "
                    "set dataset_source=DatasetSource.CONFIG.value."
                )

            self.dataset = dataset
            # Use the provided dataset name if config doesn't have one
            if not self.dataset_name:
                self.dataset_name = getattr(dataset, "dataset_name", "Custom Dataset")
            # Handle iterable datasets that don't support len()
            if isinstance(dataset, BaseIterableDataset):
                logger.info(f"Using provided streaming dataset '{self.dataset_name}'")
            else:
                logger.info(
                    f"Using provided dataset '{self.dataset_name}' with "
                    f"{len(dataset)} conversations"
                )
        elif config.dataset_source == DatasetSource.CONFIG:
            # Config mode: load dataset from config parameters
            if dataset is not None:
                raise ValueError(
                    f"Dataset provided but config.dataset_source is "
                    f"'{config.dataset_source.value}'. When using "
                    f"DatasetSource.CONFIG, do not pass a dataset to the "
                    f"constructor. Set dataset_source=DatasetSource.DIRECT "
                    f"if you want to use the provided dataset."
                )

            # Load dataset with the tokenizer
            self.dataset = load_dataset_from_config(config, self.tokenizer)
            logger.info(f"Loaded dataset from config: {self.dataset_name}")
        else:
            raise ValueError(f"Invalid dataset_source: {config.dataset_source}")

        self.sample_analyzers = self._initialize_sample_analyzers()

        # Initialize dataframe analyzer with sample analyzers
        self.dataframe_analyzer = DataFrameAnalyzer(self.sample_analyzers)

        # Initialize analysis results as None
        self._analysis_results: Optional[DatasetAnalysisResult] = None
        self._merged_df: Optional[pd.DataFrame] = None
        self._message_df: Optional[pd.DataFrame] = None
        self._conversation_df: Optional[pd.DataFrame] = None
        self._analysis_summary: Optional[dict[str, Any]] = None

        # Decimal precision for rounding metrics
        self._decimal_precision = 2

    def _get_schema_for_dataset(self) -> dict:
        """Get column schema configuration based on dataset type.

        Detects the appropriate schema based on the dataset class inheritance.
        Based on analysis of all 60 Oumi datasets:
        - 37 datasets (SFT/Vision-SFT/GRPO) convert to conversation format → use 'oumi'
        - 23 datasets (pretraining/DPO/KTO) maintain original structure → use specific

        Returns:
            Dictionary mapping column names to their configuration.
        """
        # Detect dataset type based on the dataset class
        dataset_type = self._detect_dataset_type()

        try:
            return get_schema_for_format(dataset_type)
        except ValueError:
            # Fallback to conversation schema for unknown types
            logger.warning(
                f"Unknown dataset type '{dataset_type}', using conversation schema"
            )
            return get_schema_for_format("oumi")

    def _detect_dataset_type(self) -> str:
        """Detect the dataset type based on the dataset class and configuration.

        Returns:
            String indicating the dataset type for schema selection.
        """
        if self.dataset is None:
            # No dataset provided, use config format or default to conversation
            return getattr(self.config, "dataset_format", None) or "oumi"

        # Check dataset class inheritance hierarchy for accurate detection
        dataset_class_bases = [base.__name__ for base in self.dataset.__class__.__mro__]

        # Datasets that convert to conversation format during loading
        if any(
            base in dataset_class_bases
            for base in [
                "BaseSftDataset",
                "VisionLanguageSftDataset",
                "BaseExperimentalGrpoDataset",
            ]
        ):
            return "oumi"  # All convert to conversation format

        # Datasets that maintain original structure
        elif "BasePretrainingDataset" in dataset_class_bases:
            return "pretraining"
        elif any(
            base in dataset_class_bases
            for base in ["BaseDpoDataset", "VisionLanguageDpoDataset"]
        ):
            return "dpo"
        elif "BaseExperimentalKtoDataset" in dataset_class_bases:
            return "kto"
        else:
            # Check if we have explicit format from config
            config_format = getattr(self.config, "dataset_format", None)
            if config_format in [
                "alpaca",
                "prompt_response",
                "dpo",
                "pretraining",
                "kto",
            ]:
                return config_format
            else:
                # Default to conversation format for unknown SFT-like datasets
                return "oumi"

    def _initialize_sample_analyzers(self) -> dict[str, Any]:
        """Initialize sample analyzer plugins from configuration.

        Returns:
            Dictionary mapping analyzer IDs to analyzer instances
        """
        sample_analyzers = {}
        for analyzer_params in self.config.analyzers:
            try:
                # Get the analyzer class from the registry
                analyzer_class = REGISTRY.get_sample_analyzer(analyzer_params.id)
                if analyzer_class is None:
                    raise ValueError(
                        f"Sample analyzer '{analyzer_params.id}' not found in registry"
                    )

                # Prepare parameters for analyzer constructor
                analyzer_kwargs = dict(analyzer_params.params)

                if self.tokenizer is not None:
                    analyzer_kwargs["tokenizer"] = self.tokenizer

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

    def analyze_dataset(self) -> None:
        """Analyze the dataset and store results internally.

        This method performs both message-level and conversation-level analysis
        using the configured sample analyzers. Each analyzer processes entire
        conversations and returns metrics for both individual messages and
        conversations as a whole. Results are stored internally and can be
        accessed via the query() method.

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

        # Handle iterable datasets differently to avoid downloading everything
        if isinstance(self.dataset, BaseIterableDataset):
            # For iterable datasets, we can't get the total length without iterating
            # So we'll use the sample_count directly and iterate only what we need
            conversations_to_analyze = (
                self.config.sample_count or 1000
            )  # Default limit for streaming
            total_conversations = None  # Unknown for iterable datasets
            logger.info(
                f"Analyzing up to {conversations_to_analyze} conversations from "
                f"streaming dataset"
            )
        else:
            # For map datasets, we can get the total length
            total_conversations = len(self.dataset)
            conversations_to_analyze = min(
                total_conversations, self.config.sample_count or total_conversations
            )
            logger.info(
                f"Analyzing {conversations_to_analyze} of {total_conversations} "
                f"conversations"
            )

        dataframe_list, total_items, items_to_analyze = self._prepare_dataframe_list(
            conversations_to_analyze
        )

        analysis_result = self.dataframe_analyzer.analyze_dataframe_list(
            input_data_list=dataframe_list,
            merge_on=["conversation_index", "conversation_id"],
        )
        self._merged_df = analysis_result.merged_df
        self._message_df = analysis_result.messages_df
        self._conversation_df = analysis_result.conversations_df

        self._analysis_results = DatasetAnalysisResult(
            dataset_name=self.dataset_name or "",
            total_conversations=total_conversations or conversations_to_analyze,
            conversations_analyzed=conversations_to_analyze,
        )

        # Generate and store the analysis summary after metrics are computed
        self._analysis_summary = self._generate_analysis_summary()

    def _prepare_dataframe_list(
        self, max_items: Optional[int] = None
    ) -> tuple[list[DataFrameWithSchema], int, int]:
        """Prepare DataFrameWithSchema list from input source with optional limiting.

        Args:
            max_items: Maximum number of items to analyze (None for no limit)

        Returns:
            Tuple of (dataframe_list, total_items, items_to_analyze)
        """
        if self.dataset is not None:
            # Conversation dataset input - convert to DataFrames
            if isinstance(self.dataset, BaseIterableDataset):
                # For iterable datasets, we can't get the total length
                total_items = max_items or 1000  # Use max_items or default
                items_to_analyze = total_items
                logger.info(
                    f"Converting streaming dataset with up to {items_to_analyze} items"
                )
            else:
                # For map datasets, we can get the total length
                total_items = len(self.dataset)
                logger.info(f"Converting conversation dataset with {total_items} items")

                # Determine how many items to analyze
                items_to_analyze = total_items
                if max_items is not None:
                    items_to_analyze = min(total_items, max_items)
                    if items_to_analyze < total_items:
                        logger.info(
                            f"Limiting analysis to first {max_items} "
                            f"items (dataset has {total_items} total)"
                        )

            # Use utility function to convert dataset to DataFrames
            conversations_df, messages_df = convert_dataset_to_dataframes(
                dataset=self.dataset,
                items_to_analyze=items_to_analyze,
                dataset_name=self.dataset_name or "Unknown Dataset",
            )

            schema = self._get_schema_for_dataset()

            dataframe_list = [
                DataFrameWithSchema(conversations_df, schema, "conversations"),
                DataFrameWithSchema(messages_df, schema, "messages"),
            ]
            return dataframe_list, total_items, items_to_analyze

        else:
            raise ValueError("Either dataframes or dataset must be provided")

    @property
    def analysis_results(self) -> Optional[DatasetAnalysisResult]:
        """Get the analysis results if available.

        Returns:
            DatasetAnalysisResult if analysis has been run, None otherwise
        """
        return self._analysis_results

    def query(self, query_expression: str) -> pd.DataFrame:
        """Query the analysis results using pandas query syntax.

        Args:
            query_expression: Pandas query expression (e.g., "char_count > 10")

        Returns:
            DataFrame containing rows that match the query expression

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        # Check if analysis has been run
        if self._merged_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to query the analysis results."
            )

        # Apply the query filter
        try:
            filtered_df = self._merged_df.query(query_expression)
            logger.info(f"Query '{query_expression}' returned {len(filtered_df)} rows")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise ValueError(f"Invalid query expression: {query_expression}") from e

        return filtered_df

    @property
    def analysis_df(self) -> Union[pd.DataFrame, None]:
        """Get the merged analysis DataFrame with both message and conversation metrics.

        Returns:
            DataFrame with columns prefixed by message_ and conversation_ for each
            analyzer

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._merged_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the analysis DataFrame."
            )
        return self._merged_df

    @property
    def message_df(self) -> Union[pd.DataFrame, None]:
        """Get the message-level analysis DataFrame.

        Returns:
            DataFrame with message-level metrics prefixed by message_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._message_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the message DataFrame."
            )
        return self._message_df

    @property
    def conversation_df(self) -> Union[pd.DataFrame, None]:
        """Get the conversation-level analysis DataFrame.

        Returns:
            DataFrame with conversation-level metrics prefixed by conversation_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._conversation_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the conversation DataFrame."
            )
        return self._conversation_df

    def query_conversations(
        self,
        query_expression: str,
    ) -> pd.DataFrame:
        """Query conversation-level analysis results using pandas query expression.

        Args:
            query_expression: Pandas query expression to filter conversation analysis
                results

        Returns:
            DataFrame with filtered conversation analysis results

        Raises:
            RuntimeError: If analysis has not been run yet.

        Examples:
            # Filter for short conversations
            long_conversations = analyzer.query_conversations(
                "length_token_count > 1000"
            )
        """
        # Check if analysis has been run
        if self._conversation_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to query conversation results."
            )

        # Apply the query filter
        try:
            filtered_df = self._conversation_df.query(query_expression)
            logger.info(f"Query '{query_expression}' returned {len(filtered_df)} rows")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise ValueError(f"Invalid query expression '{query_expression}': {e}")

        return filtered_df

    def filter(
        self,
        query_expression: str,
    ) -> Union[BaseMapDataset, BaseIterableDataset]:
        """Filter the original dataset based on analysis results.

        This method uses analysis results to filter the original dataset, returning
        a new dataset object containing only the conversations that match the query.

        Args:
            query_expression: Pandas query expression to filter analysis results

        Returns:
            A new dataset object containing only the filtered conversations

        Raises:
            RuntimeError: If analysis has not been run yet.

        Examples:
            # Filter for conversations with short messages
            short_dataset = analyzer.filter("length_word_count < 10")

            # Filter for conversations with assistant messages
            assistant_dataset = analyzer.filter("role == 'assistant'")

            # Filter for conversations with long user messages
            long_user_dataset = analyzer.filter(
                "role == 'user' and length_word_count > 100"
            )
        """
        # Get filtered analysis results
        filtered_df = self.query(query_expression)

        # Get unique conversation indices from filtered results
        conversation_indices = filtered_df.conversation_index.unique().tolist()

        # Create a new dataset with only the filtered conversations
        filtered_dataset = self._create_filtered_dataset(conversation_indices)

        # Get total dataset size, handling iterable datasets
        from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

        if isinstance(self.dataset, BaseIterableDataset):
            total_size = "unknown (streaming)"
        else:
            total_size = str(len(self.dataset))

        logger.info(
            f"Filtered dataset: {len(conversation_indices)} conversations "
            f"out of {total_size} total"
        )

        return filtered_dataset

    def _create_filtered_dataset(
        self, conversation_indices: list[int]
    ) -> Union[BaseMapDataset, BaseIterableDataset]:
        """Create a new dataset containing only the specified conversations.

        Args:
            conversation_indices: List of conversation indices to include

        Returns:
            A new dataset object with the same format as the original
        """
        # Deep copy the original dataset to preserve all attributes and methods
        filtered_dataset = copy.deepcopy(self.dataset)

        # Filter the DataFrame to only include the specified conversations
        # Note: This only works for map datasets, not iterable datasets
        from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

        if isinstance(self.dataset, BaseIterableDataset):
            # For iterable datasets, we can't filter by index
            # Return the original dataset as filtering is not supported
            return filtered_dataset

        original_df = self.dataset.data
        filtered_dataset._data = original_df.iloc[conversation_indices].copy()

        # Update the dataset name to indicate it's filtered
        filtered_dataset.dataset_name = f"{self.dataset.dataset_name}_filtered"

        return filtered_dataset

    def _generate_analysis_summary(self) -> dict[str, Any]:
        """Generate a comprehensive summary of dataset analysis results.

        This method aggregates metrics from all analyzers to provide insights useful
        for assessing datasets. It computes statistics like averages,
        standard deviations, min/max values, and efficiency metrics.

        Returns:
            Dictionary containing comprehensive dataset analysis summary with:
            - Dataset overview statistics
            - Message-level aggregated metrics
            - Conversation-level aggregated metrics
        """
        # Check if we have data to analyze
        if self._merged_df is None or self._merged_df.empty:
            return {"error": "No analysis data available"}

        # TODO: Refactor summary methods to be dataset agnostic
        # Currently these methods assume conversation dataset structure with
        # messages/conversations.
        # They should be generalized to work with any dataset type and column structure.
        summary = {
            "dataset_overview": self._get_dataset_overview(),
            "message_level_summary": self._get_message_level_summary(),
            "conversation_level_summary": self._get_conversation_level_summary(),
            "conversation_turns": self._get_conversation_turns_summary(),
        }

        return summary

    @property
    def analysis_summary(self) -> dict[str, Any]:
        """Get the comprehensive analysis summary.

        Returns:
            Dictionary containing comprehensive dataset analysis summary

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._analysis_summary is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to generate the analysis summary."
            )
        return self._analysis_summary

    def _get_dataset_overview(self) -> dict[str, Any]:
        """Get basic dataset overview statistics."""
        if self._analysis_results is None:
            return {}

        return {
            "dataset_name": self._analysis_results.dataset_name,
            "total_conversations": self._analysis_results.total_conversations,
            "conversations_analyzed": self._analysis_results.conversations_analyzed,
            "dataset_coverage_percentage": round(
                100.0
                * self._analysis_results.conversations_analyzed
                / self._analysis_results.total_conversations
                if self._analysis_results.total_conversations > 0
                else 0,
                self._decimal_precision,
            ),
            "total_messages": len(self._message_df)
            if self._message_df is not None
            else 0,
            "analyzers_used": list(self.sample_analyzers.keys()),
        }

    def _get_message_level_summary(self) -> dict[str, Any]:
        """Get aggregated message-level metrics across all analyzers."""
        if self._message_df is None or self._message_df.empty:
            return {}

        # Get all analyzer columns (columns that are not base message columns)
        base_columns = {
            "conversation_index",
            "conversation_id",
            "message_index",
            "message_id",
            "role",
            "text_content",
        }

        analyzer_columns = [
            col
            for col in self._message_df.columns
            if col not in base_columns
            and pd.api.types.is_numeric_dtype(self._message_df[col])
        ]

        summary = {}

        for col in analyzer_columns:
            # Extract analyzer name and metric from column
            # Format: text_content_{analyzer}_{metric}
            # Example: text_content_length_analyzer_char_count
            parts = col.split("_")
            if len(parts) >= 5:  # text_content_analyzer_metric_type
                if parts[0] == "text" and parts[1] == "content":
                    # The analyzer name and metric are in the remaining parts
                    # For "text_content_length_analyzer_char_count":
                    # parts[2:] = ["length", "analyzer", "char", "count"]
                    # We need to find where the analyzer name ends and metric begins

                    # Look for known metric suffixes to split correctly
                    remaining_parts = parts[2:]
                    metric_suffixes = [
                        "char_count",
                        "word_count",
                        "sentence_count",
                        "token_count",
                    ]

                    analyzer_name = None
                    metric_name = None

                    # Try to find a metric suffix
                    for i in range(
                        1, len(remaining_parts)
                    ):  # Start from 1 to ensure analyzer_name is not empty
                        potential_metric = "_".join(remaining_parts[i:])
                        if any(
                            potential_metric.endswith(suffix)
                            for suffix in metric_suffixes
                        ):
                            analyzer_name = "_".join(remaining_parts[:i])
                            metric_name = f"text_content_{potential_metric}"
                            break

                    # Fallback: assume last two parts are metric
                    if analyzer_name is None:
                        if len(remaining_parts) >= 2:
                            analyzer_name = "_".join(remaining_parts[:-2])
                            metric_name = (
                                f"text_content_{remaining_parts[-2]}_"
                                f"{remaining_parts[-1]}"
                            )

                    if analyzer_name and metric_name:
                        if analyzer_name not in summary:
                            summary[analyzer_name] = {}

                        # Compute statistics for numeric columns
                        values = cast(pd.Series, self._message_df[col].dropna())
                        if len(values) > 0:
                            summary[analyzer_name][metric_name] = compute_statistics(
                                values, self._decimal_precision
                            )

        return summary

    def _get_conversation_level_summary(self) -> dict[str, Any]:
        """Get aggregated conversation-level metrics across all analyzers."""
        if self._conversation_df is None or self._conversation_df.empty:
            return {}

        # Get all analyzer columns (columns that are not base conversation columns)
        base_columns = {
            "conversation_index",
            "conversation_id",
            "num_messages",
        }

        analyzer_columns = [
            col
            for col in self._conversation_df.columns
            if col not in base_columns
            and pd.api.types.is_numeric_dtype(self._conversation_df[col])
        ]

        summary = {}

        for col in analyzer_columns:
            # Use the same parsing logic as message level summary
            # Format: text_content_{analyzer}_{metric}
            # (for conversation-level aggregated metrics)
            parts = col.split("_")
            if len(parts) >= 5:  # text_content_analyzer_metric_type
                if parts[0] == "text" and parts[1] == "content":
                    remaining_parts = parts[2:]
                    metric_suffixes = [
                        "char_count",
                        "word_count",
                        "sentence_count",
                        "token_count",
                    ]

                    analyzer_name = None
                    metric_name = None

                    # Try to find a metric suffix
                    for i in range(
                        1, len(remaining_parts)
                    ):  # Start from 1 to ensure analyzer_name is not empty
                        potential_metric = "_".join(remaining_parts[i:])
                        if any(
                            potential_metric.endswith(suffix)
                            for suffix in metric_suffixes
                        ):
                            analyzer_name = "_".join(remaining_parts[:i])
                            metric_name = f"text_content_{potential_metric}"
                            break

                    # Fallback: assume last two parts are metric
                    if analyzer_name is None:
                        if len(remaining_parts) >= 2:
                            analyzer_name = "_".join(remaining_parts[:-2])
                            metric_name = (
                                f"text_content_{remaining_parts[-2]}_"
                                f"{remaining_parts[-1]}"
                            )

                    if analyzer_name and metric_name:
                        if analyzer_name not in summary:
                            summary[analyzer_name] = {}

                        # Compute statistics for numeric columns
                        values = cast(pd.Series, self._conversation_df[col].dropna())
                        if len(values) > 0:
                            summary[analyzer_name][metric_name] = compute_statistics(
                                values, self._decimal_precision
                            )

        return summary

    def _get_conversation_turns_summary(self) -> dict[str, Any]:
        """Get conversation turn statistics summary.

        Returns:
            Dictionary containing conversation turn statistics
        """
        if self._message_df is None or self._message_df.empty:
            return {}

        # groupby().size() always returns a Series, but we cast it because
        # type checker can't infer this
        turns_per_conversation = cast(
            pd.Series, self._message_df.groupby("conversation_id").size()
        )
        return compute_statistics(turns_per_conversation, self._decimal_precision)
