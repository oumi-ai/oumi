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
from tqdm import tqdm

from oumi.core.configs import AnalyzeConfig, DatasetSource
from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import REGISTRY
from oumi.utils.analysis_utils import (
    build_tokenizer_from_config,
    compute_statistics,
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

        # Initialize analysis results as None
        self._analysis_results: Optional[DatasetAnalysisResult] = None
        self._merged_df: Optional[pd.DataFrame] = None
        self._message_df: Optional[pd.DataFrame] = None
        self._conversation_df: Optional[pd.DataFrame] = None
        self._analysis_summary: Optional[dict[str, Any]] = None

        # Decimal precision for rounding metrics
        self._decimal_precision = 2

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

        total_conversations = len(self.dataset)
        conversations_to_analyze = min(
            total_conversations, self.config.sample_count or total_conversations
        )

        logger.info(f"Analyzing {conversations_to_analyze} conversations")

        self._compute_conversation_metrics()

        # Generate and store the analysis summary after metrics are computed
        self._analysis_summary = self._generate_analysis_summary()

    @property
    def analysis_results(self) -> Optional[DatasetAnalysisResult]:
        """Get the analysis results if available.

        Returns:
            DatasetAnalysisResult if analysis has been run, None otherwise
        """
        return self._analysis_results

    def _compute_conversation_metrics(self) -> None:
        """Compute metrics for all conversations in the dataset.

        This method processes each conversation and creates DataFrames with
        prefixed columns for each analyzer's metrics.
        """
        total_conversations = len(self.dataset)

        # Apply conversation limit if specified
        max_conversations = self.config.sample_count

        if max_conversations is not None:
            # AnalyzeConfig ensures sample_count is greater than 0
            conversations_to_analyze = min(total_conversations, max_conversations)
            logger.info(
                f"Limiting analysis to first {max_conversations} "
                f"conversations (dataset has {total_conversations} total)"
            )
        else:
            conversations_to_analyze = total_conversations

        logger.info(
            "Analyzing %d conversations for both message-level and "
            "conversation-level metrics",
            conversations_to_analyze,
        )

        # Collect DataFrames for messages and conversations
        message_dfs = []
        conversation_dfs = []

        # Use tqdm for progress monitoring
        for conv_idx in tqdm(
            range(conversations_to_analyze),
            desc=f"Analyzing conversations in {self.dataset_name}",
            unit="conv",
        ):
            conversation = self.dataset.conversation(conv_idx)
            conversation_id = conversation.conversation_id or f"conv_{conv_idx}"

            # Process each analyzer for this conversation
            conversation_has_data = False
            for analyzer_id, analyzer in self.sample_analyzers.items():
                try:
                    message_results, conversation_result = analyzer.analyze_sample(
                        conversation, self.tokenizer
                    )

                    # Convert to DataFrames with prefixed columns
                    message_df = self._convert_messages_to_df(
                        message_results, analyzer_id, conversation_id, conv_idx
                    )
                    conversation_df = self._convert_conversation_to_df(
                        conversation_result,
                        analyzer_id,
                        conversation_id,
                        conv_idx,
                    )

                    # Always add conversation_df (even if empty) to ensure conversation
                    # is represented
                    conversation_dfs.append(conversation_df)

                    # Only add message_df if it has data
                    if not message_df.empty:
                        message_dfs.append(message_df)
                        conversation_has_data = True

                except Exception as e:
                    logger.warning(
                        f"Analyzer {analyzer_id} failed for conversation "
                        f"{conv_idx}: {e}"
                    )

            # If no analyzers succeeded, add a placeholder row for this conversation
            if not conversation_has_data:
                # Create a placeholder row with only basic columns (no analyzer columns)
                placeholder_row = {
                    "conversation_id": conversation_id,
                    "conversation_index": conv_idx,
                    "message_index": 0,  # Add required message columns
                    "role": "system",  # Default role
                    "message_id": f"placeholder_{conv_idx}_0",
                    "text_content": "",  # Empty content
                }

                placeholder_df = pd.DataFrame([placeholder_row])
                message_dfs.append(placeholder_df)  # Add to message_dfs instead

        # Create final DataFrames
        if message_dfs:
            self._message_df = pd.concat(message_dfs, ignore_index=True)
        else:
            self._message_df = pd.DataFrame()

        if conversation_dfs:
            self._conversation_df = pd.concat(conversation_dfs, ignore_index=True)
        else:
            self._conversation_df = pd.DataFrame()

        # Create merged DataFrame with both message and conversation metrics
        if not self._message_df.empty and not self._conversation_df.empty:
            self._merged_df = self._message_df.merge(
                self._conversation_df,
                on=["conversation_id", "conversation_index"],
                how="left",
            )
        elif not self._message_df.empty:
            self._merged_df = self._message_df.copy()
        elif not self._conversation_df.empty:
            self._merged_df = self._conversation_df.copy()
        else:
            self._merged_df = pd.DataFrame()

        # Store metadata
        self._analysis_results = DatasetAnalysisResult(
            dataset_name=self.dataset_name or "",
            total_conversations=total_conversations,
            conversations_analyzed=conversations_to_analyze,
        )

    def _convert_messages_to_df(
        self,
        messages: list[MessageAnalysisResult],
        analyzer_id: str,
        conversation_id: str,
        conversation_index: int,
    ) -> pd.DataFrame:
        """Convert message results to DataFrame with prefixed columns."""
        if not messages:
            return pd.DataFrame()

        rows = []
        for message in messages:
            row = {
                "conversation_id": conversation_id,
                "conversation_index": conversation_index,
                "message_index": message.message_index,
                "role": message.role,
                "message_id": message.message_id,
                "text_content": message.text_content,
            }

            # Add analyzer metrics with message_ prefix
            for key, value in message.analyzer_metrics.items():
                row[f"message_{analyzer_id}_{key}"] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def _convert_conversation_to_df(
        self,
        conversation: ConversationAnalysisResult,
        analyzer_id: str,
        conversation_id: str,
        conversation_index: int,
    ) -> pd.DataFrame:
        """Convert conversation result to DataFrame with prefixed columns."""
        row = {
            "conversation_id": conversation_id,
            "conversation_index": conversation_index,
        }

        # Add analyzer metrics with conversation_ prefix
        for key, value in conversation.analyzer_metrics.items():
            row[f"conversation_{analyzer_id}_{key}"] = value

        return pd.DataFrame([row])

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
    ) -> BaseMapDataset:
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

        logger.info(
            f"Filtered dataset: {len(conversation_indices)} conversations "
            f"out of {len(self.dataset)} total"
        )

        return filtered_dataset

    def _create_filtered_dataset(
        self, conversation_indices: list[int]
    ) -> BaseMapDataset:
        """Create a new dataset containing only the specified conversations.

        Args:
            conversation_indices: List of conversation indices to include

        Returns:
            A new dataset object with the same format as the original
        """
        # Deep copy the original dataset to preserve all attributes and methods
        filtered_dataset = copy.deepcopy(self.dataset)

        # Filter the DataFrame to only include the specified conversations
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

        # Get all message-level analyzer columns
        message_columns = [
            col for col in self._message_df.columns if col.startswith("message_")
        ]

        summary = {}

        for col in message_columns:
            if col in [
                "message_index",
                "role",
                "message_id",
                "text_content",
                "conversation_id",
                "conversation_index",
            ]:
                continue

            # Extract analyzer name and metric from column
            # Format: message_{analyzer}_{metric}
            parts = col.split("_", 2)
            if len(parts) >= 3:
                analyzer_name = parts[1]
                metric_name = "_".join(parts[2:])

                if analyzer_name not in summary:
                    summary[analyzer_name] = {}

                # Compute statistics for numeric columns
                if pd.api.types.is_numeric_dtype(self._message_df[col]):
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

        # Get all conversation-level analyzer columns
        conversation_columns = [
            col
            for col in self._conversation_df.columns
            if col.startswith("conversation_")
        ]

        summary = {}

        for col in conversation_columns:
            if col in ["conversation_id", "conversation_index"]:
                continue

            # Extract analyzer name and metric from column
            # Format: conversation_{analyzer}_{metric}
            parts = col.split("_", 2)
            if len(parts) >= 3:
                analyzer_name = parts[1]
                metric_name = "_".join(parts[2:])

                if analyzer_name not in summary:
                    summary[analyzer_name] = {}

                # Compute statistics for numeric columns
                if pd.api.types.is_numeric_dtype(self._conversation_df[col]):
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
