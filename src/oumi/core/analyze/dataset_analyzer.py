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

import pandas as pd

from oumi.core.analyze.analysis_results import (
    AnalysisResultsManager,
    DatasetAnalysisResult,
)
from oumi.core.analyze.config_reader import ConfigReader
from oumi.core.analyze.conversation_handler import ConversationHandler
from oumi.core.analyze.dataframe_analyzer import DataFrameAnalyzer, DataFrameWithSchema
from oumi.core.analyze.summary_generator import SummaryGenerator
from oumi.core.configs import AnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.logging import logger


class DatasetAnalyzer:
    """Orchestrates the analysis of datasets using multiple sample analyzers."""

    def __init__(
        self,
        config: AnalyzeConfig,
        dataset: Optional[BaseMapDataset] = None,
        dataframes: Optional[list[DataFrameWithSchema]] = None,
        schema: Optional[dict] = None,
    ):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzeConfig object containing analysis parameters
            dataset: Optional pre-loaded dataset for conversation data
            dataframes: Optional list of DataFrameWithSchema objects for direct analysis
            schema: Optional column schema dict for explicit field types
        """
        # Use ConfigReader to initialize components
        config_reader = ConfigReader()
        components = config_reader.read_config(
            config=config,
            dataset=dataset,
            schema=schema,
        )

        # Initialize attributes from components
        self.dataset = components.dataset
        self._dataframes = dataframes
        self.schema = components.schema or schema or {}
        self.sample_analyzers = components.sample_analyzers
        self.tokenizer = components.tokenizer
        self.dataset_name = components.dataset_name
        self.split = config.split
        self.sample_count = config.sample_count

        # Initialize conversation handler for conversation-specific processing
        self.conversation_handler = ConversationHandler(
            tokenizer=self.tokenizer,
            schema=self.schema,
        )

        # Initialize DataFrame analyzer for core analysis logic
        self.dataframe_analyzer = DataFrameAnalyzer(
            sample_analyzers=self.sample_analyzers,
        )

        # Initialize summary generator for analysis summaries
        self.summary_generator = SummaryGenerator(decimal_precision=2)

        # Initialize results manager for storing and accessing results
        self.results_manager = AnalysisResultsManager()

    def analyze_dataset(self) -> None:
        """Analyze the dataset.

        This method performs analysis using the configured sample analyzers.

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

        max_items = self.sample_count
        dataframe_list, total_items, items_to_analyze = self._prepare_dataframe_list(
            max_items
        )

        self._analysis_result = self.dataframe_analyzer.analyze_dataframe_list(
            input_data_list=dataframe_list,
            merge_on="item_index",
        )

        self._analysis_results = DatasetAnalysisResult(
            dataset_name=self.dataset_name or "",
            total_conversations=total_items,
            conversations_analyzed=items_to_analyze,
        )

        analysis_summary = self.summary_generator.generate_analysis_summary(
            analysis_df=self._analysis_result.merged_df,
            conversations_df=self._analysis_result.conversations_df,
            messages_df=self._analysis_result.messages_df,
            analysis_results=self._analysis_results,
            sample_analyzers=self.sample_analyzers,
        )

        self.results_manager.store_results(
            analysis_results=self._analysis_results,
            analysis_df=self._analysis_result.merged_df,
            conversations_df=self._analysis_result.conversations_df,
            messages_df=self._analysis_result.messages_df,
            analysis_summary=analysis_summary,
            dataset=self.dataset,
        )

    @property
    def analysis_results(self) -> Optional[DatasetAnalysisResult]:
        """Get the analysis results if available.

        Returns:
            DatasetAnalysisResult if analysis has been run, None otherwise
        """
        return self.results_manager.analysis_results

    def _prepare_dataframe_list(
        self, max_items: Optional[int] = None
    ) -> tuple[list[DataFrameWithSchema], int, int]:
        """Prepare DataFrameWithSchema list from input source with optional limiting.

        Args:
            max_items: Maximum number of items to analyze (None for no limit)

        Returns:
            Tuple of (dataframe_list, total_items, items_to_analyze)
        """
        if self._dataframes is not None:
            # Direct DataFrameWithSchema list provided
            total_items = 0
            if self._dataframes:
                # Use the first DataFrame to determine total items
                # Assume all DataFrames have the same number of items
                total_items = len(self._dataframes[0].dataframe)

            logger.info(
                f"Using provided DataFrameWithSchema list with {total_items} items"
            )

            # Apply limits if specified
            items_to_analyze = total_items
            if max_items is not None:
                items_to_analyze = min(total_items, max_items)
                if items_to_analyze < total_items:
                    logger.info(
                        f"Limiting analysis to first {max_items} "
                        f"items (dataset has {total_items} total)"
                    )

            limited_dataframes = self._apply_limits_to_dataframes(
                self._dataframes, items_to_analyze
            )
            return limited_dataframes, total_items, items_to_analyze

        elif self.dataset is not None:
            # Conversation dataset input - convert to DataFrames
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

            # Use ConversationHandler to convert dataset to DataFrames
            conversations_df, messages_df = (
                self.conversation_handler.convert_dataset_to_dataframes(
                    dataset=self.dataset,
                    items_to_analyze=items_to_analyze,
                    dataset_name=self.dataset_name,
                )
            )

            dataframe_list = [
                DataFrameWithSchema(conversations_df, self.schema, "conversations"),
                DataFrameWithSchema(messages_df, self.schema, "messages"),
            ]
            return dataframe_list, total_items, items_to_analyze

        else:
            raise ValueError("Either dataframes or dataset must be provided")

    def _apply_limits_to_dataframes(
        self, dataframe_list: list[DataFrameWithSchema], items_to_analyze: int
    ) -> list[DataFrameWithSchema]:
        """Apply item limits to all DataFrames.

        Args:
            dataframe_list: List of DataFrameWithSchema objects
            items_to_analyze: Number of items to analyze

        Returns:
            List of DataFrameWithSchema objects with limits applied
        """
        if not dataframe_list:
            return dataframe_list

        limited_dataframes = []
        for df_with_schema in dataframe_list:
            limited_df = df_with_schema.dataframe.iloc[:items_to_analyze].copy()

            limited_dataframes.append(
                DataFrameWithSchema(
                    limited_df, df_with_schema.schema, df_with_schema.name
                )
            )

        return limited_dataframes

    def query(self, query_expression: str) -> pd.DataFrame:
        """Query the analysis results using pandas query syntax.

        Args:
            query_expression: Pandas query expression (e.g., "char_count > 10")

        Returns:
            DataFrame containing rows that match the query expression

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        return self.results_manager.query(query_expression)

    @property
    def analysis_df(self) -> Optional[pd.DataFrame]:
        """Get the merged analysis DataFrame with both message and conversation metrics.

        Returns:
            DataFrame with columns prefixed by message_ and conversation_ for each
            analyzer

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        return self.results_manager.analysis_df

    @property
    def messages_df(self) -> Optional[pd.DataFrame]:
        """Get the messages-level analysis DataFrame.

        Returns:
            DataFrame with message-level metrics prefixed by message_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        return self.results_manager.messages_df

    @property
    def conversations_df(self) -> Optional[pd.DataFrame]:
        """Get the conversations-level analysis DataFrame.

        Returns:
            DataFrame with conversation-level metrics prefixed by conversation_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        return self.results_manager.conversations_df

    def query_items(
        self,
        query_expression: str,
    ) -> pd.DataFrame:
        """Query item-level analysis results using pandas query expression.

        Args:
            query_expression: Pandas query expression to filter item analysis
                results

        Returns:
            DataFrame with filtered item analysis results

        Raises:
            RuntimeError: If analysis has not been run yet.

        Examples:
            # Filter for long conversations
            long_conversations = analyzer.query_items(
                "item_length_token_count > 1000"
            )
        """
        return self.results_manager.query_items(query_expression)

    def query_rows(
        self,
        query_expression: str,
    ) -> pd.DataFrame:
        """Query row-level analysis results using pandas query expression.

        Args:
            query_expression: Pandas query expression to filter row analysis
                results

        Returns:
            DataFrame with filtered row analysis results

        Raises:
            RuntimeError: If analysis has not been run yet.

        Examples:
            # Filter for long messages
            long_messages = analyzer.query_rows(
                "row_length_word_count > 100"
            )
        """
        return self.results_manager.query_rows(query_expression)

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
        return self.results_manager.filter_dataset(query_expression)

    @property
    def analysis_summary(self) -> dict[str, Any]:
        """Get the comprehensive analysis summary.

        Returns:
            Dictionary containing comprehensive dataset analysis summary

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        return self.results_manager.analysis_summary
