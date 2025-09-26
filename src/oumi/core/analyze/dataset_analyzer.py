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
        items_df: Optional[pd.DataFrame] = None,
        rows_df: Optional[pd.DataFrame] = None,
        schema: Optional[dict] = None,
    ):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzeConfig object containing analysis parameters
            dataset: Optional pre-loaded dataset for conversation data
            items_df: Optional DataFrame with items (conversations, evaluation pairs,
                etc.)
            rows_df: Optional DataFrame with rows (messages, fields, etc.) within items
            schema: Optional column schema dict for explicit field types
        """
        # Use ConfigReader to initialize all components
        config_reader = ConfigReader()
        components = config_reader.read_config(
            config=config,
            dataset=dataset,
            items_df=items_df,
            rows_df=rows_df,
            column_config=schema,
        )

        # Initialize attributes from components
        self.config = components.config
        self.dataset = components.dataset
        self._items_df = components.items_df
        self._rows_df = components.rows_df
        self.schema = components.column_config
        self.sample_analyzers = components.sample_analyzers
        self.tokenizer = components.tokenizer
        self.dataset_name = components.dataset_name
        self.split = config.split

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

        self._compute_conversation_metrics()

        # Generate and store all results
        analysis_summary = self.summary_generator.generate_analysis_summary(
            analysis_df=self._analysis_df,
            items_df=self._items_df,
            rows_df=self._rows_df,
            analysis_results=self._analysis_results,
            sample_analyzers=self.sample_analyzers,
        )

        # Store all results in the results manager
        if self._items_df is None or self._rows_df is None:
            raise RuntimeError(
                "Analysis DataFrames are None. Error in analysis process."
            )

        self.results_manager.store_results(
            analysis_results=self._analysis_results,
            analysis_df=self._analysis_df,
            items_df=self._items_df,
            rows_df=self._rows_df,
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

    def _compute_conversation_metrics(self) -> None:
        """Compute metrics for all items using DataFrame-based analyzers."""
        if self._items_df is not None and self._rows_df is not None:
            # Direct DataFrames input - use them directly
            total_items = len(self._items_df)
            items_to_analyze = total_items
            logger.info(f"Using direct DataFrames with {total_items} items")
        elif self.dataset is not None:
            # Conversation dataset input
            total_items = len(self.dataset)
            items_to_analyze = total_items
        else:
            raise ValueError("Either dataset or (items_df, rows_df) must be provided")

        # Apply item limit if specified
        max_items = self.config.sample_count if self.config else None

        if max_items is not None:
            items_to_analyze = min(total_items, max_items)
            logger.info(
                f"Limiting analysis to first {max_items} "
                f"items (dataset has {total_items} total)"
            )

        logger.info(
            "Analyzing %d items using DataFrame-based analyzers",
            items_to_analyze,
        )

        if self._items_df is not None and self._rows_df is not None:
            # Direct DataFrames - process them directly
            self._process_direct_dataframes(items_to_analyze)
        else:
            # Conversation dataset - prepare and analyze using handlers
            self._process_conversation_dataset(items_to_analyze)

        # Store metadata
        self._analysis_results = DatasetAnalysisResult(
            dataset_name=self.dataset_name or "",
            total_conversations=total_items,
            conversations_analyzed=items_to_analyze,
        )

    def _process_direct_dataframes(self, items_to_analyze: int) -> None:
        """Process direct DataFrames input using DataFrameAnalyzer."""
        if self.items_df is None or self.rows_df is None:
            raise ValueError("Both items_df and rows_df must be provided")

        # Work with copies to avoid modifying original data
        items_df: pd.DataFrame = self.items_df.copy()
        rows_df: pd.DataFrame = self.rows_df.copy()

        # Limit to requested number of items
        if items_to_analyze < len(items_df):
            item_indices = items_df["item_index"].iloc[:items_to_analyze].tolist()
            items_df = pd.DataFrame(items_df[items_df["item_index"].isin(item_indices)])
            rows_df = pd.DataFrame(rows_df[rows_df["item_index"].isin(item_indices)])

        # Use DataFrameAnalyzer for the core analysis logic
        analysis_result = self.dataframe_analyzer.analyze_dataframe_list(
            input_data_list=[
                DataFrameWithSchema(items_df, self.schema, "items"),
                DataFrameWithSchema(rows_df, self.schema, "rows"),
            ],
            merge_on="item_index",
        )

        # Store processed DataFrames
        self._items_df = analysis_result.items_df
        self._rows_df = analysis_result.rows_df
        self._analysis_df = analysis_result.merged_df

    def _process_conversation_dataset(self, items_to_analyze: int) -> None:
        """Process conversation dataset using handlers and analyzers."""
        if self.dataset is None:
            raise ValueError("Dataset must be provided for conversation processing")

        # Step 1: Use ConversationHandler to convert dataset to DataFrames
        complete_items_df, complete_rows_df = (
            self.conversation_handler.convert_dataset_to_dataframes(
                dataset=self.dataset,
                items_to_analyze=items_to_analyze,
                dataset_name=self.dataset_name,
            )
        )

        # Step 2: Use DataFrameAnalyzer to analyze the complete DataFrames
        analysis_result = self.dataframe_analyzer.analyze_dataframe_list(
            input_data_list=[
                DataFrameWithSchema(complete_items_df, self.schema, "items"),
                DataFrameWithSchema(complete_rows_df, self.schema, "rows"),
            ],
            merge_on="item_index",
        )

        # Store the processed DataFrames
        self._items_df = analysis_result.items_df
        self._rows_df = analysis_result.rows_df
        self._analysis_df = analysis_result.merged_df

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
    def rows_df(self) -> Optional[pd.DataFrame]:
        """Get the rows-level analysis DataFrame.

        Returns:
            DataFrame with row-level metrics prefixed by row_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        return self.results_manager.rows_df

    @property
    def items_df(self) -> Optional[pd.DataFrame]:
        """Get the items-level analysis DataFrame.

        Returns:
            DataFrame with item-level metrics prefixed by item_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        return self.results_manager.items_df

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
