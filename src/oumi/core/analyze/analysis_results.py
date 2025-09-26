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
from typing import Any, Optional, Union

import pandas as pd

from oumi.core.analyze.query_filter import QueryFilter
from oumi.core.datasets import BaseMapDataset


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


class AnalysisResultsManager:
    """Manages analysis results and provides access to DataFrames and summaries.

    This class encapsulates all the result management logic including:
    - Storing analysis DataFrames
    - Providing property access to results
    - Delegating queries to QueryFilter
    - Managing analysis metadata
    """

    def __init__(self):
        """Initialize the results manager."""
        # Analysis results
        self._analysis_results: Optional[DatasetAnalysisResult] = None
        self._analysis_df: Optional[pd.DataFrame] = None
        self._conversations_df: Optional[pd.DataFrame] = None
        self._messages_df: Optional[pd.DataFrame] = None
        self._analysis_summary: Optional[dict[str, Any]] = None

        # Query filter for result querying
        self._query_filter = QueryFilter()

    def store_results(
        self,
        analysis_results: DatasetAnalysisResult,
        analysis_df: pd.DataFrame,
        conversations_df: pd.DataFrame,
        messages_df: pd.DataFrame,
        analysis_summary: dict[str, Any],
        dataset: Optional[BaseMapDataset] = None,
    ) -> None:
        """Store all analysis results.

        Args:
            analysis_results: Metadata about the analysis
            analysis_df: Merged analysis DataFrame
            conversations_df: Conversations-level DataFrame
            messages_df: Messages-level DataFrame
            analysis_summary: Analysis summary dictionary
            dataset: Optional original dataset for filtering
        """
        self._analysis_results = analysis_results
        self._analysis_df = analysis_df
        self._conversations_df = conversations_df
        self._messages_df = messages_df
        self._analysis_summary = analysis_summary

        # Update query filter with new data
        self._query_filter.update_data(
            analysis_df=analysis_df,
            conversations_df=conversations_df,
            messages_df=messages_df,
            dataset=dataset,
        )

    @property
    def analysis_results(self) -> Optional[DatasetAnalysisResult]:
        """Get the analysis results if available.

        Returns:
            DatasetAnalysisResult if analysis has been run, None otherwise
        """
        return self._analysis_results

    @property
    def analysis_df(self) -> Union[pd.DataFrame, None]:
        """Get the merged analysis DataFrame with both message and conversation metrics.

        Returns:
            DataFrame with columns prefixed by message_ and conversation_ for each
            analyzer

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._analysis_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the analysis DataFrame."
            )
        return self._analysis_df

    @property
    def messages_df(self) -> Union[pd.DataFrame, None]:
        """Get the messages-level analysis DataFrame.

        Returns:
            DataFrame with message-level metrics prefixed by message_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._messages_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the messages DataFrame."
            )
        return self._messages_df

    @property
    def conversations_df(self) -> Union[pd.DataFrame, None]:
        """Get the conversations-level analysis DataFrame.

        Returns:
            DataFrame with conversation-level metrics prefixed by conversation_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._conversations_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the conversations DataFrame."
            )
        return self._conversations_df

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

    def query(self, query_expression: str) -> pd.DataFrame:
        """Query the analysis results using pandas query syntax.

        Args:
            query_expression: Pandas query expression (e.g., "char_count > 10")

        Returns:
            DataFrame containing rows that match the query expression

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        return self._query_filter.query(query_expression)

    def query_items(self, query_expression: str) -> pd.DataFrame:
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
        return self._query_filter.query_items(query_expression)

    def query_rows(self, query_expression: str) -> pd.DataFrame:
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
        return self._query_filter.query_rows(query_expression)

    def filter_dataset(self, query_expression: str) -> BaseMapDataset:
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
        return self._query_filter.filter_dataset(query_expression)
