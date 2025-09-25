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
from typing import Optional, Union

import pandas as pd

from oumi.core.datasets import BaseMapDataset
from oumi.utils.logging import logger


class QueryFilter:
    """Handles querying and filtering of analysis results."""

    def __init__(
        self,
        analysis_df: Optional[pd.DataFrame] = None,
        items_df: Optional[pd.DataFrame] = None,
        rows_df: Optional[pd.DataFrame] = None,
        dataset: Optional[BaseMapDataset] = None,
    ):
        """Initialize the QueryFilter with analysis data.

        Args:
            analysis_df: Merged analysis DataFrame with both message and
                conversation metrics
            items_df: Items-level analysis DataFrame
            rows_df: Rows-level analysis DataFrame
            dataset: Original dataset for filtering operations
        """
        self._analysis_df = analysis_df
        self._items_df = items_df
        self._rows_df = rows_df
        self._dataset = dataset

    def update_data(
        self,
        analysis_df: Optional[pd.DataFrame] = None,
        items_df: Optional[pd.DataFrame] = None,
        rows_df: Optional[pd.DataFrame] = None,
        dataset: Optional[BaseMapDataset] = None,
    ) -> None:
        """Update the data used for querying and filtering.

        Args:
            analysis_df: Merged analysis DataFrame
            items_df: Items-level analysis DataFrame
            rows_df: Rows-level analysis DataFrame
            dataset: Original dataset
        """
        if analysis_df is not None:
            self._analysis_df = analysis_df
        if items_df is not None:
            self._items_df = items_df
        if rows_df is not None:
            self._rows_df = rows_df
        if dataset is not None:
            self._dataset = dataset

    def query(self, query_expression: str) -> pd.DataFrame:
        """Query the analysis results using pandas query syntax.

        Args:
            query_expression: Pandas query expression (e.g., "char_count > 10")

        Returns:
            DataFrame containing rows that match the query expression

        Raises:
            RuntimeError: If analysis data is not available.
            ValueError: If the query expression is invalid.
        """
        # Check if analysis has been run
        if self._analysis_df is None:
            raise RuntimeError(
                "Analysis data is not available. Please ensure analysis has been run "
                "and data has been provided to the QueryFilter."
            )

        # Apply the query filter
        try:
            filtered_df = self._analysis_df.query(query_expression)
            logger.info(f"Query '{query_expression}' returned {len(filtered_df)} rows")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise ValueError(f"Invalid query expression: {query_expression}") from e

        return filtered_df

    def query_items(self, query_expression: str) -> pd.DataFrame:
        """Query item-level analysis results using pandas query expression.

        Args:
            query_expression: Pandas query expression to filter item analysis results

        Returns:
            DataFrame with filtered item analysis results

        Raises:
            RuntimeError: If items data is not available.
            ValueError: If the query expression is invalid.

        Examples:
            # Filter for long conversations
            long_conversations = query_filter.query_items(
                "item_length_token_count > 1000"
            )
        """
        # Check if analysis has been run
        if self._items_df is None:
            raise RuntimeError(
                "Items data is not available. Please ensure analysis has been run "
                "and items data has been provided to the QueryFilter."
            )

        # Apply the query filter
        try:
            filtered_df = self._items_df.query(query_expression)
            logger.info(f"Query '{query_expression}' returned {len(filtered_df)} rows")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise ValueError(f"Invalid query expression '{query_expression}': {e}")

        return filtered_df

    def query_rows(self, query_expression: str) -> pd.DataFrame:
        """Query row-level analysis results using pandas query expression.

        Args:
            query_expression: Pandas query expression to filter row analysis results

        Returns:
            DataFrame with filtered row analysis results

        Raises:
            RuntimeError: If rows data is not available.
            ValueError: If the query expression is invalid.

        Examples:
            # Filter for long messages
            long_messages = query_filter.query_rows(
                "row_length_word_count > 100"
            )
        """
        # Check if analysis has been run
        if self._rows_df is None:
            raise RuntimeError(
                "Rows data is not available. Please ensure analysis has been run "
                "and rows data has been provided to the QueryFilter."
            )

        # Apply the query filter
        try:
            filtered_df = self._rows_df.query(query_expression)
            logger.info(f"Query '{query_expression}' returned {len(filtered_df)} rows")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise ValueError(f"Invalid query expression '{query_expression}': {e}")

        return filtered_df

    def filter_dataset(self, query_expression: str) -> BaseMapDataset:
        """Filter the original dataset based on analysis results.

        This method uses analysis results to filter the original dataset, returning
        a new dataset object containing only the conversations that match the query.

        Args:
            query_expression: Pandas query expression to filter analysis results

        Returns:
            A new dataset object containing only the filtered conversations

        Raises:
            RuntimeError: If analysis data or dataset is not available.
            ValueError: If the dataset cannot be filtered.

        Examples:
            # Filter for conversations with short messages
            short_dataset = query_filter.filter_dataset("length_word_count < 10")

            # Filter for conversations with assistant messages
            assistant_dataset = query_filter.filter_dataset("role == 'assistant'")

            # Filter for conversations with long user messages
            long_user_dataset = query_filter.filter_dataset(
                "role == 'user' and length_word_count > 100"
            )
        """
        # Get filtered analysis results
        filtered_df = self.query(query_expression)

        # Get unique sample indices from filtered results
        item_indices = filtered_df.item_index.unique().tolist()

        # Create a new dataset with only the filtered conversations
        if self._dataset is None:
            raise RuntimeError("Dataset is not available for filtering")

        filtered_dataset = self._create_filtered_dataset(item_indices)

        logger.info(
            f"Filtered dataset: {len(item_indices)} items "
            f"out of {len(self._dataset)} total"
        )

        return filtered_dataset

    def _create_filtered_dataset(self, item_indices: list[int]) -> BaseMapDataset:
        """Create a new dataset containing only the specified samples.

        Args:
            item_indices: List of item indices to include

        Returns:
            A new dataset object with the same format as the original

        Raises:
            ValueError: If dataset is not available.
        """
        if self._dataset is None:
            raise ValueError("Dataset is not available, cannot create filtered dataset")

        # Deep copy the original dataset to preserve all attributes and methods
        filtered_dataset = copy.deepcopy(self._dataset)

        # Filter the DataFrame to only include the specified samples
        original_df = self._dataset.data
        filtered_dataset._data = original_df.iloc[item_indices].copy()

        # Update the dataset name to indicate it's filtered
        filtered_dataset.dataset_name = f"{self._dataset.dataset_name}_filtered"

        return filtered_dataset

    @property
    def analysis_df(self) -> Union[pd.DataFrame, None]:
        """Get the merged analysis DataFrame.

        Returns:
            DataFrame with columns prefixed by message_ and conversation_ for each
            analyzer
        """
        return self._analysis_df

    @property
    def items_df(self) -> Union[pd.DataFrame, None]:
        """Get the items-level analysis DataFrame.

        Returns:
            DataFrame with item-level metrics
        """
        return self._items_df

    @property
    def rows_df(self) -> Union[pd.DataFrame, None]:
        """Get the rows-level analysis DataFrame.

        Returns:
            DataFrame with row-level metrics
        """
        return self._rows_df

    @property
    def dataset(self) -> Union[BaseMapDataset, None]:
        """Get the original dataset.

        Returns:
            The original dataset used for filtering
        """
        return self._dataset
