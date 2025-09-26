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
from typing import Any, Optional, Union

import pandas as pd

from oumi.utils.logging import logger


@dataclass
class DataFrameWithSchema:
    """A DataFrame paired with its schema for analysis.

    Attributes:
        dataframe: The pandas DataFrame to analyze
        schema: Column schema defining types and content types
        name: Optional name for tracking/debugging purposes
    """

    dataframe: pd.DataFrame
    schema: dict
    name: Optional[str] = None


@dataclass
class AnalysisResult:
    """Result of DataFrame analysis containing processed DataFrames.

    Attributes:
        dataframes: Dictionary mapping names to processed DataFrames
        merged_df: DataFrame with merged analysis results from all input DataFrames
    """

    dataframes: dict[str, pd.DataFrame]
    merged_df: pd.DataFrame

    @property
    def conversations_df(self) -> pd.DataFrame:
        """Get the 'conversations' DataFrame."""
        return self.dataframes.get("conversations", pd.DataFrame())

    @property
    def messages_df(self) -> pd.DataFrame:
        """Get the 'messages' DataFrame."""
        return self.dataframes.get("messages", pd.DataFrame())


class DataFrameAnalyzer:
    """Core DataFrame analysis engine.

    This class encapsulates the essence of analysis: applying sample analyzers
    to pandas DataFrames with column configuration.
    """

    def __init__(
        self,
        sample_analyzers: dict[str, Any],
    ):
        """Initialize the DataFrame analyzer.

        Args:
            sample_analyzers: Dictionary of sample analyzers to apply
        """
        self.sample_analyzers = sample_analyzers

    def analyze_dataframe(
        self,
        input_data: DataFrameWithSchema,
    ) -> pd.DataFrame:
        """Apply analyzers to a DataFrame.

        Args:
            input_data: DataFrameWithSchema containing DataFrame and its schema
        Returns:
            DataFrame with analysis results added
        """
        if input_data.dataframe.empty:
            return input_data.dataframe.copy()

        result_df = input_data.dataframe.copy()
        for analyzer_id, analyzer in self.sample_analyzers.items():
            try:
                result_df = analyzer.analyze_sample(
                    result_df,
                    schema=input_data.schema,
                )
            except Exception as e:
                logger.warning(f"Analyzer {analyzer_id} failed: {e}")

        return result_df

    def analyze_dataframe_list(
        self,
        input_data_list: list[DataFrameWithSchema],
        merge_on: Union[str, list[str]],
    ) -> AnalysisResult:
        """Apply analyzers to a list of DataFrames with their schemas and merge results.

        This is a general method that can handle any number of DataFrames,
        each with its own schema, analyze each one, and then merge them sequentially.

        Args:
            input_data_list: List of DataFrameWithSchema objects to analyze and merge
            merge_on: Column(s) to merge on - can be a string or list of strings
        Returns:
            AnalysisResult with processed DataFrames and final merged result
        """
        if not input_data_list:
            return AnalysisResult(
                dataframes={},
                merged_df=pd.DataFrame(),
            )

        # Apply analyzers to all DataFrames using their respective schemas
        processed_dataframes = []
        dataframes_dict = {}

        for input_data in input_data_list:
            processed_df = self.analyze_dataframe(input_data)
            processed_dataframes.append(processed_df)

            # Store in dictionary with name if provided
            if input_data.name:
                dataframes_dict[input_data.name] = processed_df

        # Merge all DataFrames sequentially
        merged_df = self._merge_dataframe_list(processed_dataframes, merge_on)

        return AnalysisResult(
            dataframes=dataframes_dict,
            merged_df=merged_df,
        )

    def _merge_dataframe_list(
        self,
        dataframes: list[pd.DataFrame],
        merge_on: Union[str, list[str]],
    ) -> pd.DataFrame:
        """Merge a list of DataFrames sequentially.

        Args:
            dataframes: List of DataFrames to merge
            merge_on: Column(s) to merge on - can be a string or list of strings
        Returns:
            Final merged DataFrame
        """
        if not dataframes:
            return pd.DataFrame()

        # Filter out empty DataFrames
        non_empty_dfs = [df for df in dataframes if not df.empty]

        if not non_empty_dfs:
            return pd.DataFrame()

        if len(non_empty_dfs) == 1:
            return non_empty_dfs[0].copy()

        # Start with the first DataFrame and merge the rest sequentially
        result_df = non_empty_dfs[0].copy()

        for df in non_empty_dfs[1:]:
            # Normalize merge_on to a list for consistent handling
            merge_columns = [merge_on] if isinstance(merge_on, str) else merge_on

            # Check if all merge columns exist in both DataFrames
            merge_cols_in_result = all(
                col in result_df.columns for col in merge_columns
            )
            merge_cols_in_df = all(col in df.columns for col in merge_columns)

            if merge_cols_in_result and merge_cols_in_df:
                # Merge on the specified column(s)
                result_df = result_df.merge(df, on=merge_columns, how="left")
            else:
                # If merge columns don't exist, just concatenate
                # This handles cases where DataFrames have different structures
                result_df = pd.concat([result_df, df], ignore_index=True)

        return result_df
