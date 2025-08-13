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
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from oumi.core.configs import AnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import REGISTRY
from oumi.utils.analysis_utils import load_dataset_from_config
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

    def __init__(self, config: AnalyzeConfig):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzeConfig object containing all analysis parameters
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.split = config.split
        self.tokenizer = config.tokenizer

        self.dataset = load_dataset_from_config(config)

        self.sample_analyzers = self._initialize_sample_analyzers()

        # Initialize analysis results as None
        self._analysis_results: Optional[DatasetAnalysisResult] = None
        self._merged_df: Optional[pd.DataFrame] = None
        self._message_df: Optional[pd.DataFrame] = None
        self._conversation_df: Optional[pd.DataFrame] = None

        # Decimal precision for rounding metrics
        self.decimal_precision = 2

    def set_decimal_precision(self, precision: int) -> None:
        """Set the decimal precision for rounding metrics in analysis summary.

        Args:
            precision: Number of decimal places to round to (e.g., 2 for 0.12,
                3 for 0.123)
        """
        if not isinstance(precision, int) or precision < 0:
            raise ValueError("Decimal precision must be a non-negative integer")
        self.decimal_precision = precision

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
        """
        # Run analysis if not already done
        if self._merged_df is None:
            logger.info("Analysis not yet run, starting analysis...")
            self.analyze_dataset()
            # After analysis, _merged_df should be populated
            assert self._merged_df is not None

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
        """
        if self._merged_df is None:
            logger.info("Analysis not yet run, starting analysis...")
            self.analyze_dataset()
        return self._merged_df

    @property
    def message_df(self) -> Union[pd.DataFrame, None]:
        """Get the message-level analysis DataFrame.

        Returns:
            DataFrame with message-level metrics prefixed by message_
        """
        if self._message_df is None:
            logger.info("Analysis not yet run, starting analysis...")
            self.analyze_dataset()
        return self._message_df

    @property
    def conversation_df(self) -> Union[pd.DataFrame, None]:
        """Get the conversation-level analysis DataFrame.

        Returns:
            DataFrame with conversation-level metrics prefixed by conversation_
        """
        if self._conversation_df is None:
            logger.info("Analysis not yet run, starting analysis...")
            self.analyze_dataset()
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

        Examples:
            # Filter for short conversations
            long_conversations = analyzer.query_conversations(
                "length_token_count > 1000"
            )
        """
        # Run analysis if not already done
        if self._conversation_df is None:
            logger.info("Analysis not yet run, starting analysis...")
            self.analyze_dataset()
            # After analysis, _conversation_df should be populated
            assert self._conversation_df is not None

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

    def get_analysis_summary(self) -> dict[str, Any]:
        """Generate a comprehensive summary of dataset analysis results.

        This method aggregates metrics from all analyzers to provide insights useful
        for assessing datasets. It computes statistics like averages,
        standard deviations, min/max values, and efficiency metrics.

        Returns:
            Dictionary containing comprehensive dataset analysis summary with:
            - Dataset overview statistics
            - Message-level aggregated metrics
            - Conversation-level aggregated metrics
            - Tokenization efficiency metrics
            - Quality indicators and recommendations
        """
        # Ensure analysis has been run
        if self._merged_df is None:
            logger.info("Analysis not yet run, starting analysis...")
            self.analyze_dataset()
            assert self._merged_df is not None

        if self._merged_df.empty:
            return {"error": "No analysis data available"}

        summary = {
            "dataset_overview": self._get_dataset_overview(),
            "message_level_summary": self._get_message_level_summary(),
            "conversation_level_summary": self._get_conversation_level_summary(),
            "tokenization_efficiency": self._get_tokenization_efficiency(),
            "quality_indicators": self._get_quality_indicators(),
            "recommendations": self._get_recommendations(),
        }

        return summary

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
                self.decimal_precision,
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
                    values = self._message_df[col].dropna()
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = {
                            "count": len(values),
                            "mean": round(float(values.mean()), self.decimal_precision),
                            "std": round(float(values.std()), self.decimal_precision),
                            "min": float(values.min()),
                            "max": float(values.max()),
                            "median": round(
                                float(values.median()), self.decimal_precision
                            ),
                            "q25": round(
                                float(values.quantile(0.25)), self.decimal_precision
                            ),
                            "q75": round(
                                float(values.quantile(0.75)), self.decimal_precision
                            ),
                        }

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
                    values = self._conversation_df[col].dropna()
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = {
                            "count": len(values),
                            "mean": round(float(values.mean()), self.decimal_precision),
                            "std": round(float(values.std()), self.decimal_precision),
                            "min": float(values.min()),
                            "max": float(values.max()),
                            "median": round(
                                float(values.median()), self.decimal_precision
                            ),
                            "q25": round(
                                float(values.quantile(0.25)), self.decimal_precision
                            ),
                            "q75": round(
                                float(values.quantile(0.75)), self.decimal_precision
                            ),
                        }

        # Add conversation turn statistics if available
        if self._message_df is not None and not self._message_df.empty:
            turns_per_conversation = self._message_df.groupby("conversation_id").size()
            # Handle pandas Series operations with proper type conversion
            mean_val = turns_per_conversation.mean()
            std_val = turns_per_conversation.std()
            min_val = turns_per_conversation.min()
            max_val = turns_per_conversation.max()
            median_val = turns_per_conversation.median()
            q25_val = turns_per_conversation.quantile(0.25)
            q75_val = turns_per_conversation.quantile(0.75)

            summary["conversation_turns"] = {
                "count": len(turns_per_conversation),
                "mean": round(float(mean_val), self.decimal_precision)
                if mean_val is not None
                else 0.0,
                "std": round(float(std_val), self.decimal_precision)
                if std_val is not None
                else 0.0,
                "min": int(min_val) if min_val is not None else 0,
                "max": int(max_val) if max_val is not None else 0,
                "median": round(float(median_val), self.decimal_precision)
                if median_val is not None
                else 0.0,
                "q25": round(float(q25_val), self.decimal_precision)
                if q25_val is not None
                else 0.0,
                "q75": round(float(q75_val), self.decimal_precision)
                if q75_val is not None
                else 0.0,
            }

        return summary

    def _get_tokenization_efficiency(self) -> dict[str, Any]:
        """Get tokenization efficiency metrics."""
        efficiency = {}

        # Check if we have token count data
        if self._merged_df is None:
            return {"note": "No merged data available"}

        token_columns = [
            col
            for col in self._merged_df.columns
            if "token_count" in col
            and pd.api.types.is_numeric_dtype(self._merged_df[col])
        ]

        if not token_columns:
            return {"note": "No token count data available"}

        for col in token_columns:
            if col in [
                "conversation_id",
                "conversation_index",
                "message_index",
                "role",
                "message_id",
                "text_content",
            ]:
                continue

            # Extract analyzer name from column
            parts = col.split("_")
            if len(parts) >= 3:
                analyzer_name = parts[1]

                if analyzer_name not in efficiency:
                    efficiency[analyzer_name] = {}

                # Get corresponding count columns for efficiency calculations
                word_col = col.replace("token_count", "word_count")
                sentence_col = col.replace("token_count", "sentence_count")
                char_col = col.replace("token_count", "char_count")

                if word_col in self._merged_df.columns:
                    # Calculate tokens per word
                    token_word_df = self._merged_df[[col, word_col]].dropna()
                    if len(token_word_df) > 0:
                        tokens_per_word = token_word_df[col] / token_word_df[word_col]
                        # Handle pandas Series operations with proper type conversion
                        # Convert to numpy arrays and use numpy methods for type safety
                        # Type ignore for pandas Series methods that Pyright doesn't
                        # recognize
                        tokens_array = tokens_per_word.to_numpy()  # type: ignore
                        mean_val = float(np.mean(tokens_array))
                        std_val = float(np.std(tokens_array))
                        min_val = float(np.min(tokens_array))
                        max_val = float(np.max(tokens_array))
                        median_val = float(np.median(tokens_array))

                        efficiency[analyzer_name]["tokens_per_word"] = {
                            "mean": round(float(mean_val), self.decimal_precision)
                            if mean_val is not None
                            else 0.0,
                            "std": round(float(std_val), self.decimal_precision)
                            if std_val is not None
                            else 0.0,
                            "min": round(float(min_val), self.decimal_precision)
                            if min_val is not None
                            else 0.0,
                            "max": round(float(max_val), self.decimal_precision)
                            if max_val is not None
                            else 0.0,
                            "median": round(float(median_val), self.decimal_precision)
                            if median_val is not None
                            else 0.0,
                        }

                if char_col in self._merged_df.columns:
                    # Calculate tokens per character
                    token_char_df = self._merged_df[[col, char_col]].dropna()
                    if len(token_char_df) > 0:
                        tokens_per_char = token_char_df[col] / token_char_df[char_col]
                        # Handle pandas Series operations with proper type conversion
                        # Convert to numpy arrays and use numpy methods for type safety
                        # Type ignore for pandas Series methods that Pyright doesn't
                        # recognize
                        tokens_array = tokens_per_char.to_numpy()  # type: ignore
                        mean_val = float(np.mean(tokens_array))
                        std_val = float(np.std(tokens_array))
                        min_val = float(np.min(tokens_array))
                        max_val = float(np.max(tokens_array))
                        median_val = float(np.median(tokens_array))

                        efficiency[analyzer_name]["tokens_per_char"] = {
                            "mean": round(float(mean_val), self.decimal_precision)
                            if mean_val is not None
                            else 0.0,
                            "std": round(float(std_val), self.decimal_precision)
                            if std_val is not None
                            else 0.0,
                            "min": round(float(min_val), self.decimal_precision)
                            if min_val is not None
                            else 0.0,
                            "max": round(float(max_val), self.decimal_precision)
                            if max_val is not None
                            else 0.0,
                            "median": round(float(median_val), self.decimal_precision)
                            if median_val is not None
                            else 0.0,
                        }

                if sentence_col in self._merged_df.columns:
                    # Calculate tokens per sentence
                    token_sentence_df = self._merged_df[[col, sentence_col]].dropna()
                    if len(token_sentence_df) > 0:
                        tokens_per_sentence = (
                            token_sentence_df[col] / token_sentence_df[sentence_col]
                        )
                        # Handle pandas Series operations with proper type conversion
                        # Convert to numpy arrays and use numpy methods for type safety
                        # Type ignore for pandas Series methods that Pyright doesn't
                        # recognize
                        tokens_array = tokens_per_sentence.to_numpy()  # type: ignore
                        mean_val = float(np.mean(tokens_array))
                        std_val = float(np.std(tokens_array))
                        min_val = float(np.min(tokens_array))
                        max_val = float(np.max(tokens_array))
                        median_val = float(np.median(tokens_array))

                        efficiency[analyzer_name]["tokens_per_sentence"] = {
                            "mean": round(float(mean_val), self.decimal_precision)
                            if mean_val is not None
                            else 0.0,
                            "std": round(float(std_val), self.decimal_precision)
                            if std_val is not None
                            else 0.0,
                            "min": round(float(min_val), self.decimal_precision)
                            if min_val is not None
                            else 0.0,
                            "max": round(float(max_val), self.decimal_precision)
                            if max_val is not None
                            else 0.0,
                            "median": round(float(median_val), self.decimal_precision)
                            if median_val is not None
                            else 0.0,
                        }

                # Calculate compression ratio and efficiency score if we have
                # multiple metrics
                efficiency_metrics = []
                if "tokens_per_word" in efficiency[analyzer_name]:
                    efficiency_metrics.append(
                        efficiency[analyzer_name]["tokens_per_word"]["mean"]
                    )
                if "tokens_per_char" in efficiency[analyzer_name]:
                    efficiency_metrics.append(
                        efficiency[analyzer_name]["tokens_per_char"]["mean"]
                    )
                if "tokens_per_sentence" in efficiency[analyzer_name]:
                    efficiency_metrics.append(
                        efficiency[analyzer_name]["tokens_per_sentence"]["mean"]
                    )

        return efficiency

    def _get_quality_indicators(self) -> dict[str, Any]:
        """Get quality indicators for the dataset."""
        indicators = {}

        if self._merged_df is None or self._merged_df.empty:
            return {"note": "No data available for quality assessment"}

        # Message length distribution analysis
        length_columns = [
            col
            for col in self._merged_df.columns
            if any(
                metric in col
                for metric in ["char_count", "word_count", "sentence_count"]
            )
        ]

        if length_columns:
            indicators["message_length_distribution"] = {}
            for col in length_columns:
                if pd.api.types.is_numeric_dtype(self._merged_df[col]):
                    values = self._merged_df[col].dropna()
                    if len(values) > 0:
                        # Identify potential outliers (beyond 3 standard deviations)
                        mean_val = values.mean()
                        std_val = values.std()
                        outliers = values[
                            (values < mean_val - 3 * std_val)
                            | (values > mean_val + 3 * std_val)
                        ]

                        indicators["message_length_distribution"][col] = {
                            "outlier_count": len(outliers),
                            "outlier_percentage": round(
                                len(outliers) / len(values) * 100,
                                self.decimal_precision,
                            ),
                            "outlier_threshold_low": round(
                                mean_val - 3 * std_val, self.decimal_precision
                            ),
                            "outlier_threshold_high": round(
                                mean_val + 3 * std_val, self.decimal_precision
                            ),
                        }

        # Role distribution analysis
        if "role" in self._merged_df.columns:
            role_counts = self._merged_df["role"].value_counts()
            indicators["role_distribution"] = {
                "total_messages": len(self._merged_df),
                "role_counts": role_counts.to_dict(),
                "role_percentages": (
                    role_counts / len(self._merged_df) * 100
                ).to_dict(),
            }

        # Conversation length consistency
        if self._message_df is not None and not self._message_df.empty:
            turns_per_conv = self._message_df.groupby("conversation_id").size()
            indicators["conversation_length_consistency"] = {
                "single_turn_conversations": int((turns_per_conv == 1).sum()),
                "single_turn_percentage": round(
                    float((turns_per_conv == 1).sum() / len(turns_per_conv) * 100),
                    self.decimal_precision,
                ),
                "very_long_conversations": int((turns_per_conv > 10).sum()),
                "very_long_percentage": round(
                    float((turns_per_conv > 10).sum() / len(turns_per_conv) * 100),
                    self.decimal_precision,
                ),
            }

        return indicators

    def _get_recommendations(self) -> dict[str, Any]:
        """Generate recommendations based on the analysis."""
        recommendations = []

        if self._merged_df is None or self._merged_df.empty:
            return {"note": "No data available for recommendations"}

        # Check for potential issues and provide recommendations
        indicators = self._get_quality_indicators()

        # Message length recommendations
        if "message_length_distribution" in indicators:
            for metric, stats in indicators["message_length_distribution"].items():
                if stats["outlier_percentage"] > 10:
                    recommendations.append(
                        {
                            "type": "message_length_outliers",
                            "metric": metric,
                            "issue": f"High percentage of outliers "
                            f"({stats['outlier_percentage']:.1f}%)",
                            "recommendation": "Consider filtering or normalizing "
                            "extremely long/short messages",
                        }
                    )

        # Role distribution recommendations
        if "role_distribution" in indicators:
            role_dist = indicators["role_distribution"]
            if (
                "user" in role_dist["role_percentages"]
                and "assistant" in role_dist["role_percentages"]
            ):
                user_pct = role_dist["role_percentages"]["user"]
                assistant_pct = role_dist["role_percentages"]["assistant"]

                if abs(user_pct - assistant_pct) > 20:
                    recommendations.append(
                        {
                            "type": "role_imbalance",
                            "issue": f"Significant role imbalance "
                            f"(User: {user_pct:.1f}%, "
                            f"Assistant: {assistant_pct:.1f}%)",
                            "recommendation": "Consider balancing the dataset or "
                            "adjusting training weights",
                        }
                    )

        # Conversation length recommendations
        if "conversation_length_consistency" in indicators:
            conv_stats = indicators["conversation_length_consistency"]
            if conv_stats["single_turn_percentage"] > 50:
                recommendations.append(
                    {
                        "type": "conversation_diversity",
                        "issue": f"High percentage of single-turn conversations "
                        f"({conv_stats['single_turn_percentage']:.1f}%)",
                        "recommendation": "Consider including more multi-turn "
                        "conversations for better training",
                    }
                )

        # Tokenization efficiency recommendations
        efficiency = self._get_tokenization_efficiency()
        for analyzer, metrics in efficiency.items():
            if "tokens_per_word" in metrics:
                tpw_mean = metrics["tokens_per_word"]["mean"]
                if tpw_mean > 2.0:
                    recommendations.append(
                        {
                            "type": "tokenization_efficiency",
                            "analyzer": analyzer,
                            "issue": f"High tokens per word ratio ({tpw_mean:.2f})",
                            "recommendation": "Consider using a more efficient "
                            "tokenizer or preprocessing text",
                        }
                    )

        return {
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
        }
