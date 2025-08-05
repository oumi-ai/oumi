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
from tqdm import tqdm

from oumi.core.analyze.types import (
    ConversationAnalysisResult,
    DatasetAnalysisResult,
    MessageAnalysisResult,
)
from oumi.core.configs import AnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import REGISTRY
from oumi.utils.analysis_utils import load_dataset_from_config
from oumi.utils.logging import logger


class DatasetAnalyzer:
    """Orchestrates the analysis of datasets using multiple sample analyzers."""

    def __init__(self, config: AnalyzeConfig):
        """Initialize the dataset analyzer.

        Args:
            config: Configuration for the analysis
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.split = config.split
        self.tokenizer = config.tokenizer

        # Load dataset from config
        self.dataset = load_dataset_from_config(config)

        # Initialize sample analyzers
        self.sample_analyzers = {}
        for analyzer_config in config.analyzers:
            analyzer_class = REGISTRY.get_sample_analyzer(analyzer_config.id)
            if analyzer_class is None:
                raise ValueError(
                    f"Sample analyzer '{analyzer_config.id}' not found in registry"
                )

            # Prepare parameters for analyzer constructor
            analyzer_kwargs = dict(analyzer_config.params)

            # Add tokenizer if provided in config
            if self.tokenizer is not None:
                analyzer_kwargs["tokenizer"] = self.tokenizer

            analyzer = analyzer_class(**analyzer_kwargs)
            self.sample_analyzers[analyzer_config.id] = analyzer

        # Analysis results
        self._analysis_results: Optional[DatasetAnalysisResult] = None
        self._merged_df: Optional[pd.DataFrame] = None
        self._message_df: Optional[pd.DataFrame] = None
        self._conversation_df: Optional[pd.DataFrame] = None

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

        # Validate sample_count
        if self.config.sample_count is not None and self.config.sample_count <= 0:
            raise ValueError("sample_count must be positive")

        conversations_to_analyze = min(
            total_conversations, self.config.sample_count or total_conversations
        )

        logger.info(f"Analyzing {conversations_to_analyze} conversations")

        # Analyze all conversations in a single pass
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
            conversations_to_analyze = min(total_conversations, max_conversations)
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
                    analyzer_result = analyzer.compute_metrics(
                        conversation, self.tokenizer
                    )

                    # Convert to DataFrames with prefixed columns
                    message_df = self._convert_messages_to_df(
                        analyzer_result.messages, analyzer_id, conversation_id, conv_idx
                    )
                    conversation_df = self._convert_conversation_to_df(
                        analyzer_result.conversation,
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
