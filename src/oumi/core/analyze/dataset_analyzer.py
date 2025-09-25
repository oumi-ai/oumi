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

from oumi.core.analyze.column_types import ColumnType, ContentType
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


@dataclass
class MessageAnalysisResult:
    """Result of analyzing a single message.

    Attributes:
        message_index: Index of the message in the conversation
        role: Role of the message (user, assistant, system, etc.)
        message_id: Unique identifier for the message
        text_content: The text content of the message
        analyzer_metrics: Dictionary of metrics computed by analyzers
    """

    message_index: int
    role: str
    message_id: str
    text_content: str
    analyzer_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


@dataclass
class ConversationAnalysisResult:
    """Result of analyzing a conversation.

    Attributes:
        analyzer_metrics: Dictionary of metrics computed by analyzers
    """

    analyzer_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


class DatasetAnalyzer:
    """Orchestrates the analysis of datasets using multiple sample analyzers."""

    def __init__(
        self,
        config: AnalyzeConfig,
        dataset: Optional[BaseMapDataset] = None,
        items_df: Optional[pd.DataFrame] = None,
        rows_df: Optional[pd.DataFrame] = None,
        column_config: Optional[dict] = None,
    ):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzeConfig object containing analysis parameters
            dataset: Optional pre-loaded dataset for conversation data
            items_df: Optional DataFrame with items (conversations, evaluation pairs,
                etc.)
            rows_df: Optional DataFrame with rows (messages, fields, etc.) within items
            column_config: Optional column configuration dict for explicit field types
        """
        # Initialize basic attributes
        self.config = config
        self.column_config = column_config or {}
        self.dataset = dataset
        self._items_df = items_df
        self._rows_df = rows_df

        self.dataset_name = config.dataset_name
        self.split = config.split

        # Build tokenizer from config if provided
        self.tokenizer = build_tokenizer_from_config(config.tokenizer_config)

        if config.dataset_source == DatasetSource.DIRECT:
            # Direct mode: must provide either dataset OR
            # (items_df, rows_df, column_config)
            if dataset is not None:
                # Use provided dataset
                self.dataset = dataset
                # Use the provided dataset name if config doesn't have one
                if not self.dataset_name:
                    self.dataset_name = getattr(
                        dataset, "dataset_name", "Custom Dataset"
                    )
                logger.info(
                    f"Using provided dataset '{self.dataset_name}' with "
                    f"{len(dataset)} conversations"
                )
            elif (
                items_df is not None
                and rows_df is not None
                and column_config is not None
            ):
                self._items_df = items_df
                self._rows_df = rows_df
                self.column_config = column_config
                self.dataset = None
                self._initialize_direct_dataframes()
                logger.info(
                    f"Using direct DataFrames input with {len(items_df)} items "
                    f"and {len(rows_df)} rows"
                )
            else:
                raise ValueError(
                    "Config specifies dataset_source=DatasetSource.DIRECT but neither "
                    "dataset nor (items_df, rows_df, column_config) were provided. "
                    "Please provide either a dataset or all three DataFrame parameters."
                )
        elif config.dataset_source == DatasetSource.CONFIG:
            if dataset is not None:
                raise ValueError(
                    f"Dataset provided but config.dataset_source is "
                    f"'{config.dataset_source.value}'. When using "
                    f"DatasetSource.CONFIG, do not pass a dataset to the "
                    f"constructor. Set dataset_source=DatasetSource.DIRECT "
                    f"if you want to use the provided dataset."
                )
            self.dataset = load_dataset_from_config(config, self.tokenizer)
            logger.info(f"Loaded dataset from config: {self.dataset_name}")
        else:
            raise ValueError(f"Invalid dataset_source: {config.dataset_source}")

        self.item_analyzers = self._initialize_item_analyzers()

        # Initialize analysis results as None
        self._analysis_results: Optional[DatasetAnalysisResult] = None
        self._analysis_df: Optional[pd.DataFrame] = None
        self._items_df: Optional[pd.DataFrame] = None
        self._rows_df: Optional[pd.DataFrame] = None
        self._analysis_summary: Optional[dict[str, Any]] = None

        # Decimal precision for rounding metrics
        self._decimal_precision = 2

    def _initialize_direct_dataframes(self) -> None:
        """Initialize analyzer with direct DataFrames input."""
        logger.info(
            f"Using direct DataFrames input with {len(self.items_df or [])} items "
            f"and {len(self.rows_df or [])} rows"
        )

        # Validate DataFrames have required columns
        self._validate_dataframes()

        # Setup analysis fields from column config
        self._setup_analysis_fields_from_config()

    def _initialize_dataset_input(self) -> None:
        """Initialize analyzer with dataset input (conversation format)."""
        logger.info(f"Using dataset input: {self.dataset_name}")

        # Setup analysis fields from column config
        self._setup_analysis_fields_from_config()

    def _initialize_config_dataset(self) -> None:
        """Initialize analyzer with config-based dataset loading."""
        # Load dataset with the tokenizer
        self.dataset = load_dataset_from_config(self.config, self.tokenizer)

        logger.info(f"Loaded dataset from config: {self.dataset_name}")

        # Setup analysis fields from column config
        self._setup_analysis_fields_from_config()

    def _setup_analysis_fields_from_config(self) -> None:
        """Setup analysis fields based on column configuration."""
        if not self.column_config:
            # For conversation format, use default column config
            if self.dataset is not None:
                self.column_config = self._get_conversation_column_config()
            else:
                raise ValueError(
                    "Column configuration is required for direct DataFrames"
                )

        # Validate column config
        self._validate_column_config()

        # Setup analysis fields
        self.text_fields = [
            col
            for col, config in self.column_config.items()
            if config.get("content_type") == ContentType.TEXT
        ]
        self.image_fields = [
            col
            for col, config in self.column_config.items()
            if config.get("content_type") == ContentType.IMAGE
        ]
        self.numeric_fields = [
            col
            for col, config in self.column_config.items()
            if config.get("content_type") == ContentType.NUMERIC
        ]
        self.audio_fields = [
            col
            for col, config in self.column_config.items()
            if config.get("content_type") == ContentType.AUDIO
        ]
        self.video_fields = [
            col
            for col, config in self.column_config.items()
            if config.get("content_type") == ContentType.VIDEO
        ]

        # Metadata fields are those with content_type="metadata"
        self.metadata_fields = [
            col
            for col, config in self.column_config.items()
            if config.get("content_type") == ContentType.METADATA
        ]

        logger.info(
            f"Setup analysis fields from config: text={self.text_fields}, "
            f"metadata={self.metadata_fields}"
        )

    def _get_conversation_column_config(self) -> dict:
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

    def _validate_dataframes(self) -> None:
        """Validate that the provided DataFrames have required columns."""
        if self.items_df is None or self.rows_df is None:
            raise ValueError("Both items_df and rows_df must be provided")

        # Validate items_df
        required_item_cols = ["item_index", "item_id", "item_type"]
        missing_item_cols = [
            col for col in required_item_cols if col not in self.items_df.columns
        ]
        if missing_item_cols:
            raise ValueError(f"items_df missing required columns: {missing_item_cols}")

        # Validate rows_df
        required_row_cols = ["item_index", "row_index", "content"]
        missing_row_cols = [
            col for col in required_row_cols if col not in self.rows_df.columns
        ]
        if missing_row_cols:
            raise ValueError(f"rows_df missing required columns: {missing_row_cols}")

        # Validate that all row item_index values exist in items_df
        row_item_indices = set(self.rows_df["item_index"].unique())
        item_indices = set(self.items_df["item_index"].unique())
        missing_items = row_item_indices - item_indices
        if missing_items:
            raise ValueError(
                f"rows_df references items not in items_df: {missing_items}"
            )

    def _validate_column_config(self) -> None:
        """Validate that column_config is properly formatted."""
        for col_name, config in self.column_config.items():
            assert "type" in config, f"Column {col_name} must have 'type'"
            assert "content_type" in config, (
                f"Column {col_name} must have 'content_type'"
            )
            # content_type can be "metadata" (not analyzed) or a valid content type

    def _conversation_to_df(
        self, conversation, conversation_id: str, conv_idx: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert a conversation to items and rows DataFrames.

        Args:
            conversation: The conversation object to convert
            conversation_id: The conversation ID
            conv_idx: The conversation index

        Returns:
            Tuple of (items_df, rows_df)
        """
        # Set column config for conversation format
        self.column_config = self._get_conversation_column_config()

        return self._conversation_to_items_rows(conversation, conversation_id, conv_idx)

    def _conversation_to_items_rows(
        self, conversation, conversation_id: str, conv_idx: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert a conversation to items and rows DataFrames with proper dtypes."""
        # Create rows DataFrame (one row per message)
        row_rows = []
        for msg_idx, message in enumerate(conversation.messages):
            row_data = {
                "item_index": conv_idx,
                "row_index": msg_idx,
                "role": message.role.value,
                "content": message.content,
            }

            # Add any additional message metadata
            if hasattr(message, "metadata") and message.metadata:
                row_data.update(message.metadata)

            row_rows.append(row_data)

        # Create rows_df with proper dtypes
        if row_rows:
            rows_df = pd.DataFrame(row_rows)
        else:
            rows_df = pd.DataFrame(
                {"item_index": [], "row_index": [], "role": [], "content": []}
            )

        # Set proper dtypes from column_config
        self._set_dtypes_from_config(rows_df)

        # Create items DataFrame (one row per conversation)
        item_data = {
            "item_index": conv_idx,
            "item_id": conversation_id,
            "item_type": "conversation",
        }

        # Add rendered sample for token counting if tokenizer is available
        if self.tokenizer is not None:
            try:
                rendered_sample = self._render_conversation_for_tokens(conversation)
                item_data["rendered_item"] = rendered_sample
            except Exception as e:
                logger.warning(
                    f"Failed to render conversation {conversation_id} for token "
                    f"counting: {e}"
                )

        # Add any additional metadata from the conversation
        if hasattr(conversation, "metadata") and conversation.metadata:
            item_data.update(conversation.metadata)

        # Create items_df with proper dtypes
        items_df = pd.DataFrame([item_data])

        # Set proper dtypes from column_config
        self._set_dtypes_from_config(items_df)

        return items_df, rows_df

    def _render_conversation_for_tokens(self, conversation) -> str:
        """Render a conversation using the tokenizer's chat template for token counting.

        Args:
            conversation: The conversation object to render

        Returns:
            Rendered conversation string
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for conversation rendering")

        # Use the tokenizer's chat template to render the conversation
        prompt_text = self.tokenizer.apply_chat_template(
            conversation,  # type: ignore
            tokenize=False,
            add_generation_prompt=False,
        )
        return str(prompt_text)

    def _initialize_item_analyzers(self) -> dict[str, Any]:
        """Initialize sample analyzer plugins from configuration.

        Returns:
            Dictionary mapping analyzer IDs to analyzer instances
        """
        item_analyzers = {}
        if self.config is None or self.config.analyzers is None:
            return item_analyzers
        for analyzer_params in self.config.analyzers:
            try:
                # Get the analyzer class from the registry
                analyzer_class = REGISTRY.get_sample_analyzer(analyzer_params.id)
                if analyzer_class is None:
                    raise ValueError(
                        f"Item analyzer '{analyzer_params.id}' not found in registry"
                    )

                # Prepare parameters for analyzer constructor
                analyzer_kwargs = dict(analyzer_params.params)

                if self.tokenizer is not None:
                    analyzer_kwargs["tokenizer"] = self.tokenizer

                # Create analyzer instance with keyword arguments
                item_analyzer = analyzer_class(**analyzer_kwargs)
                item_analyzers[analyzer_params.id] = item_analyzer
                logger.info(f"Initialized item analyzer: {analyzer_params.id}")
            except Exception as e:
                logger.error(
                    f"Failed to initialize item analyzer {analyzer_params.id}: {e}"
                )
                logger.error(f"Analyzer configuration: {analyzer_params}")
        return item_analyzers

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
        if not self.item_analyzers:
            raise ValueError(
                "No analyzers configured for analysis. Please add at least one "
                "analyzer to the configuration before calling analyze_dataset()."
            )

        logger.info(f"Starting analysis of dataset: {self.dataset_name}")
        logger.info(
            f"Using {len(self.item_analyzers)} item analyzers: "
            f"{list(self.item_analyzers.keys())}"
        )

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
            # Conversation dataset - convert and process
            self._process_conversation_dataset(items_to_analyze)

        # Store metadata
        self._analysis_results = DatasetAnalysisResult(
            dataset_name=self.dataset_name or "",
            total_conversations=total_items,
            conversations_analyzed=items_to_analyze,
        )

    def _process_direct_dataframes(self, items_to_analyze: int) -> None:
        """Process direct DataFrames input."""
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

        # Process each analyzer
        for analyzer_id, analyzer in self.item_analyzers.items():
            try:
                # Apply row-level analysis
                if not rows_df.empty:
                    rows_df = analyzer.analyze(
                        rows_df,
                        self.column_config,
                    )

                # Apply item-level analysis
                if not items_df.empty:
                    items_df = analyzer.analyze(
                        items_df,
                        self.column_config,
                    )

            except Exception as e:
                logger.warning(f"Analyzer {analyzer_id} failed: {e}")

        # Store processed DataFrames
        self._items_df = items_df
        self._rows_df = rows_df

        # Create merged DataFrame
        if self._rows_df is not None and self._items_df is not None:
            if not self._rows_df.empty and not self._items_df.empty:
                self._analysis_df = self._rows_df.merge(
                    self._items_df, on=["item_index"], how="left"
                )
            elif not self._rows_df.empty:
                self._analysis_df = self._rows_df.copy()
            elif not self._items_df.empty:
                self._analysis_df = self._items_df.copy()
        else:
            self._analysis_df = pd.DataFrame()

    def _process_conversation_dataset(self, items_to_analyze: int) -> None:
        """Process conversation dataset input."""
        if self.dataset is None:
            raise ValueError("Dataset must be provided for conversation processing")

        items_dfs = []
        rows_dfs = []

        for item_idx in tqdm(
            range(items_to_analyze),
            desc=f"Analyzing items in {self.dataset_name}",
            unit="item",
        ):
            conversation = self.dataset.conversation(item_idx)
            conversation_id = conversation.conversation_id or f"conv_{item_idx}"
            items_df, rows_df = self._conversation_to_df(
                conversation, conversation_id, item_idx
            )

            # Process each analyzer for this item
            for analyzer_id, analyzer in self.item_analyzers.items():
                try:
                    # Apply row-level analysis
                    if not rows_df.empty:
                        rows_df = analyzer.analyze(
                            rows_df,
                            column_config=self.column_config,
                        )

                    # Apply item-level analysis
                    items_df = analyzer.analyze(
                        items_df,
                        column_config=self.column_config,
                    )

                except Exception as e:
                    logger.warning(
                        f"Analyzer {analyzer_id} failed for item {conversation_id}: {e}"
                    )

            # Add to collection DataFrames
            if not rows_df.empty:
                rows_dfs.append(rows_df)
            items_dfs.append(items_df)

        # Create final DataFrames
        if rows_dfs:
            self._rows_df = pd.concat(rows_dfs, ignore_index=True)
        else:
            self._rows_df = pd.DataFrame()

        if items_dfs:
            self._items_df = pd.concat(items_dfs, ignore_index=True)
        else:
            self._items_df = pd.DataFrame()

        # Create merged DataFrame with both row and item metrics
        if not self._rows_df.empty and not self._items_df.empty:
            # Use item_index for merging
            merge_on = ["item_index"]

            self._analysis_df = self._rows_df.merge(
                self._items_df,
                on=merge_on,
                how="left",
            )
        elif not self._rows_df.empty:
            self._analysis_df = self._rows_df.copy()
        elif not self._items_df.empty:
            self._analysis_df = self._items_df.copy()
        else:
            self._analysis_df = pd.DataFrame()

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
        if self._analysis_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to query the analysis results."
            )

        # Apply the query filter
        try:
            filtered_df = self._analysis_df.query(query_expression)
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
        if self._analysis_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the analysis DataFrame."
            )
        return self._analysis_df

    @property
    def rows_df(self) -> Union[pd.DataFrame, None]:
        """Get the rows-level analysis DataFrame.

        Returns:
            DataFrame with row-level metrics prefixed by row_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._rows_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the rows DataFrame."
            )
        return self._rows_df

    @property
    def items_df(self) -> Union[pd.DataFrame, None]:
        """Get the items-level analysis DataFrame.

        Returns:
            DataFrame with item-level metrics prefixed by item_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._items_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the items DataFrame."
            )
        return self._items_df

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
        # Check if analysis has been run
        if self._items_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to query item results."
            )

        # Apply the query filter
        try:
            filtered_df = self._items_df.query(query_expression)
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

        # Get unique sample indices from filtered results
        item_indices = filtered_df.item_index.unique().tolist()

        # Create a new dataset with only the filtered conversations
        if self.dataset is None:
            raise ValueError("Dataset is None, cannot filter")
        filtered_dataset = self._create_filtered_dataset(item_indices)

        logger.info(
            f"Filtered dataset: {len(item_indices)} items "
            f"out of {len(self.dataset)} total"
        )

        return filtered_dataset

    def _create_filtered_dataset(self, item_indices: list[int]) -> BaseMapDataset:
        """Create a new dataset containing only the specified samples.

        Args:
            item_indices: List of item indices to include

        Returns:
            A new dataset object with the same format as the original
        """
        if self.dataset is None:
            raise ValueError("Dataset is None, cannot create filtered dataset")

        # Deep copy the original dataset to preserve all attributes and methods
        filtered_dataset = copy.deepcopy(self.dataset)

        # Filter the DataFrame to only include the specified samples
        original_df = self.dataset.data
        filtered_dataset._data = original_df.iloc[item_indices].copy()

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
        if self._analysis_df is None or self._analysis_df.empty:
            return {"error": "No analysis data available"}

        summary = {
            "dataset_overview": self._get_dataset_overview(),
            "row_level_summary": self._get_row_level_summary(),
            "item_level_summary": self._get_item_level_summary(),
            "item_turns": self._get_item_turns_summary(),
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
            "total_rows": len(self._rows_df) if self._rows_df is not None else 0,
            "analyzers_used": list(self.item_analyzers.keys()),
        }

    def _get_row_level_summary(self) -> dict[str, Any]:
        """Get aggregated row-level metrics across all analyzers."""
        if self._rows_df is None or self._rows_df.empty:
            return {}

        # Get all row-level analyzer columns
        row_columns = [col for col in self._rows_df.columns if col.startswith("row_")]

        summary = {}

        for col in row_columns:
            if col in [
                "row_index",
                "content",
                "item_index",
            ]:
                continue

            # Extract analyzer name and metric from column
            # Format: row_{analyzer}_{metric}
            parts = col.split("_", 2)
            if len(parts) >= 3:
                analyzer_name = parts[1]
                metric_name = "_".join(parts[2:])

                if analyzer_name not in summary:
                    summary[analyzer_name] = {}

                # Compute statistics for numeric columns
                if pd.api.types.is_numeric_dtype(self._rows_df[col]):
                    values = cast(pd.Series, self._rows_df[col].dropna())
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = compute_statistics(
                            values, self._decimal_precision
                        )

        return summary

    def _get_item_level_summary(self) -> dict[str, Any]:
        """Get aggregated item-level metrics across all analyzers."""
        if self._items_df is None or self._items_df.empty:
            return {}

        # Get all item-level analyzer columns
        item_columns = [
            col for col in self._items_df.columns if col.startswith("item_")
        ]

        summary = {}

        for col in item_columns:
            if col in ["item_index"]:
                continue

            # Extract analyzer name and metric from column
            # Format: item_{analyzer}_{metric}
            parts = col.split("_", 2)
            if len(parts) >= 3:
                analyzer_name = parts[1]
                metric_name = "_".join(parts[2:])

                if analyzer_name not in summary:
                    summary[analyzer_name] = {}

                # Compute statistics for numeric columns
                if pd.api.types.is_numeric_dtype(self._items_df[col]):
                    values = cast(pd.Series, self._items_df[col].dropna())
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = compute_statistics(
                            values, self._decimal_precision
                        )

        return summary

    def _get_item_turns_summary(self) -> dict[str, Any]:
        """Get item turn statistics summary.

        Returns:
            Dictionary containing item turn statistics
        """
        if self._rows_df is None or self._rows_df.empty:
            return {}

        # Use item_index for grouping
        if "item_index" not in self._rows_df.columns:
            return {}

        # groupby().size() always returns a Series, but we cast it because
        # type checker can't infer this
        turns_per_item = cast(pd.Series, self._rows_df.groupby("item_index").size())
        return compute_statistics(turns_per_item, self._decimal_precision)

    def _set_dtypes_from_config(self, df: pd.DataFrame) -> None:
        """Set DataFrame column dtypes based on column_config.

        Args:
            df: DataFrame to set dtypes for
        """
        if not self.column_config:
            return

        # Simple mapping from type values to pandas operations
        dtype_mapping = {
            ColumnType.STRING: lambda col: col.astype("string"),
            ColumnType.INT: lambda col: col.astype("int64"),
            ColumnType.FLOAT: lambda col: col.astype("float64"),
            ColumnType.TIMESTAMP: lambda col: pd.to_datetime(col),
            ColumnType.BOOL: lambda col: col.astype("boolean"),
            ColumnType.CATEGORICAL: lambda col: col.astype("category"),
            ColumnType.OBJECT: lambda col: col,  # Keep as-is
        }

        for col_name, col_config in self.column_config.items():
            if col_name in df.columns and "type" in col_config:
                dtype = col_config["type"]
                try:
                    if dtype in dtype_mapping:
                        df[col_name] = dtype_mapping[dtype](df[col_name])
                except Exception as e:
                    logger.warning(
                        f"Failed to set dtype '{dtype}' for column '{col_name}': {e}"
                    )
