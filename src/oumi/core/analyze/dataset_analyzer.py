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
class FieldAnalysisResult:
    """Result of analyzing a single field in a sample.

    Attributes:
        field_name: Name of the field that was analyzed
        field_index: Index of the field within the sample
        text_content: The text content of the field
        analyzer_metrics: Dictionary containing analyzer metrics for this field
    """

    field_name: str
    field_index: int
    text_content: str
    analyzer_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the analysis result to a dictionary with flattened analyzer metrics.

        Returns:
            Dictionary representation of the analysis result
        """
        return asdict(self)


@dataclass
class SampleAnalysisResult:
    """Result of analyzing a sample as a whole.

    Attributes:
        sample_id: Unique identifier for the sample
        analyzer_metrics: Dictionary containing analyzer metrics for the sample
    """

    sample_id: str
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

    def __init__(
        self, 
        config: AnalyzeConfig, 
        dataset: Optional[BaseMapDataset] = None,
        samples: Optional[list[dict]] = None
    ):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzeConfig object containing all analysis parameters
            dataset: Optional pre-loaded dataset for conversation data
            samples: Optional list of dictionary samples for dictionary-based analysis
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.split = config.split

        # Build tokenizer from config if provided
        self.tokenizer = build_tokenizer_from_config(config.tokenizer_config)

        # Determine data type and initialize accordingly
        if samples is not None:
            # DataFrame-based analysis
            self.data_type = "dataframe"
            self.samples = samples
            self.dataset = None
            
            # Auto-detect field configuration if not provided
            if not config.text_fields and samples:
                self.text_fields = self._auto_detect_text_fields(samples[0])
            else:
                self.text_fields = config.text_fields or []
            
            self.id_field = config.id_field
            self.metadata_fields = config.metadata_fields or []
            
            logger.info(
                f"Using dictionary-based analysis with {len(samples)} samples, "
                f"text_fields: {self.text_fields}"
            )
            
        elif config.dataset_source == DatasetSource.DIRECT:
            # Conversation-based analysis with provided dataset
            self.data_type = "conversation"
            self.samples = None
            
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
            
            # Set data type from config
            self.data_type = config.data_type
            
            # Set default text fields
            if not config.text_fields:
                if self.data_type == "dataframe":
                    # For DataFrame data, use all string columns as text fields
                    if hasattr(dataset, 'to_pandas'):
                        df = dataset.to_pandas()
                    elif isinstance(dataset, pd.DataFrame):
                        df = dataset
                    else:
                        df = None
                    
                    if df is not None and len(df) > 0:
                        # Get all string columns
                        string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                        self.text_fields = string_cols
                    else:
                        self.text_fields = []
                else:
                    # Auto-detect text fields from the first conversation
                    if len(dataset) > 0:
                        first_conv = dataset.conversation(0)
                        self.text_fields = self._auto_detect_conversation_fields(first_conv)
                    else:
                        self.text_fields = []
            else:
                self.text_fields = config.text_fields
                
            self.id_field = config.id_field
            self.metadata_fields = config.metadata_fields or []
            
            # Handle DataFrame data
            if self.data_type == "dataframe":
                if hasattr(dataset, 'to_pandas'):
                    self.samples = dataset.to_pandas()
                elif isinstance(dataset, pd.DataFrame):
                    self.samples = dataset
                else:
                    raise ValueError("For data_type='dataframe', dataset must be a pandas DataFrame or have a to_pandas() method")
                logger.info(
                    f"Using provided DataFrame '{self.dataset_name}' with "
                    f"{len(self.samples)} samples, text_fields: {self.text_fields}"
                )
            else:
                logger.info(
                    f"Using provided dataset '{self.dataset_name}' with "
                    f"{len(dataset)} conversations, text_fields: {self.text_fields}"
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
            self.data_type = "conversation"  # Config-loaded datasets are conversation-based
            self.samples = None
            
            # Set default text fields for conversation analysis
            if not config.text_fields:
                # Auto-detect text fields from the first conversation
                if len(self.dataset) > 0:
                    first_conv = self.dataset.conversation(0)
                    self.text_fields = self._auto_detect_conversation_fields(first_conv)
                else:
                    self.text_fields = []
            else:
                self.text_fields = config.text_fields
                
            self.id_field = config.id_field
            self.metadata_fields = config.metadata_fields or []
            
            logger.info(f"Loaded dataset from config: {self.dataset_name}")
        else:
            raise ValueError(f"Invalid dataset_source: {config.dataset_source}")

        self.sample_analyzers = self._initialize_sample_analyzers()

        # Initialize analysis results as None
        self._analysis_results: Optional[DatasetAnalysisResult] = None
        self._merged_df: Optional[pd.DataFrame] = None
        self._fields_df: Optional[pd.DataFrame] = None
        self._samples_df: Optional[pd.DataFrame] = None
        self._analysis_summary: Optional[dict[str, Any]] = None

        # Decimal precision for rounding metrics
        self._decimal_precision = 2

    def _auto_detect_text_fields(self, sample: dict) -> list[str]:
        """Auto-detect text fields in a sample dictionary."""
        text_fields = []
        for key, value in sample.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                text_fields.append(key)
        return text_fields

    def _auto_detect_conversation_fields(self, conversation) -> list[str]:
        """Auto-detect text fields from a conversation object."""
        # For DataFrame approach, we use role names as field names
        # The field DataFrame will handle multiple rows with same field name
        text_fields = []
        for message in conversation.messages:
            field_name = message.role.value
            if field_name not in text_fields:
                text_fields.append(field_name)
        return text_fields

    def _conversation_to_df(self, conversation, conversation_id: str, conv_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert a conversation or dictionary sample directly to field and sample DataFrames.
        
        Args:
            conversation: The conversation object or dictionary to convert
            conversation_id: The conversation/sample ID
            conv_idx: The conversation/sample index
            
        Returns:
            Tuple of (field_df, sample_df)
        """
        if self.data_type == "dataframe":
            # DataFrame data is already in correct format, just add sample_index
            # Preserve original dtypes from input DataFrame
            field_df = conversation.copy()
            field_df["sample_index"] = conv_idx
            field_df["sample_index"] = field_df["sample_index"].astype("int64")
            
            sample_df = conversation.copy()
            sample_df["sample_index"] = conv_idx
            sample_df["sample_index"] = sample_df["sample_index"].astype("int64")
            sample_df["conversation_id"] = conversation_id
            sample_df["conversation_id"] = sample_df["conversation_id"].astype("string")
            
            return field_df, sample_df
        else:
            # Handle conversation data
            return self._conversation_to_dataframes(conversation, conversation_id, conv_idx)
    
    def _conversation_to_dataframes(self, conversation, conversation_id: str, conv_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert a conversation to field and sample DataFrames with proper dtypes."""
        # Create field-level DataFrame (one row per message)
        field_rows = []
        for msg_idx, message in enumerate(conversation.messages):
            field_rows.append({
                "sample_index": conv_idx,
                "field_name": message.role.value,
                "field_index": msg_idx,
                "text_content": message.content,
            })
        
        # Create field_df with proper dtypes
        if field_rows:
            field_df = pd.DataFrame(field_rows)
        else:
            field_df = pd.DataFrame(columns=["sample_index", "field_name", "field_index", "text_content"])
        
        # Set proper dtypes
        field_df["sample_index"] = field_df["sample_index"].astype("int64")
        field_df["field_name"] = field_df["field_name"].astype("string")
        field_df["field_index"] = field_df["field_index"].astype("int64")
        field_df["text_content"] = field_df["text_content"].astype("string")
        
        # Create sample-level DataFrame (one row per sample)
        sample_data = {
            "sample_index": conv_idx,  # Integer index for efficient DataFrame operations
            "conversation_id": conversation_id,  # conversation_id from the dataset
        }
        
        
        # Add rendered sample for token counting if tokenizer is available
        if self.tokenizer is not None:
            try:
                rendered_sample = self._render_conversation_for_tokens(conversation)
                sample_data["rendered_sample"] = rendered_sample
            except Exception as e:
                logger.warning(f"Failed to render conversation {conversation_id} for token counting: {e}")
        
        # Add any additional metadata from the conversation
        if conversation.metadata:
            sample_data.update(conversation.metadata)
        
        # Create sample_df with proper dtypes
        sample_df = pd.DataFrame([sample_data])
        
        # Set proper dtypes
        sample_df["sample_index"] = sample_df["sample_index"].astype("int64")
        sample_df["conversation_id"] = sample_df["conversation_id"].astype("string")
        
        
        # Set rendered_sample to string if present
        if "rendered_sample" in sample_df.columns:
            sample_df["rendered_sample"] = sample_df["rendered_sample"].astype("string")
        
        return field_df, sample_df
    

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
        """Compute metrics for all conversations using DataFrame-based analyzers."""
        if self.data_type == "dataframe":
            if self.samples is None:
                raise ValueError("Samples is None for DataFrame analysis")
            total_conversations = len(self.samples)
        else:
            if self.dataset is None:
                raise ValueError("Dataset is None for conversation analysis")
            total_conversations = len(self.dataset)

        # Apply conversation limit if specified
        max_conversations = self.config.sample_count

        if max_conversations is not None:
            conversations_to_analyze = min(total_conversations, max_conversations)
            logger.info(
                f"Limiting analysis to first {max_conversations} "
                f"conversations (dataset has {total_conversations} total)"
            )
        else:
            conversations_to_analyze = total_conversations

        logger.info(
            "Analyzing %d samples using DataFrame-based analyzers",
            conversations_to_analyze,
        )

        # Collect DataFrames for fields and samples
        field_dfs = []
        sample_dfs = []

        # Use tqdm for progress monitoring
        for conv_idx in tqdm(
            range(conversations_to_analyze),
            desc=f"Analyzing samples in {self.dataset_name}",
            unit="sample",
        ):
            if self.data_type == "dataframe" and self.samples is None:
                raise ValueError("Samples is None")
            elif self.data_type != "dataframe" and self.dataset is None:
                raise ValueError("Dataset is None")
            
            if self.data_type == "dataframe":
                # Handle DataFrame data - use the row directly
                sample_df = self.samples.iloc[conv_idx:conv_idx+1]  # Get single row as DataFrame
                sample_id = sample_df["id"].iloc[0] if "id" in sample_df.columns else f"sample_{conv_idx}"
                field_df, sample_df = self._conversation_to_df(sample_df, sample_id, conv_idx)
            else:
                # Handle conversation data
                conversation = self.dataset.conversation(conv_idx)
                conversation_id = conversation.conversation_id or f"conv_{conv_idx}"
                field_df, sample_df = self._conversation_to_df(conversation, conversation_id, conv_idx)

            # Process each analyzer for this sample
            for analyzer_id, analyzer in self.sample_analyzers.items():
                try:
                    # Apply field-level analysis
                    if not field_df.empty:
                        field_df = analyzer.analyze_fields(field_df, self.text_fields, self.tokenizer)
                    
                    # Apply sample-level analysis
                    sample_df = analyzer.analyze_sample(sample_df, self.text_fields, self.tokenizer)

                except Exception as e:
                    current_sample_id = conversation_id if self.data_type == "conversation" else sample_id
                    logger.warning(
                        f"Analyzer {analyzer_id} failed for sample "
                        f"{current_sample_id}: {e}"
                    )

            # Add to collection DataFrames
            if not field_df.empty:
                field_dfs.append(field_df)
            sample_dfs.append(sample_df)

        # Create final DataFrames
        if field_dfs:
            self._fields_df = pd.concat(field_dfs, ignore_index=True)
        else:
            self._fields_df = pd.DataFrame()

        if sample_dfs:
            self._samples_df = pd.concat(sample_dfs, ignore_index=True)
        else:
            self._samples_df = pd.DataFrame()

        # Create merged DataFrame with both field and sample metrics
        if not self._fields_df.empty and not self._samples_df.empty:
            # Use sample_index for merging
            merge_on = ["sample_index"]

            self._merged_df = self._fields_df.merge(
                self._samples_df,
                on=merge_on,
                how="left",
            )
        elif not self._fields_df.empty:
            self._merged_df = self._fields_df.copy()
        elif not self._samples_df.empty:
            self._merged_df = self._samples_df.copy()
        else:
            self._merged_df = pd.DataFrame()

        # Store metadata
        self._analysis_results = DatasetAnalysisResult(
            dataset_name=self.dataset_name or "",
            total_conversations=total_conversations,
            conversations_analyzed=conversations_to_analyze,
        )


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
    def fields_df(self) -> Union[pd.DataFrame, None]:
        """Get the field-level analysis DataFrame.

        Returns:
            DataFrame with field-level metrics prefixed by field_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._fields_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the fields DataFrame."
            )
        return self._fields_df

    @property
    def samples_df(self) -> Union[pd.DataFrame, None]:
        """Get the sample-level analysis DataFrame.

        Returns:
            DataFrame with sample-level metrics prefixed by sample_

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._samples_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the samples DataFrame."
            )
        return self._samples_df

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
        if self._samples_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to query conversation results."
            )

        # Apply the query filter
        try:
            filtered_df = self._samples_df.query(query_expression)
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
        sample_indices = filtered_df.sample_index.unique().tolist()

        # Create a new dataset with only the filtered conversations
        if self.dataset is None:
            raise ValueError("Dataset is None, cannot filter")
        filtered_dataset = self._create_filtered_dataset(sample_indices)

        logger.info(
            f"Filtered dataset: {len(sample_indices)} samples "
            f"out of {len(self.dataset)} total"
        )

        return filtered_dataset

    def _create_filtered_dataset(
        self, sample_indices: list[int]
    ) -> BaseMapDataset:
        """Create a new dataset containing only the specified samples.

        Args:
            sample_indices: List of sample indices to include

        Returns:
            A new dataset object with the same format as the original
        """
        if self.dataset is None:
            raise ValueError("Dataset is None, cannot create filtered dataset")
        
        # Deep copy the original dataset to preserve all attributes and methods
        filtered_dataset = copy.deepcopy(self.dataset)

        # Filter the DataFrame to only include the specified samples
        original_df = self.dataset.data
        filtered_dataset._data = original_df.iloc[sample_indices].copy()

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
            "field_level_summary": self._get_field_level_summary(),
            "sample_level_summary": self._get_sample_level_summary(),
            "sample_turns": self._get_sample_turns_summary(),
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
            "total_fields": len(self._fields_df)
            if self._fields_df is not None
            else 0,
            "analyzers_used": list(self.sample_analyzers.keys()),
        }

    def _get_field_level_summary(self) -> dict[str, Any]:
        """Get aggregated field-level metrics across all analyzers."""
        if self._fields_df is None or self._fields_df.empty:
            return {}

        # Get all field-level analyzer columns
        message_columns = [
            col for col in self._fields_df.columns if col.startswith("field_")
        ]

        summary = {}

        for col in message_columns:
            if col in [
                "field_name",
                "field_index", 
                "text_content",
                "sample_index",
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
                if pd.api.types.is_numeric_dtype(self._fields_df[col]):
                    values = cast(pd.Series, self._fields_df[col].dropna())
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = compute_statistics(
                            values, self._decimal_precision
                        )

        return summary

    def _get_sample_level_summary(self) -> dict[str, Any]:
        """Get aggregated sample-level metrics across all analyzers."""
        if self._samples_df is None or self._samples_df.empty:
            return {}

        # Get all sample-level analyzer columns
        sample_columns = [
            col
            for col in self._samples_df.columns
            if col.startswith("sample_")
        ]

        summary = {}

        for col in sample_columns:
            if col in ["sample_index"]:
                continue

            # Extract analyzer name and metric from column
            # Format: sample_{analyzer}_{metric}
            parts = col.split("_", 2)
            if len(parts) >= 3:
                analyzer_name = parts[1]
                metric_name = "_".join(parts[2:])

                if analyzer_name not in summary:
                    summary[analyzer_name] = {}

                # Compute statistics for numeric columns
                if pd.api.types.is_numeric_dtype(self._samples_df[col]):
                    values = cast(pd.Series, self._samples_df[col].dropna())
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = compute_statistics(
                            values, self._decimal_precision
                        )

        return summary

    def _get_sample_turns_summary(self) -> dict[str, Any]:
        """Get conversation turn statistics summary.

        Returns:
            Dictionary containing conversation turn statistics
        """
        if self._fields_df is None or self._fields_df.empty:
            return {}

        # Use sample_index for grouping
        if "sample_index" not in self._fields_df.columns:
            return {}

        # groupby().size() always returns a Series, but we cast it because
        # type checker can't infer this
        turns_per_conversation = cast(
            pd.Series, self._fields_df.groupby("sample_index").size()
        )
        return compute_statistics(turns_per_conversation, self._decimal_precision)
