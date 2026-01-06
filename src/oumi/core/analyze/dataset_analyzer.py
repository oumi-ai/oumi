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
from typing import Any, cast

import pandas as pd

from oumi.builders.models import build_tokenizer
from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.dataframe_analyzer import DataFrameAnalyzer, DataFrameWithSchema
from oumi.core.configs import AnalyzeConfig, ModelParams
from oumi.core.datasets import BaseMapDataset
from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset
from oumi.core.registry import REGISTRY
from oumi.utils.analysis_utils import (
    compute_statistics,
    compute_statistics_with_distribution,
    convert_dataset_to_dataframes,
    get_schema_for_format,
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

    def __init__(
        self,
        config: AnalyzeConfig,
        dataset: BaseMapDataset | None = None,
        skip_llm_analyzers: bool = False,
        skip_remote_llm_analyzers: bool = False,
    ):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzeConfig object containing all analysis parameters
            dataset: Optional pre-loaded dataset. If provided, this dataset will be used
                    instead of loading from the config.
            skip_llm_analyzers: If True, skip analyzers that require LLM inference.
            skip_remote_llm_analyzers: If True, skip analyzers that require remote
                LLM APIs (e.g., OpenAI, Anthropic). Local model analyzers are allowed.
        """
        self.config = config
        self.skip_llm_analyzers = skip_llm_analyzers
        self.skip_remote_llm_analyzers = skip_remote_llm_analyzers
        self.dataset_name = config.dataset_name
        self.split = config.split

        # Build tokenizer from config if provided
        self.tokenizer = self._build_tokenizer(config)

        # Use provided dataset or load from config
        if dataset is not None:
            # Dataset provided directly
            self.dataset = dataset
            # Use the provided dataset name if config doesn't have one
            if not self.dataset_name:
                self.dataset_name = getattr(dataset, "dataset_name", "Custom Dataset")
            # Handle iterable datasets that don't support len()
            if isinstance(dataset, BaseIterableDataset):
                logger.info(f"Using provided streaming dataset '{self.dataset_name}'")
            else:
                logger.info(
                    f"Using provided dataset '{self.dataset_name}' with "
                    f"{len(dataset)} conversations"
                )
        else:
            # Load dataset from config parameters
            self.dataset = load_dataset_from_config(config, self.tokenizer)
            logger.info(f"Loaded dataset from config: {self.dataset_name}")

        self.sample_analyzers = self._initialize_sample_analyzers()

        # Initialize dataframe analyzer with sample analyzers
        self.dataframe_analyzer = DataFrameAnalyzer(self.sample_analyzers)

        # Initialize analysis results as None
        self._analysis_results: DatasetAnalysisResult | None = None
        self._merged_df: pd.DataFrame | None = None
        self._message_df: pd.DataFrame | None = None
        self._conversation_df: pd.DataFrame | None = None
        self._merged_schema: dict | None = None
        self._analysis_summary: dict[str, Any] | None = None

        # Decimal precision for rounding metrics
        self._decimal_precision = 2

    def _build_tokenizer(self, config: AnalyzeConfig):
        """Build a tokenizer from the analyze config.

        Args:
            config: AnalyzeConfig containing tokenizer settings.

        Returns:
            Built tokenizer or None if no tokenizer is configured.
        """
        if not config.tokenizer_name:
            return None

        model_params = ModelParams(
            model_name=config.tokenizer_name,
            tokenizer_kwargs=config.tokenizer_kwargs,
            trust_remote_code=config.trust_remote_code,
        )
        tokenizer = build_tokenizer(model_params)
        logger.info(f"Built tokenizer for model: {config.tokenizer_name}")
        return tokenizer

    def _get_schema_for_dataset(self) -> dict:
        """Get column schema configuration based on dataset type.

        Detects the appropriate schema based on the dataset class inheritance.
        Based on analysis of all 60 Oumi datasets:
        - 37 datasets (SFT/Vision-SFT/GRPO) convert to conversation format → use 'oumi'
        - 23 datasets (pretraining/DPO/KTO) maintain original structure → use specific

        Returns:
            Dictionary mapping column names to their configuration.

        Raises:
            ValueError: If dataset type cannot be determined.
        """
        dataset_type = self._detect_dataset_type()

        try:
            return get_schema_for_format(dataset_type)
        except ValueError:
            raise ValueError(f"Unknown dataset type '{dataset_type}'.")

    def _detect_dataset_type(self) -> str:
        """Detect the dataset type based on the dataset class.

        Returns:
            String indicating the dataset type for schema selection.
        """
        if self.dataset is None:
            # No dataset provided, default to oumi format
            return "oumi"

        # Check dataset class inheritance hierarchy for accurate detection
        dataset_class_bases = [base.__name__ for base in self.dataset.__class__.__mro__]

        # Datasets that convert to conversation format during loading
        if any(
            base in dataset_class_bases
            for base in [
                "BaseSftDataset",
                "VisionLanguageSftDataset",
                "BaseExperimentalGrpoDataset",
            ]
        ):
            return "oumi"  # All convert to conversation format

        # Datasets that maintain original structure
        elif "BasePretrainingDataset" in dataset_class_bases:
            return "pretraining"
        elif any(
            base in dataset_class_bases
            for base in ["BaseDpoDataset", "VisionLanguageDpoDataset"]
        ):
            return "dpo"
        elif "BaseExperimentalKtoDataset" in dataset_class_bases:
            return "kto"
        else:
            # Default to conversation format for unknown SFT-like datasets
            return "oumi"

    def _initialize_sample_analyzers(self) -> dict[str, Any]:
        """Initialize sample analyzer plugins from configuration.

        Returns:
            Dictionary mapping analyzer IDs to analyzer instances

        Raises:
            RuntimeError: If any analyzer fails to initialize.
        """
        sample_analyzers = {}
        failed_analyzers: list[tuple[str, str]] = []
        skipped_analyzers: list[str] = []

        for analyzer_params in self.config.analyzers:
            try:
                analyzer_class = REGISTRY.get_sample_analyzer(analyzer_params.id)
                if analyzer_class is None:
                    raise ValueError(
                        f"Sample analyzer '{analyzer_params.id}' not found in registry"
                    )

                # instance_id is now always set (auto-populated if not specified)
                instance_id = analyzer_params.instance_id

                # Check if this analyzer requires LLM and should be skipped
                if self.skip_llm_analyzers and getattr(
                    analyzer_class, "requires_llm", False
                ):
                    skipped_analyzers.append(instance_id)
                    logger.info(
                        f"Skipping LLM-based analyzer: {instance_id} "
                        f"(type: {analyzer_params.id}, --skip-llm flag set)"
                    )
                    continue

                # Check if this analyzer requires remote LLM API and should be skipped
                if self.skip_remote_llm_analyzers and getattr(
                    analyzer_class, "requires_remote_llm", False
                ):
                    skipped_analyzers.append(instance_id)
                    logger.info(
                        f"Skipping remote LLM analyzer: {instance_id} "
                        f"(type: {analyzer_params.id}, --skip-remote-llm flag set)"
                    )
                    continue

                # Prepare parameters for analyzer constructor
                analyzer_kwargs = dict(analyzer_params.params)

                if self.tokenizer is not None:
                    analyzer_kwargs["tokenizer"] = self.tokenizer

                # Create analyzer instance with keyword arguments
                sample_analyzer = analyzer_class(**analyzer_kwargs)

                # Set the instance ID on the analyzer so it can use it for column naming
                sample_analyzer.analyzer_id = instance_id

                # Store analyzer instance using the instance_id
                sample_analyzers[instance_id] = sample_analyzer

                # Log with type info if instance_id differs from type
                if instance_id != analyzer_params.id:
                    log_msg = f"Initialized sample analyzer: {instance_id} (type: {analyzer_params.id})"
                else:
                    log_msg = f"Initialized sample analyzer: {instance_id}"
                logger.info(log_msg)
            except Exception as e:
                logger.error(
                    f"Failed to initialize sample analyzer {analyzer_params.id}: {e}"
                )
                logger.error(f"Analyzer configuration: {analyzer_params}")
                failed_analyzers.append((analyzer_params.id, str(e)))

        if failed_analyzers:
            error_details = "\n".join(
                f"  - {name}: {error}" for name, error in failed_analyzers
            )
            raise RuntimeError(
                f"Failed to initialize {len(failed_analyzers)} analyzer(s):\n"
                f"{error_details}"
            )

        if skipped_analyzers:
            logger.info(
                f"Skipped {len(skipped_analyzers)} LLM-based analyzer(s): "
                f"{', '.join(skipped_analyzers)}"
            )

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

        # Handle iterable datasets differently to avoid downloading everything
        if isinstance(self.dataset, BaseIterableDataset):
            # For iterable datasets, we can't get the total length without iterating
            # So we'll use the sample_count directly and iterate only what we need
            conversations_to_analyze = (
                self.config.sample_count or 1000
            )  # Default limit for streaming
            total_conversations = None  # Unknown for iterable datasets
            logger.info(
                f"Analyzing up to {conversations_to_analyze} conversations from "
                f"streaming dataset"
            )
        else:
            # For map datasets, we can get the total length
            total_conversations = len(self.dataset)
            conversations_to_analyze = min(
                total_conversations, self.config.sample_count or total_conversations
            )
            logger.info(
                f"Analyzing {conversations_to_analyze} of {total_conversations} "
                f"conversations"
            )

        dataframe_list, total_items, items_to_analyze = self._prepare_dataframe_list(
            conversations_to_analyze
        )

        analysis_result = self.dataframe_analyzer.analyze_dataframe_list(
            input_data_list=dataframe_list,
            merge_on=["conversation_index", "conversation_id"],
        )
        self._merged_df = analysis_result.merged_df
        self._message_df = analysis_result.messages_df
        self._conversation_df = analysis_result.conversations_df
        self._merged_schema = analysis_result.merged_schema

        self._analysis_results = DatasetAnalysisResult(
            dataset_name=self.dataset_name or "",
            total_conversations=total_conversations or conversations_to_analyze,
            conversations_analyzed=conversations_to_analyze,
        )

        # Generate and store the analysis summary after metrics are computed
        self._analysis_summary = self._generate_analysis_summary()

    def _prepare_dataframe_list(
        self, max_items: int | None = None
    ) -> tuple[list[DataFrameWithSchema], int, int]:
        """Prepare DataFrameWithSchema list from input source with optional limiting.

        Args:
            max_items: Maximum number of items to analyze (None for no limit)

        Returns:
            Tuple of (dataframe_list, total_items, items_to_analyze)
        """
        if self.dataset is not None:
            # Conversation dataset input - convert to DataFrames
            if isinstance(self.dataset, BaseIterableDataset):
                # For iterable datasets, we can't get the total length
                total_items = max_items or 1000  # Use max_items or default
                items_to_analyze = total_items
                logger.info(
                    f"Converting streaming dataset with up to {items_to_analyze} items"
                )
            else:
                # For map datasets, we can get the total length
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

            # Use utility function to convert dataset to DataFrames
            conversations_df, messages_df = convert_dataset_to_dataframes(
                dataset=self.dataset,
                items_to_analyze=items_to_analyze,
                dataset_name=self.dataset_name or "Unknown Dataset",
                chat_template=self.config.chat_template,
            )

            # Get base schema for the dataset type
            schema = self._get_schema_for_dataset()

            # Augment schema with any additional columns (e.g., metadata fields)
            # that may be present in the dataframes but not in the base schema
            from oumi.utils.analysis_utils import augment_schema_with_dataframe_columns

            # Augment with columns from both dataframes
            if not conversations_df.empty:
                schema = augment_schema_with_dataframe_columns(schema, conversations_df)
            if not messages_df.empty:
                schema = augment_schema_with_dataframe_columns(schema, messages_df)

            dataframe_list = [
                DataFrameWithSchema(conversations_df, schema, "conversations"),
                DataFrameWithSchema(messages_df, schema, "messages"),
            ]
            return dataframe_list, total_items, items_to_analyze

        else:
            raise ValueError("Either dataframes or dataset must be provided")

    @property
    def analysis_results(self) -> DatasetAnalysisResult | None:
        """Get the analysis results if available.

        Returns:
            DatasetAnalysisResult if analysis has been run, None otherwise
        """
        return self._analysis_results

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
    def analysis_df(self) -> pd.DataFrame | None:
        """Get the merged analysis DataFrame with both message and conversation metrics.

        Returns:
            DataFrame with columns prefixed by ``message_`` and ``conversation_`` for
            each analyzer.

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
    def message_df(self) -> pd.DataFrame | None:
        """Get the message-level analysis DataFrame.

        Returns:
            DataFrame with message-level metrics prefixed by ``message_``

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
    def conversation_df(self) -> pd.DataFrame | None:
        """Get the conversation-level analysis DataFrame.

        Returns:
            DataFrame with conversation-level metrics prefixed by ``conversation_``

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._conversation_df is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the conversation DataFrame."
            )
        return self._conversation_df

    def get_schema(self) -> dict:
        """Get the schema for the analysis results.

        Returns:
            Dictionary containing the schema for the merged DataFrame, combining
            schemas from all input DataFrames including analyzer-generated columns.

        Raises:
            RuntimeError: If analysis has not been run yet.
        """
        if self._merged_schema is None:
            raise RuntimeError(
                "Analysis has not been run yet. Please call analyze_dataset() first "
                "to access the merged schema."
            )
        return self._merged_schema

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

        Examples::

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
    ) -> BaseMapDataset | BaseIterableDataset:
        """Filter the original dataset based on analysis results.

        This method uses analysis results to filter the original dataset, returning
        a new dataset object containing only the conversations that match the query.

        Args:
            query_expression: Pandas query expression to filter analysis results

        Returns:
            A new dataset object containing only the filtered conversations

        Raises:
            RuntimeError: If analysis has not been run yet.

        Examples::

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

        # Get total dataset size, handling iterable datasets
        from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

        if isinstance(self.dataset, BaseIterableDataset):
            total_size = "unknown (streaming)"
        else:
            total_size = str(len(self.dataset))

        logger.info(
            f"Filtered dataset: {len(conversation_indices)} conversations "
            f"out of {total_size} total"
        )

        return filtered_dataset

    def _create_filtered_dataset(
        self, conversation_indices: list[int]
    ) -> BaseMapDataset | BaseIterableDataset:
        """Create a new dataset containing only the specified conversations.

        Args:
            conversation_indices: List of conversation indices to include

        Returns:
            A new dataset object with the same format as the original

        Raises:
            NotImplementedError: If the dataset is an iterable/streaming dataset.
        """
        from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

        if isinstance(self.dataset, BaseIterableDataset):
            raise NotImplementedError(
                "Filtering is not supported for iterable/streaming datasets."
            )

        # Deep copy the original dataset to preserve all attributes and methods
        filtered_dataset = copy.deepcopy(self.dataset)

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
            - Recommendations (if enabled)
        """
        # Check if we have data to analyze
        if self._merged_df is None or self._merged_df.empty:
            return {"error": "No analysis data available"}

        # TODO: Refactor summary methods to be dataset agnostic
        # Currently these methods assume conversation dataset structure with
        # messages/conversations.
        # They should be generalized to work with any dataset type and column structure.
        summary = {
            "dataset_overview": self._get_dataset_overview(),
            "message_level_summary": self._get_level_summary(self._message_df),
            "conversation_level_summary": self._get_level_summary(
                self._conversation_df
            ),
            "conversation_turns": self._get_conversation_turns_summary(),
        }

        # Generate observations (key findings summary)
        summary["observations"] = self._generate_observations(summary)

        # Generate recommendations if enabled in config
        if getattr(self.config, "generate_recommendations", True):
            summary["recommendations"] = self._generate_recommendations(summary)

        return summary

    def _generate_recommendations(
        self, summary: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on analysis results.

        Args:
            summary: The analysis summary dict.

        Returns:
            List of recommendation dictionaries.
        """
        try:
            from oumi.core.analyze.recommendations import RecommendationsEngine

            # Use outlier_threshold from config if available
            outlier_threshold = getattr(self.config, "outlier_threshold", 3.0)
            engine = RecommendationsEngine(outlier_std_threshold=outlier_threshold)
            msg_df = (
                self._message_df if self._message_df is not None else pd.DataFrame()
            )
            conv_df = (
                self._conversation_df
                if self._conversation_df is not None
                else pd.DataFrame()
            )
            recommendations = engine.generate_recommendations(
                message_df=msg_df,
                conversation_df=conv_df,
                analysis_summary=summary,
            )
            return [rec.to_dict() for rec in recommendations]
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            return []

    def _generate_observations(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate observations (key findings) based on analysis results.

        Args:
            summary: The analysis summary dict.

        Returns:
            List of observation dictionaries.
        """
        try:
            from oumi.core.analyze.observations import ObservationsEngine

            engine = ObservationsEngine()
            msg_df = (
                self._message_df if self._message_df is not None else pd.DataFrame()
            )
            conv_df = (
                self._conversation_df
                if self._conversation_df is not None
                else pd.DataFrame()
            )
            observations = engine.generate_observations(
                message_df=msg_df,
                conversation_df=conv_df,
                analysis_summary=summary,
            )
            return [obs.to_dict() for obs in observations]
        except Exception as e:
            logger.warning(f"Failed to generate observations: {e}")
            return []

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

    def _get_computable_columns(self, df: pd.DataFrame) -> list[str]:
        """Get computable columns from DataFrame using schema information.

        A computable column is one that has content_type == NUMERIC in the schema.

        Args:
            df: DataFrame to analyze

        Returns:
            List of tuples (column_name, schema_info) for computable columns
        """
        if self._merged_schema is None:
            raise RuntimeError(
                "Schema not available. Please call analyze_dataset() first."
            )

        computable_columns = []
        for col in df.columns:
            # Check schema for this column
            col_schema = self._merged_schema.get(col)
            if col_schema is None:
                continue

            # Only include columns with ContentType.NUMERIC
            content_type = col_schema.get("content_type")
            if content_type == ContentType.NUMERIC:
                computable_columns.append(col)

        return computable_columns

    def _get_level_summary(self, df: pd.DataFrame | None) -> dict[str, Any]:
        """Get aggregated metrics for a given DataFrame level.

        Uses schema information to better identify and group computable columns.

        Args:
            df: DataFrame to analyze (message_df or conversation_df)

        Returns:
            Dictionary mapping metric names to their statistics
        """
        if df is None or df.empty:
            return {}

        summary = {}
        for col in self._get_computable_columns(df):
            # Compute statistics for numeric columns
            values = cast(pd.Series, df[col].dropna())
            if len(values) > 0:
                summary[col] = compute_statistics(values, self._decimal_precision)

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
