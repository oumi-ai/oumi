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

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.sample_analyzer import AnalyzerRegistry
from oumi.core.configs import AnalyzeConfig
from oumi.utils.analysis_utils import (
    compute_sample_level_analysis,
    load_dataset_from_config,
    save_results,
)
from oumi.utils.logging import logger


class DatasetAnalyzer:
    """Orchestrates dataset analysis by creating and managing sample analyzers."""

    # Default output filename for analysis results
    DEFAULT_OUTPUT_FILENAME = "sample_level_results.json"

    def __init__(self, config: AnalyzeConfig):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzeConfig object containing all analysis parameters
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.split = config.split

        self.dataset = load_dataset_from_config(config)
        self.sample_analyzers = self._initialize_sample_analyzers()
        self._output_path = str(
            Path(self.config.output_path) / self.DEFAULT_OUTPUT_FILENAME
        )
        self._cached_results_df: Optional[pd.DataFrame] = None

    def _initialize_sample_analyzers(self):
        """Initialize sample analyzer plugins from configuration."""
        sample_analyzers = {}
        for analyzer_config in self.config.analyzers:
            try:
                config_dict = {
                    "id": analyzer_config.id,
                    **analyzer_config.config,
                }
                sample_analyzer = AnalyzerRegistry.create_analyzer(
                    analyzer_config.id, config_dict
                )
                sample_analyzers[analyzer_config.id] = sample_analyzer
                logger.info(f"Initialized sample analyzer: {analyzer_config.id}")
            except Exception as e:
                logger.error(
                    f"Failed to initialize sample analyzer {analyzer_config.id}: {e}"
                )
                logger.error(f"Analyzer configuration: {analyzer_config}")
        return sample_analyzers

    def analyze_dataset(self) -> dict[str, Any]:
        """Analyze the dataset and return analysis results.

        This method performs sample-level analysis using the configured sample
        analyzers. Each sample analyzer processes individual messages and returns
        metrics for each message.

        Returns:
            Dict[str, Any]: Analysis results containing sample-level metrics and
            insights.
        """
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

        # Step 1: Per-sample (message) level analysis
        logger.info("Step 1: Computing per-sample (message) level analysis...")

        sample_results = compute_sample_level_analysis(
            self.dataset, self.config, self.sample_analyzers
        )

        # Save sample-level results
        save_results(
            sample_results,
            self._output_path,
        )
        logger.info(f"Sample-level results saved to: {self._output_path}")

        # Cache the results for immediate querying
        from pandas import DataFrame

        messages = sample_results.get("messages", [])
        messages_df = DataFrame(messages)
        self._cached_results_df = messages_df

        final_results = {
            "dataset_name": self.dataset_name,
            "sample_level_results": sample_results,
        }
        return final_results

    def load_jsonl_results(self) -> pd.DataFrame:
        """Load analysis results from the main JSONL file into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the analysis results (messages only,
                         metadata is excluded).

        Raises:
            FileNotFoundError: If the JSONL file doesn't exist.
            ValueError: If the JSONL file is malformed.
        """
        # Only use cache for the main output file
        if self._cached_results_df is not None:
            logger.info(f"Using cached results from {self._output_path}")
            return self._cached_results_df

        file_path_obj = Path(self._output_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"JSONL file not found: {self._output_path}")

        try:
            with open(file_path_obj, encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) < 1:
                    raise ValueError("JSONL file is empty")
                data = []
                for line in lines[1:]:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON line: {e}")
                        continue
            if not data:
                raise ValueError("No valid message data found in JSONL file")
            messages_df = pd.DataFrame(data)
            logger.info(f"Loaded {len(messages_df)} messages from {self._output_path}")
            self._cached_results_df = messages_df
            return messages_df
        except Exception as e:
            logger.error(f"Failed to load JSONL results from {self._output_path}: {e}")
            raise

    def query_results(self, query_expression: str) -> pd.DataFrame:
        """Query analysis results using pandas query expression.

        Args:
            query_expression: Pandas query expression to filter the DataFrame.

        Returns:
            pd.DataFrame: Query results as a DataFrame.

        Raises:
            FileNotFoundError: If the JSONL file doesn't exist.
            ValueError: If the query expression is invalid or execution fails.

        Example:
            # Get all user messages
            results = analyzer.query_results("role == 'user'")

            # Get messages with word count > 10
            results = analyzer.query_results("length_word_count > 10")

            # Get messages from specific conversation
            results = analyzer.query_results("conversation_id == 'conv_0'")

            # Complex query with multiple conditions
            results = analyzer.query_results(
                "role == 'user' and length_word_count > 5"
            )
        """
        try:
            # Load the data into a DataFrame using the main output file
            messages_df = self.load_jsonl_results()

            # Execute pandas query
            result = messages_df.query(query_expression)

            logger.info(f"Executed pandas query, returned {len(result)} rows")
            return result

        except Exception as e:
            logger.error(f"Failed to execute pandas query: {e}")
            raise ValueError(f"Pandas query execution failed: {e}")

    def clear_cache(self) -> None:
        """Clear the cached results DataFrame."""
        self._cached_results_df = None
        logger.info("Cleared cached results")
