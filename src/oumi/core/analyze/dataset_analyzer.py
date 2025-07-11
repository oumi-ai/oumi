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

from pathlib import Path
from typing import Any

from oumi.core.analyze.sample_analyzer import AnalyzerRegistry
from oumi.core.configs import DatasetAnalyzeConfig
from oumi.utils.analysis_utils import (
    compute_sample_level_analysis,
    load_dataset_from_config,
    save_results,
)
from oumi.utils.logging import logger


class DatasetAnalyzer:
    """Orchestrates dataset analysis by creating and managing sample analyzers."""

    def __init__(self, config: DatasetAnalyzeConfig):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: DatasetAnalyzeConfig object containing all analysis parameters
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.split = config.split

        self.dataset = load_dataset_from_config(config)
        self.sample_analyzers = self._initialize_sample_analyzers()

    def _initialize_sample_analyzers(self):
        """Initialize sample analyzer plugins from configuration."""
        sample_analyzers = {}
        for analyzer_config in self.config.analyzers:
            if analyzer_config.enabled:
                try:
                    config_dict = {
                        "id": analyzer_config.id,
                        "enabled": analyzer_config.enabled,
                        **analyzer_config.config,
                    }
                    sample_analyzer = AnalyzerRegistry.create_analyzer(
                        analyzer_config.id, config_dict
                    )
                    sample_analyzers[analyzer_config.id] = sample_analyzer
                    logger.info(f"Initialized sample analyzer: {analyzer_config.id}")
                except Exception as e:
                    logger.error(
                        f"Failed to initialize sample analyzer "
                        f"{analyzer_config.id}: {e}"
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

        # Get max conversations from the config's sample_count
        max_conversations = self.config.sample_count

        conversations_to_analyze = (
            min(total_conversations, max_conversations)
            if max_conversations
            else total_conversations
        )

        logger.info(f"Analyzing {conversations_to_analyze} conversations")

        # Step 1: Per-sample (message) level analysis
        logger.info("Step 1: Computing per-sample (message) level analysis...")

        sample_results = compute_sample_level_analysis(
            self.dataset, self.config, self.sample_analyzers
        )

        # Save sample-level results
        output_filename = "sample_level_results.json"
        output_path = Path(self.config.output_path) / output_filename
        save_results(
            sample_results,
            str(output_path),
            "json",
        )
        logger.info(f"Sample-level results saved to: {output_filename}")

        final_results = {
            "dataset_name": self.dataset_name,
            "sample_level_results": sample_results,
        }
        return final_results
