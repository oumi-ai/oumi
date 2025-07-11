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

from oumi.core.analyzers import AnalyzerRegistry
from oumi.core.configs import AnalyzerConfig
from oumi.utils.analysis_utils import (
    ConversationHelper,
    compute_sample_level_analysis,
    generate_timestamped_filename,
    load_dataset_from_config,
    save_results,
)
from oumi.utils.logging import logger


class Analyzer:
    """Base class for dataset analysis functionality using plugin-style analyzers."""

    def __init__(self, config: AnalyzerConfig):
        """Initialize the dataset analyzer with configuration.

        Args:
            config: AnalyzerConfig object containing all analysis parameters
        """
        self.config = config
        self.dataset_name = config.input.name
        self.split = config.input.split
        self.dataset = load_dataset_from_config(config)
        self.conv = ConversationHelper(self.dataset, self.dataset_name)
        self.analyzers = self._initialize_analyzers()

    def _initialize_analyzers(self):
        """Initialize analyzer plugins from configuration."""
        analyzers = {}
        for analyzer_config in self.config.analyzers:
            if analyzer_config.enabled:
                try:
                    config_dict = {
                        "id": analyzer_config.id,
                        "enabled": analyzer_config.enabled,
                        **analyzer_config.config,
                    }
                    analyzer = AnalyzerRegistry.create_analyzer(
                        analyzer_config.id, config_dict
                    )
                    analyzers[analyzer_config.id] = analyzer
                    logger.info(f"Initialized analyzer: {analyzer_config.id}")
                except Exception as e:
                    logger.error(
                        f"Failed to initialize analyzer {analyzer_config.id}: {e}"
                    )
                    logger.error(f"Analyzer configuration: {analyzer_config}")
        return analyzers

    def analyze_dataset(self) -> dict[str, Any]:
        """Analyze the dataset and return analysis results.

        This method performs sample-level analysis using the configured plugin
        analyzers. Each analyzer processes individual messages and returns metrics
        for each message.

        Returns:
            Dict[str, Any]: Analysis results containing sample-level metrics and
            insights.
        """
        logger.info(f"Starting analysis of dataset: {self.dataset_name}")
        logger.info(
            f"Using {len(self.analyzers)} analyzers: {list(self.analyzers.keys())}"
        )

        total_conversations = len(self.dataset)
        max_conversations = self.config.input.max_conversations
        conversations_to_analyze = (
            min(total_conversations, max_conversations)
            if max_conversations
            else total_conversations
        )

        logger.info(f"Analyzing {conversations_to_analyze} conversations")

        # Step 1: Per-sample (message) level analysis
        logger.info("Step 1: Computing per-sample (message) level analysis...")

        sample_results = compute_sample_level_analysis(
            self.dataset, self.config, self.analyzers
        )

        # Save sample-level results
        if hasattr(self.config, "outputs") and self.config.outputs.sample_level_output:
            sample_output_path = generate_timestamped_filename(
                self.config.outputs.sample_level_output,
                self.config.outputs.save_format,
            )
            full_sample_path = Path(self.config.outputs.path) / sample_output_path
            save_results(
                sample_results,
                str(full_sample_path),
                self.config.outputs.save_format,
            )
            logger.info(f"Sample-level results saved to: {sample_output_path}")

        final_results = {
            "dataset_name": self.dataset_name,
            "sample_level_results": sample_results,
        }
        return final_results
