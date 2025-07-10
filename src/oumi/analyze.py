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

from oumi.core.configs import AnalyzerConfig
from oumi.utils.analysis_utils import (
    ConversationHelper,
    compute_aggregation_analysis,
    compute_sample_level_analysis,
    generate_timestamped_filename,
    load_dataset_from_config,
    save_results,
)
from oumi.utils.logging import logger


class Analyzer:
    """Base class for dataset analysis functionality."""

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

    def analyze_dataset(self) -> dict[str, Any]:
        """Analyze the dataset and return analysis results.

        This method performs a two-step analysis:
        1. Per-sample (message) level analysis
        2. Aggregation at conversation and global levels

        Returns:
            Dict[str, Any]: Analysis results containing various metrics and insights.
        """
        verbose = self.config.verbose

        if verbose:
            logger.info(f"Starting analysis of dataset: {self.dataset_name}")

        # Step 1: Per-sample (message) level analysis
        if verbose:
            logger.info("Step 1: Computing per-sample (message) level analysis...")

        sample_results = compute_sample_level_analysis(self.dataset, self.config)

        # Save sample-level results
        if hasattr(self.config, "outputs") and self.config.outputs.analysis_output:
            sample_output_path = generate_timestamped_filename(
                f"{self.config.outputs.analysis_output}_sample_level",
                self.config.outputs.save_format,
            )

            # Combine path with filename
            full_sample_path = Path(self.config.outputs.path) / sample_output_path

            save_results(
                sample_results,
                str(full_sample_path),
                self.config.outputs.save_format,
                verbose,
            )

            if verbose:
                logger.info(f"Sample-level results saved to: {sample_output_path}")

        # Step 2: Aggregation at conversation and global levels
        if verbose:
            logger.info("Step 2: Computing aggregation analysis...")

        aggregation_results = compute_aggregation_analysis(sample_results, self.config)

        # Save aggregation results
        if hasattr(self.config, "outputs") and self.config.outputs.aggregation_output:
            # Combine path with filename
            aggregation_filename = generate_timestamped_filename(
                self.config.outputs.aggregation_output, self.config.outputs.save_format
            )
            full_aggregation_path = (
                Path(self.config.outputs.path) / aggregation_filename
            )

            save_results(
                aggregation_results,
                str(full_aggregation_path),
                self.config.outputs.save_format,
                verbose,
            )

            if verbose:
                logger.info(f"Aggregation results saved to: {aggregation_filename}")

        # Combine results for return
        final_results = {
            "dataset_name": self.dataset_name,
            "sample_level_results": sample_results,
            "aggregation_results": aggregation_results,
        }

        return final_results
