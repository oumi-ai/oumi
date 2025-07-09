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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from oumi.core.configs.base_config import BaseConfig


@dataclass
class DatasetSchema:
    """Schema configuration for dataset structure."""

    type: str = "conversation"
    """Type of dataset structure. Currently only supports 'conversation'."""


@dataclass
class InputConfig:
    """Input configuration for dataset loading."""

    name: str = MISSING
    """Dataset name (required).

    This field is used to retrieve the appropriate class from the dataset registry
    that can be used to instantiate and preprocess the data.

    If the dataset is registered in Oumi's registry, it will be loaded from there.
    Otherwise, it will be automatically loaded from HuggingFace Hub.
    """

    split: str = "train"
    """Dataset split to use (e.g., 'train', 'test', 'validation')."""

    max_conversations: Optional[int] = None
    """Maximum number of conversations to analyze.

    If None, analyzes all conversations in the dataset.
    If set to a positive integer, only analyzes the first N conversations.
    Useful for large datasets where you want to sample a subset for analysis.
    """

    schema: DatasetSchema = field(default_factory=DatasetSchema)
    """Schema configuration for dataset structure."""


@dataclass
class OutputConfig:
    """Output configuration for analysis results."""

    path: str = "."
    """Directory path where output files will be saved.

    Defaults to current directory ('.').
    """

    analysis_output: str = "analysis_results_[timestamp].parquet"
    """Path for sample-level analysis results."""

    aggregation_output: str = "aggregations_results_[timestamp].json"
    """Path for aggregated results."""

    save_format: str = "json"
    """Format to save the analysis results.

    Options:
        - "json": Save as JSON file
        - "yaml": Save as YAML file
        - "csv": Save as CSV (for tabular data)
        - "parquet": Save as Parquet file
    """


@dataclass
class LanguageDetectionConfig:
    """Configuration for language detection analysis."""

    enabled: bool = True
    """Whether to enable language detection."""

    confidence_threshold: float = 0.2
    """Minimum confidence threshold for language detection."""

    top_k: int = 3
    """Number of top languages to detect per sample."""

    multilingual_flag: dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "min_num_languages": 2}
    )
    """Configuration for multilingual sample detection."""


@dataclass
class LanguageAggregationConfig:
    """Configuration for language aggregation metrics."""

    distribution: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "min_samples": 10,
            "report_top_n": 10,
            "include_other_bucket": True,
        }
    )
    """Language distribution aggregation configuration."""

    minority_alert: dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "threshold_percent": 5.0}
    )
    """Minority language alert configuration."""

    confidence_statistics: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "stats": ["mean", "stddev", "percentile_10", "percentile_90"],
        }
    )
    """Language detection confidence statistics configuration."""

    multilingual_samples: dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "common_language_pairs": True}
    )
    """Multilingual sample analysis configuration."""


@dataclass
class LengthMetricsConfig:
    """Configuration for length-related metrics."""

    enabled: bool = True
    """Whether length metrics are enabled."""

    char_count: bool = True
    """Whether to compute character count."""

    word_count: bool = True
    """Whether to compute word count."""

    sentence_count: bool = True
    """Whether to compute sentence count."""

    token_count: bool = False
    """Whether to compute token count."""


@dataclass
class SafetyTypeConfig:
    enabled: bool = True
    include_default: bool = True
    custom_keywords: list[str] = field(default_factory=list)
    custom_regexes: list[str] = field(default_factory=list)


@dataclass
class SafetyMetricsConfig:
    """Configuration for safety-related metrics."""

    enabled: bool = True
    profanity: SafetyTypeConfig = field(default_factory=SafetyTypeConfig)
    slurs: SafetyTypeConfig = field(default_factory=SafetyTypeConfig)
    explicit: SafetyTypeConfig = field(default_factory=SafetyTypeConfig)
    hate_speech: SafetyTypeConfig = field(default_factory=SafetyTypeConfig)
    pii: SafetyTypeConfig = field(default_factory=SafetyTypeConfig)


@dataclass
class SampleLevelMetrics:
    """Configuration for sample-level metrics organized by category."""

    language: LanguageDetectionConfig = field(default_factory=LanguageDetectionConfig)
    """Language detection configuration."""

    length: LengthMetricsConfig = field(default_factory=LengthMetricsConfig)
    """Length-related metrics configuration."""

    safety: SafetyMetricsConfig = field(default_factory=SafetyMetricsConfig)
    """Safety-related metrics configuration."""


@dataclass
class AggregationMetrics:
    """Configuration for aggregation-level metrics."""

    language: LanguageAggregationConfig = field(
        default_factory=LanguageAggregationConfig
    )
    """Language aggregation configuration."""


@dataclass
class AnalyzerConfig(BaseConfig):
    """Configuration for dataset analysis and aggregation."""

    input: InputConfig = field(default_factory=InputConfig)
    """Input configuration for dataset sources."""

    outputs: OutputConfig = field(default_factory=OutputConfig)
    """Output configuration for analysis results."""

    sample_level_metrics: SampleLevelMetrics = field(default_factory=SampleLevelMetrics)
    """Configuration for sample-level metrics."""

    aggregation_metrics: AggregationMetrics = field(default_factory=AggregationMetrics)
    """Configuration for aggregation-level metrics."""

    verbose: bool = False
    """Whether to enable verbose output during analysis."""

    analysis_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the analysis functions."""

    def __post_init__(self):
        """Validates the configuration parameters."""
        # Validate input configuration
        if not self.input.name:
            raise ValueError("input.name is required")

        # Validate language detection configuration
        if not isinstance(self.sample_level_metrics.language.enabled, bool):
            raise ValueError("sample_level_metrics.language.enabled must be a boolean")

        if (
            self.sample_level_metrics.language.confidence_threshold < 0
            or self.sample_level_metrics.language.confidence_threshold > 1
        ):
            raise ValueError(
                "sample_level_metrics.language.confidence_threshold must be "
                "between 0 and 1"
            )

        if self.sample_level_metrics.language.top_k <= 0:
            raise ValueError("sample_level_metrics.language.top_k must be positive")

        # Validate output configuration
        valid_save_formats = ["json", "yaml", "csv", "parquet"]
        if self.outputs.save_format not in valid_save_formats:
            raise ValueError(
                f"outputs.save_format must be one of {valid_save_formats}, "
                f"got {self.outputs.save_format}"
            )
