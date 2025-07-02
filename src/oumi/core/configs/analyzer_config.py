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
from typing import Any

from omegaconf import MISSING

from oumi.core.configs.base_config import BaseConfig


@dataclass
class DatasetSchema:
    """Schema configuration for dataset structure."""

    type: str = "conversation"  # "single_turn" or "conversation"
    """Type of dataset structure."""

    fields: dict[str, str] = field(
        default_factory=lambda: {
            "text_field": "text",
            "conversation_field": "messages",
            "conversation_id_field": "id",
            "role_field": "role",
            "content_field": "content",
        }
    )
    """Field mappings for dataset structure."""


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

    schema: DatasetSchema = field(default_factory=DatasetSchema)
    """Schema configuration for dataset structure."""


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration for text analysis."""

    normalize_whitespace: bool = True
    """Whether to normalize whitespace in text."""

    lowercase: bool = False
    """Whether to convert text to lowercase."""

    remove_special_chars: bool = False
    """Whether to remove special characters, emojis, and non-linguistic tokens."""


@dataclass
class OutputConfig:
    """Output configuration for analysis results."""

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
class SampleLevelMetrics:
    """Configuration for sample-level metrics."""

    language: LanguageDetectionConfig = field(default_factory=LanguageDetectionConfig)
    """Language detection configuration."""


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

    analyze_version: str = "v1.0.0"
    """Version of the analysis configuration."""

    input: InputConfig = field(default_factory=InputConfig)
    """Input configuration for dataset sources."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    """Preprocessing configuration for text analysis."""

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

        if self.input.schema.type not in ["single_turn", "conversation"]:
            raise ValueError(
                f"input.schema.type must be one of ['single_turn', 'conversation'], "
                f"got {self.input.schema.type}"
            )

        # Validate preprocessing configuration
        if not isinstance(self.preprocessing.normalize_whitespace, bool):
            raise ValueError("preprocessing.normalize_whitespace must be a boolean")

        if not isinstance(self.preprocessing.lowercase, bool):
            raise ValueError("preprocessing.lowercase must be a boolean")

        if not isinstance(self.preprocessing.remove_special_chars, bool):
            raise ValueError("preprocessing.remove_special_chars must be a boolean")

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
