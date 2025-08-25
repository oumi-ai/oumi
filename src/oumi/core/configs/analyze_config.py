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
from enum import Enum
from typing import Any, Optional

from omegaconf import MISSING

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.base_params import BaseParams


class DatasetSource(Enum):
    """Source of the dataset for analysis."""

    CONFIG = "config"
    """Load dataset from config parameters (dataset_name, dataset_path, etc.)"""
    DIRECT = "direct"
    """Pass dataset directly to DatasetAnalyzer.__init__()"""


@dataclass
class SampleAnalyzerParams(BaseParams):
    """Params for a single sample analyzer plugin."""

    id: str = MISSING
    """Unique identifier for the analyzer."""

    params: dict[str, Any] = field(default_factory=dict)
    """Analyzer-specific parameters passed to the analyzer constructor."""


@dataclass
class AnalyzeConfig(BaseConfig):
    """Configuration for dataset analysis and aggregation."""

    # Required field - must come first
    dataset_source: DatasetSource = MISSING
    """Source of the dataset for analysis. Use CONFIG to load from config parameters
    or DIRECT to pass dataset directly to DatasetAnalyzer.__init__().

    This field is required and must be explicitly set.
    """

    # Simple fields for common use cases
    dataset_name: Optional[str] = None
    """Dataset name."""

    dataset_path: Optional[str] = None
    """Path to a custom dataset file (JSON or JSONL format).
    If provided, this takes precedence over dataset_name for loading custom datasets.
    """

    dataset_format: Optional[str] = None
    """Format of the custom dataset. Either 'oumi' (conversation format) or 'alpaca'.
    Only used when dataset_path is provided.
    """

    split: str = "train"
    """The split of the dataset to load.
    This is typically one of "train", "test", or "validation". Defaults to "train".
    """

    subset: Optional[str] = None
    """The subset of the dataset to load. If None, uses the base dataset."""

    sample_count: Optional[int] = None
    """The number of examples to sample from the dataset.
    If None, uses the full dataset. If specified, must be non-negative.
    """

    output_path: str = "."
    """Directory path where output files will be saved.

    Defaults to current directory ('.').
    """

    analyzers: list[SampleAnalyzerParams] = field(default_factory=list)
    """List of analyzer configurations (plugin-style)."""

    tokenizer_config: Optional[dict[str, Any]] = None
    """Tokenizer configuration for building a tokenizer.
    If None, no tokenizer will be used.

    Expected format:
    {
        "model_name": "gpt2",  # Required: model name for tokenizer
        "tokenizer_kwargs": {},  # Optional: additional tokenizer parameters
        "trust_remote_code": False  # Optional: whether to trust remote code
    }
    """

    # Add processor parameters for vision-language datasets
    processor_name: Optional[str] = None
    """Processor name for vision-language datasets."""

    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    """Processor-specific parameters."""

    trust_remote_code: bool = False
    """Whether to trust remote code for processor loading."""

    is_multimodal: Optional[bool] = None
    """If True, treat the dataset as multimodal (vision-language) when using a
    custom dataset_path. If False, treat as text-only.
    """

    def __post_init__(self):
        """Validates the configuration parameters."""
        if self.dataset_source == DatasetSource.CONFIG:
            # Only require dataset info when loading from config
            if not self.dataset_name and not self.dataset_path:
                raise ValueError(
                    "Either 'dataset_name' or 'dataset_path' must be provided when "
                    "dataset_source=DatasetSource.CONFIG"
                )
        else:
            # When using direct dataset, dataset_name is optional but recommended
            if not self.dataset_name:
                self.dataset_name = "Custom Dataset"

        # Validate dataset_format requirements
        if self.dataset_path is not None:
            if self.dataset_format is None:
                raise ValueError(
                    "'dataset_format' must be specified when using 'dataset_path'. "
                    "Use 'oumi' for conversation format or 'alpaca' for instruction "
                    "format."
                )
            elif self.dataset_format not in ["oumi", "alpaca"]:
                raise ValueError("'dataset_format' must be either 'oumi' or 'alpaca'")

            # Require explicit is_multimodal setting for custom datasets
            if self.is_multimodal is None:
                raise ValueError(
                    "'is_multimodal' must be specified when using 'dataset_path'. "
                    "Set to 'True' for vision-language datasets or 'False' for "
                    "text-only datasets."
                )

            # Additional validation for multimodal
            if self.is_multimodal is True:
                # Currently VLJsonlinesDataset expects oumi conversation format
                if self.dataset_format != "oumi":
                    raise ValueError(
                        "Multimodal datasets require dataset_format='oumi'"
                    )
                if not self.processor_name:
                    raise ValueError(
                        "'processor_name' must be specified when 'is_multimodal' "
                        "is True"
                    )

        # Validate sample_count
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("`sample_count` must be greater than 0.")

        # Validate analyzer configurations
        analyzer_ids = set()
        for analyzer in self.analyzers:
            # Validate analyzer ID
            if not analyzer.id:
                raise ValueError("Analyzer 'id' must be provided")
            if analyzer.id in analyzer_ids:
                raise ValueError(f"Duplicate analyzer ID found: '{analyzer.id}'")
            analyzer_ids.add(analyzer.id)
