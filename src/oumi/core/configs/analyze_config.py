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

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import MISSING

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.test_params import TestParams


class DatasetSource(Enum):
    """Source of the dataset for analysis.

    .. deprecated::
        This enum is deprecated and will be removed in a future release.
        The dataset source is now automatically determined based on whether
        a dataset is passed directly to DatasetAnalyzer.__init__().
    """

    CONFIG = "config"
    """Load dataset from config parameters (dataset_name, dataset_path, etc.)"""
    DIRECT = "direct"
    """Pass dataset directly to DatasetAnalyzer.__init__()"""


@dataclass
class SampleAnalyzerParams(BaseParams):
    """Params for a single sample analyzer plugin."""

    id: str = MISSING
    """Analyzer type identifier (registered name in the registry)."""

    instance_id: str | None = None
    """Optional unique instance identifier for this analyzer instance.

    If provided, this ID will be used to uniquely identify this analyzer instance
    and will be used in generated column names. This allows multiple instances of
    the same analyzer type with different configurations.

    If None, the 'id' field will be used as the instance identifier (auto-populated
    in __post_init__).
    """

    params: dict[str, Any] = field(default_factory=dict)
    """Analyzer-specific parameters passed to the analyzer constructor."""

    def __post_init__(self):
        """Auto-populate instance_id if not provided."""
        # If instance_id is not set, use the analyzer type id
        if self.instance_id is None:
            self.instance_id = self.id


@dataclass
class AnalyzeConfig(BaseConfig):
    """Configuration for dataset analysis and aggregation."""

    dataset_source: DatasetSource | None = None
    """Source of the dataset for analysis.

    .. deprecated::
        This field is deprecated and will be removed in a future release.
        The dataset source is now automatically determined based on whether
        a dataset is passed directly to DatasetAnalyzer.__init__().
    """

    dataset_format: str | None = None
    """Format of the custom dataset.

    .. deprecated::
        This field is deprecated and will be removed in a future release.
        The dataset format is now automatically detected from the file contents.
    """

    dataset_name: str | None = None
    """Dataset name."""

    dataset_path: str | None = None
    """Path to a custom dataset file (JSON or JSONL format).
    If provided, this takes precedence over dataset_name for loading custom datasets.
    """

    split: str = "train"
    """The split of the dataset to load.
    This is typically one of "train", "test", or "validation". Defaults to "train".
    """

    subset: str | None = None
    """The subset of the dataset to load. If None, uses the base dataset."""

    sample_count: int | None = None
    """The number of examples to sample from the dataset.
    If None, uses the full dataset. If specified, must be non-negative.
    """

    output_path: str = "."
    """Directory path where output files will be saved.

    Defaults to current directory ('.').
    """

    analyzers: list[SampleAnalyzerParams] = field(default_factory=list)
    """List of analyzer configurations (plugin-style)."""

    # Tokenizer configuration
    tokenizer_name: str | None = None
    """The name or path of the tokenizer to use for token counting metrics.

    If None, no tokenizer will be used. This is typically a model identifier
    from HuggingFace Hub (e.g., "openai-community/gpt2").
    """

    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the tokenizer constructor."""

    tokenizer_config: dict[str, Any] | None = None
    """Tokenizer configuration for building a tokenizer.

    .. deprecated::
        This field is deprecated and will be removed in a future release.
        Use 'tokenizer_name' and 'tokenizer_kwargs' instead.
    """

    chat_template: str | None = None
    """Optional chat template to use for conversation rendering.

    When specified, conversations will be formatted using the specified chat template
    (e.g., 'chat_ml', 'llama3-instruct', 'default') instead of the simple
    'ROLE: content' format. This is useful for LLM judge analyzers to see
    conversations in the same format the model was trained on.

    Common templates:
    - 'chat_ml': ChatML format with <|im_start|> tokens
    - 'llama3-instruct': Llama 3 instruction format
    - 'default': Basic template with role markers

    If None, uses simple 'ROLE: content' format for backward compatibility.
    """

    # Processor parameters for vision-language datasets
    processor_name: str | None = None
    """Processor name for vision-language datasets.

    If provided, the dataset will be treated as multimodal (vision-language).
    """

    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    """Processor-specific parameters."""

    trust_remote_code: bool = False
    """Whether to trust remote code for tokenizer/processor loading."""

    is_multimodal: bool | None = None
    """Whether to treat the dataset as multimodal (vision-language).

    .. deprecated::
        This field is deprecated and will be removed in a future release.
        Multimodality is now automatically detected based on whether
        'processor_name' is provided.
    """

    # User-defined tests
    tests: list[TestParams] = field(default_factory=list)
    """List of user-defined tests to run on analysis results.

    Tests allow declarative specification of quality checks to run on the
    analyzed dataset. Each test defines conditions that samples must meet.

    Example:
        tests:
          - id: no_pii
            type: percentage
            metric: quality__has_pii
            condition: "== True"
            max_percentage: 1.0
            severity: high
            title: "PII detected in dataset"

    Supported test types:
        - threshold: Compare metric against a value
        - percentage: Check % of samples matching a condition
        - distribution: Check distribution properties
        - regex: Match regex pattern against text
        - contains: Check for substrings in text
        - query: Execute pandas query expression
        - outliers: Detect statistical outliers
        - composite: Combine multiple tests
        - python: Custom Python function

    See TestParams for full documentation of available options.
    """

    generate_report: bool = False
    """Whether to generate an HTML report with interactive visualizations.

    Requires the 'plotly' package: pip install 'oumi[analyze_advanced]'
    The report includes charts, tables, and recommendations in a self-contained
    HTML file.
    """

    report_title: str | None = None
    """Custom title for the HTML report.

    If not provided, a default title will be generated based on the dataset name.
    """

    def __post_init__(self):
        """Validates the configuration parameters."""
        # Emit deprecation warnings for deprecated fields
        if self.dataset_source is not None:
            warnings.warn(
                "The 'dataset_source' field is deprecated and will be removed in a "
                "future release. The dataset source is now automatically determined "
                "based on whether a dataset is passed directly to "
                "DatasetAnalyzer.__init__(). This field is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.dataset_format is not None:
            warnings.warn(
                "The 'dataset_format' field is deprecated and will be removed in a "
                "future release. The dataset format is now automatically detected "
                "from the file contents. This field is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Handle deprecated tokenizer_config field
        if self.tokenizer_config is not None:
            warnings.warn(
                "The 'tokenizer_config' field is deprecated and will be removed in a "
                "future release. Use 'tokenizer_name' and 'tokenizer_kwargs' instead. "
                "Values from 'tokenizer_config' will be used for this run.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Migrate values from tokenizer_config to new fields if not already set
            if self.tokenizer_name is None and "model_name" in self.tokenizer_config:
                self.tokenizer_name = self.tokenizer_config["model_name"]
            if (
                not self.tokenizer_kwargs
                and "tokenizer_kwargs" in self.tokenizer_config
            ):
                self.tokenizer_kwargs = self.tokenizer_config["tokenizer_kwargs"]
            # trust_remote_code from tokenizer_config only applies if not explicitly set
            if (
                "trust_remote_code" in self.tokenizer_config
                and self.tokenizer_config["trust_remote_code"]
                and not self.trust_remote_code
            ):
                self.trust_remote_code = self.tokenizer_config["trust_remote_code"]

        # Handle deprecated is_multimodal field
        if self.is_multimodal is not None:
            warnings.warn(
                "The 'is_multimodal' field is deprecated and will be removed in a "
                "future release. Multimodality is now automatically detected based "
                "on whether 'processor_name' is provided. This field is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Validate sample_count
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("`sample_count` must be greater than 0.")

        # Validate analyzer configurations
        instance_ids = set()
        for analyzer in self.analyzers:
            # Validate analyzer ID (type)
            if not analyzer.id:
                raise ValueError("Analyzer 'id' must be provided")

            # instance_id is now always set (auto-populated in __post_init__)
            # Check for duplicate instance IDs
            if analyzer.instance_id in instance_ids:
                raise ValueError(
                    f"Duplicate analyzer instance ID found: '{analyzer.instance_id}'. "
                    f"Each analyzer instance must have a unique 'instance_id'."
                )
            instance_ids.add(analyzer.instance_id)

        # Validate test configurations
        test_ids = set()
        for test in self.tests:
            # Check for duplicate test IDs
            if test.id in test_ids:
                raise ValueError(
                    f"Duplicate test ID found: '{test.id}'. "
                    f"Each test must have a unique 'id'."
                )
            test_ids.add(test.id)

            # Validate individual test (detailed validation in TestParams)
            test.finalize_and_validate()
