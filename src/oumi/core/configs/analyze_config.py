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
class SampleAnalyzeConfig:
    """Configuration for a single sample analyzer plugin."""

    id: str = MISSING
    """Unique identifier for the analyzer."""

    enabled: bool = True
    """Whether this analyzer is enabled."""

    config: dict[str, Any] = field(default_factory=dict)
    """Analyzer-specific configuration parameters."""


@dataclass
class DatasetAnalyzeConfig(BaseConfig):
    """Configuration for dataset analysis and aggregation."""

    # Simple fields for common use cases
    dataset_name: Optional[str] = None
    """Dataset name for simple single-dataset analysis.

    If provided, will create a simple DataParams configuration automatically.
    Ignored if 'data' is explicitly provided.
    """

    split: str = "train"
    """Dataset split to use for analysis (e.g., 'train', 'test', 'validation').

    Used when dataset_name is provided, or applied to datasets in 'data' if not
    explicitly set in individual datasets.
    """

    sample_count: Optional[int] = None
    """Maximum number of conversations to analyze.

    If None, analyzes all conversations in the dataset.
    If set to a positive integer, only analyzes the first N conversations.
    Used when dataset_name is provided, or applied to datasets in 'data' if not
    explicitly set in individual datasets.
    """

    output_path: str = "."
    """Directory path where output files will be saved.

    Defaults to current directory ('.').
    """

    analyzers: list[SampleAnalyzeConfig] = field(default_factory=list)
    """List of analyzer configurations (plugin-style)."""

    def __post_init__(self):
        """Validates the configuration parameters."""
        if not self.dataset_name:
            raise ValueError("'dataset_name' must be provided")

        # Validate analyzer configurations
        for analyzer in self.analyzers:
            if not analyzer.id:
                raise ValueError("Each analyzer must have a unique 'id'")
