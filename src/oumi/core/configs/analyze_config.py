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
from oumi.core.configs.params.base_params import BaseParams


@dataclass
class SampleAnalyzerParams(BaseParams):
    """Params for a single sample analyzer plugin."""

    id: str = MISSING
    """Unique identifier for the analyzer."""

    config: dict[str, Any] = field(default_factory=dict)
    """Analyzer-specific configuration parameters."""


@dataclass
class AnalyzeConfig(BaseConfig):
    """Configuration for dataset analysis and aggregation."""

    # Simple fields for common use cases
    dataset_name: Optional[str] = None
    """Dataset name."""

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

    def __post_init__(self):
        """Validates the configuration parameters."""
        if not self.dataset_name:
            raise ValueError("'dataset_name' must be provided")

        # Validate analyzer configurations
        analyzer_ids = set()
        for analyzer in self.analyzers:
            # Validate analyzer ID
            if not analyzer.id:
                raise ValueError("Analyzer 'id' must be provided")
            if analyzer.id in analyzer_ids:
                raise ValueError(f"Duplicate analyzer ID found: '{analyzer.id}'")
            analyzer_ids.add(analyzer.id)
