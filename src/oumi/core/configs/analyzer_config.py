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


@dataclass
class OutputConfig:
    """Output configuration for analysis results."""

    path: str = "."
    """Directory path where output files will be saved.

    Defaults to current directory ('.').
    """


@dataclass
class AnalyzerPluginConfig:
    """Configuration for a single analyzer plugin."""

    id: str = MISSING
    """Unique identifier for the analyzer."""

    enabled: bool = True
    """Whether this analyzer is enabled."""

    config: dict[str, Any] = field(default_factory=dict)
    """Analyzer-specific configuration parameters."""


@dataclass
class AnalyzerConfig(BaseConfig):
    """Configuration for dataset analysis and aggregation."""

    input: InputConfig = field(default_factory=InputConfig)
    """Input configuration for dataset sources."""

    outputs: OutputConfig = field(default_factory=OutputConfig)
    """Output configuration for analysis results."""

    analyzers: list[AnalyzerPluginConfig] = field(default_factory=list)
    """List of analyzer configurations (plugin-style)."""

    def __post_init__(self):
        """Validates the configuration parameters."""
        # Validate input configuration
        if not self.input.name:
            raise ValueError("input.name is required")

        # Validate analyzer configurations
        for analyzer in self.analyzers:
            if not analyzer.id:
                raise ValueError("Each analyzer must have a unique 'id'")
