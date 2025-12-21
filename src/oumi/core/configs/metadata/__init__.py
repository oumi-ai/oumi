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

"""Config metadata module for config discoverability and documentation."""

from oumi.core.configs.metadata.comment_parser import (
    parse_metadata_comments,
    parse_tags,
)
from oumi.core.configs.metadata.config_metadata import (
    ConfigMetadata,
    ConfigType,
    FinetuningType,
    TrainingMethod,
)
from oumi.core.configs.metadata.extractor import MetadataExtractor
from oumi.core.configs.metadata.vram_estimator import (
    estimate_training_vram_gb,
    estimate_vram_from_config,
    get_recommended_gpus,
    get_vram_tier,
)

__all__ = [
    "ConfigMetadata",
    "ConfigType",
    "FinetuningType",
    "MetadataExtractor",
    "TrainingMethod",
    "estimate_training_vram_gb",
    "estimate_vram_from_config",
    "get_recommended_gpus",
    "get_vram_tier",
    "parse_metadata_comments",
    "parse_tags",
]
