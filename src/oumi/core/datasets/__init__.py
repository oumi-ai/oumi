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

"""Core datasets module for the Oumi (Open Universal Machine Intelligence) library.

This module provides base classes for different types of datasets used in
the Oumi framework. These base classes serve as foundations for creating custom
datasets for various machine learning tasks.

These base classes can be extended to create custom datasets tailored to specific
machine learning tasks within the Oumi framework.

For more detailed information on each class, please refer to their respective
documentation.
"""

from oumi.core.datasets.base_dpo_dataset import BaseExperimentalDpoDataset
from oumi.core.datasets.base_grpo_dataset import BaseExperimentalGrpoDataset
from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset
from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.datasets.base_pretraining_dataset import BasePretrainingDataset
from oumi.core.datasets.base_sft_dataset import BaseSftDataset
from oumi.core.datasets.packed_sft_dataset import PackedSftDataset
from oumi.core.datasets.pretraining_async_text_dataset import (
    PretrainingAsyncTextDataset,
)
from oumi.core.datasets.vision_language_dataset import VisionLanguageSftDataset

__all__ = [
    "BaseExperimentalDpoDataset",
    "BaseExperimentalGrpoDataset",
    "BaseIterableDataset",
    "BaseMapDataset",
    "BasePretrainingDataset",
    "BaseSftDataset",
    "PackedSftDataset",
    "PretrainingAsyncTextDataset",
    "VisionLanguageSftDataset",
]
