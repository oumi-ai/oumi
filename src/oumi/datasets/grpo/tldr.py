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

from oumi.core.datasets.base_grpo_dataset import BaseExperimentalGrpoDataset
from oumi.core.registry import register_dataset


@register_dataset("trl-lib/tldr")
class TldrGrpoDataset(BaseExperimentalGrpoDataset):
    """Dataset class for the `trl-lib/tldr` dataset."""

    default_dataset = "trl-lib/tldr"
