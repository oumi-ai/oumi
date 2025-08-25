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

from oumi.core.datasets import VisionLanguageDpoDataset
from oumi.core.registry import register_dataset


@register_dataset("openbmb/RLAIF-V-Dataset")
class OpenbmbRlaifVDataset(VisionLanguageDpoDataset):
    """Preprocess the RLAIF-V dataset for DPO.

    See Also:
        For more information on how to use this dataset, refer to:
        - Huggingface hub: https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset
    """

    default_dataset = "openbmb/RLAIF-V-Dataset"

    def __init__(self, *args, **kwargs):
        """Initialize the OpenbmbRlaifVDataset."""
        super().__init__(
            *args,
            images_key="image",
            prompt_key="question",
            chosen_key="chosen",
            rejected_key="rejected",
            **kwargs,
        )
