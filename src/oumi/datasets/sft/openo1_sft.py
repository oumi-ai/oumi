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

"""OpenO1 synthetic reasoning SFT dataset."""

from oumi.core.registry.registry import register_dataset
from oumi.datasets.sft.prompt_response import PromptResponseDataset


@register_dataset("O1-OPEN/OpenO1-SFT")
class OpenO1SFTDataset(PromptResponseDataset):
    """Synthetic reasoning SFT dataset."""

    default_dataset = "O1-OPEN/OpenO1-SFT"

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Initializes a dataset for OpenO1SFT from HuggingFace."""
        super().__init__(
            hf_dataset_path="O1-OPEN/OpenO1-SFT",
            prompt_column="instruction",
            response_column="output",
            **kwargs,
        )
