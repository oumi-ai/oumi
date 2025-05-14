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

from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("roneneldan/TinyStories")
class TinyStoriesDataset(BasePretrainingDataset):
    """TinyStoriesDataset class for loading and processing the TinyStories dataset.

    This dataset contains synthetically generated short stories with a small
    vocabulary, created by GPT-3.5 and GPT-4. It is designed for text generation
    tasks and is available in English.

    See Also:
        - Paper: https://arxiv.org/abs/2305.07759
        - Huggingface hub: https://huggingface.co/datasets/roneneldan/TinyStories

    Note:
        The dataset is available under the CDLA-Sharing-1.0 license.
    """

    default_dataset = "roneneldan/TinyStories"
