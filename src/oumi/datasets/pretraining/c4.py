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


@register_dataset("allenai/c4")
class C4Dataset(BasePretrainingDataset):
    """A dataset for pretraining on the Colossal Clean Crawled Corpus (C4).

    The C4 dataset is based on the Common Crawl dataset and is available in
    multiple variants: 'en', 'en.noclean', 'en.noblocklist', 'realnewslike',
    and 'multilingual' (mC4). It is intended for pretraining language models
    and word representations.

    For more details and download instructions, visit:
    https://huggingface.co/datasets/allenai/c4

    References:
        Paper: https://arxiv.org/abs/1910.10683

    Data Fields:
        - url: URL of the source as a string
        - text: Text content as a string
        - timestamp: Timestamp as a string

    Dataset Variants:
        - en: 305GB
        - en.noclean: 2.3TB
        - en.noblocklist: 380GB
        - realnewslike: 15GB
        - multilingual (mC4): 9.7TB (108 subsets, one per language)

    The dataset is released under the ODC-BY license and is subject to the
    Common Crawl terms of use.
    """

    default_dataset = "allenai/c4"
