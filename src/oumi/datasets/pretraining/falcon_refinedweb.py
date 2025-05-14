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


@register_dataset("tiiuae/falcon-refinedweb")
class FalconRefinedWebDataset(BasePretrainingDataset):
    """A massive English web dataset built by TII for pretraining large language models.

    The Falcon RefinedWeb dataset is created through stringent filtering and
    large-scale deduplication of CommonCrawl. It contains about 1B instances
    (968M individual web pages) for a total of 2.8TB of clean text data.

    This dataset is intended primarily for pretraining large language models and
    can be used on its own or augmented with curated sources.

    Dataset Link:
        https://huggingface.co/datasets/tiiuae/falcon-refinedweb

    Paper:
        https://arxiv.org/abs/2306.01116

    Features:
        - content (str): The processed and cleaned text contained in the page.
        - url (str): The URL of the webpage crawled to produce the sample.
        - timestamp (timestamp[s]): Timestamp of when the webpage was crawled by
          CommonCrawl.
        - dump (str): The CommonCrawl dump the sample is a part of.
        - segment (str): The CommonCrawl segment the sample is a part of.
        - image_urls (List[List[str]]): A list of elements in the type
          [image_url, image_alt_text] for all images found in the content.

    Usage:
        from datasets import load_dataset
        rw = load_dataset("tiiuae/falcon-refinedweb")

    License:
        ODC-By 1.0

    Note:
        - This public extract is about ~500GB to download, requiring 2.8TB of
          local storage once unpacked.
        - The dataset may contain sensitive information and biased content.
        - No canonical splits are provided for this dataset.
    """

    default_dataset = "tiiuae/falcon-refinedweb"
