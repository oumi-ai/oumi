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


@register_dataset("togethercomputer/RedPajama-Data-1T")
class RedPajamaDataV1Dataset(BasePretrainingDataset):
    """RedPajama is a clean-room, fully open-source implementation of the LLaMa dataset.

    This dataset contains approximately 1.2 trillion tokens from various sources:
    Commoncrawl (878B), C4 (175B), GitHub (59B), ArXiv (28B), Wikipedia (24B),
    and StackExchange (20B) :footcite:`2023_redpajama`.

    The dataset is primarily in English, though the Wikipedia slice contains
    multiple languages.

    Dataset Structure:
        .. code-block:: python

            {
                "text": str,
                "meta": {
                    "url": str,
                    "timestamp": str,
                    "source": str,
                    "language": str,
                    ...
                },
                "red_pajama_subset": str
            }

    Subsets:
        - common_crawl
        - c4
        - github
        - arxiv
        - wikipedia
        - stackexchange

    See Also:
        - For more information on dataset creation and source data, please refer
          to the RedPajama GitHub repository:
          https://github.com/togethercomputer/RedPajama-Data
        - Hugging Face dataset page:
          https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T

    Note:
        The 'book' config is defunct and no longer accessible due to reported
        copyright infringement for the Book3 dataset contained in this config.

    Note:
        Please refer to the licenses of the data subsets you use. Links to the
        respective licenses can be found in the README.

    Citations:
        .. footbibliography::
    """

    default_dataset = "togethercomputer/RedPajama-Data-1T"
