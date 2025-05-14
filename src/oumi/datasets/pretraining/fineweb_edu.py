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


@register_dataset("HuggingFaceFW/fineweb-edu")
class FineWebEduDataset(BasePretrainingDataset):
    """FineWeb-Edu: A high-quality educational dataset filtered from web content.

    This dataset contains 1.3 trillion tokens of educational web pages filtered
    from the FineWeb dataset using an educational quality classifier. It aims to
    provide the finest collection of educational content from the web
    :footcite:`2024_fineweb_edu`.

    The dataset is available in multiple configurations:
      - Full dataset (default)
      - Individual CommonCrawl dumps (e.g. CC-MAIN-2024-10)
      - Sample subsets (10BT, 100BT, 350BT tokens)

    Key Features:
      - 1.3 trillion tokens of educational content
      - Filtered using a classifier trained on LLama3-70B-Instruct annotations
      - Outperforms other web datasets on educational benchmarks

    See Also:
      - Huggingface hub page: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

    Note:
      The dataset is released under the Open Data Commons Attribution License
      (ODC-By) v1.0.

    Citations:
      .. footbibliography::
    """

    default_dataset = "HuggingFaceFW/fineweb-edu"
