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


@register_dataset("togethercomputer/RedPajama-Data-V2")
class RedPajamaDataV2Dataset(BasePretrainingDataset):
    """RedPajama V2 Dataset for training large language models.

    This dataset includes over 100B text documents from 84 CommonCrawl snapshots,
    processed using the CCNet pipeline. It contains 30B documents with quality
    signals and 20B deduplicated documents :footcite:`2023_redpajama`.

    The dataset is available in English, German, French, Italian, and Spanish.

    Key Features:
        - Over 100B text documents
        - 30B documents with quality annotations
        - 20B unique documents after deduplication
        - Estimated 50.6T tokens in total (30.4T after deduplication)
        - Quality signals for filtering and analysis
        - Minhash signatures for fuzzy deduplication

    See Also:
        - Hugging Face dataset page:
          https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
        - Blog post: https://together.ai/blog/redpajama-data-v2
        - GitHub repo: https://github.com/togethercomputer/RedPajama-Data

    Note:
        - License: Common Crawl Foundation Terms of Use:
          https://commoncrawl.org/terms-of-use
        - Code: Apache 2.0 license

    Citations:
        .. footbibliography::
    """

    default_dataset = "togethercomputer/RedPajama-Data-V2"
