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


@register_dataset("wikimedia/wikipedia")
class WikipediaDataset(BasePretrainingDataset):
    """Dataset containing cleaned Wikipedia articles in multiple languages.

    This dataset is built from the Wikipedia dumps (https://dumps.wikimedia.org/)
    with one subset per language, each containing a single train split.
    Each example contains the content of one full Wikipedia article
    with cleaning to strip markdown and unwanted sections (references, etc.).

    Data Fields:
        id (str): ID of the article.
        url (str): URL of the article.
        title (str): Title of the article.
        text (str): Text content of the article.

    Note:
        All configurations contain a single 'train' split.

    Languages:
        The dataset supports numerous languages. For a full list, see:
        https://meta.wikimedia.org/wiki/List_of_Wikipedias

    Note:
        The dataset is licensed under the GNU Free Documentation License (GFDL) and
        the Creative Commons Attribution-Share-Alike 3.0 License.

    See Also:
        - Homepage: https://dumps.wikimedia.org
        - Hugging Face Hub: https://huggingface.co/datasets/wikimedia/wikipedia
    """

    default_dataset = "wikimedia/wikipedia"
