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


@register_dataset("Salesforce/wikitext")
class WikiTextDataset(BasePretrainingDataset):
    """WikiText language modeling dataset.

    The WikiText dataset is a collection of over 100 million tokens extracted from
    verified Good and Featured articles on Wikipedia. It is available in two sizes:
    WikiText-2 (2 million tokens) and WikiText-103 (103 million tokens). Each size
    comes in two variants: raw (for character-level work) and processed (for
    word-level work) :footcite:`2016_pointer_sentinel`.

    The dataset is well-suited for models that can take advantage of long-term
    dependencies, as it is composed of full articles and retains original case,
    punctuation, and numbers.

     Data Fields:
        text (str): The text content of the dataset.

    See Also:
        - Hugging Face Hub: https://huggingface.co/datasets/Salesforce/wikitext

    Note:
        The dataset is licensed under the Creative Commons Attribution-ShareAlike
        License (CC BY-SA 4.0).

    Citations:
        .. footbibliography::
    """

    default_dataset = "Salesforce/wikitext"
