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


@register_dataset("EleutherAI/pile")
class PileV1Dataset(BasePretrainingDataset):
    """The Pile: An 825 GiB diverse, open source language modeling dataset.

    The Pile is a large-scale English language dataset consisting of 22 smaller,
    high-quality datasets combined together. It is designed for training large
    language models and supports various natural language processing tasks
    :footcite:`2020_thepile,2022_thepile_datasheet`.

    Data Fields:
      text (str): The main text content.
      meta (dict): Metadata about the instance, including 'pile_set_name'.

    Key Features:
        - 825 GiB of diverse text data
        - Primarily in English
        - Supports text generation and fill-mask tasks
        - Includes various subsets like enron_emails, europarl, free_law, etc.

    Subsets:
      - all
      - enron_emails
      - europarl
      - free_law
      - hacker_news
      - nih_exporter
      - pubmed
      - pubmed_central
      - ubuntu_irc
      - uspto
      - github

    Splits:
      - train
      - validation
      - test

    See Also:
      - Homepage: https://pile.eleuther.ai/
      - HuggingFace hub: https://huggingface.co/datasets/EleutherAI/pile

    Warning:
        This dataset contains text from various sources and may include
        personal or sensitive information. Users should consider potential biases
        and limitations when using this dataset.

    Citations:
        .. footbibliography::
    """

    default_dataset = "EleutherAI/pile"
