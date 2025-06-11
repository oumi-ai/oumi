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

from oumi.core.registry import register_dataset
from oumi.datasets.sft.alpaca import AlpacaDataset


@register_dataset("uiuc-convai/CoALM-IT")
class CoALMDataset(AlpacaDataset):
    """Dataset class for the UIUC CoALM dataset.

    This dataset follows the same structure as the Alpaca dataset, with
    instruction, input, and output fields. It is designed for training
    Conversational Agentic Language Models (CoALM) that can handle both
    task-oriented dialogue and function calling.


    Dataset Sources:
        - Paper: https://arxiv.org/abs/2502.08820
        - Project Page: https://emrecanacikgoz.github.io/CoALM/
        - Repository: https://github.com/oumi-ai/oumi/tree/main/configs/projects/calm
        - Dataset: https://huggingface.co/datasets/uiuc-convai/CoALM-IT

    Examples:
        >>> from oumi.datasets import CoALMDataset
        >>> dataset = CoALMDataset()
        >>> # The dataset will be loaded from HuggingFace with the path
        >>> # "uiuc-convai/CoALM-IT" and transformed into the Oumi
        >>> # conversation format automatically.
    """

    default_dataset = "uiuc-convai/CoALM-IT"
