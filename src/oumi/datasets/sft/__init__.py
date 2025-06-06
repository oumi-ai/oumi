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

"""Supervised fine-tuning datasets module."""

from oumi.datasets.sft.alpaca import AlpacaDataset
from oumi.datasets.sft.aya import AyaDataset
from oumi.datasets.sft.chatqa import ChatqaDataset, ChatqaTatqaDataset
from oumi.datasets.sft.chatrag_bench import ChatRAGBenchDataset
from oumi.datasets.sft.coalm import CoALMDataset
from oumi.datasets.sft.dolly import ArgillaDollyDataset
from oumi.datasets.sft.huggingface import HuggingFaceDataset
from oumi.datasets.sft.magpie import ArgillaMagpieUltraDataset, MagpieProDataset
from oumi.datasets.sft.openo1_sft import OpenO1SFTDataset
from oumi.datasets.sft.prompt_response import PromptResponseDataset
from oumi.datasets.sft.sft_jsonlines import TextSftJsonLinesDataset
from oumi.datasets.sft.tulu3_sft_mixture import Tulu3MixtureDataset
from oumi.datasets.sft.ultrachat import UltrachatH4Dataset
from oumi.datasets.sft.wildchat import WildChatDataset

__all__ = [
    "AlpacaDataset",
    "ArgillaDollyDataset",
    "ArgillaMagpieUltraDataset",
    "AyaDataset",
    "ChatqaDataset",
    "ChatqaTatqaDataset",
    "ChatRAGBenchDataset",
    "CoALMDataset",
    "HuggingFaceDataset",
    "MagpieProDataset",
    "OpenO1SFTDataset",
    "PromptResponseDataset",
    "TextSftJsonLinesDataset",
    "Tulu3MixtureDataset",
    "UltrachatH4Dataset",
    "WildChatDataset",
]
