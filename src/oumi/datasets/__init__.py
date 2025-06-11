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

"""Datasets module for the Oumi (Open Universal Machine Intelligence) library.

This module provides various dataset implementations for use in the Oumi framework.
These datasets are designed for different machine learning tasks and can be used
with the models and training pipelines provided by Oumi.

For more information on the available datasets and their usage, see the
:mod:`oumi.datasets` documentation.

Each dataset is implemented as a separate class, inheriting from appropriate base
classes in the :mod:`oumi.core.datasets` module. For usage examples and more detailed
information on each dataset, please refer to their respective class documentation.

See Also:
    - :mod:`oumi.models`: Compatible models for use with these datasets.
    - :mod:`oumi.core.datasets`: Base classes for dataset implementations.

Example:
    >>> from oumi.datasets import AlpacaDataset
    >>> from torch.utils.data import DataLoader
    >>> dataset = AlpacaDataset()
    >>> train_loader = DataLoader(dataset, batch_size=32)
"""

from oumi.datasets.debug import (
    DebugClassificationDataset,
    DebugPretrainingDataset,
    DebugSftDataset,
)
from oumi.datasets.evaluation import AlpacaEvalDataset
from oumi.datasets.grpo.letter_count import LetterCountGrpoDataset
from oumi.datasets.grpo.tldr import TldrGrpoDataset
from oumi.datasets.preference_tuning.orpo_dpo_mix import OrpoDpoMix40kDataset
from oumi.datasets.pretraining.c4 import C4Dataset
from oumi.datasets.pretraining.dolma import DolmaDataset
from oumi.datasets.pretraining.falcon_refinedweb import FalconRefinedWebDataset
from oumi.datasets.pretraining.fineweb_edu import FineWebEduDataset
from oumi.datasets.pretraining.pile import PileV1Dataset
from oumi.datasets.pretraining.red_pajama_v1 import RedPajamaDataV1Dataset
from oumi.datasets.pretraining.red_pajama_v2 import RedPajamaDataV2Dataset
from oumi.datasets.pretraining.slim_pajama import SlimPajamaDataset
from oumi.datasets.pretraining.starcoder import StarCoderDataset
from oumi.datasets.pretraining.the_stack import TheStackDataset
from oumi.datasets.pretraining.tiny_stories import TinyStoriesDataset
from oumi.datasets.pretraining.tiny_textbooks import TinyTextbooksDataset
from oumi.datasets.pretraining.wikipedia import WikipediaDataset
from oumi.datasets.pretraining.wikitext import WikiTextDataset
from oumi.datasets.pretraining.youtube_commons import YouTubeCommonsDataset
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
from oumi.datasets.vision_language.coco_captions import COCOCaptionsDataset
from oumi.datasets.vision_language.docmatix import DocmatixDataset
from oumi.datasets.vision_language.flickr30k import Flickr30kDataset
from oumi.datasets.vision_language.geometry3k import Geometry3kDataset
from oumi.datasets.vision_language.huggingface import HuggingFaceVisionDataset
from oumi.datasets.vision_language.llava_instruct_mix_vsft import (
    LlavaInstructMixVsftDataset,
)
from oumi.datasets.vision_language.lmms_lab_multimodal_open_r1 import (
    LmmsLabMultimodalOpenR1Dataset,
)
from oumi.datasets.vision_language.mnist_sft import MnistSftDataset
from oumi.datasets.vision_language.pixmo_ask_model_anything import (
    PixmoAskModelAnythingDataset,
)
from oumi.datasets.vision_language.pixmo_cap import PixmoCapDataset
from oumi.datasets.vision_language.pixmo_cap_qa import PixmoCapQADataset
from oumi.datasets.vision_language.the_cauldron import TheCauldronDataset
from oumi.datasets.vision_language.vision_jsonlines import VLJsonlinesDataset
from oumi.datasets.vision_language.vqav2_small import Vqav2SmallDataset

__all__ = [
    "AlpacaDataset",
    "AlpacaEvalDataset",
    "ArgillaDollyDataset",
    "ArgillaMagpieUltraDataset",
    "AyaDataset",
    "C4Dataset",
    "ChatqaDataset",
    "ChatqaTatqaDataset",
    "ChatRAGBenchDataset",
    "CoALMDataset",
    "COCOCaptionsDataset",
    "DebugClassificationDataset",
    "DebugPretrainingDataset",
    "DebugSftDataset",
    "DocmatixDataset",
    "DolmaDataset",
    "FalconRefinedWebDataset",
    "FineWebEduDataset",
    "Flickr30kDataset",
    "Flickr30kDataset",
    "Geometry3kDataset",
    "HuggingFaceDataset",
    "HuggingFaceVisionDataset",
    "LetterCountGrpoDataset",
    "LlavaInstructMixVsftDataset",
    "LmmsLabMultimodalOpenR1Dataset",
    "MagpieProDataset",
    "MnistSftDataset",
    "OpenO1SFTDataset",
    "OrpoDpoMix40kDataset",
    "PileV1Dataset",
    "PixmoAskModelAnythingDataset",
    "PixmoCapDataset",
    "PixmoCapQADataset",
    "PromptResponseDataset",
    "RedPajamaDataV1Dataset",
    "RedPajamaDataV2Dataset",
    "SlimPajamaDataset",
    "StarCoderDataset",
    "TextSftJsonLinesDataset",
    "TheCauldronDataset",
    "TheStackDataset",
    "TinyStoriesDataset",
    "TinyTextbooksDataset",
    "TldrGrpoDataset",
    "Tulu3MixtureDataset",
    "UltrachatH4Dataset",
    "VLJsonlinesDataset",
    "Vqav2SmallDataset",
    "WikipediaDataset",
    "WikiTextDataset",
    "WildChatDataset",
    "YouTubeCommonsDataset",
]
