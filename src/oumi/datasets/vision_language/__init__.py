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

"""Vision-Language datasets module."""

from oumi.datasets.vision_language.coco_captions import COCOCaptionsDataset
from oumi.datasets.vision_language.docmatix import DocmatixDataset
from oumi.datasets.vision_language.flickr30k import Flickr30kDataset
from oumi.datasets.vision_language.llava_instruct_mix_vsft import (
    LlavaInstructMixVsftDataset,
)
from oumi.datasets.vision_language.mnist_sft import MnistSftDataset
from oumi.datasets.vision_language.the_cauldron import TheCauldronDataset
from oumi.datasets.vision_language.vision_jsonlines import VLJsonlinesDataset
from oumi.datasets.vision_language.vqav2_small import Vqav2SmallDataset

__all__ = [
    "COCOCaptionsDataset",
    "DocmatixDataset",
    "Flickr30kDataset",
    "LlavaInstructMixVsftDataset",
    "MnistSftDataset",
    "VLJsonlinesDataset",
    "Vqav2SmallDataset",
    "TheCauldronDataset",
]
