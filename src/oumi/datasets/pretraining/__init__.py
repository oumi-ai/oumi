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

"""Pretraining datasets module."""

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

__all__ = [
    "C4Dataset",
    "DolmaDataset",
    "FalconRefinedWebDataset",
    "FineWebEduDataset",
    "PileV1Dataset",
    "RedPajamaDataV1Dataset",
    "RedPajamaDataV2Dataset",
    "SlimPajamaDataset",
    "StarCoderDataset",
    "TheStackDataset",
    "TinyStoriesDataset",
    "TinyTextbooksDataset",
    "WikipediaDataset",
    "WikiTextDataset",
    "YouTubeCommonsDataset",
]
