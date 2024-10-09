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
    >>> dataset = AlpacaDataset()
    >>> train_loader = DataLoader(dataset, batch_size=32)
"""

from oumi.datasets.alpaca import AlpacaDataset
from oumi.datasets.chatqa import ChatqaDataset
from oumi.datasets.chatrag_bench import ChatRAGBenchDataset
from oumi.datasets.debug import DebugClassificationDataset, DebugPretrainingDataset
from oumi.datasets.dolly import ArgillaDollyDataset
from oumi.datasets.magpie import ArgillaMagpieUltraDataset, MagpieProDataset
from oumi.datasets.sft_jsonlines import TextSftJsonLinesDataset
from oumi.datasets.vision_language.coco_captions import COCOCaptionsDataset
from oumi.datasets.vision_language.flickr30k import Flickr30kDataset
from oumi.datasets.vision_language.llava_instruct_mix_vsft import (
    LlavaInstructMixVsftDataset,
)
from oumi.datasets.vision_language.vision_jsonlines import VLJsonlinesDataset

__all__ = [
    "AlpacaDataset",
    "ChatqaDataset",
    "ChatRAGBenchDataset",
    "DebugClassificationDataset",
    "DebugPretrainingDataset",
    "COCOCaptionsDataset",
    "Flickr30kDataset",
    "LlavaInstructMixVsftDataset",
    "VLJsonlinesDataset",
    "ArgillaDollyDataset",
    "ArgillaMagpieUltraDataset",
    "MagpieProDataset",
    "TextSftJsonLinesDataset",
]
