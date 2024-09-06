"""Datasets module for the LeMa (Learning Machines) library.

This module provides various dataset implementations for use in the LeMa framework.
These datasets are designed for different machine learning tasks and can be used
with the models and training pipelines provided by LeMa.

For more information on the available datasets and their usage, see the
:mod:`lema.datasets` documentation.

Each dataset is implemented as a separate class, inheriting from appropriate base
classes in the :mod:`lema.core.datasets` module. For usage examples and more detailed
information on each dataset, please refer to their respective class documentation.

See Also:
    - :mod:`lema.models`: Compatible models for use with these datasets.
    - :mod:`lema.core.datasets`: Base classes for dataset implementations.

Example:
    >>> from lema.datasets import AlpacaDataset
    >>> dataset = AlpacaDataset()
    >>> train_loader = DataLoader(dataset, batch_size=32)
"""

from lema.datasets.alpaca import AlpacaDataset
from lema.datasets.chatqa import ChatqaDataset
from lema.datasets.chatrag_bench import ChatRAGBenchDataset
from lema.datasets.debug import DebugClassificationDataset, DebugPretrainingDataset
from lema.datasets.vision_language.coco_captions import COCOCaptionsDataset
from lema.datasets.vision_language.flickr30k import Flickr30kDataset
from lema.datasets.vision_language.vision_jsonlines import JsonlinesDataset

__all__ = [
    "AlpacaDataset",
    "ChatqaDataset",
    "ChatRAGBenchDataset",
    "DebugClassificationDataset",
    "DebugPretrainingDataset",
    "COCOCaptionsDataset",
    "Flickr30kDataset",
    "JsonlinesDataset",
]
