"""Core datasets module for the LeMa (Learning Machines) library.

This module provides base classes for different types of datasets used in
the LeMa framework. These base classes serve as foundations for creating custom
datasets for various machine learning tasks.

These base classes can be extended to create custom datasets tailored to specific
machine learning tasks within the LeMa framework.

For more detailed information on each class, please refer to their respective
documentation.
"""

from lema.core.datasets.base_dataset import BaseLMSftDataset, BaseMapDataset
from lema.core.datasets.iterable_dataset import (
    BaseIterableDataset,
    BasePretrainingIterableDataset,
)
from lema.core.datasets.vision_language_dataset import VisionLanguageSftDataset

__all__ = [
    "BaseIterableDataset",
    "BaseLMSftDataset",
    "BaseMapDataset",
    "BasePretrainingIterableDataset",
    "VisionLanguageSftDataset",
]
