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

from typing import Optional

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import VisionLanguageDpoDataset
from oumi.core.registry import register_dataset
from oumi.utils.io_utils import load_jsonlines


@register_dataset("vision_dpo_jsonl")
class VisionDpoJsonlinesDataset(VisionLanguageDpoDataset):
    """VisionDpoJsonlinesDataset for loading Vision-Language DPO data in Oumi format.

    This dataset class is designed to work with JSON Lines (.jsonl) files containing
    Vision-Language Direct Preference Optimization (DPO) data. It supports loading data
    either from a file or from a provided list of data samples.

    See `VisionLanguageDpoDataset` for more details.

    Example::

        dataset = VisionDpoJsonlinesDataset(
            dataset_path="data/dataset_examples/vision_language_dpo_format.jsonl"
        )
    """

    default_dataset = "vision_dpo_jsonl"

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        data: Optional[list[dict]] = None,
        **kwargs,
    ):
        """Initialize the VisionDpoJsonlinesDataset.

        Args:
            dataset_name: Name of the dataset (for registry purposes).
            dataset_path: Path to the JSONL file containing vision DPO data.
            data: List of data samples to use instead of loading from file.
            **kwargs: Additional arguments passed to the parent class.

        Raises:
            ValueError: If neither dataset_path nor data is provided.
        """
        if dataset_path is not None and data is not None:
            raise ValueError("Only one of dataset_path or data must be provided")

        if data is not None:
            rows = data

        elif dataset_path is not None:
            rows = load_jsonlines(dataset_path)
        else:
            raise ValueError("Either dataset_path or data must be provided")

        self._data = pd.DataFrame(rows)

        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, **kwargs)

    @override
    def _load_data(self) -> pd.DataFrame:
        """Load the data from the provided samples."""
        # data is already loaded in the constructor, no need to load again
        return self._data
