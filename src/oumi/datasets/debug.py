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

import time
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from oumi.core.datasets.base_dpo_dataset import BaseExperimentalDpoDataset
from oumi.core.datasets.base_pretraining_dataset import BasePretrainingDataset
from oumi.core.datasets.base_sft_dataset import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("debug_classfication")
class DebugClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_size: int = 1000,
        feature_dim: int = 128,
        data_type: str = "float32",
        num_classes: int = 10,
        preprocessing_time_ms: float = 0,
        **kwargs,
    ):
        """Initialize a DebugClassificationDataset.

        This dataset generates random data and labels for debugging purposes.

        Args:
            dataset_size: The size of the dataset.
            feature_dim: The dimension of each feature.
            data_type: The data type of the dataset.
                Supported values are "float32", "float16", and "bfloat16".
            num_classes: The number of classes in the dataset.
            preprocessing_time_ms: The time taken for preprocessing
                in milliseconds.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the data_type is not supported.
        """
        self.size = dataset_size
        self.feature_dim = feature_dim
        self.data_type = data_type
        self.num_classes = num_classes
        self.preprocessing_time_ms = preprocessing_time_ms

        if self.data_type == "float32":
            dtype = torch.float32
        elif self.data_type == "float16":
            dtype = torch.float16
        elif self.data_type == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

        self.data = torch.randn(self.size, self.feature_dim, dtype=dtype)
        self.labels = torch.randint(0, self.num_classes, (self.size,))

    def __len__(self):
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx):
        """Return the data and label at the given index."""
        if self.preprocessing_time_ms > 0:
            time.sleep(self.preprocessing_time_ms * 1000)
        return {"features": self.data[idx], "labels": self.labels[idx]}


@register_dataset("debug_pretraining")
class DebugPretrainingDataset(BasePretrainingDataset):
    default_dataset = "debug_pretraining"

    def __init__(
        self,
        dataset_size: int = 1000,
        **kwargs,
    ):
        """Initializes a DebugPretrainingDataset.

        Args:
            dataset_size: The size of the dataset.
            **kwargs: Additional keyword arguments.

        """
        self.size = dataset_size

        super().__init__(**kwargs)

    def _load_data(self) -> list[dict]:
        return [{"text": f"This is document number {idx}."} for idx in range(self.size)]


@register_dataset("debug_sft")
class DebugSftDataset(BaseSftDataset):
    default_dataset = "debug_sft"

    def __init__(
        self,
        dataset_size: int = 5,
        **kwargs,
    ):
        """Initializes a DebugSftDataset."""
        self.size = dataset_size

        super().__init__(**kwargs)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transforms the example into a Conversation object."""
        return Conversation(
            messages=[
                Message(
                    role=Role.USER, content=(example.get("user_message", None) or "")
                ),
                Message(
                    role=Role.ASSISTANT,
                    content=(example.get("assistant_message", None) or ""),
                ),
            ]
        )

    @override
    def _load_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_message": [
                    f"Hello, how are you? (Document number {idx})"
                    for idx in range(self.size)
                ],
                "assistant_message": [
                    f"I'm fine, thank you! (Document number {idx})"
                    for idx in range(self.size)
                ],
            }
        )


@register_dataset("debug_dpo")
class DebugDpoDataset(BaseExperimentalDpoDataset):
    default_dataset = "debug_dpo"

    def __init__(
        self,
        dataset_size: int = 5,
        **kwargs,
    ):
        """Initializes a DebugSftDataset."""
        self.size = dataset_size

        super().__init__(**kwargs)

    def transform_preference(self, sample: dict) -> dict:
        """Transforms the sample into a preference dict."""
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }

    @override
    def _load_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "prompt": [
                    f"Hello, how are you? (Document number {idx})"
                    for idx in range(self.size)
                ],
                "chosen": [
                    f"I'm fine, thank you! (Document number {idx})"
                    for idx in range(self.size)
                ],
                "rejected": [
                    f"fine (Document number {idx})" for idx in range(self.size)
                ],
            }
        )
