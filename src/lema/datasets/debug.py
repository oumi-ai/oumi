import time

import torch
from torch.utils.data import Dataset

from lema.core.registry import register_dataset


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
class DebugPretrainingDataset(Dataset):
    def __init__(
        self,
        dataset_size: int = 1000,
        vocab_size: int = 10000,
        sequence_length: int = 1024,
        preprocessing_time_ms: float = 0,
        **kwargs,
    ):
        """Initializes a DebugPretrainingDataset.

        Args:
            dataset_size: The size of the dataset.
            vocab_size: The size of the vocabulary.
            sequence_length: The length of each sequence.
            preprocessing_time_ms: The time taken for preprocessing in milliseconds.
            **kwargs: Additional keyword arguments.

        """
        self.size = dataset_size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.preprocessing_time_ms = preprocessing_time_ms

        self.data = torch.randint(
            low=0, high=self.vocab_size, size=(self.size, self.sequence_length)
        )

    def __len__(self):
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx):
        """Return the data and label at the given index."""
        if self.preprocessing_time_ms > 0:
            time.sleep(self.preprocessing_time_ms * 1000)
        return {"input_ids": self.data[idx], "labels": self.data[idx]}
