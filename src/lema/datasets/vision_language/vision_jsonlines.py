import os
from typing import Optional

import pandas as pd

from lema.core.datasets import VisionLanguageSftDataset
from lema.core.registry import register_dataset
from lema.core.types.turn import Conversation


@register_dataset("vision_language_jsonl")
class JsonlinesDataset(VisionLanguageSftDataset):
    default_dataset = "custom"

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        data: Optional[list] = None,
        data_column: str = "messages",
        **kwargs,
    ):
        """Initializes a new instance of the JsonlinesDataset class."""
        self.data_column = data_column
        self.dataset_path = dataset_path

        if dataset_path is not None and data is not None:
            raise ValueError(
                "Either dataset_path or data must be provided, but not both"
            )

        if data is not None:
            self._data = pd.DataFrame({self.data_column: data})

        elif dataset_path is not None:
            if not os.path.isfile(dataset_path):
                raise ValueError(f"Dataset path does not exist: {dataset_path}")

            if not dataset_path.endswith(".jsonl"):
                raise ValueError("Dataset path must end with .jsonl")

            self._data = pd.read_json(dataset_path, lines=True)

            if self.data_column not in self._data.columns:
                raise ValueError(f"Data column {self.data_column} not found in dataset")
        else:
            raise ValueError("Dataset path or data must be provided")

        super().__init__(**kwargs)

    def _load_data(self) -> pd.DataFrame:
        # no-op, data is already loaded in __init__
        return self._data

    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        return Conversation(messages=example["messages"])
