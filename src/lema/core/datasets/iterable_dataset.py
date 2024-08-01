import abc
from typing import Any, Dict, Iterable, Optional

import datasets
import torch
from torch.utils.data import IterDataPipe
from transformers import PreTrainedTokenizerBase

from lema.utils.logging import logger


class BaseIterableDataset(IterDataPipe, abc.ABC):
    data: Iterable[Any]
    dataset_name_or_path: str
    default_dataset: Optional[str] = None
    default_subset: Optional[str] = None

    def __init__(
        self,
        *,
        dataset_name_or_path: Optional[str],
        subset: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseDataset class."""
        if len(kwargs) > 0:
            logger.debug(
                f"Unknown arguments: {', '.join(kwargs.keys())}. "
                "Please check the class constructor for supported arguments."
            )

        dataset_name_or_path = dataset_name_or_path or self.default_dataset

        if dataset_name_or_path is None:
            raise ValueError(
                "Please specify a dataset_name_or_path or "
                "set the default_dataset class attribute."
            )

        self.dataset_name_or_path = dataset_name_or_path
        self.dataset_subset = subset or self.default_subset
        self.split = split
        self.data = self._load_data()

    def __iter__(self):
        """Iterates over the dataset."""
        for item in self.data:
            yield self.transform(item)

    @abc.abstractmethod
    def transform(self, sample: Any) -> Dict[str, Any]:
        """Preprocesses the inputs in the given sample.

        Args:
            sample (Any): A sample from the dataset.

        Returns:
            dict: A dictionary containing the preprocessed input data.
        """
        raise NotImplementedError

    def _load_data(self) -> Iterable[Any]:
        """Loads the dataset from the specified source."""
        return datasets.load_dataset(
            path=self.dataset_name_or_path,
            name=self.dataset_subset,
            split=self.split,
            streaming=True,
        )


class BasePretrainingIterableDataset(BaseIterableDataset):
    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        dataset_text_field: str = "text",
        **kwargs,
    ):
        """Initializes a new instance of the BasePretrainingIterableDataset class."""
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.dataset_text_field = dataset_text_field

    def __iter__(self):
        """Iterates over the dataset."""
        buffer = []
        for sample in self.data:
            buffer.extend(self.tokenize(sample[self.dataset_text_field]))
            while len(buffer) >= self.seq_length:
                yield self.create_sample(buffer[: self.seq_length])
                buffer = buffer[self.seq_length :]

    def tokenize(self, text: str) -> list:
        """Tokenizes the given text."""
        return self.tokenizer.encode(text)

    def create_sample(self, tokens: list) -> Dict[str, torch.Tensor]:
        """Creates a sample from the given tokens."""
        input_ids = torch.tensor(tokens)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
