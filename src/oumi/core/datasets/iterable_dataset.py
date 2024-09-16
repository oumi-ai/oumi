import abc
from typing import Any, Dict, Iterable, List, Optional

import datasets
import torch
from torch.utils.data import IterDataPipe

from oumi.core.tokenizers import BaseTokenizer
from oumi.utils.logging import logger


class BaseIterableDataset(IterDataPipe, abc.ABC):
    """Abstract base class for iterable datasets."""

    _data: Iterable[Any]
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
        """Initializes a new instance of the BaseIterableDataset class."""
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
        self._data = self._load_data()

    #
    # Main API
    #
    def __iter__(self):
        """Iterates over the dataset."""
        for item in self.data:
            yield self.transform(item)

    def iter_raw(self):
        """Iterates over the raw dataset."""
        yield from self.data

    def to_hf(self) -> datasets.IterableDataset:
        """Converts the dataset to a Hugging Face dataset."""
        return datasets.IterableDataset.from_generator(self.__iter__)

    @property
    def data(self) -> Iterable[Any]:
        """Returns the underlying dataset data."""
        return self._data

    #
    # Abstract Methods
    #
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
    """Abstract base class for pretraining iterable datasets."""

    def __init__(
        self,
        *,
        tokenizer: BaseTokenizer,
        seq_length: int,
        dataset_text_field: str = "text",
        append_concat_token: bool = True,
        add_special_tokens: bool = True,
        skip_last: bool = True,
        **kwargs,
    ):
        """Initializes a new instance of the BasePretrainingIterableDataset class."""
        if append_concat_token and tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer must have an EOS token if append_concat_token is enabled."
            )

        self.concat_token_id = tokenizer.eos_token_id if append_concat_token else None

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self._dataset_text_field = dataset_text_field
        self._append_concat_token = append_concat_token
        self._add_special_tokens = add_special_tokens
        self._skip_last = skip_last

        super().__init__(**kwargs)

    def __iter__(self):
        """Iterates over the dataset and yields samples of a specified sequence length.

        The underlying dataset is a stream of documents. Each document is expected to
        containt a text field `self._dataset_text_field` that will be tokenized.
        Training sampels are then yielded in sequences of length `self.seq_length`.

        Given this iterator might return samples from different documents, we optionally
        use `self.concat_token_id` to separate the sequences from different documents.
        """
        buffer = []
        for document in self.data:
            if self._append_concat_token and len(buffer) > 0:
                # We started preprocessing a new document
                # so we need to append the concatenation token to mark the end
                # of the previous document.
                buffer.append(self.concat_token_id)

            # Pre-process and tokenize the document
            document_tokens = self.transform(document[self._dataset_text_field])
            buffer.extend(document_tokens)

            # Yield sequences of the specified length.
            # Otherwise, resume pre-processing the next document.
            while len(buffer) >= self.seq_length:
                # We have enough tokens to create a fully packed sample
                yield self._create_training_sample(buffer[: self.seq_length])
                buffer = buffer[self.seq_length :]

        # Finished iterating on the dataset, yield the remaining buffer
        if len(buffer) > 0:
            if not self._skip_last or len(buffer) == self.seq_length:
                yield self._create_training_sample(buffer)

    def transform(self, sample: Any) -> List[int]:
        """Preprocesses the inputs in the given sample."""
        return self.tokenize(sample)

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes the given text.

        Should not apply any padding/truncation to allow for packing.
        """
        return self.tokenizer.encode(
            text=text,
            return_tensors=None,
            max_length=None,
            padding=False,
            truncation=False,
            add_special_tokens=self._add_special_tokens,
        )

    def _create_training_sample(self, tokens: list) -> Dict[str, torch.Tensor]:
        """Creates a training sample from the given tokens."""
        input_ids = torch.tensor(tokens)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
