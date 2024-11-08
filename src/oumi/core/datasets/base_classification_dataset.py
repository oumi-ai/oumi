import re
from abc import ABC, abstractmethod
from typing import Literal, Optional, Union, cast

import pandas as pd

import torch

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.types.classification import Classification
from oumi.utils.logging import logger


class BaseClassificationDataset(BaseMapDataset, ABC):
    """In-memory dataset for SFT data."""

    default_dataset = None

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        return_tensors: bool = False,
        text_col: str = "text",
        label_col: str = "labels",
        num_labels: int = 2,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseSftDataset class."""
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
        )

        self._text_col = text_col
        self._label_col = label_col
        self._tokenizer = tokenizer
        self._return_tensors = "pt" if return_tensors else None
        self._num_labels = num_labels

        self._data = self._load_data()

    #
    # Properties
    #
    @property
    def text_col(self) -> str:
        """Gets the text target column.

        The generated text will be stored in this column.
        """
        return self._text_col
    
    @property
    def label_col(self) -> str:
        """Gets the text target column.

        The generated text will be stored in this column.
        """
        return self._label_col

    #
    # Main API
    #
    def input(self, idx: int) -> str:
        """Returns the prompt at the specified index.

        Args:
            idx (int): The index of the input to retrieve.

        Returns:
            str: The input at the specified index.
        """
        return self.tokenize(self.classification(idx), tokenize=False)[self.text_col]

    def classification(self, idx: int) -> Classification:
        """Returns the classification at the specified index.

        Args:
            idx (int): The index of the classification to retrieve.

        Returns:
            str: The classification at the specified index.
        """
        sample = self.raw(idx)
        return self.transform_classification(sample)

    #
    # Pre-processing
    #
    def transform(self, sample: pd.Series) -> dict:
        """Preprocesses the inputs in the given sample."""
        return self.tokenize(self.transform_classification(sample))

    #
    # Abstract Methods
    #
    @abstractmethod
    def transform_classification(self, example: Union[dict, pd.Series]) -> Classification:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and instruction.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        raise NotImplementedError

    def tokenize(
        self,
        sample: Union[dict, pd.Series, Classification],
        tokenize: bool = True,
    ) -> dict:
        """Applies the chat template carried by the tokenizer to the input example.

        Args:
            sample (Dict): Mapping `messages` to a List containing the (ordered)
                messages exchanged within a single chat dialogue.
                Each item of example["messages"] is a dict mapping the `content` of the
                message and the `role` of the one relayed it.
                E.g., role == 'user' or role == 'assistant'.
            tokenize (bool): Whether to tokenize the messages or not.

        Raises:
            NotImplementedError: Currently only the `sft` task mode is supported.
            ValueError: if requested `task` is not in "sft" or "generation"

        Returns:
            Dict: It adds a `text` key in the input `example` dictionary, mapped to
            a string carrying the `messages` to the tokenizer's chat format.
        """
        if self._tokenizer is None:
            raise ValueError("Tokenizer is required for tokenization.")

        if isinstance(sample, Classification):
            classification = sample
        else:
            if isinstance(sample, pd.Series):
                sample = sample.to_dict()

            if isinstance(sample, dict):
                classification = Classification.from_dict(sample)
            else:
                raise ValueError(
                    "Input samples must be a Classification or a dict with "
                    "'input' and 'label' keys."
                )

        return self._tokenize(classification, tokenize)

    def _tokenize(
        self, sample: Classification, tokenize: bool = True
    ) -> dict:
        if self._tokenizer is None:
            raise ValueError("Tokenizer is required for tokenization.")
        
        if not tokenize:
            return {
                self.text_col: sample.input,
                self.label_col: sample.label,
            }
        
        results = self._tokenizer(sample.input,
                                  return_tensors=self._return_tensors,
                                  truncation=True)
        results[self.label_col] = torch.tensor([sample.label])
        return cast(dict, results)
