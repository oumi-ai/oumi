from abc import ABC, abstractmethod
from typing import Literal, Optional, Union, cast

import pandas as pd

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.types.conversation import Conversation


class BaseSftDataset(BaseMapDataset, ABC):
    """In-memory dataset for SFT data.

    The SFT datasets are expected to be in the following format:
    WIP.

    The exected output is a tokenized prompt
    """

    default_dataset = None

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        task: Literal["sft", "generation", "auto"] = "auto",
        return_tensors: bool = False,
        text_col: str = "text",
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseSftDataset class."""
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
        )

        self._task = task
        self._text_col = text_col
        self._tokenizer = tokenizer
        self._return_tensors = "pt" if return_tensors else None
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
    def task(self) -> str:
        """Gets the task mode for the dataset.

        The generated prompt is often different for generation vs SFT tasks.
        """
        return self._task

    #
    # Main API
    #

    def prompt(self, idx: int) -> str:
        """Returns the prompt at the specified index.

        Args:
            idx (int): The index of the prompt to retrieve.

        Returns:
            str: The prompt at the specified index.
        """
        return self.tokenize(self.conversation(idx), tokenize=False)[self.text_col]

    def conversation(self, idx: int) -> Conversation:
        """Returns the conversation at the specified index.

        Args:
            idx (int): The index of the conversation to retrieve.

        Returns:
            str: The conversation at the specified index.
        """
        sample = self.raw(idx)
        return self.transform_conversation(sample)

    #
    # Abstract Methods
    #
    @abstractmethod
    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and instruction.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        raise NotImplementedError

    #
    # Pre-processing
    #
    def transform(self, sample: pd.Series) -> dict:
        """Preprocesses the inputs in the given sample."""
        return self.tokenize(self.transform_conversation(sample))

    def tokenize(
        self,
        samples: Union[dict, pd.Series, Conversation],
        tokenize: bool = True,
    ) -> dict:
        """Applies the chat template carried by the tokenizer to the input example.

        Args:
            samples (Dict): Mapping `messages` to a List containing the (ordered)
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

        results = self._tokenizer.apply_chat_template(
            samples,  # type: ignore
            tokenize=tokenize,
            return_dict=tokenize,
            return_tensors=self._return_tensors,
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            add_generation_prompt=(self.task == "generation"),
        )

        if tokenize:
            return cast(dict, results)
        else:
            return {
                self.text_col: results,
            }
