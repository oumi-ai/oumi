import os
from abc import ABC, abstractmethod
from typing import Literal, Optional, Union, cast

import datasets
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from lema.core.types.turn import Conversation, Message, Role
from lema.logging import logger


#
# Base Map Dataset
#
class BaseMapDataset(Dataset, ABC):
    data: pd.DataFrame
    dataset_name_or_path: str
    default_dataset: Optional[str] = None
    supported_datasets = {}

    def __init__(
        self,
        *,
        dataset_name_or_path: Optional[str],
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
        self.split = split

    #
    # Main API
    #
    def __getitem__(self, idx: int) -> dict:
        """Gets the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: The item at the specified index.
        """
        sample = self.raw(idx)
        processed = self.transform(sample)
        return processed

    def __len__(self) -> int:
        """Gets the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data)

    def raw(self, idx: int) -> Union[dict, pd.Series]:
        """Returns the raw data at the specified index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            pd.Series: The raw data at the specified index.
        """
        return self.data.iloc[idx]

    def as_generator(self):
        """Returns a generator for the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def to_hf(self) -> datasets.Dataset:
        """Converts the dataset to a Hugging Face dataset."""
        return cast(
            datasets.Dataset, datasets.Dataset.from_generator(self.as_generator)
        )

    #
    # Abstract Methods
    #
    @abstractmethod
    def transform(self, sample: Union[dict, pd.Series]) -> dict:
        """Preprocesses the inputs in the given sample.

        Args:
            sample (dict): A dictionary containing the input data.

        Returns:
            dict: A dictionary containing the preprocessed input data.
        """
        raise NotImplementedError

    #
    # Data Loading
    #
    def _load_data(self) -> pd.DataFrame:
        """Loads the dataset from the specified source.

        Returns:
            dict: The loaded dataset.
        """
        if os.path.exists(self.dataset_name_or_path):
            if self.dataset_name_or_path.endswith(".jsonl"):
                return self._load_jsonl_dataset(self.dataset_name_or_path)

            elif self.dataset_name_or_path.endswith(".parquet"):
                return self._load_parquet_dataset(self.dataset_name_or_path)

            else:
                raise ValueError(
                    f"File format not supported for {self.dataset_name_or_path}"
                )

        return self._load_hf_hub_dataset(self.dataset_name_or_path)

    def _load_hf_hub_dataset(self, path: str) -> pd.DataFrame:
        """Loads the dataset from the specified Hugging Face Hub source.

        Returns:
            dict: The loaded dataset.
        """
        splits_or_dataset = datasets.load_dataset(path=path)

        if isinstance(
            splits_or_dataset, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            raise ValueError("IterableDataset is not supported with this class.")

        # Grab a single dataset split
        if isinstance(splits_or_dataset, datasets.Dataset):
            dataset = splits_or_dataset

        elif self.split is not None:
            dataset = splits_or_dataset[self.split]

        elif len(splits_or_dataset) == 1:
            dataset = splits_or_dataset.values().__iter__().__next__()

        else:
            raise ValueError(
                "Multiple splits found in the dataset. Please specify a single split. "
                f"Available splits: {list(splits_or_dataset.keys())}"
            )

        return cast(pd.DataFrame, dataset.to_pandas())

    def _load_jsonl_dataset(self, path: str) -> pd.DataFrame:
        return pd.read_json(path, lines=True)

    def _load_parquet_dataset(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)


class BaseLMSftDataset(BaseMapDataset, ABC):
    """In-memory dataset for SFT data.

    The SFT datasets are expected to be in the following format:
    WIP.

    The exected output is a tokenized prompt
    """

    default_dataset = None

    def __init__(
        self,
        *,
        dataset_name_or_path: Optional[str] = None,
        split: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        task: Literal["sft", "generation", "auto"] = "auto",
        return_tensors: bool = False,
        text_col: str = "text",
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseSftDataset class."""
        super().__init__(
            dataset_name_or_path=dataset_name_or_path, split=split, **kwargs
        )

        self.task = task
        self.text_col = text_col
        self._tokenizer = tokenizer
        self.return_tensors = "pt" if return_tensors else None
        self.data = self._load_data()

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
    def transform(self, sample: pd.Series) -> Union[dict, pd.Series]:
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
            return_tensors=self.return_tensors,
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            padding=True,
            add_generation_prompt=(self.task == "generation"),
        )

        if tokenize:
            return cast(dict, results)
        else:
            return {
                self.text_col: results,
            }


#
# Custom Dataset
#
class LemaSftDataset(BaseLMSftDataset):
    pass


class AlpacaDataset(BaseLMSftDataset):
    system_prompt = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
    )

    default_dataset = "tatsu-lab/alpaca"

    supported_datasets = {"yahma/alpaca-cleaned", "tatsu-lab/alpaca"}

    def __init__(
        self,
        *,
        include_system_prompt: bool = True,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the AlpacaDataset class."""
        self.include_system_prompt = include_system_prompt

        super().__init__(**kwargs)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and instruction.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        messages = []

        # Use default aplaca user prompt template
        if example.get("input") is not None and len(example["input"]) > 0:
            # This example has both an instruction and a user input.
            user_prompt = """{instruction}\n\n### Input:\n{input}""".format(
                instruction=example["instruction"], input=example["input"]
            )
        else:
            user_prompt = cast(str, example["instruction"])

        model_output = cast(str, example["output"])

        # Create message list
        if self.include_system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        messages.append(Message(role=Role.USER, content=user_prompt))
        messages.append(Message(role=Role.ASSISTANT, content=model_output))

        return Conversation(messages=messages)


class ChatQADataset(BaseLMSftDataset):
    default_dataset = "nvidia/ChatQA-Training-Data"

    supported_datasets = {"nvidia/ChatQA-Training-Data"}

    def format_inputs(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and instruction.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        messages = []

        for message in example["messages"]:
            messages.append(Message(role=message["role"], content=message["content"]))

        for response in example["answers"]:
            messages.append({"role": Role.ASSISTANT, "content": response})

        return Conversation(messages=messages)
