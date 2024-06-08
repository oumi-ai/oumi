from typing import Callable, Union

import transformers
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from trl.trainer import ConstantLengthDataset

from lema.core.types import DataParams, TrainingConfig
from lema.datasets.alpaca import alpaca_preprocessing_fn  # TODO: pull from registry
from lema.datasets.trl_dpo_preprocessor import trl_dpo_chat_preprocessor_fn
from lema.datasets.ultrachat_200k import trl_sft_ultrachat_200k_preprocessor_fn


def build_prompt_generation_fn(
    function_name: str, tokenizer: transformers.PreTrainedTokenizerBase
) -> Callable:
    """Build a prompt generation function.

    Args:
        function_name (str): The name of the prompt generation function.
        tokenizer: The tokenizer object used for tokenization.

    Returns:
        The prompt generation function corresponding to the given function_name.

    Raises:
        ValueError: If the function_name is unknown.
    """
    # TODO: this should be pulled from registry
    if function_name == "alpaca":
        return alpaca_preprocessing_fn(tokenizer)
    elif function_name == "trl_sft_ultrachat_200k":
        return trl_sft_ultrachat_200k_preprocessor_fn(tokenizer)
    elif function_name == "trl_dpo":
        return trl_dpo_chat_preprocessor_fn(tokenizer)

    raise ValueError(f"Unknown prompt generation function: {function_name}")


def build_dataset(
    config: TrainingConfig,
    tokenizer: transformers.PreTrainedTokenizerBase,
    **kwargs,
) -> Union[
    ConstantLengthDataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
]:
    """Build a dataset for training.

    Args:
        config: The training config.
        tokenizer: The tokenizer object to use for preprocessing.
        kwargs: Keyword arguments.

    Returns:
        dataset: The built dataset for training.
    """
    data_params: DataParams = config.data
    # TODO: should return all splits
    dataset = load_dataset(
        data_params.dataset_name,
        name=data_params.dataset_config,
        streaming=data_params.stream,
        split=data_params.split,
    )

    preprocessing_fn = None
    if data_params.preprocessing_function_name:
        preprocessing_fn = build_prompt_generation_fn(
            data_params.preprocessing_function_name, tokenizer
        )

    if data_params.pack:
        # Fetch max sequence length. If not specified, defaults to 1024.
        dataset_kwargs = {}
        if config.model.model_max_length:
            dataset_kwargs["seq_length"] = config.model.model_max_length
        # Our preprocessing functions take a dict as input and return a dict as output.
        # formatting_func must return a str, so we fetch the target str from the dict.
        if preprocessing_fn:
            dataset_kwargs["formatting_func"] = lambda x: preprocessing_fn(x)[
                data_params.text_col
            ]
        dataset = ConstantLengthDataset(
            tokenizer,
            dataset,
            dataset_text_field=data_params.text_col,
            **dataset_kwargs,
        )
    elif data_params.preprocessing_function_name:
        dataset = dataset.map(
            preprocessing_fn, **data_params.preprocessing_function_kwargs
        )

    return dataset
