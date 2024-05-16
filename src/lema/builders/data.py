from typing import Callable, Optional

import transformers
from datasets import Dataset, load_dataset

from lema.datasets.alpaca import alpaca_preprocessing_fn  # TODO: pull from registry


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

    raise ValueError(f"Unknown prompt generation function: {function_name}")


def build_dataset(
    dataset_name: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    preprocessing_function_name: Optional[str] = None,
    **kwargs,
) -> Dataset:
    """Build a dataset for training.

    Args:
        dataset_name: The name of the dataset to load.
        tokenizer: The tokenizer object to use for preprocessing.
        preprocessing_function_name: The name of the preprocessing
            function to apply to the dataset.
        **kwargs: Additional keyword arguments to pass to the dataset mapping function.

    Returns:
        dataset: The built dataset for training.
    """
    # TODO: should return all splits
    dataset = load_dataset(dataset_name, split="train")

    if preprocessing_function_name:
        preprocessing_fn = build_prompt_generation_fn(
            preprocessing_function_name, tokenizer
        )
        dataset = dataset.map(preprocessing_fn, batched=True, **kwargs)

    return dataset
