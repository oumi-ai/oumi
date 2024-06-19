import copy
from typing import Callable, List, Optional, Sequence, TypeVar, Union, cast

import transformers
from datasets import (
    Dataset,
    IterableDataset,
    ReadInstruction,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)
from trl.trainer import ConstantLengthDataset

from lema.core.types import DataParams, DatasetParams, MixtureStrategy, TrainingConfig
from lema.datasets.alpaca import alpaca_preprocessing_fn  # TODO: pull from registry
from lema.datasets.trl_dpo_preprocessor import trl_dpo_chat_preprocessor_fn
from lema.datasets.ultrachat_200k import trl_sft_ultrachat_200k_preprocessor_fn

DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)


def build_prompt_generation_fn(
    function_name: str, tokenizer: transformers.PreTrainedTokenizerBase
) -> Callable:
    """Builds a prompt generation function.

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
    seed: Optional[int] = None,
    **kwargs,
) -> Union[ConstantLengthDataset, DatasetType]:
    """Builds a dataset for training.

    Args:
        config: The training config.
        tokenizer: The tokenizer object to use for preprocessing.
        seed: If specified, a seed used for random sampling.
        kwargs: Keyword arguments.

    Returns:
        dataset: The built dataset for training.
    """
    data_params: DataParams = config.data

    # TODO: should return all splits
    datasets = [
        _preprocess_dataset(
            _sample_dataset(dataset_params, data_params.stream),
            dataset_params,
            tokenizer,
        )
        for dataset_params in data_params.datasets
    ]
    mixture_proportions = [
        dataset.mixture_proportion for dataset in data_params.datasets
    ]

    # Interleave datasets using mixture_strategy.
    dataset = _mix_datasets(
        datasets,
        mixture_proportions,
        data_params.mixture_strategy,
        seed,
    )
    if data_params.pack:
        # Fetch max sequence length. If not specified, defaults to 1024.
        dataset_kwargs = {}
        if config.model.model_max_length:
            dataset_kwargs["seq_length"] = config.model.model_max_length
        dataset = ConstantLengthDataset(
            tokenizer,
            dataset,
            dataset_text_field=data_params.text_col,
            **dataset_kwargs,
        )
    return dataset


def _mix_datasets(
    dataset_list: List[DatasetType],
    mixture_proportions: Sequence[Optional[float]],
    mixture_strategy: str,
    seed: Optional[int],
) -> DatasetType:
    """Joins multiple datasets using the provided `mixture_strategy`."""
    if any([proportion is None for proportion in mixture_proportions]):
        # All datasets should be concatenated when no proportion is specified.
        return concatenate_datasets(dataset_list)
    else:
        # All mixture_proportions are not None.
        mixture_proportions = cast(List[float], mixture_proportions)
        # Interleave datasets using the specified proportions and mixture strategy.
        return interleave_datasets(
            dataset_list,
            probabilities=mixture_proportions,
            seed=seed,
            stopping_strategy=(MixtureStrategy(mixture_strategy).get_literal_value()),
        )


def _sample_dataset(
    dataset_params: DatasetParams,
    streaming: bool,
) -> DatasetType:
    """Loads and samples the specified dataset."""
    if dataset_params.sample_count is not None:
        read_instructions = ReadInstruction(
            dataset_params.split, to=dataset_params.sample_count, unit="abs"
        )
        return cast(
            DatasetType,
            load_dataset(
                dataset_params.dataset_name,
                name=dataset_params.dataset_config,
                streaming=streaming,
                split=read_instructions.to_spec(),
            ),
        )
    elif dataset_params.sample_proportion is not None:
        # Get the number of complete datasets required for oversampling.
        oversampling_copies = int(dataset_params.sample_proportion)
        # Get the remaining percentage as a value in [0.0, 100.0].
        dataset_percentage = 100 * (dataset_params.sample_proportion % 1)
        read_instructions = ReadInstruction(
            dataset_params.split, rounding="closest", to=dataset_percentage, unit="%"
        )
        dataset_list = []
        if oversampling_copies > 0:
            dataset = cast(
                DatasetType,
                load_dataset(
                    dataset_params.dataset_name,
                    name=dataset_params.dataset_config,
                    streaming=streaming,
                    split=dataset_params.split,
                ),
            )
            dataset_list = [copy.deepcopy(dataset) for _ in range(oversampling_copies)]
        # Load the remaining dataset piece.
        proportioned_dataset = cast(
            DatasetType,
            load_dataset(
                dataset_params.dataset_name,
                name=dataset_params.dataset_config,
                streaming=streaming,
                split=read_instructions.to_spec(),
            ),
        )
        dataset_list.append(proportioned_dataset)
        return concatenate_datasets(dataset_list)
    else:
        # No sampling.
        return cast(
            DatasetType,
            load_dataset(
                dataset_params.dataset_name,
                name=dataset_params.dataset_config,
                streaming=streaming,
                split=dataset_params.split,
            ),
        )


def _preprocess_dataset(
    dataset: DatasetType,
    dataset_params: DatasetParams,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> DatasetType:
    """Applies preprocessing to a dataset given an optional preprocessing function."""
    if dataset_params.preprocessing_function_name:
        preprocessing_fn = build_prompt_generation_fn(
            dataset_params.preprocessing_function_name, tokenizer
        )
        return dataset.map(
            preprocessing_fn, **dataset_params.preprocessing_function_kwargs
        )
    else:
        return dataset
