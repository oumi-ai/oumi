from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union, cast

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

from lema.core.types import (
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
    TrainingConfig,
)
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
    dataset_split: DatasetSplit,
    seed: Optional[int] = None,
    **kwargs,
) -> Union[ConstantLengthDataset, DatasetType]:
    """Builds a dataset for the specified split.

    Args:
        config: The training config.
        tokenizer: The tokenizer object to use for preprocessing.
        dataset_split: The split of the dataset to load.
        seed: If specified, a seed used for random sampling.
        kwargs: Keyword arguments.

    Returns:
        dataset: The built dataset for `dataset_split`.
    """
    dataset_split_params: DatasetSplitParams = config.data.get_split(dataset_split)

    datasets = [
        _preprocess_dataset(
            _sample_dataset(dataset_params, dataset_split_params.stream),
            dataset_params,
            tokenizer,
        )
        for dataset_params in dataset_split_params.datasets
    ]
    mixture_proportions = [
        dataset.mixture_proportion for dataset in dataset_split_params.datasets
    ]

    # Interleave datasets using mixture_strategy.
    dataset = _mix_datasets(
        datasets,
        mixture_proportions,
        dataset_split_params.mixture_strategy,
        seed,
    )
    if dataset_split_params.pack:
        # Fetch max sequence length. If not specified, defaults to 1024.
        dataset_kwargs = {}
        if config.model.model_max_length:
            dataset_kwargs["seq_length"] = config.model.model_max_length
        dataset = ConstantLengthDataset(
            tokenizer,
            dataset,
            dataset_text_field=dataset_split_params.target_col,
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
    stream: bool,
) -> DatasetType:
    """Loads and samples the specified dataset."""
    if dataset_params.sample_count is None:
        # No sampling.
        return cast(
            DatasetType,
            load_dataset(
                dataset_params.dataset_name,
                name=dataset_params.subset,
                streaming=stream,
                split=dataset_params.split,
            ),
        )
    if stream:
        dataset = cast(
            IterableDataset,
            load_dataset(
                dataset_params.dataset_name,
                name=dataset_params.subset,
                streaming=stream,
                split=dataset_params.split,
            ),
        )
        generator = _build_iterable_dataset_sampler(
            dataset, dataset_params.sample_count
        )
        return cast(
            DatasetType, IterableDataset.from_generator(generator, dataset.features)
        )
    # Cast the ReadInstruction to Any as Huggingface type annotations are not
    # up to date with their documenation. ReadInstruction can be passed as a
    # split when loading datasets.
    read_instructions: Any = ReadInstruction(
        dataset_params.split, to=dataset_params.sample_count, unit="abs"
    )
    dataset = cast(
        Dataset,
        load_dataset(
            dataset_params.dataset_name,
            name=dataset_params.subset,
            streaming=stream,
            split=read_instructions,
        ),
    )
    if dataset.num_rows < dataset_params.sample_count:
        oversampling_copies = int(dataset_params.sample_count / dataset.num_rows)
        dataset_list = [
            cast(
                DatasetType,
                load_dataset(
                    dataset_params.dataset_name,
                    name=dataset_params.subset,
                    streaming=stream,
                    split=dataset_params.split,
                ),
            )
            for _ in range(oversampling_copies)
        ]
        remaining_rows = dataset_params.sample_count % dataset.num_rows
        if remaining_rows > 0:
            split_read_instructions: Any = ReadInstruction(
                dataset_params.split, to=remaining_rows, unit="abs"
            )
            sampled_dataset: DatasetType = cast(
                DatasetType,
                load_dataset(
                    dataset_params.dataset_name,
                    name=dataset_params.subset,
                    streaming=stream,
                    split=split_read_instructions,
                ),
            )
            dataset_list.append(sampled_dataset)
        return concatenate_datasets(dataset_list)
    else:
        return cast(DatasetType, dataset)


def _preprocess_dataset(
    dataset: DatasetType,
    dataset_params: DatasetParams,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> DatasetType:
    """Applies preprocessing to a dataset given an optional preprocessing function."""
    if dataset_params.preprocessing_function_name is None:
        return dataset
    preprocessing_fn = build_prompt_generation_fn(
        dataset_params.preprocessing_function_name, tokenizer
    )
    return dataset.map(preprocessing_fn, **dataset_params.preprocessing_function_kwargs)


def _build_iterable_dataset_sampler(dataset: IterableDataset, n: int) -> Callable:
    """Returns a generator that supports oversampling an IterableDataset."""

    def _generator():
        generation_count = 0
        while generation_count < n:
            for generation in dataset:
                generation_count += 1
                yield generation
                if generation_count >= n:
                    break

    return _generator
