import copy
import warnings
from typing import Callable, List, Optional, Sequence, TypeVar, Union, cast

import datasets
from trl.trainer import ConstantLengthDataset

from lema.core.registry import REGISTRY
from lema.core.types import (
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
    TrainingConfig,
)
from lema.core.types.base_tokenizer import BaseTokenizer
from lema.datasets.alpaca import alpaca_preprocessing_fn
from lema.datasets.chatqa import chatqa_preprocessor_fn
from lema.datasets.pretraining_async_text_dataset import PretrainingAsyncTextDataset
from lema.datasets.prompt_response_sft_preprocessor_factory import (
    PromptResponseSftPreprocessorFactory,
)
from lema.datasets.trl_dpo_preprocessor import trl_dpo_chat_preprocessor_fn
from lema.datasets.ultrachat_200k import trl_sft_ultrachat_200k_preprocessor_fn

DatasetType = TypeVar("DatasetType", datasets.Dataset, datasets.IterableDataset)


def build_prompt_generation_fn(
    function_name: str, tokenizer: BaseTokenizer
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
    prompt_response_factory = PromptResponseSftPreprocessorFactory(tokenizer)

    if function_name == "alpaca":
        warnings.warn(
            "The 'alpaca' prompt generation function is deprecated and will be removed "
            "in a future release. Please use 'AlpacaDataset' instead.",
            DeprecationWarning,
        )
        return alpaca_preprocessing_fn(tokenizer)
    elif function_name == "trl_sft_ultrachat_200k":
        return trl_sft_ultrachat_200k_preprocessor_fn(tokenizer)
    elif function_name == "aya":
        return prompt_response_factory.get_preprocessing_fn(
            prompt_key="inputs",
            response_key="targets",
        )
    elif function_name == "trl_dpo":
        return trl_dpo_chat_preprocessor_fn(tokenizer)
    elif function_name == "chatqa":
        warnings.warn(
            "The 'chatqa' prompt generation function is deprecated and will be removed "
            "in a future release. Please use 'ChatQADataset' instead.",
            DeprecationWarning,
        )
        return chatqa_preprocessor_fn(tokenizer)

    raise ValueError(f"Unknown prompt generation function: {function_name}")


def build_dataset(
    config: TrainingConfig,
    tokenizer: BaseTokenizer,
    dataset_split: DatasetSplit,
    seed: Optional[int] = None,
    **kwargs,
) -> Union[ConstantLengthDataset, DatasetType, PretrainingAsyncTextDataset]:
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
            _sample_dataset(
                _load_dataset(
                    dataset_params=dataset_params,
                    stream=dataset_split_params.stream,
                    tokenizer=tokenizer,
                ),
                dataset_params=dataset_params,
                stream=dataset_split_params.stream,
            ),
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
        dataset_split_params.seed,
    )
    if dataset_split_params.pack:
        # Fetch max sequence length. If not specified, defaults to 1024.
        dataset_kwargs = {}
        if config.model.model_max_length:
            dataset_kwargs["seq_length"] = config.model.model_max_length

        if dataset_split_params.experimental_use_async_dataset:
            dataset = PretrainingAsyncTextDataset(
                tokenizer,
                dataset,
                dataset_text_field=dataset_split_params.target_col,
                **dataset_kwargs,
            )
        else:
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
        return datasets.concatenate_datasets(dataset_list)
    else:
        # All mixture_proportions are not None.
        mixture_proportions = cast(List[float], mixture_proportions)
        # Interleave datasets using the specified proportions and mixture strategy.
        return datasets.interleave_datasets(
            dataset_list,
            probabilities=mixture_proportions,
            seed=seed,
            stopping_strategy=(MixtureStrategy(mixture_strategy).get_literal_value()),
        )


def _sample_dataset(
    dataset: Union[
        datasets.DatasetDict,
        datasets.Dataset,
        datasets.IterableDatasetDict,
        datasets.IterableDataset,
    ],
    dataset_params: DatasetParams,
    stream: bool,
) -> DatasetType:
    """Samples the specified dataset."""
    if dataset_params.sample_count is None:
        # No sampling.
        dataset = cast(DatasetType, dataset)
        if dataset_params.shuffle:
            dataset = dataset.shuffle(dataset_params.seed)
        return dataset
    if stream:
        dataset = cast(datasets.IterableDataset, dataset)
        if dataset_params.shuffle:
            dataset = dataset.shuffle(dataset_params.seed)
        generator = _build_iterable_dataset_sampler(
            dataset, dataset_params.sample_count
        )
        return cast(
            DatasetType,
            datasets.IterableDataset.from_generator(generator, dataset.features),
        )
    dataset = cast(datasets.Dataset, dataset)
    if dataset.num_rows >= dataset_params.sample_count:
        if dataset_params.shuffle:
            dataset = dataset.shuffle(dataset_params.seed).flatten_indices()
        return cast(DatasetType, dataset.take(dataset_params.sample_count))
    # Oversample the dataset.
    oversampling_copies = int(dataset_params.sample_count // dataset.num_rows)
    dataset_list = [
        cast(datasets.Dataset, copy.deepcopy(dataset))
        for _ in range(oversampling_copies)
    ]
    remaining_rows = dataset_params.sample_count % dataset.num_rows
    if remaining_rows > 0:
        sampled_dataset = cast(datasets.Dataset, dataset)
        if dataset_params.shuffle:
            sampled_dataset = sampled_dataset.shuffle(dataset_params.seed)
        dataset_list.append(sampled_dataset.take(remaining_rows))
    oversampled_dataset = datasets.concatenate_datasets(dataset_list)
    if dataset_params.shuffle:
        oversampled_dataset = oversampled_dataset.shuffle(
            dataset_params.seed
        ).flatten_indices()
    return cast(DatasetType, oversampled_dataset)


def _preprocess_dataset(
    dataset: DatasetType,
    dataset_params: DatasetParams,
    tokenizer: BaseTokenizer,
) -> DatasetType:
    """Applies preprocessing to a dataset given an optional preprocessing function."""
    if (
        dataset_params.preprocessing_function_name is None
        or REGISTRY.get_dataset(
            dataset_params.dataset_name, subset=dataset_params.subset
        )
        is not None
    ):
        # Custom datasets handle pre-processing internally.
        return dataset
    preprocessing_fn = build_prompt_generation_fn(
        dataset_params.preprocessing_function_name, tokenizer
    )
    return dataset.map(preprocessing_fn, **dataset_params.preprocessing_function_kwargs)


def _build_iterable_dataset_sampler(
    dataset: datasets.IterableDataset, n: int
) -> Callable:
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


def _load_dataset(
    dataset_params: DatasetParams,
    stream: bool,
    tokenizer: Optional[BaseTokenizer] = None,
) -> Union[
    datasets.DatasetDict,
    datasets.Dataset,
    datasets.IterableDatasetDict,
    datasets.IterableDataset,
]:
    """Loads a dataset with the specified name and subset."""
    if not stream:
        # Streaming is not supported yet for custom datasets.
        dataset_class = REGISTRY.get_dataset(
            dataset_params.dataset_name, subset=dataset_params.subset
        )

        if dataset_class is not None:
            dataset = dataset_class(
                split=dataset_params.split,
                subset=dataset_params.subset,
                tokenizer=tokenizer,
                **dataset_params.dataset_kwargs,
            )
            return dataset.to_hf()

    return datasets.load_dataset(
        dataset_params.dataset_name,
        name=dataset_params.subset,
        split=dataset_params.split,
        streaming=stream,
        **dataset_params.dataset_kwargs,
    )
