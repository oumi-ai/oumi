import copy
from pathlib import Path
from typing import Callable, List, Optional, Sequence, TypeVar, Union, cast

import datasets
from trl.trainer import ConstantLengthDataset

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
    TrainingConfig,
)
from oumi.core.registry import REGISTRY
from oumi.core.tokenizers import BaseTokenizer
from oumi.datasets.pretraining_async_text_dataset import PretrainingAsyncTextDataset
from oumi.datasets.prompt_response_sft_preprocessor_factory import (
    PromptResponseSftPreprocessorFactory,
)
from oumi.datasets.trl_dpo_preprocessor import trl_dpo_chat_preprocessor_fn
from oumi.datasets.ultrachat_200k import trl_sft_ultrachat_200k_preprocessor_fn
from oumi.utils.hf_datasets_utils import is_cached_to_disk_hf_dataset
from oumi.utils.logging import logger

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

    if function_name == "trl_sft_ultrachat_200k":
        return trl_sft_ultrachat_200k_preprocessor_fn(tokenizer)
    elif function_name == "aya":
        return prompt_response_factory.get_preprocessing_fn(
            prompt_key="inputs",
            response_key="targets",
        )
    elif function_name == "trl_dpo":
        return trl_dpo_chat_preprocessor_fn(tokenizer)

    raise ValueError(f"Unknown prompt generation function: {function_name}")


def build_dataset_mixture(
    config: TrainingConfig,
    tokenizer: Optional[BaseTokenizer],
    dataset_split: DatasetSplit,
    seed: Optional[int] = None,
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

    if dataset_split_params.experimental_use_torch_datapipes:
        from oumi.builders.oumi_data import build_dataset_mixture as build_oumi_dataset

        logger.warning(
            "Using experimental torch datapipes preprocessing pipeline. "
            "This is currently in beta and may not be stable."
        )
        # TODO: OPE-271. Some type hackery going on here.
        # We return a torchdata.IterDataPipe instead of a HuggingFace Dataset or
        # IterableDataset. This is a temporary workaround until torchdata is stable
        # and becomes the default processign pipeline.
        return build_oumi_dataset(config, tokenizer, dataset_split, seed)  # type: ignore

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


def build_dataset_from_params(
    dataset_params: DatasetParams,
    tokenizer: Optional[BaseTokenizer],
    seed: Optional[int] = None,
    stream: bool = False,
    pack: bool = False,
    experimental_use_torch_datapipes: bool = False,
    experimental_use_async_dataset: bool = False,
) -> Union[ConstantLengthDataset, DatasetType, PretrainingAsyncTextDataset]:
    """Builds a dataset from a dataset params object.

    Please refer to `DatasetParams` & `DatasetSplitParams` for a description of
    all the arguments.
    """
    training_config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[dataset_params],
                stream=stream,
                pack=pack,
                experimental_use_async_dataset=experimental_use_async_dataset,
                experimental_use_torch_datapipes=experimental_use_torch_datapipes,
            )
        )
    )

    return build_dataset_mixture(
        config=training_config,
        dataset_split=DatasetSplit.TRAIN,
        tokenizer=tokenizer,
        seed=seed,
    )


def build_dataset(
    dataset_name: str,
    tokenizer: Optional[BaseTokenizer],
    seed: Optional[int] = None,
    stream: bool = False,
    pack: bool = False,
    experimental_use_torch_datapipes: bool = False,
    experimental_use_async_dataset: bool = False,
    **kwargs,
) -> Union[ConstantLengthDataset, DatasetType, PretrainingAsyncTextDataset]:
    """Builds a dataset from a dataset name.

    Please refer to `DatasetParams` & `DatasetSplitParams` for a description of
    the all the arguments.
    """
    dataset_params = DatasetParams(
        dataset_name=dataset_name,
        **kwargs,
    )

    return build_dataset_from_params(
        dataset_params=dataset_params,
        tokenizer=tokenizer,
        seed=seed,
        stream=stream,
        pack=pack,
        experimental_use_torch_datapipes=experimental_use_torch_datapipes,
        experimental_use_async_dataset=experimental_use_async_dataset,
    )


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
    tokenizer: Optional[BaseTokenizer],
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

    if tokenizer is None:
        raise ValueError(
            "Tokenizer is required for preprocessing but was not provided."
        )

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
                trust_remote_code=dataset_params.trust_remote_code,
                **dataset_params.dataset_kwargs,
            )
            return dataset.to_hf()

    dataset_name_or_path: Path = Path(dataset_params.dataset_name)
    if is_cached_to_disk_hf_dataset(dataset_name_or_path):
        return datasets.Dataset.load_from_disk(dataset_name_or_path)
    else:
        return datasets.load_dataset(
            dataset_params.dataset_name,
            name=dataset_params.subset,
            split=dataset_params.split,
            streaming=stream,
            trust_remote_code=dataset_params.trust_remote_code,
            **dataset_params.dataset_kwargs,
        )
