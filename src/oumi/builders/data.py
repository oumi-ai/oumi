# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from collections.abc import Sequence
from typing import Callable, Optional, TypeVar, Union, cast

import datasets

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
)
from oumi.core.datasets.base_pretraining_dataset import BasePretrainingDataset
from oumi.core.datasets.pretraining_async_text_dataset import (
    PretrainingAsyncTextDataset,
)
from oumi.core.registry import REGISTRY
from oumi.core.tokenizers import BaseTokenizer
from oumi.utils.hf_utils import is_cached_to_disk_hf_dataset
from oumi.utils.logging import logger

DatasetType = TypeVar("DatasetType", datasets.Dataset, datasets.IterableDataset)


def build_dataset_mixture(
    data_params: DataParams,
    tokenizer: Optional[BaseTokenizer],
    dataset_split: DatasetSplit,
    seq_length: Optional[int] = None,
    seed: Optional[int] = None,
) -> Union[DatasetType, PretrainingAsyncTextDataset]:
    """Builds a dataset for the specified split.

    Args:
        data_params: The data params.
        tokenizer: The tokenizer object to use for preprocessing.
        dataset_split: The split of the dataset to load.
        seq_length: The length each example will be packed to. This is only used if
            packing is requested, and the dataset isn't already packed. If not provided,
            defaults to 1024.
        seed: If specified, a seed used for random sampling.
        kwargs: Keyword arguments.

    Returns:
        dataset: The built dataset for `dataset_split`.
    """
    dataset_split_params: DatasetSplitParams = data_params.get_split(dataset_split)
    if dataset_split_params.use_torchdata:
        from oumi.builders.oumi_data import build_dataset_mixture as build_oumi_dataset

        logger.warning(
            "Using torchdata preprocessing pipeline. "
            "This is currently in beta and may not be stable."
        )
        # TODO: OPE-271. Some type hackery going on here.
        # We return a torchdata.IterDataPipe instead of a HuggingFace Dataset or
        # IterableDataset. This is a temporary workaround until torchdata is stable
        # and becomes the default processing pipeline.
        return build_oumi_dataset(data_params, tokenizer, dataset_split, seed)  # type: ignore

    # Check if the underlying dataset is already packed, or if we need to pack it
    # ourselves.
    is_packed = _is_mixture_packed(dataset_split_params)

    datasets = [
        _sample_dataset(
            _load_dataset(
                dataset_params=dataset_params,
                stream=dataset_split_params.stream,
                tokenizer=tokenizer,
            ),
            dataset_params=dataset_params,
            stream=dataset_split_params.stream,
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
    if dataset_split_params.pack and not is_packed:
        # Fetch max sequence length. If not specified, defaults to 1024.
        dataset_kwargs = {}
        if seq_length is not None:
            dataset_kwargs["seq_length"] = seq_length

        dataset = PretrainingAsyncTextDataset(
            tokenizer,
            dataset,
            **dataset_kwargs,
        )

    return dataset


def build_dataset(
    dataset_name: str,
    tokenizer: Optional[BaseTokenizer],
    seed: Optional[int] = None,
    stream: bool = False,
    pack: bool = False,
    use_torchdata: Optional[bool] = None,
    **kwargs,
) -> Union[DatasetType, PretrainingAsyncTextDataset]:
    """Builds a dataset from a dataset name.

    Please refer to `DatasetParams` & `DatasetSplitParams` for a description of
    the all the arguments.
    """
    dataset_params = DatasetParams(
        dataset_name=dataset_name,
        **kwargs,
    )
    data_params = DataParams(
        train=DatasetSplitParams(
            datasets=[dataset_params],
            stream=stream,
            pack=pack,
            use_torchdata=use_torchdata,
        )
    )

    return build_dataset_mixture(
        data_params=data_params,
        dataset_split=DatasetSplit.TRAIN,
        tokenizer=tokenizer,
        seed=seed,
    )


def _mix_datasets(
    dataset_list: list[DatasetType],
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
        mixture_proportions = cast(list[float], mixture_proportions)
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
        if dataset_params.shuffle:
            dataset = dataset.shuffle(dataset_params.seed)
        dataset = cast(DatasetType, dataset)
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
    """Loads a dataset with the specified name and subset.

    Note:
        For custom map datasets, streaming is only partially supported:
         - The full dataset is downloaded (or loaded from disk), and loaded in memory.
         - However, transformations are applied lazily in streaming mode. The raw
           dataset is not post-processed (i.e., not "transformed") before
           training starts. Instead, it's returned as `IterableDataset` with lazy
           feature generation i.e., `transform()` is called on-demand during
           training.
    """
    dataset_class = REGISTRY.get_dataset(
        dataset_params.dataset_name, subset=dataset_params.subset
    )

    if dataset_class is not None:
        dataset_kwargs = {**dataset_params.dataset_kwargs}
        if dataset_params.transform_num_workers is not None:
            dataset_kwargs["transform_num_workers"] = (
                dataset_params.transform_num_workers
            )
        # Use the dataset name override from 'dataset_kwargs' if specified (OPE-897).
        dataset_name = (
            dataset_kwargs.pop("dataset_name_override", None)
            or dataset_params.dataset_name
        )

        dataset = dataset_class(
            dataset_name=dataset_name,
            dataset_path=dataset_params.dataset_path,
            split=dataset_params.split,
            subset=dataset_params.subset,
            tokenizer=tokenizer,
            trust_remote_code=dataset_params.trust_remote_code,
            **dataset_kwargs,
        )
        return dataset.to_hf(return_iterable=stream)

    # Load a fully preprocessed (tokenized, etc) dataset from disk.
    # The raw data will be used for training, with any processing
    # other than collation (if enabled).
    dataset_path = dataset_params.dataset_path
    if dataset_path and is_cached_to_disk_hf_dataset(dataset_path):
        return datasets.Dataset.load_from_disk(dataset_path)
    else:
        return datasets.load_dataset(
            dataset_params.dataset_name,
            name=dataset_params.subset,
            split=dataset_params.split,
            streaming=stream,
            trust_remote_code=dataset_params.trust_remote_code,
            **dataset_params.dataset_kwargs,
        )


def _is_mixture_packed(dataset_split_params: DatasetSplitParams) -> bool:
    """Returns whether all datasets in the mixture are packed.

    Raises:
        ValueError: If a mixture of packed and unpacked datasets is detected.
    """
    num_packed = 0
    for dataset in dataset_split_params.datasets:
        dataset_class = REGISTRY.get_dataset(
            dataset.dataset_name, subset=dataset.subset
        )

        if dataset_class is not None and issubclass(
            dataset_class,  # type: ignore
            BasePretrainingDataset,
        ):
            num_packed += 1
    if num_packed == len(dataset_split_params.datasets):
        return True
    elif num_packed == 0:
        return False
    else:
        # Currently, registered datasets get packed and unregistered ones don't. We
        # don't support mixing both at the moment.
        raise ValueError(
            "We currently don't support mixing registered and unregistered datasets."
        )
