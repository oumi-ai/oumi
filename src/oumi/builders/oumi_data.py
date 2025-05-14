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

from typing import Optional, cast

import torch.utils.data.datapipes as dp
from torch.utils.data import IterDataPipe, MapDataPipe
from torchdata.datapipes.iter import (
    HuggingFaceHubReader,
    MultiplexerLongest,
    SampleMultiplexer,
)
from torchdata.datapipes.map.util.converter import MapToIterConverterIterDataPipe

from oumi.core.configs import (
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
)
from oumi.core.configs.params.data_params import DataParams
from oumi.core.registry import REGISTRY
from oumi.core.tokenizers import BaseTokenizer


def build_dataset_mixture(
    data_params: DataParams,
    tokenizer: Optional[BaseTokenizer],
    dataset_split: DatasetSplit,
    seed: Optional[int] = None,
) -> IterDataPipe:
    """Builds a dataset for the specified split.

    Args:
        data_params: The data params.
        tokenizer: The tokenizer object to use for preprocessing.
        dataset_split: The split of the dataset to load.
        seed: If specified, a seed used for random sampling.

    Returns:
        dataset: The built dataset for `dataset_split`.
    """
    dataset_split_params: DatasetSplitParams = data_params.get_split(dataset_split)

    if len(dataset_split_params.datasets) == 0:
        raise ValueError("No datasets specified in the split.")

    datapipes: list[IterDataPipe] = []

    for dataset_params in dataset_split_params.datasets:
        # Load the dataset
        datapipe = _load_dataset(dataset_params, dataset_split_params.stream, tokenizer)

        # Apply sampling if needed
        if dataset_params.sample_count is not None:
            datapipe = datapipe.shuffle(buffer_size=dataset_params.shuffle_buffer_size)
            datapipe = datapipe.sharding_filter()
            datapipe = datapipe.header(dataset_params.sample_count)

        datapipes.append(datapipe)

    if len(datapipes) != len(dataset_split_params.datasets):
        raise RuntimeError("Failed to load all datasets.")

    # Combine datapipes
    if len(datapipes) > 1:
        mixture_proportions = [
            dataset_params.mixture_proportion
            for dataset_params in dataset_split_params.datasets
        ]

        if any([proportion is None for proportion in mixture_proportions]):
            # All datasets should be concatenated when no proportion is specified.

            if (
                dataset_split_params.mixture_strategy
                == MixtureStrategy.FIRST_EXHAUSTED.value
            ):
                # Yields one element at a time from each input Iterable DataPipes
                # one element from the 1st input DataPipe, then one element
                # from the 2nd DataPipe in the next iteration, etc.
                # It ends when the shortest input DataPipe is exhausted.
                combined_datapipe = dp.iter.Multiplexer(*datapipes)
            elif (
                dataset_split_params.mixture_strategy
                == MixtureStrategy.ALL_EXHAUSTED.value
            ):
                # Yields one element at a time from each input Iterable DataPipes:
                # one element from the 1st input DataPipe, then one element
                # from the 2nd DataPipe in the next iteration, etc.
                # Ends when all input DataPipes are exhausted.
                combined_datapipe = MultiplexerLongest(*datapipes)
            else:
                raise ValueError(
                    "Unsupported mixture strategy: "
                    f"{dataset_split_params.mixture_strategy}"
                )
        else:
            # All mixture_proportions are not None.
            mixture_proportions = cast(list[float], mixture_proportions)
            mixture = {
                datapipe: mixture_proportion
                for mixture_proportion, datapipe in zip(mixture_proportions, datapipes)
            }
            # We need to cast here as SampleMultiplexer expects a torchdata.IterDataPipe
            # and not torch.utils.data.IterDataPipe. This is a temporary workaround
            # until torchdata is updated to use torch.utils.data.IterDataPipe or
            # SampleMultiplexer is moved to torch.utils.data
            combined_datapipe = SampleMultiplexer(mixture, seed=seed)  # type: ignore
    else:
        combined_datapipe = datapipes[0]

    # Apply packing if needed
    # TODO: handle pre-packed datasets, non-iterable datasets
    # Need to add `seq_length as an argument passed in from build_dataset_mixture
    # if dataset_split_params.pack:
    #     combined_datapipe = combined_datapipe.batch(seq_length)
    #     combined_datapipe = combined_datapipe.map(
    #         functools.partial(pack_tokens, tokenizer=tokenizer)
    #     )

    return cast(IterDataPipe, combined_datapipe)


def _load_dataset(
    dataset_params: DatasetParams,
    stream: bool,
    tokenizer: Optional[BaseTokenizer] = None,
) -> IterDataPipe:
    """Loads a dataset and wraps it in a DataPipe if necessary."""
    # First, try to load a custom dataset from the REGISTRY
    dataset_class = REGISTRY.get_dataset(
        dataset_params.dataset_name, subset=dataset_params.subset
    )

    if dataset_class is not None:
        dataset_kwargs = {**dataset_params.dataset_kwargs}
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

        if isinstance(dataset, MapDataPipe):
            # TODO: should we keep map datasets as is?
            return MapToIterConverterIterDataPipe(dataset)
        else:
            return dataset

    # If not a custom dataset, try loading from Hugging Face
    # We need to cast here as HuggingFaceHubReader inherits from torchdata.IterDataPipe
    # and not torch.utils.data.IterDataPipe. This is a temporary workaround until
    # torchdata is updated to use torch.utils.data.IterDataPipe or HuggingFaceHubReader
    # is moved to torch.utils.data
    return cast(
        IterDataPipe,
        HuggingFaceHubReader(
            dataset=dataset_params.dataset_name,
            name=dataset_params.subset,
            split=dataset_params.split,
            streaming=stream,
            **dataset_params.dataset_kwargs,
        ),
    )
