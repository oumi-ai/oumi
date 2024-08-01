from typing import List, Optional, cast

import torch.utils.data.datapipes as dp
import transformers
from torch.utils.data import IterDataPipe, MapDataPipe
from torchdata.datapipes.iter import HuggingFaceHubReader, SampleMultiplexer
from torchdata.datapipes.map.util.converter import MapToIterConverterIterDataPipe

from lema.core.registry import REGISTRY
from lema.core.types import (
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
    TrainingConfig,
)


def build_dataset(
    config: TrainingConfig,
    tokenizer: transformers.PreTrainedTokenizerBase,
    dataset_split: DatasetSplit,
    seed: Optional[int] = None,
    **kwargs,
) -> IterDataPipe:
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

    datapipes: List[IterDataPipe] = []

    for dataset_params in dataset_split_params.datasets:
        # Load the dataset
        datapipe = _load_dataset(dataset_params, dataset_split_params.stream, tokenizer)

        # Apply sampling if needed
        if dataset_params.sample_count is not None:
            datapipe = datapipe.shuffle(buffer_size=dataset_params.shuffle_buffer_size)
            datapipe = datapipe.sharding_filter()
            datapipe = datapipe.header(dataset_params.sample_count)

        # Apply preprocessing
        # if dataset_params.preprocessing_function_name:
        #     preprocessing_fn = build_prompt_generation_fn(
        #         dataset_params.preprocessing_function_name, tokenizer
        #     )
        #     datapipe = datapipe.map(preprocessing_fn)

        datapipes.append(datapipe)

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
                combined_datapipe = dp.iter.Multiplexer(*datapipes)
            else:
                # TODO: implement multiplexer with longest strategy
                # for now just concat
                combined_datapipe = dp.iter.Concater(*datapipes)
        else:
            # All mixture_proportions are not None.
            mixture_proportions = cast(List[float], mixture_proportions)
            mixture = {
                datapipe: mixture_proportion
                for mixture_proportion, datapipe in zip(mixture_proportions, datapipes)
            }
            combined_datapipe = SampleMultiplexer(mixture, seed=seed)  # type: ignore
    else:
        combined_datapipe = datapipes[0]

    # Apply packing if needed
    # if dataset_split_params.pack:
    #     combined_datapipe = combined_datapipe.batch(config.model.model_max_length)
    #     combined_datapipe = combined_datapipe.map(
    #         functools.partial(pack_tokens, tokenizer=tokenizer)
    #     )

    return cast(IterDataPipe, combined_datapipe)


def _load_dataset(
    dataset_params: DatasetParams,
    stream: bool,
    tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
) -> IterDataPipe:
    """Loads a dataset and wraps it in a DataPipe if necessary."""
    # First, try to load a custom dataset from the REGISTRY
    dataset_class = REGISTRY.get_dataset(
        dataset_params.dataset_name, subset=dataset_params.subset
    )

    if dataset_class is not None:
        # Custom dataset handling
        dataset = dataset_class(
            split=dataset_params.split,
            subset=dataset_params.subset,
            tokenizer=tokenizer,
        )

        if isinstance(dataset, MapDataPipe):
            return MapToIterConverterIterDataPipe(dataset)
        else:
            return dataset

    # If not a custom dataset, try loading from Hugging Face
    return HuggingFaceHubReader(
        dataset=dataset_params.dataset_name,
        name=dataset_params.subset,
        split=dataset_params.split,
        streaming=stream,
    )  # type: ignore
