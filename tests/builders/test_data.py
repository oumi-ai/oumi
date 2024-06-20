from typing import List, Union

import pytest
from datasets import (
    Dataset,
    IterableDataset,
)
from trl.trainer import ConstantLengthDataset

from lema.builders import (
    build_dataset,
    build_tokenizer,
)
from lema.core.types import (
    DataParams,
    DatasetParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)

pytestmark = pytest.mark.parametrize("streaming", [True, False])


def _get_default_config(
    datasets: List[DatasetParams],
    streaming: bool,
    packing: bool = False,
) -> TrainingConfig:
    return TrainingConfig(
        data=DataParams(
            datasets=datasets,
            text_col="prompt",
            stream=streaming,
            pack=packing,
        ),
        model=ModelParams(
            model_name="openai-community/gpt2",
            model_max_length=1024,
        ),
        training=TrainingParams(
            trainer_type=TrainerType.TRL_SFT,
            max_steps=3,
        ),
    )


def _get_dataset_size(
    dataset: Union[Dataset, IterableDataset, ConstantLengthDataset],
    streaming: bool,
    packing=False,
) -> int:
    if streaming:
        if packing:
            assert isinstance(dataset, ConstantLengthDataset)
        else:
            assert isinstance(dataset, IterableDataset)
        example_count = 0
        for _ in dataset:
            example_count += 1
        return example_count
    else:
        assert isinstance(dataset, Dataset)
        return dataset.num_rows


def test_data_single_dataset(streaming: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
            )
        ],
        streaming,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert _get_dataset_size(dataset, streaming) == 51760


def test_data_multiple_datasets(streaming: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
            ),
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
            ),
        ],
        streaming,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert _get_dataset_size(dataset, streaming) == 51760 * 2  # Duplicated dataset


def test_data_multiple_datasets_local_sample(streaming: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
                sample_count=5,
            ),
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
                sample_count=51761,  # oversample by 1.
            ),
        ],
        streaming,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert _get_dataset_size(dataset, streaming) == 5 + 51761


def test_data_multiple_datasets_local_mixed(streaming: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
                sample_count=5,
                mixture_proportion=0.1,
            ),
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
                sample_count=50,
                mixture_proportion=0.4,
            ),
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
                sample_count=5,
                mixture_proportion=0.5,
            ),
        ],
        streaming,
    )
    config.data.mixture_strategy = "first_exhausted"
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    # The dataset size should be small. We stop merging when the smallest dataset is
    # exhausted.
    assert _get_dataset_size(dataset, streaming) == 9


def test_data_multiple_datasets_local_mixed_all_exhausted(streaming: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
                sample_count=5,
                mixture_proportion=0.1,
            ),
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
                sample_count=50,
                mixture_proportion=0.4,
            ),
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
                sample_count=5,
                mixture_proportion=0.5,
            ),
        ],
        streaming,
    )
    config.data.mixture_strategy = "all_exhausted"
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    # The dataset size should be larger. We stop merging when all datasets have been
    # exhausted.
    assert _get_dataset_size(dataset, streaming) == 124


def test_data_multiple_datasets_mixed_exception(streaming: bool):
    with pytest.raises(Exception):
        config = _get_default_config(
            [
                DatasetParams(
                    dataset_name="yahma/alpaca-cleaned",
                    preprocessing_function_name="alpaca",
                    split="train",
                    sample_count=5,
                    mixture_proportion=0.5,
                ),
                DatasetParams(
                    dataset_name="yahma/alpaca-cleaned",
                    preprocessing_function_name="alpaca",
                    split="train",
                    sample_count=50,
                    mixture_proportion=0.4,
                ),
                DatasetParams(
                    dataset_name="yahma/alpaca-cleaned",
                    preprocessing_function_name="alpaca",
                    split="train",
                    sample_count=5,
                    mixture_proportion=0.5,
                ),
            ],
            streaming,
        )
        config.data.mixture_strategy = "first_exhausted"


def test_data_multiple_datasets_packing(streaming: bool):
    if not streaming:
        with pytest.raises(Exception):
            _ = _get_default_config(
                [
                    DatasetParams(
                        dataset_name="yahma/alpaca-cleaned",
                        preprocessing_function_name="alpaca",
                        split="train",
                        sample_count=5,
                        mixture_proportion=0.1,
                    ),
                    DatasetParams(
                        dataset_name="yahma/alpaca-cleaned",
                        preprocessing_function_name="alpaca",
                        split="train",
                        sample_count=50,
                        mixture_proportion=0.4,
                    ),
                    DatasetParams(
                        dataset_name="yahma/alpaca-cleaned",
                        preprocessing_function_name="alpaca",
                        split="train",
                        sample_count=5,
                        mixture_proportion=0.5,
                    ),
                ],
                streaming,
                packing=True,
            )
    else:
        config = _get_default_config(
            [
                DatasetParams(
                    dataset_name="yahma/alpaca-cleaned",
                    preprocessing_function_name="alpaca",
                    split="train",
                    sample_count=5,
                    mixture_proportion=0.1,
                ),
                DatasetParams(
                    dataset_name="yahma/alpaca-cleaned",
                    preprocessing_function_name="alpaca",
                    split="train",
                    sample_count=50,
                    mixture_proportion=0.4,
                ),
                DatasetParams(
                    dataset_name="yahma/alpaca-cleaned",
                    preprocessing_function_name="alpaca",
                    split="train",
                    sample_count=5,
                    mixture_proportion=0.5,
                ),
            ],
            streaming,
            packing=True,
        )
        config.data.mixture_strategy = "first_exhausted"
        tokenizer = build_tokenizer(config.model)
        dataset = build_dataset(config, tokenizer, seed=1)
        # The packed dataset should be even smaller.
        assert _get_dataset_size(dataset, streaming, packing=True) == 2
