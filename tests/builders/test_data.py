from typing import List

from datasets import (
    Dataset,
    IterableDataset,
)

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


def _get_default_config(datasets: List[DatasetParams]) -> TrainingConfig:
    return TrainingConfig(
        data=DataParams(
            datasets=datasets,
            text_col="prompt",
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


def _get_dataset_size(dataset: IterableDataset) -> int:
    example_count = 0
    for _ in dataset:
        example_count += 1
    return example_count


def test_data_single_dataset():
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
            )
        ]
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 51760


def test_data_single_dataset_streamed():
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="yahma/alpaca-cleaned",
                preprocessing_function_name="alpaca",
                split="train",
            )
        ]
    )
    config.data.stream = True
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert isinstance(dataset, IterableDataset)
    assert dataset.n_shards == 1


def test_data_multiple_datasets():
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
        ]
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 51760 * 2  # Duplicated dataset


def test_data_multiple_datasets_local_sample():
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
        ]
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 5 + 51761


def test_data_multiple_datasets_streamed_sample():
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
                sample_count=51761,  # oversample by 1
            ),
        ]
    )
    config.data.stream = True
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert isinstance(dataset, IterableDataset)
    assert _get_dataset_size(dataset) == 5 + 51761


def test_data_multiple_datasets_local_mixed():
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
        ]
    )
    config.data.mixture_strategy = "first_exhausted"
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert isinstance(dataset, Dataset)
    # The dataset size should small. We stop merging when the smallest dataset is
    # exhausted.
    assert dataset.num_rows == 9


def test_data_multiple_datasets_streamed_mixed():
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
        ]
    )
    config.data.mixture_strategy = "first_exhausted"
    config.data.stream = True
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert isinstance(dataset, IterableDataset)
    # The dataset size should small. We stop merging when the smallest dataset is
    # exhausted.
    assert _get_dataset_size(dataset) == 9
