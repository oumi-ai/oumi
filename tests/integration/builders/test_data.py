from typing import Union

import datasets
import pytest
from trl.trainer import ConstantLengthDataset

from oumi.builders import (
    build_dataset,
    build_dataset_from_params,
    build_dataset_mixture,
    build_tokenizer,
)
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from oumi.core.datasets.base_pretraining_dataset import BasePretrainingDataset
from oumi.core.datasets.pretraining_async_text_dataset import (
    PretrainingAsyncTextDataset,
)

pytestmark = pytest.mark.parametrize("stream", [True, False])


def _get_default_config(
    datasets: list[DatasetParams],
    stream: bool,
    split: DatasetSplit,
    pack: bool = False,
    model_max_length: int = 64,
) -> TrainingConfig:
    dataset_split_params = DatasetSplitParams(
        datasets=datasets,
        stream=stream,
        pack=pack,
        use_async_dataset=pack,
    )
    base_config = TrainingConfig(
        data=DataParams(),
        model=ModelParams(
            model_name="MlpEncoder",
            model_max_length=model_max_length,
            tokenizer_pad_token="<|endoftext|>",
            tokenizer_name="gpt2",
        ),
        training=TrainingParams(
            trainer_type=TrainerType.HF,
            max_steps=3,
        ),
    )
    if split == DatasetSplit.TRAIN:
        base_config.data.train = dataset_split_params
    elif split == DatasetSplit.TEST:
        base_config.data.test = dataset_split_params
    elif split == DatasetSplit.VALIDATION:
        base_config.data.validation = dataset_split_params
    return base_config


def _get_dataset_size(
    dataset: Union[
        datasets.Dataset,
        datasets.IterableDataset,
        ConstantLengthDataset,
        PretrainingAsyncTextDataset,
    ],
    stream: bool,
    pack: bool = False,
) -> int:
    if pack:
        assert isinstance(
            dataset,
            (
                ConstantLengthDataset,
                PretrainingAsyncTextDataset,
                BasePretrainingDataset,
            ),
        ), f"Expected a packable dataset, got {type(dataset)}"

    if pack or stream:
        example_count = 0
        for _ in dataset:
            example_count += 1
        return example_count

    elif isinstance(dataset, datasets.Dataset):
        return dataset.num_rows
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def test_data_single_dataset_in_mixture(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="debug_sft",
                dataset_kwargs={"dataset_size": 5},
            )
        ],
        stream,
        DatasetSplit.TRAIN,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.TRAIN)
    assert _get_dataset_size(dataset, stream) == 5


def test_data_single_dataset_from_kwargs(stream: bool):
    config = _get_default_config(
        [],
        stream,
        DatasetSplit.TRAIN,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(
        dataset_name="debug_sft",
        dataset_kwargs={"dataset_size": 6},
        tokenizer=tokenizer,
        stream=stream,
    )
    assert _get_dataset_size(dataset, stream) == 6


def test_data_single_dataset_from_params(stream: bool):
    config = _get_default_config(
        [],
        stream,
        DatasetSplit.TRAIN,
    )

    dataset_params = DatasetParams(
        dataset_name="debug_sft", dataset_kwargs={"dataset_size": 5}
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_from_params(
        dataset_params=dataset_params,
        tokenizer=tokenizer,
        stream=stream,
    )
    assert _get_dataset_size(dataset, stream) == 5


def test_data_multiple_datasets(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(dataset_name="debug_sft", dataset_kwargs={"dataset_size": 5}),
            DatasetParams(dataset_name="debug_sft", dataset_kwargs={"dataset_size": 7}),
        ],
        stream,
        DatasetSplit.TEST,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.TEST)
    assert _get_dataset_size(dataset, stream) == 5 + 7


def test_data_multiple_datasets_local_sample(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="debug_sft",
                dataset_kwargs={"size": 10},
                sample_count=5,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                dataset_kwargs={"size": 200},
                sample_count=201,  # oversample by 1.
            ),
        ],
        stream,
        DatasetSplit.VALIDATION,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.VALIDATION)
    assert _get_dataset_size(dataset, stream) == 5 + 201


def test_data_multiple_datasets_shuffle_different_seeds(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=5,
                shuffle=True,
                seed=1,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=5,
                shuffle=True,
                seed=2,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=5,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=5,
            ),
        ],
        stream,
        DatasetSplit.VALIDATION,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.VALIDATION)
    assert _get_dataset_size(dataset, stream) == 20
    # Read all the data to handle streaming / nonstreaming in a unified manner.
    data = []
    for val in dataset:
        data.append(val)
    # The third and fourth splits are the same. The first two splits are unique.
    assert data[0] != data[5]
    assert data[0] != data[10]
    assert data[0] != data[15]
    assert data[5] != data[10]
    assert data[5] != data[15]
    assert data[10] == data[15]


def test_data_multiple_datasets_local_mixed(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=5,
                mixture_proportion=0.1,
                trust_remote_code=True,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=50,
                mixture_proportion=0.4,
                trust_remote_code=True,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=5,
                mixture_proportion=0.5,
                trust_remote_code=True,
            ),
        ],
        stream,
        DatasetSplit.TRAIN,
    )
    config.data.get_split(DatasetSplit.TRAIN).mixture_strategy = "first_exhausted"
    config.data.get_split(DatasetSplit.TRAIN).seed = 1
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.TRAIN)
    # The dataset size should be small. We stop merging when the smallest dataset is
    # exhausted.
    assert _get_dataset_size(dataset, stream) == 9


def test_data_multiple_datasets_local_mixed_all_exhausted(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=5,
                mixture_proportion=0.1,
                trust_remote_code=True,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=50,
                mixture_proportion=0.4,
                trust_remote_code=True,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=5,
                mixture_proportion=0.5,
                trust_remote_code=True,
            ),
        ],
        stream,
        DatasetSplit.TRAIN,
    )
    config.data.get_split(DatasetSplit.TRAIN).mixture_strategy = "all_exhausted"
    config.data.get_split(DatasetSplit.TRAIN).seed = 1
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.TRAIN)
    # The dataset size should be larger. We stop merging when all datasets have been
    # exhausted.
    assert _get_dataset_size(dataset, stream) == 124


def test_data_multiple_datasets_mixed_exception(stream: bool):
    # Expect an exception when the sum of mixture_proportion > 1.0 .
    with pytest.raises(Exception):
        config = _get_default_config(
            [
                DatasetParams(
                    dataset_name="debug_sft",
                    sample_count=5,
                    mixture_proportion=0.5,
                ),
                DatasetParams(
                    dataset_name="debug_sft",
                    sample_count=50,
                    mixture_proportion=0.4,
                ),
                DatasetParams(
                    dataset_name="debug_sft",
                    sample_count=5,
                    mixture_proportion=0.5,
                ),
            ],
            stream,
            DatasetSplit.TEST,
        )
        config.data.get_split(DatasetSplit.TEST).mixture_strategy = "first_exhausted"


def test_data_multiple_datasets_different_mix_seeds(stream: bool):
    datasets = []
    for seed in range(1, 3):
        config = _get_default_config(
            [
                DatasetParams(
                    dataset_name="debug_sft",
                    sample_count=5,
                    mixture_proportion=0.1,
                ),
                DatasetParams(
                    dataset_name="debug_sft",
                    sample_count=50,
                    mixture_proportion=0.4,
                ),
                DatasetParams(
                    dataset_name="debug_sft",
                    sample_count=5,
                    mixture_proportion=0.5,
                ),
            ],
            stream,
            DatasetSplit.TRAIN,
        )
        config.data.get_split(DatasetSplit.TRAIN).mixture_strategy = "first_exhausted"
        config.data.get_split(DatasetSplit.TRAIN).seed = seed
        tokenizer = build_tokenizer(config.model)
        datasets.append(build_dataset_mixture(config, tokenizer, DatasetSplit.TRAIN))
    dataset_a = datasets[0]
    dataset_b = datasets[1]
    assert _get_dataset_size(dataset_a, stream) != _get_dataset_size(dataset_b, stream)


def test_data_multiple_datasets_packing(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=50,
                mixture_proportion=0.1,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=50,
                mixture_proportion=0.4,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                sample_count=50,
                mixture_proportion=0.5,
            ),
        ],
        stream,
        DatasetSplit.TEST,
        model_max_length=1024,
        pack=True,
    )
    config.data.get_split(DatasetSplit.TEST).mixture_strategy = "first_exhausted"
    config.data.get_split(DatasetSplit.TEST).seed = 1
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.TEST)
    # The packed dataset should be even smaller.
    assert _get_dataset_size(dataset, stream, pack=True) == 3
