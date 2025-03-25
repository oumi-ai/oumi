from typing import Union

import pytest
from datasets import Dataset, IterableDataset
from trl.trainer import ConstantLengthDataset

from oumi.builders import (
    build_dataset,
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
) -> TrainingConfig:
    dataset_split_params = DatasetSplitParams(
        datasets=datasets,
        target_col="question",
        stream=stream,
        pack=pack,
    )
    base_config = TrainingConfig(
        data=DataParams(),
        model=ModelParams(
            model_name="MlpEncoder",
            tokenizer_name="gpt2",
            load_pretrained_weights=False,
            model_max_length=1024,
            tokenizer_pad_token="<|endoftext|>",
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
        Dataset, IterableDataset, ConstantLengthDataset, PretrainingAsyncTextDataset
    ],
    stream: bool,
    pack: bool = False,
) -> int:
    if stream or pack:
        if pack:
            assert isinstance(
                dataset,
                (
                    ConstantLengthDataset,
                    BasePretrainingDataset,
                    PretrainingAsyncTextDataset,
                ),
            )
        else:
            assert isinstance(dataset, (IterableDataset))
        example_count = 0
        for _ in dataset:
            example_count += 1
        return example_count
    else:
        assert isinstance(dataset, Dataset)
        return dataset.num_rows


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
    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TRAIN,
        seq_length=config.model.model_max_length,
    )
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

    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(
        dataset_name="debug_sft",
        tokenizer=tokenizer,
        stream=stream,
        dataset_kwargs={"dataset_size": 5},
    )
    assert _get_dataset_size(dataset, stream) == 5


def test_data_multiple_datasets(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="tasksource/mmlu",
                subset="abstract_algebra",
                split="test",
            ),
            DatasetParams(
                dataset_name="tasksource/mmlu",
                subset="abstract_algebra",
                split="test",
            ),
        ],
        stream,
        DatasetSplit.TEST,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TEST,
        seq_length=config.model.model_max_length,
    )
    assert _get_dataset_size(dataset, stream) == 100 * 2  # Duplicated dataset


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
    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.VALIDATION,
        seq_length=config.model.model_max_length,
    )
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
    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.VALIDATION,
        seq_length=config.model.model_max_length,
    )
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
    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TRAIN,
        seq_length=config.model.model_max_length,
    )
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
    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TRAIN,
        seq_length=config.model.model_max_length,
    )
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
        datasets.append(
            build_dataset_mixture(
                config.data,
                tokenizer,
                DatasetSplit.TRAIN,
                seq_length=config.model.model_max_length,
            )
        )
    dataset_a = datasets[0]
    dataset_b = datasets[1]
    assert _get_dataset_size(dataset_a, stream) != _get_dataset_size(dataset_b, stream)


def test_data_multiple_datasets_packing(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="debug_sft",
                dataset_kwargs={"dataset_size": 50},
                sample_count=50,
                mixture_proportion=0.1,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                dataset_kwargs={"dataset_size": 50},
                sample_count=50,
                mixture_proportion=0.4,
            ),
            DatasetParams(
                dataset_name="debug_sft",
                dataset_kwargs={"dataset_size": 50},
                sample_count=50,
                mixture_proportion=0.5,
            ),
        ],
        stream,
        DatasetSplit.TEST,
        pack=True,
    )
    config.data.get_split(DatasetSplit.TEST).mixture_strategy = "first_exhausted"
    config.data.get_split(DatasetSplit.TEST).seed = 1
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TEST,
        seq_length=config.model.model_max_length,
    )
    # The packed dataset should be even smaller.
    assert _get_dataset_size(dataset, stream, pack=True) == 3


def test_packing_without_streaming_with_sft_dataset(stream: bool):
    """Test that packing works regardless of streaming flag"""
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="debug_sft", dataset_kwargs={"dataset_size": 50}
                    )
                ],
                pack=True,
                stream=stream,
            )
        ),
        model=ModelParams(model_name="gpt2", model_max_length=128),
    )

    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TRAIN,
        seq_length=config.model.model_max_length,
    )

    # Verify it returns a PretrainingAsyncTextDataset
    assert isinstance(dataset, PretrainingAsyncTextDataset)

    # Verify we can iterate through it
    items = []
    for idx, item in enumerate(dataset):
        items.append(item)
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert len(item["input_ids"]) == 128

    assert len(items) == 11  # number of packed samples in the dataset


def test_packing_without_streaming_with_pretraining_dataset(stream: bool):
    """Test that packing works regardless of streaming flag"""

    if not stream:
        pytest.skip("Iterable datasets must be streamed")

    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="debug_pretraining",
                        dataset_kwargs={"dataset_size": 50, "seq_length": 128},
                    )
                ],
                pack=True,
                stream=stream,
            )
        ),
        model=ModelParams(model_name="gpt2", model_max_length=128),
    )

    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TRAIN,
        seq_length=config.model.model_max_length,
    )

    # Verify it returns a IterableDataset
    assert isinstance(dataset, IterableDataset)

    # Verify we can iterate through it
    items = []
    for idx, item in enumerate(dataset):
        items.append(item)
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert len(item["input_ids"]) == 128

    assert len(items) == 2  # number of packed samples in the dataset


@pytest.mark.skip(
    reason="FIXME: this test is inconsistent, and fails depending on cache state"
)
def test_mixed_dataset_packing(stream: bool):
    """Test packing with mixed datasets"""
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="debug_sft",
                        dataset_kwargs={"dataset_size": 50},
                        mixture_proportion=0.6,
                    ),
                    DatasetParams(
                        dataset_name="debug_sft",
                        dataset_kwargs={"dataset_size": 30},
                        mixture_proportion=0.4,
                    ),
                ],
                pack=True,
                stream=stream,
                seed=1,
            )
        ),
        model=ModelParams(model_name="gpt2", model_max_length=128),
    )

    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TRAIN,
        seq_length=config.model.model_max_length,
    )

    # Verify type and basic functionality
    assert isinstance(dataset, PretrainingAsyncTextDataset)

    # Check interleaving is working by sampling first few items
    items = []
    for idx, item in enumerate(dataset):
        items.append(item)
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert len(item["input_ids"]) == 128

    assert len(items) > 0
    assert len(items) == 15
