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

pytestmark = pytest.mark.parametrize("stream", [True, False])


def _get_default_config(
    datasets: List[DatasetParams],
    stream: bool,
    pack: bool = False,
) -> TrainingConfig:
    return TrainingConfig(
        data=DataParams(
            datasets=datasets,
            text_col="question",
            stream=stream,
            pack=pack,
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
    stream: bool,
    pack: bool = False,
) -> int:
    if stream:
        if pack:
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


def test_data_single_dataset(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
            )
        ],
        stream,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert _get_dataset_size(dataset, stream) == 100


def test_data_multiple_datasets(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
            ),
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
            ),
        ],
        stream,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert _get_dataset_size(dataset, stream) == 100 * 2  # Duplicated dataset


def test_data_multiple_datasets_local_sample(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
                sample_count=5,
            ),
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
                sample_count=201,  # oversample by 1.
            ),
        ],
        stream,
    )
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    assert _get_dataset_size(dataset, stream) == 5 + 201


def test_data_multiple_datasets_local_mixed(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="cais/mmlu",
                dataset_config="abstract_algebra",
                split="test",
                sample_count=5,
                mixture_proportion=0.1,
            ),
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
                sample_count=50,
                mixture_proportion=0.4,
            ),
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
                sample_count=5,
                mixture_proportion=0.5,
            ),
        ],
        stream,
    )
    config.data.mixture_strategy = "first_exhausted"
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    # The dataset size should be small. We stop merging when the smallest dataset is
    # exhausted.
    assert _get_dataset_size(dataset, stream) == 9


def test_data_multiple_datasets_local_mixed_all_exhausted(stream: bool):
    config = _get_default_config(
        [
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
                sample_count=5,
                mixture_proportion=0.1,
            ),
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
                sample_count=50,
                mixture_proportion=0.4,
            ),
            DatasetParams(
                dataset_name="tasksource/mmlu",
                dataset_config="abstract_algebra",
                split="test",
                sample_count=5,
                mixture_proportion=0.5,
            ),
        ],
        stream,
    )
    config.data.mixture_strategy = "all_exhausted"
    tokenizer = build_tokenizer(config.model)
    dataset = build_dataset(config, tokenizer, seed=1)
    # The dataset size should be larger. We stop merging when all datasets have been
    # exhausted.
    assert _get_dataset_size(dataset, stream) == 124


def test_data_multiple_datasets_mixed_exception(stream: bool):
    # Expect an exception when the sum of mixture_proportion > 1.0 .
    with pytest.raises(Exception):
        config = _get_default_config(
            [
                DatasetParams(
                    dataset_name="tasksource/mmlu",
                    dataset_config="abstract_algebra",
                    split="test",
                    sample_count=5,
                    mixture_proportion=0.5,
                ),
                DatasetParams(
                    dataset_name="tasksource/mmlu",
                    dataset_config="abstract_algebra",
                    split="test",
                    sample_count=50,
                    mixture_proportion=0.4,
                ),
                DatasetParams(
                    dataset_name="tasksource/mmlu",
                    dataset_config="abstract_algebra",
                    split="test",
                    sample_count=5,
                    mixture_proportion=0.5,
                ),
            ],
            stream,
        )
        config.data.mixture_strategy = "first_exhausted"


def test_data_multiple_datasets_packing(stream: bool):
    if stream:
        config = _get_default_config(
            [
                DatasetParams(
                    dataset_name="tasksource/mmlu",
                    dataset_config="abstract_algebra",
                    split="test",
                    sample_count=50,
                    mixture_proportion=0.1,
                ),
                DatasetParams(
                    dataset_name="tasksource/mmlu",
                    dataset_config="abstract_algebra",
                    split="test",
                    sample_count=50,
                    mixture_proportion=0.4,
                ),
                DatasetParams(
                    dataset_name="tasksource/mmlu",
                    dataset_config="abstract_algebra",
                    split="test",
                    sample_count=50,
                    mixture_proportion=0.5,
                ),
            ],
            stream,
            pack=True,
        )
        config.data.mixture_strategy = "first_exhausted"
        tokenizer = build_tokenizer(config.model)
        dataset = build_dataset(config, tokenizer, seed=1)
        # The packed dataset should be even smaller.
        assert _get_dataset_size(dataset, stream, pack=True) == 3
    else:
        # Raise an exception as streaming is requried for packing.
        with pytest.raises(Exception):
            _ = _get_default_config(
                [
                    DatasetParams(
                        dataset_name="tasksource/mmlu",
                        dataset_config="abstract_algebra",
                        split="test",
                        sample_count=5,
                        mixture_proportion=0.1,
                    ),
                    DatasetParams(
                        dataset_name="tasksource/mmlu",
                        dataset_config="abstract_algebra",
                        split="test",
                        sample_count=50,
                        mixture_proportion=0.4,
                    ),
                    DatasetParams(
                        dataset_name="tasksource/mmlu",
                        dataset_config="abstract_algebra",
                        split="test",
                        sample_count=5,
                        mixture_proportion=0.5,
                    ),
                ],
                stream,
                pack=True,
            )
