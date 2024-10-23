import pytest
import torch
import torch.utils.data.datapipes as dp
from datasets import Dataset as HFDataset
from torch.utils.data import IterDataPipe

import oumi.builders.oumi_data
from oumi.builders.oumi_data import _load_dataset, build_dataset_mixture
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    ModelParams,
    TrainingConfig,
)
from oumi.core.datasets import BaseIterableDataset, BaseMapDataset
from oumi.core.registry import register_dataset
from oumi.core.tokenizers import BaseTokenizer


#
# Toy datasets
#
def create_small_dataset(size=10):
    return [{"text": f"Sample text {i}", "label": i % 2} for i in range(size)]


@register_dataset("small_map_dataset")
class SmallMapDataset(BaseMapDataset):
    def __init__(
        self, size: int = 10, split=None, subset=None, tokenizer=None, dataset_path=None
    ):
        self._data = create_small_dataset(size)  # type: ignore

    def __getitem__(self, index):
        return self.data[index]

    def transform(self, x):
        return x


@register_dataset("small_iterable_dataset")
class SmallIterableDataset(BaseIterableDataset):
    def __init__(
        self,
        size: int = 10,
        split=None,
        subset=None,
        tokenizer=None,
        dataset_path=None,
    ):
        self._data = create_small_dataset(size)

    def transform(self, x):
        return x


class SimpleTokenizer(BaseTokenizer):
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([ord(c) for c in text])}


def create_hf_dataset(size=10):
    data = create_small_dataset(size)
    return HFDataset.from_dict(
        {
            "text": [item["text"] for item in data],
            "label": [item["label"] for item in data],
        }
    )


def create_training_config(datasets):
    config = TrainingConfig()
    config.data = DataParams(train=DatasetSplitParams(datasets=datasets))
    return config


# Helper function to create a DatasetParams object
def create_dataset_params(dataset_name, subset=None, split="train"):
    return DatasetParams(dataset_name=dataset_name, subset=subset, split=split)


# Patch HuggingFaceHubReader to use our local HF dataset
def mock_hf_hub_reader(dataset, name, split, streaming):
    hf_dataset = create_hf_dataset()
    return dp.iter.IterableWrapper(hf_dataset)


@pytest.fixture
def tokenizer() -> BaseTokenizer:
    return SimpleTokenizer()


@pytest.fixture
def base_config():
    return TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[DatasetParams(dataset_name="dummy", split="train")]
            )
        ),
        model=ModelParams(model_name="gpt2", tokenizer_name="gpt2"),
    )


#
# Tests
#
def test_load_dataset_map(tokenizer):
    dataset_params = create_dataset_params("small_map_dataset")
    result = _load_dataset(dataset_params, False, tokenizer)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 10


def test_load_dataset_iterable(tokenizer):
    dataset_params = create_dataset_params("small_iterable_dataset")
    result = _load_dataset(dataset_params, True, tokenizer)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 10


def test_load_dataset_huggingface(tokenizer, monkeypatch):
    monkeypatch.setattr(
        oumi.builders.oumi_data,
        "HuggingFaceHubReader",
        mock_hf_hub_reader,
    )

    dataset_params = create_dataset_params("huggingface_dataset")
    result = _load_dataset(dataset_params, False, tokenizer)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 10


def test_build_dataset_mixture_single(tokenizer):
    config = create_training_config([create_dataset_params("small_map_dataset")])
    result = build_dataset_mixture(config, tokenizer, DatasetSplit.TRAIN)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 10


def test_build_dataset_mixture_multiple(tokenizer):
    config = create_training_config(
        [
            create_dataset_params("small_map_dataset"),
            create_dataset_params("small_iterable_dataset"),
        ]
    )
    result = build_dataset_mixture(config, tokenizer, DatasetSplit.TRAIN)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 20  # 10 from each dataset


def test_build_dataset_mixture_sampling(tokenizer):
    dataset_params = create_dataset_params("small_map_dataset")
    dataset_params.sample_count = 5
    dataset_params.shuffle_buffer_size = 10
    config = create_training_config([dataset_params])
    result = build_dataset_mixture(config, tokenizer, DatasetSplit.TRAIN)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 5


def test_build_dataset_mixture(tokenizer):
    config = create_training_config(
        [
            create_dataset_params("small_map_dataset"),
            create_dataset_params("small_iterable_dataset"),
        ]
    )
    config.data.train.datasets[0].mixture_proportion = 0.7
    config.data.train.datasets[1].mixture_proportion = 0.3
    result = build_dataset_mixture(config, tokenizer, DatasetSplit.TRAIN, seed=42)
    assert isinstance(result, IterDataPipe)
    samples = list(result)
    assert len(samples) == 20


def test_build_dataset_mixture_with_no_datasets(base_config, tokenizer):
    base_config.data.train.datasets = []
    with pytest.raises(ValueError):
        build_dataset_mixture(base_config, tokenizer, DatasetSplit.TRAIN)


def test_build_dataset_mixture_with_multiple_datasets_different_sizes(
    base_config, tokenizer
):
    base_config.data.train.datasets = [
        DatasetParams(
            dataset_name="small_map_dataset",
            split="train",
            sample_count=100,
            dataset_kwargs={"size": 500},
        ),
        DatasetParams(
            dataset_name="small_map_dataset",
            split="train",
            sample_count=200,
            dataset_kwargs={"size": 500},
        ),
    ]
    # The first dataset will be exhausted first
    base_config.data.train.mixture_strategy = "first_exhausted"
    dataset = build_dataset_mixture(base_config, tokenizer, DatasetSplit.TRAIN)
    assert len(list(dataset)) == 200

    # All datasets will be exhausted
    base_config.data.train.mixture_strategy = "all_exhausted"
    dataset = build_dataset_mixture(base_config, tokenizer, DatasetSplit.TRAIN)
    assert len(list(dataset)) == 300
