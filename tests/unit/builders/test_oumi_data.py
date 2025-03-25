from typing import Optional, Union

import pytest
import torch
import torch.utils.data.datapipes as dp
from datasets import Dataset as HFDataset
from torch.utils.data import IterDataPipe
from typing_extensions import override

import oumi.builders.oumi_data
from oumi.builders.oumi_data import _load_dataset, build_dataset_mixture
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
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
        self,
        size: int = 11,
        split=None,
        subset=None,
        tokenizer=None,
        dataset_name=None,
        dataset_path=None,
        trust_remote_code: bool = False,
    ):
        self._data = create_small_dataset(size)  # type: ignore

    @override
    def __getitem__(self, index):
        return self.data[index]

    @override
    def transform(self, x):
        return x


@register_dataset("small_iterable_dataset")
class SmallIterableDataset(BaseIterableDataset):
    def __init__(
        self,
        size: int = 9,  # Use a different default size (vs SmallMapDataset)
        split=None,
        subset=None,
        tokenizer=None,
        dataset_name=None,
        dataset_path=None,
        trust_remote_code: bool = False,
    ):
        self._data = create_small_dataset(size)

    @override
    def transform(self, x):
        return x


@register_dataset("custom_proxy_dataset")
class CustomProxyIterableDataset(BaseIterableDataset):
    def __init__(
        self,
        *,
        dataset_name: Optional[str],
        dataset_path: Optional[str] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        trust_remote_code: bool = False,
        transform_num_workers: Optional[Union[str, int]] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        **kwargs,
    ):
        if dataset_name is None:
            raise ValueError("`dataset_name` must be provided")
        elif split is None:
            raise ValueError("`split` must be provided")
        elif len(kwargs) > 0:
            raise ValueError(f"`kwargs` must be empty. Actual: {kwargs}")

        self._inner_dataset = _load_dataset(
            DatasetParams(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                subset=subset,
                split=split,
                trust_remote_code=trust_remote_code,
                transform_num_workers=transform_num_workers,
            ),
            stream=True,
            tokenizer=tokenizer,
        )

    @override
    def __iter__(self):
        return self._inner_dataset.__iter__()

    @override
    def transform(self, x):
        return self._inner_dataset.transform(x)


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


def create_data_params(datasets):
    return DataParams(train=DatasetSplitParams(datasets=datasets))


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
def base_data_params():
    return DataParams(
        train=DatasetSplitParams(
            datasets=[DatasetParams(dataset_name="dummy", split="train")]
        )
    )


#
# Tests
#
def test_load_dataset_map(tokenizer):
    dataset_params = create_dataset_params("small_map_dataset")
    result = _load_dataset(dataset_params, stream=False, tokenizer=tokenizer)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 11


def test_load_dataset_iterable(tokenizer):
    dataset_params = create_dataset_params("small_iterable_dataset")
    result = _load_dataset(dataset_params, stream=True, tokenizer=tokenizer)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 9


def test_load_custom_proxy_map_dataset_using_name_override(tokenizer):
    dataset_params = create_dataset_params("custom_proxy_dataset")
    dataset_params.dataset_kwargs["dataset_name_override"] = "small_map_dataset"
    result = _load_dataset(dataset_params, stream=True, tokenizer=tokenizer)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 11


def test_load_custom_proxy_iterable_dataset_using_name_override(tokenizer):
    dataset_params = create_dataset_params("custom_proxy_dataset")
    dataset_params.dataset_kwargs["dataset_name_override"] = "small_iterable_dataset"
    result = _load_dataset(dataset_params, stream=True, tokenizer=tokenizer)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 9


def test_load_dataset_huggingface(tokenizer, monkeypatch):
    monkeypatch.setattr(
        oumi.builders.oumi_data,
        "HuggingFaceHubReader",
        mock_hf_hub_reader,
    )

    dataset_params = create_dataset_params("huggingface_dataset")
    result = _load_dataset(dataset_params, stream=False, tokenizer=tokenizer)
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 10


def test_build_dataset_mixture_single(tokenizer):
    data_params = create_data_params([create_dataset_params("small_map_dataset")])
    result = build_dataset_mixture(
        data_params,
        tokenizer,
        DatasetSplit.TRAIN,
    )
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 11


def test_build_dataset_mixture_multiple(tokenizer):
    dataset_params1 = create_dataset_params("small_map_dataset")
    dataset_params2 = create_dataset_params("small_iterable_dataset")
    data_params = create_data_params(
        [
            dataset_params1,
            dataset_params2,
        ]
    )
    assert data_params.train.mixture_strategy == MixtureStrategy.FIRST_EXHAUSTED
    result = build_dataset_mixture(
        data_params,
        tokenizer,
        DatasetSplit.TRAIN,
    )
    assert isinstance(result, IterDataPipe)
    # It's 18 (9*2), not 20 (11+9) because of FIRST_EXHAUSTED strategy
    assert len(list(result)) == 18


def test_build_dataset_mixture_sampling(tokenizer):
    dataset_params = create_dataset_params("small_map_dataset")
    dataset_params.sample_count = 5
    dataset_params.shuffle_buffer_size = 10
    data_params = create_data_params([dataset_params])
    result = build_dataset_mixture(
        data_params,
        tokenizer,
        DatasetSplit.TRAIN,
    )
    assert isinstance(result, IterDataPipe)
    assert len(list(result)) == 5


def test_build_dataset_mixture(tokenizer):
    data_params = create_data_params(
        [
            create_dataset_params("small_map_dataset"),
            create_dataset_params("small_iterable_dataset"),
        ]
    )
    data_params.train.datasets[0].mixture_proportion = 0.7
    data_params.train.datasets[1].mixture_proportion = 0.3
    result = build_dataset_mixture(data_params, tokenizer, DatasetSplit.TRAIN, seed=42)
    assert isinstance(result, IterDataPipe)
    samples = list(result)
    assert len(samples) == 20


def test_build_dataset_mixture_with_no_datasets(base_data_params, tokenizer):
    base_data_params.train.datasets = []
    with pytest.raises(ValueError):
        build_dataset_mixture(base_data_params, tokenizer, DatasetSplit.TRAIN)


def test_build_dataset_mixture_with_multiple_datasets_different_sizes(
    base_data_params, tokenizer
):
    base_data_params.train.datasets = [
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
    base_data_params.train.mixture_strategy = "first_exhausted"
    dataset = build_dataset_mixture(base_data_params, tokenizer, DatasetSplit.TRAIN)
    assert len(list(dataset)) == 200

    # All datasets will be exhausted
    base_data_params.train.mixture_strategy = "all_exhausted"
    dataset = build_dataset_mixture(base_data_params, tokenizer, DatasetSplit.TRAIN)
    assert len(list(dataset)) == 300
