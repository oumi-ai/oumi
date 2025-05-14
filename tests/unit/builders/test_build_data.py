import tempfile
from pathlib import Path
from typing import Optional, Union

import datasets
import pytest
from typing_extensions import override

from oumi.builders.data import (
    build_dataset,
    build_dataset_mixture,
)
from oumi.builders.models import build_tokenizer
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
)
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.datasets import BaseIterableDataset, BaseMapDataset
from oumi.core.registry import register_dataset
from oumi.core.tokenizers import BaseTokenizer


#
# Toy datasets
#
def create_small_dataset(size=10):
    return [{"text": f"Sample text {i}", "label": i % 2} for i in range(size)]


@register_dataset("small_map_dataset_for_build_data_testing")
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


@register_dataset("small_iterable_dataset_for_build_data_testing")
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


@register_dataset("custom_proxy_dataset_for_build_data_testing")
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

        self._inner_dataset = build_dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            stream=True,
            # DatasetParams kwargs below
            dataset_path=dataset_path,
            subset=subset,
            split=split,
            trust_remote_code=trust_remote_code,
            transform_num_workers=transform_num_workers,
        )

    @override
    def __iter__(self):
        return self._inner_dataset.__iter__()

    @override
    def transform(self, x):
        raise NotImplementedError("Not implemented!")


@pytest.fixture
def gpt2_tokenizer():
    tokenizer = build_tokenizer(
        ModelParams(
            model_name="openai-community/gpt2",
            torch_dtype_str="float16",
            trust_remote_code=False,
            chat_template="default",
            tokenizer_pad_token="<|endoftext|>",
        )
    )
    assert tokenizer.pad_token_id is not None
    assert isinstance(tokenizer.pad_token_id, int)
    return tokenizer


@pytest.fixture
def sample_conversations_jsonl(single_turn_conversation):
    """Creates a temporary JSONL file with sample conversations."""
    conversations = [
        single_turn_conversation,
        single_turn_conversation,
    ]

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        import jsonlines

        with jsonlines.Writer(f) as writer:
            for conv in conversations:
                writer.write(conv.to_dict())

    yield Path(f.name)
    Path(f.name).unlink()  # Cleanup temp file


@pytest.mark.parametrize(
    "stream",
    [
        False,
        True,
    ],
)
def test_build_dataset_conversations(
    sample_conversations_jsonl, gpt2_tokenizer, stream: bool
):
    """Test building dataset from conversations format JSONL."""
    dataset = build_dataset(
        dataset_name="text_sft_jsonl",
        tokenizer=gpt2_tokenizer,
        dataset_path=str(sample_conversations_jsonl),
        stream=stream,
    )
    if stream:
        assert isinstance(dataset, datasets.IterableDataset)
    else:
        assert isinstance(dataset, datasets.Dataset)

    # Convert to list to access items
    items = list(dataset)
    assert len(items) == 2

    # Check first conversation
    assert isinstance(items[0], dict)
    assert isinstance(items[1], dict)


def test_load_dataset_map(gpt2_tokenizer):
    result = build_dataset(
        dataset_name="small_map_dataset_for_build_data_testing",
        tokenizer=gpt2_tokenizer,
        split="train",
    )
    assert isinstance(result, datasets.Dataset), f"Type: {type(result)}"
    assert len(list(result)) == 11


def test_load_dataset_iterable(gpt2_tokenizer):
    result = build_dataset(
        dataset_name="small_iterable_dataset_for_build_data_testing",
        tokenizer=gpt2_tokenizer,
        stream=True,
        split="train",
    )
    assert isinstance(result, datasets.IterableDataset), f"Type: {type(result)}"
    assert len(list(result)) == 9


def test_load_custom_proxy_map_dataset_using_name_override(gpt2_tokenizer):
    result = build_dataset(
        dataset_name="custom_proxy_dataset_for_build_data_testing",
        tokenizer=gpt2_tokenizer,
        stream=True,
        split="train",
        dataset_kwargs={
            "dataset_name_override": "small_map_dataset_for_build_data_testing"
        },
    )
    assert isinstance(result, datasets.IterableDataset), f"Type: {type(result)}"
    assert len(list(result)) == 11


def test_load_custom_proxy_iterable_dataset_using_name_override(gpt2_tokenizer):
    result = build_dataset(
        dataset_name="custom_proxy_dataset_for_build_data_testing",
        tokenizer=gpt2_tokenizer,
        stream=True,
        split="train",
        dataset_kwargs={
            "dataset_name_override": "small_iterable_dataset_for_build_data_testing"
        },
    )
    assert isinstance(result, datasets.IterableDataset), f"Type: {type(result)}"
    assert len(list(result)) == 9


def test_build_dataset_invalid_path():
    """Test building dataset with invalid file path."""
    with pytest.raises(FileNotFoundError):
        build_dataset(
            dataset_name="text_sft_jsonl",
            tokenizer=None,
            dataset_path="nonexistent.jsonl",
        )


@pytest.mark.parametrize(
    "stream",
    [
        False,
        True,
    ],
)
def test_build_dataset_mixture(
    sample_conversations_jsonl, gpt2_tokenizer, stream: bool
):
    """Test building a mixture of datasets with specified proportions."""
    # Create config with dataset mixture
    data_params = DataParams(
        train=DatasetSplitParams(
            datasets=[
                DatasetParams(
                    dataset_name="text_sft_jsonl",
                    dataset_path=str(sample_conversations_jsonl),
                    mixture_proportion=0.7,
                ),
                DatasetParams(
                    dataset_name="text_sft",
                    dataset_path=str(sample_conversations_jsonl),
                    mixture_proportion=0.3,
                ),
            ],
            mixture_strategy="all_exhausted",
            seed=42,
            stream=stream,
        )
    )

    dataset = build_dataset_mixture(
        data_params=data_params,
        tokenizer=gpt2_tokenizer,
        dataset_split=DatasetSplit.TRAIN,
    )
    if stream:
        assert isinstance(dataset, datasets.IterableDataset)
    else:
        assert isinstance(dataset, datasets.Dataset)

    # Convert to list to access items
    items = list(dataset)

    # Check that we have items from both datasets
    assert len(items) == 4
