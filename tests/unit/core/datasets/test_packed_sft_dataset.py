from unittest.mock import Mock

import pytest
import torch

from oumi.core.datasets.base_sft_dataset import BaseSftDataset
from oumi.core.datasets.packed_sft_dataset import PackedSftDataset
from oumi.core.types.conversation import Conversation, Message, Role


class MockBaseSftDataset(BaseSftDataset):
    """Mock dataset for testing PackedSftDataset."""

    dataset_name = "mock"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = self._load_data()  # type: ignore

    def _load_data(self):
        return [
            {
                "input_ids": [1, 2, 3],
                "labels": [1, 2, 3],
            },
            {
                "input_ids": [4, 5],
                "labels": [4, 5],
            },
            {
                "input_ids": [6, 7, 8, 9],
                "labels": [6, 7, 8, 9],
            },
        ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def transform_conversation(self, example):
        return Conversation(messages=[Message(role=Role.USER, content="test")])


@pytest.fixture
def mock_base_dataset():
    return MockBaseSftDataset(
        dataset_name="mock",
        tokenizer=Mock(),
    )


def test_packed_sft_dataset_initialization(mock_base_dataset):
    """Test basic initialization of PackedSftDataset."""
    with pytest.raises(ValueError, match="pad_token_id"):
        PackedSftDataset(
            base_dataset=mock_base_dataset,
            max_seq_len=5,
            enable_padding=True,
        )

    dataset = PackedSftDataset(
        base_dataset=mock_base_dataset,
        max_seq_len=5,
        enable_padding=True,
        pad_token_id=0,
    )
    assert isinstance(dataset, PackedSftDataset)
    assert dataset._max_seq_len == 5
    assert dataset._pad_token_id == 0


def test_packed_sft_dataset_split_no_padding_no_concat(mock_base_dataset):
    dataset = PackedSftDataset(
        base_dataset=mock_base_dataset,
        max_seq_len=5,
        enable_padding=False,
        concat_token_id=None,
        show_progress=False,
        split_samples=True,
        pad_token_id=None,
    )

    assert len(dataset) == 2  # 9 tokens // max_seq_len = 5 == 2 packs

    first_pack = dataset[0]
    assert isinstance(first_pack["input_ids"], torch.Tensor)
    assert first_pack["input_ids"].tolist() == [1, 2, 3, 4, 5]
    assert first_pack["labels"].tolist() == [1, 2, 3, 4, 5]

    second_pack = dataset[1]
    assert isinstance(second_pack["input_ids"], torch.Tensor)
    assert second_pack["input_ids"].tolist() == [6, 7, 8, 9]
    assert second_pack["labels"].tolist() == [6, 7, 8, 9]


def test_packed_sft_dataset_split_with_padding_no_concat(mock_base_dataset):
    dataset = PackedSftDataset(
        base_dataset=mock_base_dataset,
        max_seq_len=5,
        enable_padding=True,
        concat_token_id=None,
        show_progress=False,
        split_samples=True,
        pad_token_id=142,
    )

    assert len(dataset) == 2  # 9 tokens // max_seq_len = 5 == 2 packs

    first_pack = dataset[0]
    assert isinstance(first_pack["input_ids"], torch.Tensor)
    assert first_pack["input_ids"].tolist() == [1, 2, 3, 4, 5]
    assert first_pack["labels"].tolist() == [1, 2, 3, 4, 5]

    second_pack = dataset[1]
    assert isinstance(second_pack["input_ids"], torch.Tensor)
    assert second_pack["input_ids"].tolist() == [6, 7, 8, 9, 142]
    assert second_pack["labels"].tolist() == [6, 7, 8, 9, -100]


def test_packed_sft_dataset_no_split_with_padding_no_concat(mock_base_dataset):
    dataset = PackedSftDataset(
        base_dataset=mock_base_dataset,
        max_seq_len=4,
        enable_padding=True,
        concat_token_id=None,
        show_progress=False,
        split_samples=False,
        pad_token_id=142,
    )

    assert len(dataset) == 3  # 3 samples, 9 tokens, but can't be split

    first_pack = dataset[0]
    assert isinstance(first_pack["input_ids"], torch.Tensor)
    assert first_pack["input_ids"].tolist() == [1, 2, 3, 142]
    assert first_pack["labels"].tolist() == [1, 2, 3, -100]

    second_pack = dataset[1]
    assert isinstance(second_pack["input_ids"], torch.Tensor)
    assert second_pack["input_ids"].tolist() == [4, 5, 142, 142]
    assert second_pack["labels"].tolist() == [4, 5, -100, -100]

    third_pack = dataset[2]
    assert isinstance(third_pack["input_ids"], torch.Tensor)
    assert third_pack["input_ids"].tolist() == [6, 7, 8, 9]
    assert third_pack["labels"].tolist() == [6, 7, 8, 9]


def test_packed_sft_dataset_no_split_no_padding_no_concat(mock_base_dataset):
    dataset = PackedSftDataset(
        base_dataset=mock_base_dataset,
        max_seq_len=4,
        enable_padding=False,
        concat_token_id=None,
        show_progress=False,
        split_samples=False,
        pad_token_id=142,
    )

    assert len(dataset) == 3  # 3 samples, 9 tokens, but can't be split

    first_pack = dataset[0]
    assert isinstance(first_pack["input_ids"], torch.Tensor)
    assert first_pack["input_ids"].tolist() == [1, 2, 3]
    assert first_pack["labels"].tolist() == [1, 2, 3]

    second_pack = dataset[1]
    assert isinstance(second_pack["input_ids"], torch.Tensor)
    assert second_pack["input_ids"].tolist() == [4, 5]
    assert second_pack["labels"].tolist() == [4, 5]

    third_pack = dataset[2]
    assert isinstance(third_pack["input_ids"], torch.Tensor)
    assert third_pack["input_ids"].tolist() == [6, 7, 8, 9]
    assert third_pack["labels"].tolist() == [6, 7, 8, 9]


def test_packed_sft_dataset_split_no_padding_with_concat(
    mock_base_dataset,
):
    dataset = PackedSftDataset(
        base_dataset=mock_base_dataset,
        max_seq_len=6,
        enable_padding=False,
        concat_token_id=42,
        show_progress=False,
        split_samples=True,
        pad_token_id=None,
    )

    assert len(dataset) == 2

    first_pack = dataset[0]
    assert isinstance(first_pack["input_ids"], torch.Tensor)
    assert first_pack["input_ids"].tolist() == [1, 2, 3, 42, 4, 5]
    assert first_pack["labels"].tolist() == [1, 2, 3, -100, 4, 5]

    second_pack = dataset[1]
    assert isinstance(second_pack["input_ids"], torch.Tensor)
    assert second_pack["input_ids"].tolist() == [6, 7, 8, 9]
    assert second_pack["labels"].tolist() == [6, 7, 8, 9]


def test_packed_sft_dataset_split_with_with_padding_with_concat(
    mock_base_dataset,
):
    dataset = PackedSftDataset(
        base_dataset=mock_base_dataset,
        max_seq_len=6,
        enable_padding=True,
        concat_token_id=42,
        show_progress=False,
        split_samples=True,
        pad_token_id=142,
    )

    assert len(dataset) == 2

    first_pack = dataset[0]
    assert isinstance(first_pack["input_ids"], torch.Tensor)
    assert first_pack["input_ids"].tolist() == [1, 2, 3, 42, 4, 5]
    assert first_pack["labels"].tolist() == [1, 2, 3, -100, 4, 5]

    second_pack = dataset[1]
    assert isinstance(second_pack["input_ids"], torch.Tensor)
    assert second_pack["input_ids"].tolist() == [6, 7, 8, 9, 142, 142]
    assert second_pack["labels"].tolist() == [6, 7, 8, 9, -100, -100]


@pytest.mark.parametrize("split_samples", [True, False])
def test_packed_dataset_with_long_sample(mock_base_dataset, split_samples):
    """Test handling of samples longer than max_seq_len."""
    long_sample = {
        "input_ids": [10] * 10,
        "labels": [10] * 10,
    }
    mock_base_dataset._data.append(long_sample)

    dataset = PackedSftDataset(
        base_dataset=mock_base_dataset,
        max_seq_len=5,
        split_samples=split_samples,
        enable_padding=False,
        show_progress=False,
    )

    if split_samples:
        # Should split long samples into multiple packs
        assert len(dataset) == 4  # 19 tokens // 5 = 4
    else:
        # Should skip long samples with warning
        assert (
            len(dataset) == 2
        )  # 0+1 packed, 2 added, 3 ignored ignored because too long == 2 packs


def test_packed_dataset_oob():
    """Test handling of out of bounds index."""
    base_dataset = MockBaseSftDataset(
        dataset_name="mock",
        tokenizer=Mock(),
    )
    base_dataset._data = [{"input_ids": [], "labels": []}]  # type: ignore

    dataset = PackedSftDataset(
        base_dataset=base_dataset,
        max_seq_len=5,
        enable_padding=False,
        show_progress=False,
    )

    assert len(dataset) == 0
    with pytest.raises(IndexError):
        _ = dataset[0]


def test_packed_dataset_empty_base_dataset():
    """Test handling of empty base dataset."""
    base_dataset = MockBaseSftDataset(
        dataset_name="mock",
        tokenizer=Mock(),
    )
    base_dataset._data = []  # type: ignore

    with pytest.raises(ValueError, match="Cannot pack empty dataset."):
        PackedSftDataset(
            base_dataset=base_dataset,
            max_seq_len=5,
            enable_padding=False,
            show_progress=False,
        )


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"input_ids": [1, 2, 3]},  # Missing labels
        {"labels": [1, 2, 3]},  # Missing input_ids
    ],
)
def test_packed_dataset_validation(invalid_data):
    """Test validation of required keys in base dataset."""

    class InvalidMockDataset(MockBaseSftDataset):
        def _load_data(self):
            return [invalid_data]

    with pytest.raises(ValueError, match="must contain"):
        PackedSftDataset(
            base_dataset=InvalidMockDataset(
                dataset_name="mock",
                tokenizer=Mock(),
                task="sft",
                return_tensors=True,
            ),
            max_seq_len=5,
            enable_padding=False,
            show_progress=False,
        )
