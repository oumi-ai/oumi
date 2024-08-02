import json
import os

import pytest
import torch
from transformers import AutoTokenizer

from lema.core.datasets.iterable_dataset import BasePretrainingIterableDataset


#
# Fixtures
#
@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


class TestDataset(BasePretrainingIterableDataset):
    def _load_data(self):
        return [
            {"text": "This is a test sentence."},
            {"text": "Another example sentence for testing."},
            {"text": "A third sentence to ensure we have enough data."},
        ]


@pytest.fixture
def test_dataset(tokenizer):
    return TestDataset(
        tokenizer=tokenizer,
        dataset_name_or_path="dummy_path",
        seq_length=10,
        dataset_text_field="text",
    )


@pytest.fixture
def create_sample_data(tmp_path):
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()

    for i in range(5):
        file_path = data_dir / f"data_{i}.json"
        with open(file_path, "w") as f:
            json.dump({"text": f"This is sample text number {i}. " * 10}, f)

    return str(data_dir)


class DiskDataset(BasePretrainingIterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.dataset_name_or_path

    def _load_data(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path) as f:
                    yield json.load(f)


#
# Tests
#
def test_initialization(tokenizer):
    dataset = TestDataset(
        tokenizer=tokenizer,
        dataset_name_or_path="dummy_path",
        seq_length=10,
        dataset_text_field="text",
    )
    assert dataset.tokenizer == tokenizer
    assert dataset.seq_length == 10
    assert dataset._dataset_text_field == "text"


def test_tokenize(test_dataset, tokenizer):
    text = "Test sentence"
    tokens = test_dataset.tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(token, int) for token in tokens)


def test_create_sample(test_dataset, tokenizer):
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sample = test_dataset._create_training_sample(tokens)
    assert isinstance(sample, dict)
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert all(isinstance(tensor, torch.Tensor) for tensor in sample.values())
    assert all(tensor.shape == torch.Size([10]) for tensor in sample.values())


def test_iter(test_dataset, tokenizer):
    samples = list(test_dataset)
    assert len(samples) > 0
    for sample in samples:
        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        assert all(isinstance(tensor, torch.Tensor) for tensor in sample.values())
        assert all(tensor.shape == torch.Size([10]) for tensor in sample.values())


def test_buffer_handling(tokenizer):
    dataset = TestDataset(
        tokenizer=tokenizer,
        dataset_name_or_path="dummy_path",
        seq_length=20,  # Longer sequence length to test buffer handling
        dataset_text_field="text",
    )
    samples = list(dataset)
    total_tokens = sum(
        len(dataset.tokenize(item["text"])) for item in dataset._load_data()
    )
    expected_samples = total_tokens // 20
    assert len(samples) == expected_samples


def test_disk_dataset(tokenizer, create_sample_data):
    dataset = DiskDataset(
        tokenizer=tokenizer,
        dataset_name_or_path=create_sample_data,
        seq_length=50,
        dataset_text_field="text",
    )

    samples = list(dataset)
    assert len(samples) > 0

    for sample in samples:
        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        assert all(isinstance(tensor, torch.Tensor) for tensor in sample.values())
        assert all(tensor.shape == torch.Size([50]) for tensor in sample.values())

    # Check if we've processed all files
    num_files = len([f for f in os.listdir(create_sample_data) if f.endswith(".json")])
    total_tokens = sum(
        len(
            dataset.tokenize(
                json.load(open(os.path.join(create_sample_data, f)))["text"]
            )
        )
        for f in os.listdir(create_sample_data)
        if f.endswith(".json")
    )
    expected_samples = total_tokens // 50
    assert len(samples) == expected_samples
    assert len(samples) >= num_files  # At least one sample per file
