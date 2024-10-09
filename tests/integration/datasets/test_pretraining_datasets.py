from collections import namedtuple

import pytest
from tqdm import tqdm
from transformers import AutoTokenizer

from oumi.core.registry import REGISTRY

TestDatasetConfig = namedtuple("TestDatasetConfig", ["dataset_name", "dataset_subset"])


USE_STREAMING: bool = True
LIMIT_SAMPLES: int = 1_000_000  # set to 0 to iterate through the entire dataset


@pytest.fixture(
    params=[
        ("wikimedia/wikipedia", "20231101.en"),
    ]
)
def dataset_fixture(request):
    dataset_name, dataset_subset = request.param
    dataset_class = REGISTRY.get_dataset(dataset_name)
    if dataset_class is None:
        pytest.fail(f"Dataset {dataset_name} not found in registry")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.eos_token = "</s>"

    return dataset_name, dataset_class(
        dataset_name_or_path=dataset_name,
        subset=dataset_subset,
        split="train",
        tokenizer=tokenizer,
        seq_length=64,
        stream=USE_STREAMING,
    )


def test_dataset_conversation(dataset_fixture):
    dataset_name, dataset = dataset_fixture
    assert dataset is not None, f"Dataset {dataset_name} is empty"
    # Iterate through all items in the dataset
    for i, batch in enumerate(tqdm(dataset)):
        assert batch is not None
        if LIMIT_SAMPLES > 0 and i >= LIMIT_SAMPLES:
            break
