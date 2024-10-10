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
        ("allenai/c4", "en"),
        ("allenai/dolma", "v1_7"),
        ("tiiuae/falcon-refinedweb", None),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        ("EleutherAI/pile", None),
        ("togethercomputer/RedPajama-Data-1T", "common_crawl"),
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
    assert dataset is not None

    # Iterate through all the items in the dataset
    for batch_idx, batch in enumerate(tqdm(dataset)):
        assert isinstance(
            batch, dict
        ), f"Invalid batch format for dataset {dataset_name} at batch {batch_idx}"
        assert (
            "input_ids" in batch
        ), f"Missing 'input_ids' in batch {batch_idx} for dataset {dataset_name}"
        assert (
            "attention_mask" in batch
        ), f"Missing 'attention_mask' in batch {batch_idx} for dataset {dataset_name}"

        if LIMIT_SAMPLES > 0 and batch_idx >= LIMIT_SAMPLES:
            break
