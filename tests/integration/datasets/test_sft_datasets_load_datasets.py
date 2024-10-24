import pytest
from transformers import AutoTokenizer

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import REGISTRY, RegistryType


def _get_all_sft_datasets_private_key() -> list[str]:
    """List all SFT datasets in the registry."""
    datasets = []
    for key, value in REGISTRY._registry.items():
        if key.registry_type == RegistryType.DATASET and issubclass(
            value, BaseSftDataset
        ):
            datasets.append(key.name)
    return datasets


@pytest.skip(
    "This test is very time consuming, and should be run manually.",
    allow_module_level=True,
)
@pytest.mark.parametrize("dataset_key", _get_all_sft_datasets_private_key())
def test_sft_datasets(dataset_key: str):
    dataset_cls = REGISTRY._registry[(dataset_key, RegistryType.DATASET)]
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    idx = 0

    # Dataset can successfully be loaded
    dataset = dataset_cls(tokenizer=tokenizer)
    assert dataset.raw(idx) is not None

    # Rows can successfully be pre-processed
    assert dataset.conversation(idx) is not None
    assert dataset.prompt(idx) is not None

    # Rows can successfully be used for training
    assert dataset[idx] is not None
