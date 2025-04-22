import pytest
from transformers import AutoTokenizer

from oumi.core.registry import REGISTRY


def is_known_dataset_issue(dataset_name: str, idx: int) -> bool:
    """Check if the issue at the given index is a known issue."""
    known_issues = {
        "mlabonne/orpo-dpo-mix-40k": [
            15438,  # identical chosen and rejected responses
            16135,  # empty rejected key
            16798,  # identical chosen and rejected responses
            2173,  # identical chosen and rejected responses
            32553,  # identical chosen and rejected responses
            32572,  # identical chosen and rejected responses
            32579,  # identical chosen and rejected responses
            32580,  # identical chosen and rejected responses
            32670,  # identical chosen and rejected responses
            32671,  # identical chosen and rejected responses
            32672,  # identical chosen and rejected responses
            33072,  # identical chosen and rejected responses
            33073,  # identical chosen and rejected responses
            33142,  # identical chosen and rejected responses
            33195,  # identical chosen and rejected responses
            33203,  # identical chosen and rejected responses
            33207,  # identical chosen and rejected responses
            33217,  # identical chosen and rejected responses
            33221,  # identical chosen and rejected responses
            33224,  # identical chosen and rejected responses
            33228,  # identical chosen and rejected responses
            33241,  # identical chosen and rejected responses
            33479,  # identical chosen and rejected responses
            33511,  # identical chosen and rejected responses
            33646,  # identical chosen and rejected responses
            33703,  # identical chosen and rejected responses
            33739,  # identical chosen and rejected responses
            33773,  # identical chosen and rejected responses
            33878,  # identical chosen and rejected responses
            33886,  # identical chosen and rejected responses
            34031,  # identical chosen and rejected responses
            34163,  # identical chosen and rejected responses
            34188,  # identical chosen and rejected responses
            34422,  # identical chosen and rejected responses
            34450,  # identical chosen and rejected responses
            34487,  # identical chosen and rejected responses
            34552,  # identical chosen and rejected responses
            34646,  # identical chosen and rejected responses
            3585,  # identical chosen and rejected responses
            3841,  # identical chosen and rejected responses
            4544,  # identical chosen and rejected responses
            5670,  # identical chosen and rejected responses
            5776,  # empty rejected key
            6210,  # identical chosen and rejected responses
        ],
    }
    return idx in known_issues.get(dataset_name, [])


@pytest.fixture(
    params=[
        "mlabonne/orpo-dpo-mix-40k",
    ]
)
def dataset_fixture(request):
    dataset_name = request.param
    dataset_class = REGISTRY.get_dataset(dataset_name)
    if dataset_class is None:
        pytest.fail(f"Dataset {dataset_name} not found in registry")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return dataset_name, dataset_class(
        dataset_name=dataset_name, split="train", tokenizer=tokenizer
    )


@pytest.mark.skip(
    reason="This test is very time consuming, and should be run manually."
)
def test_dataset_structure(dataset_fixture):
    dataset_name, dataset = dataset_fixture
    assert len(dataset) > 0, f"Dataset {dataset_name} is empty"

    # Iterate through all items in the dataset
    for idx in range(len(dataset)):
        item = dataset[idx]

        # Check that each item has the expected keys
        assert "prompt" in item, f"'prompt' not found in item at index {idx}"
        assert "chosen" in item, f"'chosen' not found in item at index {idx}"
        assert "rejected" in item, f"'rejected' not found in item at index {idx}"

        # Check that the values are strings and not empty
        assert isinstance(item["prompt"], str), (
            f"'prompt' is not a string at index {idx}"
        )
        assert len(item["prompt"]) > 0, f"'prompt' is empty at index {idx}"

        assert isinstance(item["chosen"], str), (
            f"'chosen' is not a string at index {idx}"
        )
        assert len(item["chosen"]) > 0, f"'chosen' is empty at index {idx}"

        assert isinstance(item["rejected"], str), (
            f"'rejected' is not a string at index {idx}"
        )
        if len(item["rejected"]) == 0:
            if not is_known_dataset_issue(dataset_name, idx):
                pytest.fail(f"'rejected' is empty at index {idx}")
        else:
            assert len(item["rejected"]) > 0, f"'rejected' is empty at index {idx}"

        # Check that chosen and rejected are different, accounting for known issues
        if item["chosen"] == item["rejected"]:
            if not is_known_dataset_issue(dataset_name, idx):
                pytest.fail(
                    f"'chosen' and 'rejected' are identical at index {idx}, "
                    f"and this is not a known issue."
                )
        else:
            assert item["chosen"] != item["rejected"], (
                f"'chosen' and 'rejected' are identical at index {idx}"
            )
