from typing import Any, List


def batch(dataset: List[Any], batch_size: int) -> List[List[Any]]:
    """Batch the provided dataset.

    Args:
        dataset: The dataset to batch, which is a flat list of items.
        batch_size: The desired size of each batch.

    Returns:
        A list of batches. Each batch is a list of `batch_size` items.
    """
    dataset_length = len(dataset)

    batches = []
    for dataset_index in range(0, dataset_length, batch_size):
        batches.append(dataset[dataset_index : dataset_index + batch_size])
    return batches


def unbatch(dataset: List[List[Any]]) -> List[Any]:
    """Unbatch (flatten) the provided dataset."""
    return [item for batch in dataset for item in batch]
