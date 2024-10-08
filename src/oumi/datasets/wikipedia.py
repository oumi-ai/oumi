from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("wikimedia/wikipedia")
class WikipediaDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on Wikipedia text."""
