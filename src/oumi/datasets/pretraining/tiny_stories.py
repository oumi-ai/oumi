from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("roneneldan/TinyStories")
class TinyStoriesDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the TinyStories corpus."""
