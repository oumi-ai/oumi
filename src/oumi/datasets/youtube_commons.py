from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("PleIAs/YouTube-Commons")
class YouTubeCommonsDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the YouTube-Commons corpus."""
