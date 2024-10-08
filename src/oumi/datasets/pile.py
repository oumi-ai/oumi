from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("EleutherAI/pile")
class PileDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on The Pile corpus."""
