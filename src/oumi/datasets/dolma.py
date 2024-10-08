from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("allenai/dolma")
class DolmaDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the Dolma corpus."""
