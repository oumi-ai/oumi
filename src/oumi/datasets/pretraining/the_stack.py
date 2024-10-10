from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("bigcode/the-stack")
class TheStackDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on The Stack corpus."""
