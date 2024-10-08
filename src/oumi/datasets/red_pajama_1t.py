from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("togethercomputer/RedPajama-Data-1T")
class RedPajamaData1TDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the RedPajama-Data-1T corpus."""
