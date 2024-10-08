from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("HuggingFaceFW/fineweb-edu")
class FineWebEduDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the FineWeb-Edu corpus."""
