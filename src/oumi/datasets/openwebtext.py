from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("Skylion007/openwebtext")
class OpenWebTextDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the OpenWebText corpus."""
