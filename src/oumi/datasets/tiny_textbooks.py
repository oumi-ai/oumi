from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("nampdn-ai/tiny-textbooks")
class TinyTextbooksDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the Tiny Textbooks corpus."""
