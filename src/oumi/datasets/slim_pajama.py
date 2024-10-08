from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("cerebras/SlimPajama-627B")
class SlimPajamaDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the SlimPajama-627B corpus."""
