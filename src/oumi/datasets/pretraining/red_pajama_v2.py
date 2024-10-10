from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("togethercomputer/RedPajama-Data-V2")
class RedPajamaDataV2Dataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the RedPajama-Data-V2 corpus."""
