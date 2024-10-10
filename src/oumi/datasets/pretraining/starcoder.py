from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("bigcode/starcoderdata")
class StarCoderDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the StarCoderData corpus."""
