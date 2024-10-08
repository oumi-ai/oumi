from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("tiiuae/falcon-refinedweb")
class FalconRefinedWebDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the Falcon RefinedWeb corpus."""
