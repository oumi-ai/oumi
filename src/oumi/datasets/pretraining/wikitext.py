from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("Salesforce/wikitext")
class WikiTextDataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the WikiText corpus."""
