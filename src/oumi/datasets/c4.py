from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("allenai/c4")
class C4Dataset(BasePretrainingIterableDataset):
    """A dataset for pretraining on the Colossal Clean Crawled Corpus (C4).

    C4 is a colossal, cleaned version of Common Crawl's web crawl corpus.
    It's based on the Common Crawl dataset (https://commoncrawl.org) and
    was introduced in the paper: https://arxiv.org/abs/1910.10683
    """
