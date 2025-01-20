from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("allenai/c4")
class C4Dataset(BasePretrainingDataset):
    """A dataset for pretraining on the Colossal Clean Crawled Corpus (C4).

    The C4 dataset is based on the Common Crawl dataset and is available in
    multiple variants: 'en', 'en.noclean', 'en.noblocklist', 'realnewslike',
    and 'multilingual' (mC4). It is intended for pretraining language models
    and word representations.

    For more details and download instructions, visit:
    https://huggingface.co/datasets/allenai/c4

    References:
        Paper: https://arxiv.org/abs/1910.10683

    Data Fields:
        - url: URL of the source as a string
        - text: Text content as a string
        - timestamp: Timestamp as a string

    Dataset Variants:
        - en: 305GB
        - en.noclean: 2.3TB
        - en.noblocklist: 380GB
        - realnewslike: 15GB
        - multilingual (mC4): 9.7TB (108 subsets, one per language)

    The dataset is released under the ODC-BY license and is subject to the
    Common Crawl terms of use.
    """

    default_dataset = "allenai/c4"
