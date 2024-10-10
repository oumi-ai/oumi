from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("togethercomputer/RedPajama-Data-1T")
class RedPajamaDataV1Dataset(BasePretrainingIterableDataset):
    """RedPajama is a clean-room, fully open-source implementation of the LLaMa dataset.

    This dataset contains approximately 1.2 trillion tokens from various sources:
    Commoncrawl (878B), C4 (175B), GitHub (59B), ArXiv (28B), Wikipedia (24B),
    and StackExchange (20B).

    The dataset is primarily in English, though the Wikipedia slice contains
    multiple languages.

    Dataset Structure:
    {
        "text": str,
        "meta": {
            "url": str,
            "timestamp": str,
            "source": str,
            "language": str,
            ...
        },
        "red_pajama_subset": str
    }

    Subsets:
        - common_crawl
        - c4
        - github
        - arxiv
        - wikipedia
        - stackexchange

    For more information on dataset creation and source data, please refer to
    the RedPajama GitHub repository:
    https://github.com/togethercomputer/RedPajama-Data

    Hugging Face dataset page:
    https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T

    References:
        @software{together2023redpajama,
        author = {Together Computer},
        title = {RedPajama: An Open Source Recipe to Reproduce LLaMA training dataset},
        month = April,
        year = 2023,
        url = {https://github.com/togethercomputer/RedPajama-Data}
        }

    Note:
        The 'book' config is defunct and no longer accessible due to reported
        copyright infringement for the Book3 dataset contained in this config.

    License:
        Please refer to the licenses of the data subsets you use. Links to the
        respective licenses can be found in the README.
    """
