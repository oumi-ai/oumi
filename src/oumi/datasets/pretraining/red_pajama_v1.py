from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("togethercomputer/RedPajama-Data-1T")
class RedPajamaDataV1Dataset(BasePretrainingDataset):
    """RedPajama is a clean-room, fully open-source implementation of the LLaMa dataset.

    This dataset contains approximately 1.2 trillion tokens from various sources:
    Commoncrawl (878B), C4 (175B), GitHub (59B), ArXiv (28B), Wikipedia (24B),
    and StackExchange (20B) :footcite:`2023_redpajama`.

    The dataset is primarily in English, though the Wikipedia slice contains
    multiple languages.

    Dataset Structure:
        .. code-block:: python

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

    See Also:
        - For more information on dataset creation and source data, please refer
          to the RedPajama GitHub repository:
          https://github.com/togethercomputer/RedPajama-Data
        - Hugging Face dataset page:
          https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T

    Note:
        The 'book' config is defunct and no longer accessible due to reported
        copyright infringement for the Book3 dataset contained in this config.

    Note:
        Please refer to the licenses of the data subsets you use. Links to the
        respective licenses can be found in the README.

    Citations:
        .. footbibliography::
    """

    default_dataset = "togethercomputer/RedPajama-Data-1T"
