from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("cerebras/SlimPajama-627B")
class SlimPajamaDataset(BasePretrainingDataset):
    """SlimPajama-627B: A cleaned and deduplicated version of RedPajama.

    SlimPajama is the largest extensively deduplicated, multi-corpora, open-source
    dataset for training large language models. It was created by cleaning and
    deduplicating the 1.2T token RedPajama dataset, resulting in a 627B token dataset.

    The dataset consists of 59166 jsonl files and is ~895GB compressed. It includes
    training, validation, and test splits :footcite:`2023_slimpajama`.

    Key Features:
        - 627B tokens
        - Open-source
        - Curated data sources
        - Extensive deduplication
        - Primarily English language

    Data Sources and Proportions:
        - Commoncrawl: 52.2%
        - C4: 26.7%
        - GitHub: 5.2%
        - Books: 4.2%
        - ArXiv: 4.6%
        - Wikipedia: 3.8%
        - StackExchange: 3.3%

    See Also:
        - Blog post: https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama
        - Repository: https://github.com/Cerebras/modelzoo/tree/main/modelzoo/transformers/data_processing/slimpajama
        - Hugging Face: https://huggingface.co/datasets/cerebras/SlimPajama-627B

    Dataset Structure:
        Each example is a JSON object with the following structure:

        .. code-block:: python

            {
                "text": str,
                "meta": {
                    "redpajama_set_name": str  # One of the data source names
                }
            }

    Citations:
        .. footbibliography::
    """

    default_dataset = "cerebras/SlimPajama-627B"
