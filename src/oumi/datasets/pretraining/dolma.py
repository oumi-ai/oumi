from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("allenai/dolma")
class DolmaDataset(BasePretrainingDataset):
    """Dolma: A dataset of 3 trillion tokens from diverse web content.

    Dolma :footcite:`2024_dolma` is a large-scale dataset containing
    approximately 3 trillion tokens sourced from various web content, academic
    publications, code, books, and encyclopedic materials. It is designed for
    language modeling tasks and casual language model training.

    The dataset is available in multiple versions, with v1.7 being the latest
    release used to train OLMo 7B-v1.7. It includes data from sources such as
    Common Crawl, Refined Web, StarCoder, C4, Reddit, Semantic Scholar, arXiv,
    StackExchange, and more.

    Data Fields:
      id (str): Unique identifier for the data entry.
      text (str): The main content of the data entry.
      added (str, optional): Timestamp indicating when the entry was added
        to the dataset.
      created (str, optional): Timestamp indicating when the original content
        was created.
      source (str, optional): Information about the origin or source of the
        data.

    See Also:
      - Paper: https://arxiv.org/abs/2402.00159
      - GitHub project: https://github.com/allenai/dolma
      - Hugging Face Hub: https://huggingface.co/datasets/allenai/dolma

    Note:
        The dataset is released under the ODC-BY license. Users are bound by
        the license agreements and terms of use of the original data sources.

    Citations:
      .. footbibliography::
    """

    default_dataset = "allenai/dolma"
