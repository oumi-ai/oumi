from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("togethercomputer/RedPajama-Data-V2")
class RedPajamaDataV2Dataset(BasePretrainingIterableDataset):
    """RedPajama V2 Dataset for training large language models.

    This dataset includes over 100B text documents from 84 CommonCrawl snapshots,
    processed using the CCNet pipeline. It contains 30B documents with quality
    signals and 20B deduplicated documents.

    The dataset is available in English, German, French, Italian, and Spanish.

    Key Features:
    - Over 100B text documents
    - 30B documents with quality annotations
    - 20B unique documents after deduplication
    - Estimated 50.6T tokens in total (30.4T after deduplication)
    - Quality signals for filtering and analysis
    - Minhash signatures for fuzzy deduplication

    Usage:
        from datasets import load_dataset

        # Load sample dataset
        ds = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample")

        # Load full dataset for specific languages and snapshots
        ds = load_dataset("togethercomputer/RedPajama-Data-V2",
                          name="default",
                          partition="head_middle",
                          snapshots=["2023-06", "2022-49"],
                          languages=["en", "de"])

    For more information, see the dataset card:
    https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2

    References:
        - Blog post: https://together.ai/blog/redpajama-data-v2
        - GitHub repo: https://github.com/togethercomputer/RedPajama-Data

    Citation:
        @software{together2023redpajama,
          author = {Together Computer},
          title = {RedPajama: an Open Dataset for Training Large Language Models},
          month = October,
          year = 2023,
          url = {https://github.com/togethercomputer/RedPajama-Data}
        }

    License:
        Common Crawl Foundation Terms of Use: https://commoncrawl.org/terms-of-use
        Code: Apache 2.0 license
    """
