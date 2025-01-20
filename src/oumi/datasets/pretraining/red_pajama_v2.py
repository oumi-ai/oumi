from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("togethercomputer/RedPajama-Data-V2")
class RedPajamaDataV2Dataset(BasePretrainingDataset):
    """RedPajama V2 Dataset for training large language models.

    This dataset includes over 100B text documents from 84 CommonCrawl snapshots,
    processed using the CCNet pipeline. It contains 30B documents with quality
    signals and 20B deduplicated documents :footcite:`2023_redpajama`.

    The dataset is available in English, German, French, Italian, and Spanish.

    Key Features:
        - Over 100B text documents
        - 30B documents with quality annotations
        - 20B unique documents after deduplication
        - Estimated 50.6T tokens in total (30.4T after deduplication)
        - Quality signals for filtering and analysis
        - Minhash signatures for fuzzy deduplication

    See Also:
        - Hugging Face dataset page:
          https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
        - Blog post: https://together.ai/blog/redpajama-data-v2
        - GitHub repo: https://github.com/togethercomputer/RedPajama-Data

    Note:
        - License: Common Crawl Foundation Terms of Use:
          https://commoncrawl.org/terms-of-use
        - Code: Apache 2.0 license

    Citations:
        .. footbibliography::
    """

    default_dataset = "togethercomputer/RedPajama-Data-V2"
