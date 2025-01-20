from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("HuggingFaceFW/fineweb-edu")
class FineWebEduDataset(BasePretrainingDataset):
    """FineWeb-Edu: A high-quality educational dataset filtered from web content.

    This dataset contains 1.3 trillion tokens of educational web pages filtered
    from the FineWeb dataset using an educational quality classifier. It aims to
    provide the finest collection of educational content from the web
    :footcite:`2024_fineweb_edu`.

    The dataset is available in multiple configurations:
      - Full dataset (default)
      - Individual CommonCrawl dumps (e.g. CC-MAIN-2024-10)
      - Sample subsets (10BT, 100BT, 350BT tokens)

    Key Features:
      - 1.3 trillion tokens of educational content
      - Filtered using a classifier trained on LLama3-70B-Instruct annotations
      - Outperforms other web datasets on educational benchmarks

    See Also:
      - Huggingface hub page: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

    Note:
      The dataset is released under the Open Data Commons Attribution License
      (ODC-By) v1.0.

    Citations:
      .. footbibliography::
    """

    default_dataset = "HuggingFaceFW/fineweb-edu"
