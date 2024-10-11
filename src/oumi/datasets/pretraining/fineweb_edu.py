from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("HuggingFaceFW/fineweb-edu")
class FineWebEduDataset(BasePretrainingIterableDataset):
    """FineWeb-Edu: A high-quality educational dataset filtered from web content.

    This dataset contains 1.3 trillion tokens of educational web pages filtered
    from the FineWeb dataset using an educational quality classifier. It aims to
    provide the finest collection of educational content from the web.

    The dataset is available in multiple configurations:
    - Full dataset (default)
    - Individual CommonCrawl dumps (e.g. CC-MAIN-2024-10)
    - Sample subsets (10BT, 100BT, 350BT tokens)

    Huggingface hub page:
    https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

    Key Features:
    - 1.3 trillion tokens of educational content
    - Filtered using a classifier trained on LLama3-70B-Instruct annotations
    - Outperforms other web datasets on educational benchmarks

    License:
    Open Data Commons Attribution License (ODC-By) v1.0

    Citation:
    @software{lozhkov2024fineweb-edu,
      author = {Lozhkov, Anton and Ben Allal, Loubna and von Werra,
        Leandro and Wolf, Thomas},
      title = {FineWeb-Edu},
      month = May,
      year = 2024,
      doi = { 10.57967/hf/2497 },
      url = {https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu}
    }
    """
