from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("Salesforce/wikitext")
class WikiTextDataset(BasePretrainingIterableDataset):
    """WikiText language modeling dataset.

    The WikiText dataset is a collection of over 100 million tokens extracted from
    verified Good and Featured articles on Wikipedia. It is available in two sizes:
    WikiText-2 (2 million tokens) and WikiText-103 (103 million tokens). Each size
    comes in two variants: raw (for character-level work) and processed (for
    word-level work).

    The dataset is well-suited for models that can take advantage of long-term
    dependencies, as it is composed of full articles and retains original case,
    punctuation, and numbers.

    For more information, visit the dataset page on HuggingFace Hub:
    https://huggingface.co/datasets/Salesforce/wikitext

    Data Fields:
        text (str): The text content of the dataset.

    Citation:
        @misc{merity2016pointer,
            title={Pointer Sentinel Mixture Models},
            author={Stephen Merity and Caiming Xiong
                and James Bradbury and Richard Socher},
            year={2016},
            eprint={1609.07843},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }

    License:
        Creative Commons Attribution-ShareAlike License (CC BY-SA 4.0)
    """
