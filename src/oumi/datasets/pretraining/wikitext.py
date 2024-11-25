from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("Salesforce/wikitext")
class WikiTextDataset(BasePretrainingDataset):
    """WikiText language modeling dataset.

    The WikiText dataset is a collection of over 100 million tokens extracted from
    verified Good and Featured articles on Wikipedia. It is available in two sizes:
    WikiText-2 (2 million tokens) and WikiText-103 (103 million tokens). Each size
    comes in two variants: raw (for character-level work) and processed (for
    word-level work) :footcite:`2016_pointer_sentinel`.

    The dataset is well-suited for models that can take advantage of long-term
    dependencies, as it is composed of full articles and retains original case,
    punctuation, and numbers.

     Data Fields:
        text (str): The text content of the dataset.

    See Also:
        - Hugging Face Hub: https://huggingface.co/datasets/Salesforce/wikitext

    Note:
        The dataset is licensed under the Creative Commons Attribution-ShareAlike
        License (CC BY-SA 4.0).

    Citations:
        .. footbibliography::
    """
