from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("roneneldan/TinyStories")
class TinyStoriesDataset(BasePretrainingIterableDataset):
    """TinyStoriesDataset class for loading and processing the TinyStories dataset.

    This dataset contains synthetically generated short stories with a small
    vocabulary, created by GPT-3.5 and GPT-4. It is designed for text generation
    tasks and is available in English.

    See Also:
        - Paper: https://arxiv.org/abs/2305.07759
        - Huggingface hub: https://huggingface.co/datasets/roneneldan/TinyStories

    Note:
        The dataset is available under the CDLA-Sharing-1.0 license.
    """
