from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("nampdn-ai/tiny-textbooks")
class TinyTextbooksDataset(BasePretrainingDataset):
    """A dataset of textbook-like content for training small language models.

    This dataset contains 420,000 textbook documents covering a wide range of topics
    and concepts. It provides a comprehensive and diverse learning resource for
    causal language models, focusing on quality over quantity.

    The dataset was synthesized using the Nous-Hermes-Llama2-13b model, combining
    the best of the falcon-refinedweb and minipile datasets to ensure diversity and
    quality while maintaining a small size.

    See Also:
        - Huggingface hub: https://huggingface.co/datasets/nampdn-ai/tiny-textbooks
        - Textbooks Are All You Need II: phi-1.5 technical report
          (https://arxiv.org/abs/2309.05463)
        - Falcon: A Large Language Model for Search
          (https://arxiv.org/abs/2306.01116)
        - The MiniPile Challenge for Data-Efficient Language Models
      (https://arxiv.org/abs/2304.08442)
    """

    default_dataset = "nampdn-ai/tiny-textbooks"
