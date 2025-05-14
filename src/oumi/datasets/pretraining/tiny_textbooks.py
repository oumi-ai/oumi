# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
