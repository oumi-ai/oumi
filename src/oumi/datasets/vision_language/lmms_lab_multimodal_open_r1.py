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

from oumi.core.registry import register_dataset
from oumi.datasets.vision_language.huggingface import HuggingFaceVisionDataset


@register_dataset("lmms-lab/multimodal-open-r1-8k-verified")
class LmmsLabMultimodalOpenR1Dataset(HuggingFaceVisionDataset):
    """Multimodal Open R1 8K Verified Dataset from LMMS Lab.

    A specialized dataset class for the lmms-lab/multimodal-open-r1-8k-verified dataset
    that contains multimodal reasoning problems with images, problems, and solutions.

    HuggingFace Hub: https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the LMMS Lab Multimodal Open R1 dataset.

        Args:
            **kwargs: Additional arguments passed to the parent class such as:
                - split: Dataset split to use ("train", "test", etc.)
                - system_prompt: Optional system prompt to add to conversations
                - max_length: Maximum sequence length for the dataset
                - Other HuggingFaceVisionDataset arguments
        """
        kwargs.setdefault("image_column", "image")
        kwargs.setdefault("question_column", "problem")
        kwargs.setdefault("answer_column", "solution")

        super().__init__(
            hf_dataset_path="lmms-lab/multimodal-open-r1-8k-verified",
            **kwargs,
        )
