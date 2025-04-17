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

from typing import Any, Optional

from oumi.core.feature_generators import (
    FeatureGeneratorOptions,
    VisionLanguageConversationFeatureGenerator,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types import Conversation


class VisionLanguageSftCollator:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        processor_name: str,
        *,
        processor_kwargs: Optional[dict[str, Any]] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
        truncation_side: str = "right",
        label_ignore_index: Optional[int] = None,
        allow_multi_image_inputs: bool = True,
        trust_remote_code: bool = False,
    ):
        """Custom collator for multi-modal vision-language training.

        Args:
            tokenizer: The tokenizer used for encoding the data.
            processor_name: The name of the processor to use for feature generation.
            processor_kwargs: A dictionary of processor-specific parameters.
                These parameters are passed to the processor constructor.
                They can override model-specific parameters.
            max_length: Padding length.
            truncation: Whether to truncate long inputs to `max_length`.
                If False, the long inputs are preserved as is even if they exceed
                `max_length`. Only has effect if `max_length` is specified.
            truncation_side: The side to truncate the tokens ("right" or "left").
            label_ignore_index:  If set, then label values of tokens that shouldn't
                contribute to the loss computation will be replaced by
                this special value.
            allow_multi_image_inputs: Whether to allow multi-image inputs.
            trust_remote_code: Whether to trust remote code execution for the processor.
        """
        self._allow_multi_image_inputs = allow_multi_image_inputs

        if not processor_name:
            raise ValueError("processor_name is required for VisionLanguageSftCollator")

        self._conversation_feature_generator = (
            VisionLanguageConversationFeatureGenerator(
                tokenizer=tokenizer,
                processor_name=processor_name,
                processor_kwargs=processor_kwargs,
                trust_remote_code=trust_remote_code,
                return_tensors="pt",
                truncation=truncation,
                truncation_side=truncation_side,
                max_length=max_length,
                label_ignore_index=label_ignore_index,
            )
        )

    def __call__(self, batch) -> dict[str, Any]:
        """Custom collator for multi-modal vision-language training.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        batch_size = len(batch)
        if batch_size <= 0:
            raise ValueError("Batch is empty")

        conversations: list[Conversation] = []
        for idx in range(batch_size):
            example = batch[idx]
            if "conversation_json" not in example:
                raise ValueError(
                    f"Example doesn't contain 'conversation_json' key. "
                    f"Example: {idx + 1} of {batch_size}. "
                    f"Available keys: {example.keys()}"
                )

            conversation_json = example["conversation_json"]
            conversations.append(Conversation.from_json(conversation_json))
        assert len(conversations) == batch_size

        result = self._conversation_feature_generator.transform_conversations(
            conversations,
            FeatureGeneratorOptions(allow_feature_reshape=False),
        )

        return result
