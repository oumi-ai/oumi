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

"""Vision-Language SFT collator for conversation-based multimodal training.

This module provides a collator specifically designed for supervised fine-tuning (SFT)
of vision-language models using conversation data.

Unlike VisionLanguageCollatorWithPadding which expects pre-processed features,
this collator works with raw conversation objects and handles the complete feature
generation pipeline.

Example:
    >>> from oumi.builders import build_tokenizer
    >>> from oumi.core.configs import ModelParams
    >>> tokenizer = build_tokenizer(ModelParams(model_name="llava-hf/llava-1.5-7b-hf"))
    >>> collator = VisionLanguageSftCollator(
    ...     tokenizer=tokenizer,
    ...     processor_name="llava-hf/llava-1.5-7b-hf",
    ...     max_length=512,
    ...     truncation=True
    ... )
    >>> # Expects batch items with conversation_json field
    >>> batch = collator([{"conversation_json": conversation1.to_json()}, ...])
"""

from typing import Any, Optional

from oumi.core.feature_generators import (
    FeatureGeneratorOptions,
    VisionLanguageConversationFeatureGenerator,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types import Conversation
from oumi.utils.torch_utils import pad_to_max_dim_and_stack


class VisionLanguageSftCollator:
    """Collator for vision-language SFT that processes conversation data.

    This collator is designed for supervised fine-tuning of vision-language models
    where training data comes in the form of conversations containing both text and
    images. It handles the complete pipeline from raw conversations to model-ready
    tensor batches.

    Key Features:
        - Processes Conversation objects containing text and image data
        - Uses model-specific processors to extract image features
        - Handles tokenization and feature generation in one step
        - Supports various vision-language architectures
        - Manages padding, truncation, and label masking

    The collator expects batch items with a "conversation_json" field containing
    serialized Conversation objects. These conversations can include:
        - Multiple turns of dialogue
        - Image references (paths, URLs, or base64 data)
        - System prompts and user/assistant messages
    """

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
        train_on_completions_only: bool = False,
        response_template: Optional[str] = None,
        instruction_template: Optional[str] = None,
        process_individually: bool = False,
    ):
        """Initializes the vision-language SFT collator.

        Args:
            tokenizer: The tokenizer for encoding text. Should match the model's
                tokenizer for proper token alignment.

            processor_name: Name or path of the processor to use for feature extraction.
                This should typically match the model name.
                The processor handles image preprocessing and feature extraction.

            processor_kwargs: Optional parameters to pass to the processor constructor.
                These can override default settings or model-specific parameters.

            max_length: Maximum sequence length for padding/truncation. If None,
                sequences are padded to the batch maximum. If specified, sequences
                are padded to this length and may be truncated.

            truncation: Whether to truncate sequences exceeding max_length.
                If False, long sequences are kept intact. Only applies when
                max_length is specified.

            truncation_side: Which side to truncate from ("right" or "left").
                Most models use "right" truncation, but some may require "left"
                for specific architectures or tasks.

            label_ignore_index: Value to use for masking labels in loss computation.

            allow_multi_image_inputs: Whether to support multiple images per
                conversation.
                Set to True for models like MLLaMA that handle multiple images.
                Set to False for models that only support single images per example.

            trust_remote_code: Whether to trust and execute remote code when loading
                the processor. Required for some models (e.g., Qwen2-VL) that use
                custom processing code.

            process_individually: Whether to process each conversation individually
                and then collate features by padding to max dimensions. When True:
                - Each conversation is processed separately through the feature
                    generator
                - Features are padded to the maximum size in the batch
                - Useful for models with variable-sized outputs or heterogeneous data
                - May be less efficient but more flexible than batch processing
                When False (default), conversations are processed as a batch.

            train_on_completions_only: If True, only compute loss on the assistant's
                response tokens. This enables instruction-following training where:

                - **When True**: Only assistant responses contribute to the loss.
                  User instructions, system prompts, and special tokens are masked
                  (ignored) during training. The model learns to generate appropriate
                  responses without being penalized for user input tokens.

                - **When False**: All tokens in the conversation contribute to the loss.
                  This is standard language modeling where the model learns to predict
                  the next token for the entire conversation sequence.

                This parameter is particularly useful for instruction-tuning where you
                want the model to learn response patterns without memorizing prompts.

                **Masking Strategy**: The behavior depends on whether
                instruction_template is provided:

                - **With instruction_template**: Uses multi-turn masking strategy.
                  All user turns are masked, all assistant turns are unmasked.
                  Suitable for multi-turn conversations.

                - **Without instruction_template**: Uses single-turn masking strategy.
                  Only the final assistant response is unmasked, everything else
                  is masked. Suitable for single-turn prompt-response pairs.

            response_template: The template string that marks the beginning of the
                assistant's response. Required if train_on_completions_only is True.

                This should match the exact string that appears in your tokenized
                conversation format to indicate where assistant responses start.

                For example:
                    - Phi-3: "<|assistant|>"
                    - Llama-3: "<|start_header_id|>assistant<|end_header_id|>"
                    - Custom: "Assistant: " or "AI: "

            instruction_template: The template string that marks the beginning of the
                user's instruction.

                For example:
                    - Phi-3: "<|user|>"
                    - Llama-3: "<|start_header_id|>user<|end_header_id|>"
                    - Custom: "User: " or "Human: "
        """
        self._allow_multi_image_inputs = allow_multi_image_inputs
        self._process_individually = process_individually

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
                train_on_completions_only=train_on_completions_only,
                response_template=response_template,
                instruction_template=instruction_template,
            )
        )

    def __call__(self, batch) -> dict[str, Any]:
        """Process a batch of conversation data into model-ready features.

        This method converts serialized conversations into the tensor format expected
        by vision-language models. It handles the complete pipeline:
        1. Deserializes conversation JSON strings
        2. Passes conversations to the feature generator
        3. Returns batched tensors ready for training

        Args:
            batch: List of dictionaries, where each dictionary must contain a
                "conversation_json" field with a serialized Conversation object.

                Expected format:
                [
                    {"conversation_json": '{"messages": [...], "images": [...]}'},
                    {"conversation_json": '{"messages": [...], "images": [...]}'},
                    ...
                ]

                The conversation JSON should include:
                - messages: List of message dictionaries with role and content
                - images: Optional list of image data (paths, URLs, or base64)

        Returns:
            Dictionary containing all features needed for model training:
                - "input_ids": Token IDs including image placeholders
                - "attention_mask": Attention masks for the input
                - "labels": Target labels with appropriate masking
                - "pixel_values" or model-specific image features
                - Additional model-specific features (cross_attention_mask, etc.)

            The exact keys depend on the model architecture and processor used.

        Raises:
            ValueError: If batch is empty or any item lacks "conversation_json" field.

        Example:
            >>> conversation = Conversation(messages=[
            ...     {"role": "user", "content": "What's in this image?"},
            ...     {"role": "assistant", "content": "I see a cat."}
            ... ], images=["path/to/image.jpg"])
            >>> batch_item = {"conversation_json": conversation.to_json()}
            >>> features = collator([batch_item])
            >>> print(features.keys())
            dict_keys(['input_ids', 'attention_mask', 'labels', 'pixel_values'])
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

        if self._process_individually:
            individual_results = []
            for conversation in conversations:
                single_result = (
                    self._conversation_feature_generator.transform_conversations(
                        [conversation],
                        FeatureGeneratorOptions(allow_feature_reshape=True),
                    )
                )
                individual_results.append(single_result)

            # Collate features by padding to max dimensions
            result = self._collate_individual_results(individual_results)
        else:
            result = self._conversation_feature_generator.transform_conversations(
                conversations,
                FeatureGeneratorOptions(allow_feature_reshape=False),
            )

        return result

    def _collate_individual_results(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Collate individually processed results by padding to max dimensions.

        Args:
            results: List of feature dictionaries from individual conversation
                processing

        Returns:
            Collated dictionary with padded tensors

        Raises:
            ValueError: If results have inconsistent keys or non-tensor values
        """
        if not results or len(results) == 0:
            return {}

        # Get keys from first result and verify consistency
        expected_keys = set(results[0].keys())
        for i, result in enumerate(results):
            if set(result.keys()) != expected_keys:
                raise ValueError(
                    f"Inconsistent keys in batch. Expected {expected_keys}, "
                    f"but result {i} has {set(result.keys())}"
                )

        # Collate each feature
        collated = {}
        for key in expected_keys:
            values = [result[key] for result in results]

            # Determine max variable dimensions based on feature type
            # For multi-image models, we may need 2 variable dims (num_images, seq_len)
            max_var_dims = 2 if self._allow_multi_image_inputs else 1

            # Pad and stack tensors
            collated[key] = pad_to_max_dim_and_stack(
                values, max_variable_sized_dims=max_var_dims
            )

        return collated
