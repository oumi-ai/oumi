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

import copy
from typing import Any, NamedTuple, Optional

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

from oumi.core.configs.internal.internal_model_config import (
    InternalFeatureFirstDimAction,
    InternalModelConfig,
)
from oumi.core.configs.internal.supported_models import (
    find_internal_model_config_using_model_name,
    get_default_vlm_model_config,
)
from oumi.core.feature_generators.base_feature_generator import (
    BaseConversationFeatureGenerator,
    FeatureGeneratorOptions,
)
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.tokenizers.utils import (
    mask_labels_for_completions_only,
    mask_labels_without_user_template,
)
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
)
from oumi.utils.conversation_utils import (
    load_pil_image_from_content_item,
    truncate_text_in_content_items,
)
from oumi.utils.logging import logger
from oumi.utils.str_utils import truncate_text_pieces_to_max_tokens_limit
from oumi.utils.torch_utils import get_first_dim_len


class _SpecialTokens(NamedTuple):
    """Special tokens used by `VisionLanguageFeatureGenerator`."""

    image_token: Optional[str]
    image_token_id: Optional[int]
    label_ignore_index: Optional[int]

    pad_token_id: int
    """Token id of `PAD` token."""


class VisionLanguageConversationFeatureGenerator(BaseConversationFeatureGenerator):
    """Applies `processor` to generate model inputs from an input `Conversation`."""

    def __init__(
        self,
        *,
        tokenizer: Optional[BaseTokenizer] = None,
        processor: Optional[BaseProcessor] = None,
        processor_name: Optional[str] = None,
        processor_kwargs: Optional[dict[str, Any]] = None,
        trust_remote_code: bool = False,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
        truncation_side: str = "right",
        label_ignore_index: Optional[int] = None,
        train_on_completions_only: bool = False,
        response_template: Optional[str] = None,
        instruction_template: Optional[str] = None,
    ) -> None:
        """Initializes a new instance of VisionLanguageFeatureProcessor."""
        # Importing these here to avoid circular dependencies
        from oumi.builders.processors import build_processor

        if truncation_side not in ("left", "right"):
            raise ValueError(
                f"Invalid truncation_side: '{truncation_side}'. "
                "Expected 'left' or 'right'."
            )

        self._max_length: Optional[int] = max_length
        self._truncation: bool = truncation
        self._truncation_side = truncation_side
        self._return_tensors = return_tensors

        # Completion-only training configuration
        self._train_on_completions_only = train_on_completions_only
        self._response_template = response_template
        self._instruction_template = instruction_template

        # Validate completion-only training configuration
        if self._train_on_completions_only:
            if self._response_template is None:
                raise ValueError(
                    "response_template must be provided when "
                    "train_on_completions_only=True"
                )

        if tokenizer is None:
            raise ValueError(
                f"Tokenizer must be provided for {self.__class__.__name__}"
            )
        elif not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer doesn't define `pad_token_id`.")
        elif not isinstance(tokenizer.pad_token_id, int):
            raise RuntimeError(
                "Tokenizer's `pad_token_id` is not an integer. "
                f"Type: {type(tokenizer.pad_token_id)}"
            )

        if processor is not None:
            if processor_name:
                logger.warning(
                    "Both processor and processor_name are provided. "
                    f"Ignoring processor_name: {processor_name}"
                )
            if processor_kwargs is not None and len(processor_kwargs) > 0:
                logger.warning(
                    "Both processor and processor_kwargs are provided. "
                    f"Ignoring processor_kwargs: {processor_kwargs}"
                )
        elif processor_name:
            # TODO OPE-1185 Add plumbing for processor_kwargs
            processor = build_processor(
                processor_name,
                tokenizer,
                trust_remote_code=trust_remote_code,
                processor_kwargs=processor_kwargs,
            )
        else:
            raise ValueError(
                "At least one of processor or processor_name must provided."
            )

        assert processor is not None
        if not callable(processor):
            raise ValueError("Processor is not callable!")

        self._processor: BaseProcessor = processor
        self._image_processor = self._processor.image_processor

        self._internal_model_config: InternalModelConfig = (
            find_internal_model_config_using_model_name(
                self._processor.processor_name, trust_remote_code=trust_remote_code
            )
            or get_default_vlm_model_config()
        )

        self._special_tokens: _SpecialTokens = _SpecialTokens(
            image_token=self._processor.image_token,
            image_token_id=self._processor.image_token_id,
            label_ignore_index=(
                label_ignore_index
                if label_ignore_index is not None
                else self._processor.label_ignore_index
            ),
            pad_token_id=int(tokenizer.pad_token_id),
        )

        # Tokenize templates for completion-only training
        if self._train_on_completions_only:
            assert self._response_template is not None  # Already validated above
            self._response_token_ids = self._processor.tokenizer.encode(
                self._response_template, add_special_tokens=False
            )
            if self._instruction_template is not None:
                self._instruction_token_ids = self._processor.tokenizer.encode(
                    self._instruction_template, add_special_tokens=False
                )
            else:
                self._instruction_token_ids = None

            # Log the completion-only masking strategy being used
            if self._instruction_token_ids is not None:
                logger.info(
                    "Completion-only training configured with multi-turn strategy. "
                    f"Using response template: '{self._response_template}' and "
                    f"instruction template: '{self._instruction_template}'. "
                    "All assistant responses will be unmasked for training."
                )
            else:
                logger.info(
                    "Completion-only training configured with single-turn strategy. "
                    f"Using response template: '{self._response_template}' only. "
                    "Only the last assistant response will be unmasked for training."
                )

    def _prepare_simple_model(
        self, conversation: Conversation
    ) -> tuple[Image.Image, str]:
        """Prepares the images and prompt for a simple model.

        Simple models only use the last image and text turn in the conversation. They
        don't use the chat template, so the prompt is just the last text turn.
        """
        image_turns = [turn for turn in conversation.messages if turn.contains_images()]
        text_turns = [turn for turn in conversation.messages if turn.contains_text()]

        if len(image_turns) == 0:
            raise ValueError("Conversation must contain at least one image turn")
        if len(text_turns) == 0:
            raise ValueError("Conversation must contain at least one text turn")

        last_image_item: ContentItem = image_turns[-1].image_content_items[-1]
        last_text_item: ContentItem = text_turns[-1].text_content_items[-1]

        prompt = last_text_item.content or ""
        truncated_texts = self._truncate_text_pieces([prompt])
        assert len(truncated_texts) == 1
        prompt = truncated_texts[0]
        image = self._load_image(last_image_item)

        return image, prompt

    def _prepare_instruct_model(
        self, conversation: Conversation
    ) -> tuple[list[Image.Image], str]:
        """Prepares the images and prompt for an instruct model.

        Instruct models use the chat template to generate the prompt, and can include
        multiple images and text turns.
        """
        if self._processor is None:
            raise ValueError("Processor is required for instruct model")

        # Generates the prompt using the chat template
        # including image placeholders for each image in the conversation
        messages = []
        for turn in conversation.messages:
            if turn.contains_text() or turn.contains_images():
                messages.append(turn)
            else:
                raise ValueError(
                    f"Unsupported message: {turn.id}. Contains no text and no images."
                )

        messages = self._truncate_text_in_content_items(messages)

        text_prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=False
        )

        # Loads the images from the conversation
        image_items = [
            item for turn in conversation.messages for item in turn.image_content_items
        ]
        images = [self._load_image(item) for item in image_items]

        return images, text_prompt

    def _load_image(self, image_item: ContentItem) -> Image.Image:
        """Loads an image from a message.

        Args:
            image_item (`ContentItem`): A content item representing an image.

        Returns:
            Image.Image: A PIL image.
        """
        if self._image_processor is None:
            raise ValueError("Processor required for transform")
        return load_pil_image_from_content_item(image_item)

    @override
    def transform_conversation(
        self, conversation: Conversation, options: Optional[FeatureGeneratorOptions]
    ) -> dict:
        """Transforms a single Oumi conversation into a dictionary of model inputs.

        Args:
            conversation: An input conversation.
            options: Options for the feature generator.

        Returns:
            dict: A dictionary of inputs for a model.
        """
        return self.transform_conversations([conversation], options)

    @override
    def transform_conversations(
        self,
        conversations: list[Conversation],
        options: Optional[FeatureGeneratorOptions],
    ) -> dict:
        """Transforms a list of Oumi conversations into a dictionary of model inputs.

        Args:
            conversations: An input conversation.
            options: Options for the feature generator.

        Returns:
            dict: A dictionary of inputs for a model.
        """
        if self._processor is None:
            raise ValueError("Processor required to transform a conversation")

        valid_options: FeatureGeneratorOptions = options or FeatureGeneratorOptions()

        all_images: list[list[Image.Image]] = []
        all_prompts: list[str] = []
        if self._processor.chat_template is None:
            for conversation in conversations:
                image, prompt = self._prepare_simple_model(conversation)
                all_images.append([image])
                all_prompts.append(prompt)
        else:
            for conversation in conversations:
                images, prompt = self._prepare_instruct_model(conversation)
                all_images.append(images)
                all_prompts.append(prompt)

        inputs = self._processor(
            images=[image for item in all_images for image in item],
            text=all_prompts,
            return_tensors=self._return_tensors,
            padding=True,
        )

        # Clone `input_ids` as `labels`.
        input_ids = inputs["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            inputs["labels"] = input_ids.clone()
        else:
            inputs["labels"] = copy.deepcopy(input_ids)

        # Post-process input features according to internal config.
        for (
            feature_name,
            feature_spec,
        ) in self._internal_model_config.model_input_features.items():
            if (not feature_spec.required) and (feature_name not in inputs):
                continue
            x = inputs[feature_name]

            if not isinstance(x, (list, torch.Tensor, np.ndarray)):
                raise ValueError(
                    f"Unexpected type of the feature '{feature_name}': {type(x)}"
                )

            first_dim_action = feature_spec.first_dim_action
            if (
                first_dim_action != InternalFeatureFirstDimAction.KEEP
                and not valid_options.allow_feature_reshape
            ):
                logger.debug(f"{feature_name}: Rewrote {first_dim_action} to KEEP")
                first_dim_action = InternalFeatureFirstDimAction.KEEP

            if first_dim_action in (
                InternalFeatureFirstDimAction.DROP_ALWAYS,
                InternalFeatureFirstDimAction.DROP_IF_DUMMY,
            ):
                first_dim_len = get_first_dim_len(x)
                if first_dim_len <= 0:
                    raise ValueError(
                        f"Empty first dimension for the feature '{feature_name}'."
                    )
                drop_first_dim = (
                    first_dim_action == InternalFeatureFirstDimAction.DROP_ALWAYS
                    or first_dim_len <= 1
                )
                if first_dim_len > 1 and drop_first_dim:
                    logger.warning(
                        "The first dimension (dim=0) is non-dummy for "
                        f"the feature: '{feature_name}'! "
                        f"{first_dim_action} for the first dim size: {first_dim_len}). "
                        "Only the first element is kept, others are dropped, "
                        "which may lead to data loss, and to tensor shape errors."
                    )
                if drop_first_dim:
                    inputs[feature_name] = x[0]
                else:
                    inputs[feature_name] = x
            else:
                assert first_dim_action == InternalFeatureFirstDimAction.KEEP
                inputs[feature_name] = x

        # Ignore `image_token_id`-s in the loss computation.
        if (
            self._special_tokens.label_ignore_index is not None
            and self._special_tokens.image_token_id is not None
        ):
            labels = inputs["labels"]
            image_token_id = int(self._special_tokens.image_token_id)
            label_ignore_index = int(self._special_tokens.label_ignore_index)
            if isinstance(labels, (torch.Tensor, np.ndarray)):
                # Modify in-place
                labels[labels == image_token_id] = label_ignore_index
            else:
                # Create numpy array, modify, and copy back.
                labels = np.array(labels)
                labels[labels == image_token_id] = label_ignore_index
                inputs["labels"] = labels.tolist()
        elif (
            self._internal_model_config is not None
            and self._internal_model_config.sanitize_negative_labels
        ):
            # Some VLM-s may generate negative input_ids for image tokens.
            # For example, Phi3-Vision generates `-N` input ids for
            # "<|image_N|>" tokens. It can cause CUDA errors during loss
            # computation as loss function may assume all labels are
            # within the [0, num_classes) range.
            # The code below attempts to sanitize labels by resetting all negative
            # labels to `label_ignore_index` (if provided) or to PAD token index.
            #
            # TODO OPE-701 Consider having a more general configuration per model type.
            labels = inputs["labels"]
            sanitized_label_target = int(
                self._special_tokens.pad_token_id
                if (
                    self._special_tokens.label_ignore_index is None
                    or self._special_tokens.label_ignore_index < 0
                )
                else self._special_tokens.label_ignore_index
            )
            assert sanitized_label_target >= 0
            if isinstance(labels, torch.Tensor):
                # Modify in-place
                labels[labels < 0] = sanitized_label_target
            elif isinstance(labels, np.ndarray):
                # Modify in-place
                labels[labels < 0] = sanitized_label_target
            else:
                # Create numpy array, modify, and copy back.
                labels = np.array(labels)
                labels[labels < 0] = sanitized_label_target
                inputs["labels"] = labels.tolist()

        # Apply completion-only training masking if enabled
        if self._train_on_completions_only:
            self._apply_completion_only_masking(inputs)

        return inputs.data

    def _truncate_text_in_content_items(self, messages: list[Message]) -> list[Message]:
        """Truncates text contents in Messages to `max_length` total tokens.

        Note that we have to truncate plain texts *before* we apply chat template
        as the final processed prompt is generally unsafe to truncate at arbitrary
        offset: it may break invariants (e.g., prompt contains `N` images tokens)
        leading to runtime errors in processor.
        """
        if not (
            self._truncation and self._max_length is not None and self._max_length > 0
        ):
            return messages

        return truncate_text_in_content_items(
            messages,
            tokenizer=self._processor.tokenizer,
            max_tokens=self._max_length,
            truncation_side=self._truncation_side,
        )

    def _truncate_text_pieces(self, text_pieces: list[str]) -> list[str]:
        """Truncates text pieces to total length not exceeding `max_length`."""
        if not (
            self._truncation and self._max_length is not None and self._max_length > 0
        ):
            return copy.deepcopy(text_pieces)

        return truncate_text_pieces_to_max_tokens_limit(
            text_pieces,
            tokenizer=self._processor.tokenizer,
            max_tokens=self._max_length,
            truncation_side=self._truncation_side,
        )

    def _apply_completion_only_masking(self, inputs: Any) -> None:
        """Apply masking to keep only assistant responses for loss computation."""
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")

        if labels is None or input_ids is None:
            raise ValueError(
                "Properties `labels` and `input_ids` are required for "
                "completion-only training"
            )

        # Convert to numpy for processing
        labels_array = np.array(labels)
        input_ids_array = np.array(input_ids)

        if len(labels_array.shape) == 1:
            self._mask_single_conversation(labels_array, input_ids_array)
        else:
            # Process each sequence in the batch
            for i in range(labels_array.shape[0]):
                self._mask_single_conversation(labels_array[i], input_ids_array[i])

        # Convert back to original format
        if isinstance(labels, torch.Tensor):
            inputs["labels"] = torch.from_numpy(labels_array)
        elif isinstance(labels, list):
            inputs["labels"] = labels_array.tolist()
        else:
            inputs["labels"] = labels_array

    def _mask_single_conversation(
        self, labels: np.ndarray, input_ids: np.ndarray
    ) -> None:
        """Mask a single conversation to keep only assistant responses."""
        ignore_index = int(self._special_tokens.label_ignore_index or -100)

        # Choose masking strategy based on whether instruction token IDs are available
        if hasattr(self, "_instruction_token_ids") and self._instruction_token_ids:
            mask_labels_for_completions_only(
                labels,
                self._response_token_ids,
                self._instruction_token_ids,
                ignore_index=ignore_index,
            )
        else:
            mask_labels_without_user_template(
                labels,
                self._response_token_ids,
                ignore_index=ignore_index,
            )
