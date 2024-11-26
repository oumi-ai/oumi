import copy
import io
from abc import ABC, abstractmethod
from enum import Enum
from typing import Final, NamedTuple, Optional, Union

import numpy as np
import requests
import torch
from PIL import Image
from typing_extensions import override

import oumi.core.constants as constants
from oumi.builders.processors import build_processor
from oumi.core.datasets import BaseSftDataset
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.utils.logging import logger
from oumi.utils.torch_utils import get_first_dim_len


class _SpecialTokens(NamedTuple):
    """Special tokens used by VisionLanguageSftDataset."""

    image_token: Optional[str]
    image_token_id: Optional[int]
    label_ignore_index: Optional[int]


class _FirstDimAction(Enum):
    """Enum representing how to handle the first feature dimension."""

    DROP_ALWAYS = "drop_always"
    """The first dimension is commonly dummy (length: 1) and must be dropped.

    In effect, this operation is applied: `x = x[0, ...]`, which reduces
    `x`'s rank by 1 (e.g., 3D->2D), and discards the following elements: `x[1:, ...]`.
    """

    DROP_IF_DUMMY = "drop_if_dummy"
    """Drop the first dimension only if it's dummy (length: 1)."""

    KEEP = "keep"
    """Always preserve the first dimension."""


class InputFeatureSpec(NamedTuple):
    feature_name: str
    required: bool
    first_dim_action: _FirstDimAction = _FirstDimAction.DROP_ALWAYS


_INPUT_FEATURES_LIST: Final[list[InputFeatureSpec]] = [
    InputFeatureSpec(feature_name="input_ids", required=True),
    InputFeatureSpec(
        feature_name="pixel_values",
        required=True,
        first_dim_action=_FirstDimAction.DROP_IF_DUMMY,
    ),
    InputFeatureSpec(feature_name="attention_mask", required=True),
    InputFeatureSpec(feature_name="labels", required=True),
    # Llama 3.2 Vision
    InputFeatureSpec(feature_name="aspect_ratio_ids", required=False),
    InputFeatureSpec(feature_name="aspect_ratio_mask", required=False),
    InputFeatureSpec(feature_name="cross_attention_mask", required=False),
    # Qwen2 VL
    InputFeatureSpec(feature_name="image_grid_thw", required=False),
]
_INPUT_FEATURES_DICT: Final[dict[str, InputFeatureSpec]] = {
    spec.feature_name: spec for spec in _INPUT_FEATURES_LIST
}


class VisionLanguageSftDataset(BaseSftDataset, ABC):
    """Abstract dataset for vision-language models.

    This class extends BaseSftDataset to provide functionality specific to
    vision-language tasks. It handles the processing of both image and text data.

    Note:
        This dataset is designed to work with models that can process both
        image and text inputs simultaneously, such as CLIP, BLIP, or other
        multimodal architectures.

    Example:
        >>> class MyVisionLanguageSftDataset(VisionLanguageSftDataset):
        ...     def transform_conversation(self, example: dict) -> Conversation:
        ...         # Implement the abstract method
        ...         # Convert the raw example into a Conversation object
        ...         pass
        >>>
        >>> dataset = MyVisionLanguageSftDataset(
        ...     processor_name="openai/clip-vit-base-patch32",
        ...     dataset_name="coco_captions",
        ...     split="train"
        ... )
        >>> sample = next(iter(dataset))
        >>> print(sample.keys())
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[BaseTokenizer] = None,
        processor: Optional[BaseProcessor] = None,
        processor_name: Optional[str] = None,
        label_ignore_index: Optional[int] = constants.LABEL_IGNORE_INDEX,
        limit: Optional[int] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the VisionLanguageDataset class."""
        super().__init__(tokenizer=tokenizer, **kwargs)

        if tokenizer is None:
            raise ValueError(
                f"Tokenizer must be provided for {self.__class__.__name__}"
            )

        if processor is not None:
            if processor_name:
                logger.warning(
                    "Both processor and processor_name are provided. "
                    f"Ignoring processor_name: {processor_name}"
                )
        elif processor_name:
            processor = build_processor(
                processor_name, tokenizer, trust_remote_code=trust_remote_code
            )

        self._processor: Optional[BaseProcessor] = processor
        if self._processor is not None:
            if not callable(self._processor):
                raise ValueError("Processor is not callable!")
            self._image_processor = self._processor.image_processor
        else:
            assert self._processor is None
            self._tokenizer = None  # Reset base class's member variable.
            self._image_processor = None

        self._special_tokens: _SpecialTokens = _SpecialTokens(
            image_token=(self._processor.image_token if self._processor else None),
            image_token_id=(
                self._processor.image_token_id if self._processor else None
            ),
            label_ignore_index=label_ignore_index,
        )

        if limit is not None:
            # TODO: this should be removed when we switch to datapipes.
            # Right now, we have to iterate over the whole dataset at init time,
            # Which takes way to long.
            self._data = self._data.head(limit)

    @abstractmethod
    def transform_conversation(self, example: dict) -> Conversation:
        """Transforms a raw example into an Oumi Conversation object.

        Args:
            example (dict): A dictionary representing a single conversation example.

        Returns:
            Conversation: A Conversation object representing the conversation.
        """
        raise NotImplementedError

    @override
    def transform(self, sample: dict) -> dict:
        """Transforms an Oumi conversation into a dictionary of inputs for a model.

        Args:
            sample (dict): A dictionary representing a single conversation example.

        Returns:
            dict: A dictionary of inputs for a model.
        """
        if self._processor is None:
            raise ValueError("Processor required for transform")

        conversation = self.transform_conversation(sample)

        if self._processor.chat_template is None:
            image, prompt = self._prepare_simple_model(conversation)

            inputs = self._processor(
                images=[image],
                text=[prompt],
                return_tensors=self._return_tensors,
                padding=True,
            )
        else:
            images, prompt = self._prepare_instruct_model(conversation)

            inputs = self._processor(
                images=images,
                text=[prompt],
                return_tensors=self._return_tensors,
                padding=True,
            )

        # Clone `input_ids` as `labels`.
        input_ids = inputs["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            inputs["labels"] = input_ids.clone()
        else:
            inputs["labels"] = copy.deepcopy(inputs["input_ids"])

        # Processors by default return a list of tensors for each key
        # We need to squeeze the first dimension so that it works with the data-loader
        # Images will be of shape (C, H, W) and texts will be of shape (T)
        # However, this is going to break models that support multiple images
        # TODO: OPE-355 add support for multiple images
        for feature_name, feature_spec in _INPUT_FEATURES_DICT.items():
            if (not feature_spec.required) and (feature_name not in inputs):
                continue
            x = inputs[feature_name]

            if not isinstance(x, (list, torch.Tensor, np.ndarray)):
                raise ValueError(
                    f"Unexpected type of the feature '{feature_name}': {type(x)}"
                )

            first_dim_action = feature_spec.first_dim_action

            if first_dim_action in (
                _FirstDimAction.DROP_ALWAYS,
                _FirstDimAction.DROP_IF_DUMMY,
            ):
                first_dim_len = get_first_dim_len(x)
                if first_dim_len <= 0:
                    raise ValueError(
                        f"Empty first dimension for the feature '{feature_name}'."
                    )
                drop_first_dim = (
                    first_dim_action == _FirstDimAction.DROP_ALWAYS
                    or first_dim_len <= 1
                )
                if first_dim_len > 1 and drop_first_dim:
                    logger.warning(
                        "The first dimension is non-dummy for "
                        f"the feature: '{feature_name}' (Length: {first_dim_len}). "
                        "Only the first element is kept, others are dropped, "
                        "which may lead to data loss, and to tensor shape errors."
                    )
                if drop_first_dim:
                    inputs[feature_name] = x[0]
                else:
                    inputs[feature_name] = x
            else:
                assert feature_spec.first_dim_action == _FirstDimAction.KEEP
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

        return inputs.data

    def _prepare_simple_model(
        self, conversation: Conversation
    ) -> tuple[Image.Image, str]:
        """Prepares the images and prompt for a simple model.

        Simple models only use the last image and text turn in the conversation. They
        don't use the chat template, so the prompt is just the last text turn.
        """
        image_turns = [turn for turn in conversation.messages if turn.is_image()]
        text_turns = [turn for turn in conversation.messages if turn.is_text()]

        if not image_turns:
            raise ValueError("Conversation must contain at least one image turn")
        if not text_turns:
            raise ValueError("Conversation must contain at least one text turn")

        last_image_turn = image_turns[-1]
        last_text_turn = text_turns[-1].content or ""

        prompt = last_text_turn
        image = self._load_image(last_image_turn)

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
        texts = []
        for turn in conversation.messages:
            if turn.is_text() or turn.is_image():
                texts.append(turn)
            else:
                raise ValueError(f"Unsupported message type: {turn.type}")

        text = self._processor.apply_chat_template(texts, add_generation_prompt=False)

        # Loads the images from the conversation
        images = [turn for turn in conversation.messages if turn.is_image()]
        images = [self._load_image(image) for image in images]

        return images, text

    def _load_image(self, image: Union[str, Message]) -> Image.Image:
        """Loads an image from a message.

        Args:
            image (Union[str, Message]): A string representing the image path or a
                Message object.

        Returns:
            Image.Image: A PIL image.
        """
        if self._image_processor is None:
            raise ValueError("Processor required for transform")

        if isinstance(image, str):
            image_type = Type.IMAGE_URL if image.startswith("http") else Type.IMAGE_PATH
            image = Message(type=image_type, content=image, role=Role.USER)

        if image.type == Type.IMAGE_PATH:
            if image.content is None:
                raise ValueError("Image path is None")
            image_bin = Image.open(image.content).convert("RGB")

        elif image.type == Type.IMAGE_URL:
            if image.content is None:
                raise ValueError("Image URL is None")
            try:
                response = requests.get(image.content, stream=True)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.exception(f"Failed to download image: '{image.content}'")
                raise e
            image_bin = Image.open(io.BytesIO(response.content)).convert("RGB")

        elif image.type == Type.IMAGE_BINARY:
            if image.binary is None:
                raise ValueError("Image binary is None")
            image_bin = Image.open(io.BytesIO(image.binary)).convert("RGB")

        else:
            raise ValueError(f"Unsupported image type: {image.type}")

        return image_bin
