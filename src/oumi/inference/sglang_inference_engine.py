from __future__ import annotations

import copy
import functools
import json
from typing import Any, NamedTuple

from typing_extensions import override

from oumi.builders import (
    build_processor,
    build_tokenizer,
    is_image_text_llm,
)
from oumi.core.configs import (
    GenerationParams,
    ModelParams,
    RemoteParams,
)
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.conversation_utils import base64encode_content_item_image_bytes
from oumi.utils.logging import logger


class _SamplingParams(NamedTuple):
    """It's a clone of `sglang.lang.ir.SglSamplingParams`.

    Only includes a subset of parameters supported in oumi.
    Unsupported params are left commented out for reference.
    """

    max_new_tokens: int = 128
    # min_new_tokens: int = 0
    stop: str | list[str] = ""
    stop_token_ids: list[int] | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    # top_k: int = -1  # -1 means disable
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # ignore_eos: bool = False
    # return_logprob: bool | None = None
    # logprob_start_len: int | None = None
    # top_logprobs_num: int | None = None
    # return_text_in_logprobs: bool | None = None
    json_schema: str | None = None

    # For constrained generation:
    # dtype: str | None = None
    regex: str | None = None


class SGLangInferenceEngine(RemoteInferenceEngine):
    """Engine for running SGLang inference."""

    def __init__(
        self, model_params: ModelParams, remote_params: RemoteParams, **kwargs
    ):
        """Initializes the SGL inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            remote_params: Remote server params.
            kwargs: Other keyword arguments.
        """
        self._model_params = copy.deepcopy(model_params)
        self._tokenizer = build_tokenizer(self._model_params)
        self._processor: BaseProcessor | None = None
        if is_image_text_llm(self._model_params):
            # Only enable Processor for vision language models for now.
            self._processor = build_processor(
                self._model_params.model_name,
                self._tokenizer,
                trust_remote_code=self._model_params.trust_remote_code,
            )

        # TODO Launch a local SGLLang server if requested.

        super().__init__(
            model_params=model_params, remote_params=remote_params, **kwargs
        )

    def _create_sampling_params(
        self, generation_params: GenerationParams
    ) -> _SamplingParams:
        regex: str | None = None
        json_schema: str | None = None
        if generation_params.guided_decoding is not None:
            if generation_params.guided_decoding.regex is not None:
                regex = generation_params.guided_decoding.regex
            else:
                json_schema_value = None
                if generation_params.guided_decoding.json is not None:
                    json_schema_value = generation_params.guided_decoding.json
                elif (
                    generation_params.guided_decoding.choice is not None
                    and len(generation_params.guided_decoding.choice) > 0
                ):
                    json_schema_value = {
                        "enum": generation_params.guided_decoding.choice
                    }

                if isinstance(json_schema_value, str):
                    json_schema = json_schema_value
                elif isinstance(json_schema_value, dict):
                    json_schema = json.dumps(json_schema_value, ensure_ascii=False)
                else:
                    raise ValueError(
                        "Unsupported type of generation_params.guided_decoding.json: "
                        f"{type(generation_params.guided_decoding.json)}"
                    )

        return _SamplingParams(
            max_new_tokens=generation_params.max_new_tokens,
            temperature=generation_params.temperature,
            top_p=generation_params.top_p,
            min_p=generation_params.min_p,
            frequency_penalty=generation_params.frequency_penalty,
            presence_penalty=generation_params.presence_penalty,
            stop=(generation_params.stop_strings or []),
            stop_token_ids=generation_params.stop_token_ids,
            regex=regex,
            json_schema=json_schema,
        )

    def _create_sampling_params_as_dict(
        self, generation_params: GenerationParams
    ) -> dict[str, Any]:
        return self._create_sampling_params(generation_params)._asdict()

    def _apply_chat_template_impl(self, conversation: Conversation) -> str:
        if self._processor is None:
            return self._tokenizer.apply_chat_template(
                conversation,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
            )
        return self._processor.apply_chat_template(
            conversation,  # type: ignore
            add_generation_prompt=True,
        )

    def _create_image_data_as_str(self, conversation: Conversation) -> str | None:
        image_items = [
            item for m in conversation.messages for item in m.image_content_items
        ]
        num_images = len(image_items)
        if num_images <= 0:
            return None
        elif num_images > 1:
            # FIXME OPE-355 Support multiple images
            logger.warning(
                conversation.append_id_to_string(
                    f"A conversation contains multiple images ({num_images}). "
                    "Only 1 image is currently supported. Using the last image."
                )
            )

        image_item = image_items[-1]
        if image_item.type == Type.IMAGE_BINARY:
            if not image_item.binary:
                raise ValueError(
                    conversation.append_id_to_string(
                        f"No image bytes in message: {image_item.type}"
                    )
                )
            return base64encode_content_item_image_bytes(image_item)

        assert image_item.type in (Type.IMAGE_PATH, Type.IMAGE_URL)
        image_path_or_url = image_item.content
        if not image_path_or_url:
            friendly_type_name = (
                "image path" if image_item.type == Type.IMAGE_PATH else "image URL"
            )
            raise ValueError(
                conversation.append_id_to_string(
                    f"Empty {friendly_type_name} in message: {image_item.type}"
                )
            )
        return image_path_or_url

    @override
    def _convert_conversation_to_api_input(
        self, conversation: Conversation, generation_params: GenerationParams
    ) -> dict[str, Any]:
        """Converts a conversation to SGLang Native API input.

        See https://sgl-project.github.io/references/sampling_params.html for details.

        Args:
            conversation: The Oumi Conversation object to convert.
            generation_params: Parameters for text generation.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted input for the
            SGLang server native API, including the model, messages, generation params.
        """
        # Chat templates loaded by SGLang server are generally different from oumi's
        # chat template, hence, let's apply the template here ourselves.
        prompt = self._apply_chat_template_impl(conversation)

        sampling_params_dict = self._create_sampling_params_as_dict(generation_params)
        body = {
            "text": prompt,
            "sampling_params": sampling_params_dict,
        }
        image_data = self._create_image_data_as_str(conversation)
        if image_data:
            body["image_data"] = image_data
        return body

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an SGLang Native API response to a conversation."""
        new_message = Message(
            content=response["text"],
            role=Role.ASSISTANT,
        )
        return Conversation(
            messages=[*original_conversation.messages, new_message],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )

    @override
    def _get_request_headers(self, remote_params: RemoteParams) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    @override
    @functools.cache
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "guided_decoding",
            "max_new_tokens",
            "min_p",
            "presence_penalty",
            "stop_strings",
            "stop_token_ids",
            "temperature",
            "top_p",
        }
