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

from __future__ import annotations

import functools
import json
from typing import Any, NamedTuple

import pydantic
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
from oumi.core.configs.internal.supported_models import (
    find_internal_model_config_using_model_name,
)
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.conversation_utils import (
    base64encode_content_item_image_bytes,
    load_image_bytes_to_content_item,
)


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
        self,
        model_params: ModelParams,
        *,
        remote_params: RemoteParams | None = None,
        generation_params: GenerationParams | None = None,
    ):
        """Initializes the SGL inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            remote_params: Remote server params.
            generation_params: The generation parameters to use for inference.
        """
        if remote_params is None:
            raise ValueError("remote_params is required")

        super().__init__(
            model_params=model_params,
            generation_params=generation_params,
            remote_params=remote_params,
        )

        self._tokenizer = build_tokenizer(self._model_params)
        self._processor: BaseProcessor | None = None
        self._supports_multiple_images: bool = False

        if is_image_text_llm(self._model_params):
            # Only enable Processor for vision language models for now.
            self._processor = build_processor(
                self._model_params.model_name,
                self._tokenizer,
                trust_remote_code=self._model_params.trust_remote_code,
                processor_kwargs=self._model_params.processor_kwargs,
            )
            internal_model_config = find_internal_model_config_using_model_name(
                self._model_params.model_name,
                trust_remote_code=self._model_params.trust_remote_code,
            )
            self._supports_multiple_images = (
                (internal_model_config is not None)
                and (internal_model_config.visual_config is not None)
                and internal_model_config.visual_config.supports_multiple_images
            )

        # TODO Launch a local SGLLang server if requested.

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
                elif isinstance(json_schema_value, pydantic.BaseModel) or (
                    isinstance(json_schema_value, type)
                    and issubclass(json_schema_value, pydantic.BaseModel)
                ):
                    json_schema = json.dumps(json_schema_value.model_json_schema())
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

    def _create_image_data_as_str_list(self, conversation: Conversation) -> list[str]:
        image_items = [
            item for m in conversation.messages for item in m.image_content_items
        ]
        num_images = len(image_items)
        if num_images <= 0:
            return []

        max_images = num_images if self._supports_multiple_images else 1

        if num_images > max_images:
            # If a conversation contains too many images, raise an error.
            # We can't silently discard extra images at this point
            # as many models verify that the actual number of images matches
            # the number of image tokens in text prompt.
            raise ValueError(
                conversation.append_id_to_string(
                    f"A conversation contains too many images ({num_images}). "
                    f"Max {max_images} image is allowed."
                )
            )

        result: list[str] = []
        for idx, image_item in enumerate(image_items):
            if image_item.type == Type.IMAGE_URL:
                # Preserve URL-s: leave them to SGLang server to download
                # to keep message payload size under control.
                # TODO Consider making this behaviour configurable.
                image_url = image_item.content
                if not image_url:
                    raise ValueError(
                        conversation.append_id_to_string(
                            f"Empty image URL in message: {image_item.type} "
                            f"in image item {idx + 1} of {num_images}!"
                        )
                    )
                result.append(image_url)
            else:
                image_item = load_image_bytes_to_content_item(image_item)
                if image_item.binary is None or len(image_item.binary) == 0:
                    raise ValueError(
                        conversation.append_id_to_string(
                            f"No image bytes in image item {idx + 1} of {num_images}!"
                        )
                    )
                result.append(base64encode_content_item_image_bytes(image_item))

        return result

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to SGLang Native API input.

        See https://sgl-project.github.io/references/sampling_params.html for details.

        Args:
            conversation: The Oumi Conversation object to convert.
            generation_params: Parameters for text generation.
            model_params: Ignored.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted input for the
            SGLang server native API, including the model, messages, generation params.
        """
        # Chat templates loaded by SGLang server are generally different from Oumi's
        # chat templates, hence, let's apply Oumi chat template here ourselves.
        prompt = self._apply_chat_template_impl(conversation)

        sampling_params_dict = self._create_sampling_params_as_dict(generation_params)
        body = {
            "text": prompt,
            "sampling_params": sampling_params_dict,
        }
        image_data: list[str] = self._create_image_data_as_str_list(conversation)
        if len(image_data) > 0:
            body["image_data"] = image_data if len(image_data) > 1 else image_data[0]
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
