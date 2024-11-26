import asyncio
import copy
import json
import os
from typing import Any, Optional

import aiohttp
import pydantic
from tqdm.asyncio import tqdm
from typing_extensions import override

from oumi.core.async_utils import safe_asyncio_run
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.utils.image_utils import base64encode_image_bytes, load_image_bytes_to_message

_CONTENT_KEY: str = "content"
_MESSAGE_KEY: str = "message"
_ROLE_KEY: str = "role"
_TYPE_KEY: str = "type"
_TEXT_KEY: str = "text"
_IMAGE_URL_KEY: str = "image_url"
_AUTHORIZATION_KEY: str = "Authorization"
_URL_KEY: str = "url"


class RemoteInferenceEngine(BaseInferenceEngine):
    """Engine for running inference against a server implementing the OpenAI API."""

    def __init__(self, model_params: ModelParams, remote_params: RemoteParams):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            remote_params: Remote server params.
        """
        self._model = model_params.model_name
        self._remote_params = copy.deepcopy(remote_params)

    @staticmethod
    def _get_content_for_message(message: Message) -> dict[str, Any]:
        """Returns the content for a message.

        Args:
            message: The message to get the content for.

        Returns:
            Dict[str, Any]: The content for the message.
        """
        if message.type == Type.TEXT:
            return {_TYPE_KEY: Type.TEXT.value, _TEXT_KEY: (message.content or "")}
        elif not message.is_image():
            raise ValueError(f"Unsupported message type: {message.type}")

        if not message.binary and message.type != Type.IMAGE_URL:
            message = load_image_bytes_to_message(message)

        if message.binary:
            b64_image = base64encode_image_bytes(message, add_mime_prefix=True)
            return {
                _TYPE_KEY: Type.IMAGE_URL.value,
                _IMAGE_URL_KEY: {_URL_KEY: b64_image},
            }

        assert (
            message.type == Type.IMAGE_URL
        ), f"Unexpected message type: {message.type}. Must be a code bug."
        return {
            _TYPE_KEY: Type.IMAGE_URL.value,
            _IMAGE_URL_KEY: {message.content or ""},
        }

    @staticmethod
    def _get_list_of_message_json_dicts(
        messages: list[Message],
        *,
        group_adjacent_same_role_turns: bool,
    ) -> list[dict[str, Any]]:
        """Returns a list of JSON dictionaries representing messages.

        Loads image bytes and encodes them as base64.

        Args:
            messages: The input messages.
            group_adjacent_same_role_turns: Whether to pack adjacent messages
                from the same role into a single element in output list.
                For multimodal conversations, adjacent image and text turns from
                the same role must be grouped together.

        Returns:
            list[Dict[str, Any]]: The list of messages encoded as nested JSON dicts.
        """
        num_messages = len(messages)
        result = []
        idx = 0
        while idx < num_messages:
            end_idx = idx + 1
            if group_adjacent_same_role_turns:
                while end_idx < num_messages and (
                    messages[idx].role == messages[end_idx].role
                ):
                    end_idx += 1

            item: dict[str, Any] = {
                _ROLE_KEY: messages[idx].role.value,
            }
            group_size = end_idx - idx
            if group_size == 1 and messages[idx].is_text():
                # Set "content" to a primitive string value, which is the common
                # convention for text-only models.
                item[_CONTENT_KEY] = messages[idx].content
            else:
                # Set "content" to be a list of dictionaries for more complex cases.
                content_list = []
                while idx < end_idx:
                    content_list.append(
                        RemoteInferenceEngine._get_content_for_message(messages[idx])
                    )
                    idx += 1
                item[_CONTENT_KEY] = content_list

            idx = end_idx
            result.append(item)

        return result

    def _convert_conversation_to_api_input(
        self, conversation: Conversation, generation_params: GenerationParams
    ) -> dict[str, Any]:
        """Converts a conversation to an OpenAI input.

        Documentation: https://platform.openai.com/docs/api-reference/chat/create

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for generation during inference.

        Returns:
            Dict[str, Any]: A dictionary representing the OpenAI input.
        """
        api_input = {
            "model": self._model,
            "messages": [
                {
                    _CONTENT_KEY: [self._get_content_for_message(message)],
                    _ROLE_KEY: message.role.value,
                }
                for message in conversation.messages
            ],
            "max_completion_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
            "frequency_penalty": generation_params.frequency_penalty,
            "presence_penalty": generation_params.presence_penalty,
            "n": 1,  # Number of completions to generate for each prompt.
            "seed": generation_params.seed,
            "logit_bias": generation_params.logit_bias,
        }

        if generation_params.stop_strings:
            api_input["stop"] = generation_params.stop_strings

        if generation_params.guided_decoding:
            json_schema = generation_params.guided_decoding.json

            if json_schema is not None:
                if isinstance(json_schema, type) and issubclass(
                    json_schema, pydantic.BaseModel
                ):
                    schema_name = json_schema.__name__
                    schema_value = json_schema.model_json_schema()
                elif isinstance(json_schema, dict):
                    # Use a generic name if no schema is provided.
                    schema_name = "Response"
                    schema_value = json_schema
                elif isinstance(json_schema, str):
                    # Use a generic name if no schema is provided.
                    schema_name = "Response"
                    # Try to parse as JSON string
                    schema_value = json.loads(json_schema)
                else:
                    raise ValueError(
                        f"Got unsupported JSON schema type: {type(json_schema)}"
                        "Please provide a Pydantic model or a JSON schema as a "
                        "string or dict."
                    )

                api_input["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema_value,
                    },
                }
            else:
                raise ValueError(
                    "Only JSON schema guided decoding is supported, got '%s'",
                    generation_params.guided_decoding,
                )

        return api_input

    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an API response to a conversation.

        Args:
            response: The API response to convert.
            original_conversation: The original conversation.

        Returns:
            Conversation: The conversation including the generated response.
        """
        message = response["choices"][0][_MESSAGE_KEY]
        return Conversation(
            messages=[
                *original_conversation.messages,
                Message(
                    content=message[_CONTENT_KEY],
                    role=Role(message[_ROLE_KEY]),
                    type=Type.TEXT,
                ),
            ],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )

    def _get_api_key(self, remote_params: RemoteParams) -> Optional[str]:
        if not remote_params:
            return None

        if remote_params.api_key:
            return remote_params.api_key

        if remote_params.api_key_env_varname:
            return os.environ.get(remote_params.api_key_env_varname)

        return None

    def _get_request_headers(
        self, remote_params: Optional[RemoteParams]
    ) -> dict[str, str]:
        headers = {}

        if not remote_params:
            return headers

        headers[_AUTHORIZATION_KEY] = f"Bearer {self._get_api_key(remote_params)}"
        return headers

    async def _query_api(
        self,
        conversation: Conversation,
        inference_config: InferenceConfig,
        remote_params: RemoteParams,
        semaphore: asyncio.Semaphore,
        session: aiohttp.ClientSession,
    ) -> Conversation:
        """Queries the API with the provided input.

        Args:
            conversation: The conversations to run inference on.
            inference_config: Parameters for inference.
            remote_params: Parameters for running inference against a remote API.
            semaphore: Semaphore to limit concurrent requests.
            session: The aiohttp session to use for the request.

        Returns:
            Conversation: Inference output.
        """
        assert remote_params.api_url
        async with semaphore:
            api_input = self._convert_conversation_to_api_input(
                conversation, inference_config.generation
            )
            headers = self._get_request_headers(inference_config.remote_params)
            retries = 0
            # Retry the request if it fails.
            for _ in range(remote_params.max_retries + 1):
                async with session.post(
                    remote_params.api_url,
                    json=api_input,
                    headers=headers,
                    timeout=remote_params.connection_timeout,
                ) as response:
                    response_json = await response.json()
                    if response.status == 200:
                        result = self._convert_api_output_to_conversation(
                            response_json, conversation
                        )
                        if inference_config.output_path:
                            # Write what we have so far to our scratch directory.
                            self._save_conversation(
                                result,
                                self._get_scratch_filepath(
                                    inference_config.output_path
                                ),
                            )
                        await asyncio.sleep(remote_params.politeness_policy)
                        return result
                    else:
                        retries += 1
                        await asyncio.sleep(remote_params.politeness_policy)
            raise RuntimeError(
                f"Failed to query API after {remote_params.max_retries} retries."
            )

    async def _infer(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig,
        remote_params: RemoteParams,
    ) -> list[Conversation]:
        """Runs model inference on the provided input.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.
            remote_params: Parameters for running inference against a remote API.

        Returns:
            List[Conversation]: Inference output.
        """
        # Limit number of HTTP connections to the number of workers.
        connector = aiohttp.TCPConnector(limit=remote_params.num_workers)
        # Control the number of concurrent tasks via a semaphore.
        semaphore = asyncio.BoundedSemaphore(remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._query_api(
                    conversation,
                    inference_config,
                    remote_params,
                    semaphore,
                    session,
                )
                for conversation in input
            ]

            disable_tqdm = len(tasks) < 2
            return await tqdm.gather(*tasks, disable=disable_tqdm)

    @override
    def infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig,
    ) -> list[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        if not inference_config.remote_params:
            raise ValueError("Remote params must be provided in inference config.")
        conversations = safe_asyncio_run(
            self._infer(input, inference_config, inference_config.remote_params)
        )
        if inference_config.output_path:
            self._save_conversations(conversations, inference_config.output_path)
        return conversations

    @override
    def infer_from_file(
        self, input_filepath: str, inference_config: InferenceConfig
    ) -> list[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the
        existence of input_filepath in the generation_params.

        Args:
            input_filepath: Path to the input file containing prompts for
                generation.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        if not inference_config.remote_params:
            raise ValueError("Remote params must be provided in inference config.")
        input = self._read_conversations(input_filepath)
        conversations = safe_asyncio_run(
            self._infer(input, inference_config, inference_config.remote_params)
        )
        if inference_config.output_path:
            self._save_conversations(conversations, inference_config.output_path)
        return conversations

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "guided_decoding",
            "logit_bias",
            "max_new_tokens",
            "presence_penalty",
            "seed",
            "stop_strings",
            "temperature",
            "top_p",
        }
