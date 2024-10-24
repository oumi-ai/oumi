import asyncio
import base64
import os
from typing import Any, Optional

import aiohttp
from tqdm.asyncio import tqdm

from oumi.core.async_utils import safe_asyncio_run
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role, Type

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

    def __init__(self, model_params: ModelParams):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
        """
        self._model = model_params.model_name

    def _get_content_for_message(self, message: Message) -> dict[str, Any]:
        """Returns the content for a message.

        Args:
            message: The message to get the content for.

        Returns:
            Dict[str, Any]: The content for the message.
        """
        content: dict[str, Any] = {
            _TYPE_KEY: message.type.value,
        }
        b64_image = None if message.binary is None else base64.b64encode(message.binary)

        if message.type == Type.TEXT:
            content[_TEXT_KEY] = message.content or ""
        elif message.type == Type.IMAGE_URL:
            content[_IMAGE_URL_KEY] = {
                _URL_KEY: b64_image or message.content,
            }
        elif message.type == Type.IMAGE_PATH:
            if message.content and not b64_image:
                with open(message.content, "rb") as image_file:
                    b64_image = base64.b64encode(image_file.read())
            content[_IMAGE_URL_KEY] = {
                _URL_KEY: b64_image or message.content,
            }
        elif message.type == Type.IMAGE_BINARY:
            content[_IMAGE_URL_KEY] = {
                _URL_KEY: b64_image or message.content,
            }
        else:
            raise ValueError(f"Unsupported message type: {message.type}")
        return content

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

        if remote_params.api_key is not None:
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
            headers = self._get_request_headers(
                inference_config.generation.remote_params
            )
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
        generation_params = inference_config.generation
        if not generation_params.remote_params:
            raise ValueError("Remote params must be provided in generation_params.")
        conversations = safe_asyncio_run(
            self._infer(input, inference_config, generation_params.remote_params)
        )
        if inference_config.output_path:
            self._save_conversations(conversations, inference_config.output_path)
        return conversations

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
        generation_params = inference_config.generation
        if not generation_params.remote_params:
            raise ValueError("Remote params must be provided in generation_params.")
        input = self._read_conversations(input_filepath)
        conversations = safe_asyncio_run(
            self._infer(input, inference_config, generation_params.remote_params)
        )
        if inference_config.output_path:
            self._save_conversations(conversations, inference_config.output_path)
        return conversations

    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "logit_bias",
            "max_new_tokens",
            "presence_penalty",
            "remote_params",
            "seed",
            "stop_strings",
            "temperature",
            "top_p",
        }
