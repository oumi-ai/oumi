import asyncio
import base64
from typing import Any, Dict, List

import aiohttp

from lema.core.configs import GenerationConfig, ModelParams
from lema.core.inference import BaseInferenceEngine
from lema.core.types.turn import Conversation, Message, Role, Type


class RemoteInferenceEngine(BaseInferenceEngine):
    """Engine for running inference against a server implementing the OpenAI API."""

    def __init__(self, model_params: ModelParams):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
        """
        self._model = model_params.model_name

    def _validate_generation_config(self, generation_config: GenerationConfig):
        """Validates the generation config.

        Args:
            generation_config: Configuration parameters for generation during
                inference.

        Raises:
            ValueError: If the API key or URL is not provided in the
                generation_config.
        """
        if not generation_config.api_url:
            raise ValueError("The API URL must be provided in generation_config.")
        if generation_config.num_workers < 1:
            raise ValueError(
                "Number of num_workers must be greater than or equal to 1."
            )
        if generation_config.politeness_policy < 0:
            raise ValueError("Politeness policy must be greater than or equal to 0.")
        if generation_config.connection_timeout < 0:
            raise ValueError("Connection timeout must be greater than or equal to 0.")
        if generation_config.max_retries < 0:
            raise ValueError("Max retries must be greater than or equal to 0.")

    def _get_content_for_message(self, message: Message) -> Dict[str, Any]:
        """Returns the content for a message.

        Args:
            message: The message to get the content for.

        Returns:
            Dict[str, Any]: The content for the message.
        """
        content: Dict[str, Any] = {
            "type": message.type.value,
        }
        b64_image = None if message.binary is None else base64.b64encode(message.binary)

        if message.type == Type.TEXT:
            content["text"] = message.content or ""
        elif message.type == Type.IMAGE_URL:
            content["image_url"] = {
                "url": b64_image or message.content,
            }
        elif message.type == Type.IMAGE_PATH:
            if message.content and not b64_image:
                with open(message.content, "rb") as image_file:
                    b64_image = base64.b64encode(image_file.read())
            content["image_url"] = {
                "url": b64_image or message.content,
            }
        elif message.type == Type.IMAGE_BINARY:
            content["image_url"] = {
                "url": b64_image or message.content,
            }
        else:
            raise ValueError(f"Unsupported message type: {message.type}")
        return content

    def _convert_conversation_to_openai_input(
        self, conversation: Conversation, generation_config: GenerationConfig
    ) -> Dict[str, Any]:
        """Converts a conversation to an OpenAI input.

        Args:
            conversation: The conversation to convert.
            generation_config: Configuration parameters for generation during
                inference.

        Returns:
            Dict[str, Any]: A dictionary representing the OpenAI input.
        """
        return {
            "model": self._model,
            "messages": [
                {
                    "content": [self._get_content_for_message(message)],
                    "role": message.role.value,
                }
                for message in conversation.messages
            ],
            "max_completion_tokens": generation_config.max_new_tokens,
            "n": 1,
            "seed": generation_config.seed,
        }

    def _convert_openai_output_to_conversation(
        self, response: Dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Converts an OpenAI response to a conversation.

        Args:
            response: The OpenAI response to convert.
            original_conversation: The original conversation.

        Returns:
            Conversation: The conversation including the generated response.
        """
        message = response["choices"][0]["message"]
        return Conversation(
            messages=[
                *original_conversation.messages,
                Message(
                    content=message["content"],
                    role=Role(message["role"]),
                    type=Type.TEXT,
                ),
            ],
            metadata=original_conversation.metadata,
            conversation_id=original_conversation.conversation_id,
        )

    async def _query_api(
        self,
        conversation: Conversation,
        generation_config: GenerationConfig,
        semaphore: asyncio.Semaphore,
        session: aiohttp.ClientSession,
    ) -> Conversation:
        """Queries the API with the provided input.

        Args:
            conversation: The conversations to run inference on.
            generation_config: Configuration parameters for generation during
                inference.
            semaphore: Semaphore to limit concurrent requests.
            session: The aiohttp session to use for the request.

        Returns:
            Conversation: Inference output.
        """
        assert generation_config.api_url
        async with semaphore:
            openai_input = self._convert_conversation_to_openai_input(
                conversation, generation_config
            )
            headers = {}
            if generation_config.api_key is not None:
                headers["Authorization"] = f"Bearer {generation_config.api_key}"
            retries = 0
            # Retry the request if it fails.
            while True:
                async with session.post(
                    generation_config.api_url,
                    json=openai_input,
                    headers=headers,
                    timeout=generation_config.connection_timeout,
                ) as response:
                    response_json = await response.json()
                    if response.status == 200:
                        result = self._convert_openai_output_to_conversation(
                            response_json, conversation
                        )
                        await asyncio.sleep(generation_config.politeness_policy)
                        return result
                    else:
                        retries += 1
                        if retries > generation_config.max_retries:
                            raise RuntimeError(
                                "Failed to query API after "
                                f"{generation_config.max_retries} retries."
                            )
                        await asyncio.sleep(generation_config.politeness_policy)

    async def _infer(
        self, input: List[Conversation], generation_config: GenerationConfig
    ) -> List[Conversation]:
        """Runs model inference on the provided input.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during
                inference.

        Returns:
            List[Conversation]: Inference output.
        """
        self._validate_generation_config(generation_config)
        # Limit number of HTTP connections to the number of workers.
        connector = aiohttp.TCPConnector(limit=generation_config.num_workers)
        # Control the number of concurrent tasks via a semaphore.
        semaphore = asyncio.BoundedSemaphore(generation_config.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            return await asyncio.gather(
                *[
                    self._query_api(conversation, generation_config, semaphore, session)
                    for conversation in input
                ]
            )

    def infer_online(
        self, input: List[Conversation], generation_config: GenerationConfig
    ) -> List[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during
                inference.

        Returns:
            Optional[List[Conversation]]: Inference output.
        """
        conversations = asyncio.run(self._infer(input, generation_config))
        if generation_config.output_filepath:
            self._save_conversations(conversations, generation_config.output_filepath)
        return conversations

    def infer_from_file(
        self, input_filepath: str, generation_config: GenerationConfig
    ) -> List[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the
        existence of input_filepath in the generation_config.

        Args:
            input_filepath: Path to the input file containing prompts for
                generation.
            generation_config: Configuration parameters for generation during
                inference.

        Returns:
            Optional[List[Conversation]]: Inference output.
        """
        input = self._read_conversations(input_filepath)
        conversations = asyncio.run(self._infer(input, generation_config))
        if generation_config.output_filepath:
            self._save_conversations(conversations, generation_config.output_filepath)
        return conversations
