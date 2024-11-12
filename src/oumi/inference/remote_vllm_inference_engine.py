from typing import Any, Optional

from typing_extensions import override

from oumi.core.configs import GenerationParams, RemoteParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine

_CONTENT_KEY: str = "content"
_ROLE_KEY: str = "role"


class RemoteVLLMInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against Remote vLLM."""

    @override
    def _get_api_key(self, remote_params: RemoteParams) -> str:
        """Gets the authentication token for the remote LLM."""
        return remote_params.api_key or ""

    def _get_request_headers(
        self, remote_params: Optional[RemoteParams]
    ) -> dict[str, str]:
        """Gets the request headers for the remote LLM."""
        if not remote_params:
            raise ValueError("Remote params are required for remote LLM inference.")

        headers = {
            "Authorization": f"Bearer {self._get_api_key(remote_params)}",
            "Content-Type": "application/json",
        }
        return headers

    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "logit_bias",
            "presence_penalty",
            "remote_params",
            "seed",
            "stop_strings",
            "temperature",
            "top_p",
        }

    @override
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
            # "max_completion_tokens": generation_params.max_new_tokens,
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
