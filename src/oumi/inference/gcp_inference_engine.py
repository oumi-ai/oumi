from typing import Any, Dict, Optional

from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from typing_extensions import override

from oumi.core.configs import GenerationParams, RemoteParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.logging import logger

_CONTENT_KEY: str = "content"
_ROLE_KEY: str = "role"


class GCPInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against Google Vertex AI."""

    @override
    def _get_api_key(self, remote_params: RemoteParams) -> str:
        """Gets the authentication token for GCP."""
        if remote_params.api_key:
            credentials = service_account.Credentials.from_service_account_file(
                filename=remote_params.api_key,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            credentials, _ = default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        credentials.refresh(Request())  # type: ignore
        return credentials.token  # type: ignore

    def _get_request_headers(
        self, remote_params: Optional[RemoteParams]
    ) -> Dict[str, str]:
        """Gets the request headers for GCP."""
        if not remote_params:
            raise ValueError("Remote params are required for GCP inference.")

        headers = {
            "Authorization": f"Bearer {self._get_api_key(remote_params)}",
            "Content-Type": "application/json",
        }
        return headers

    def _convert_conversation_to_api_input(
        self, conversation: Conversation, generation_params: GenerationParams
    ) -> Dict[str, Any]:
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
            "n": 1,  # Number of completions to generate for each prompt.
            "seed": generation_params.seed,
            "logit_bias": generation_params.logit_bias,
            "min_p": generation_params.min_p,
        }

        # Log warning for unsupported parameter
        if generation_params.min_p > 0.0:
            logger.warning(
                "GCPInferenceEngine does not support min_p. "
                f"Received value: min_p={generation_params.min_p}. "
                "This parameter will be ignored."
            )

        if generation_params.frequency_penalty:
            logger.warning(
                "GCPInferenceEngine does not support frequency_penalty. "
                f"Received value: "
                f"frequency_penalty={generation_params.frequency_penalty}. "
                "This parameter will be ignored."
            )

        if generation_params.presence_penalty:
            logger.warning(
                "GCPInferenceEngine does not support frequency_penalty. "
                "Received value: "
                f"presence_penalty={generation_params.presence_penalty}. "
            )

        if generation_params.stop_strings:
            api_input["stop"] = generation_params.stop_strings

        return api_input
