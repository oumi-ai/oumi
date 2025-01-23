import copy
from typing import Any, Optional

from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation
from oumi.inference.gcp_inference_engine import (
    _convert_guided_decoding_config_to_api_input,
)
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class GoogleGeminiInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against Gemini API."""

    def __init__(
        self, model_params: ModelParams, remote_params: Optional[RemoteParams] = None
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            remote_params: Remote server params.
        """
        self._model = model_params.model_name

        if remote_params is None:
            self._remote_params = RemoteParams(
                api_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                api_key_env_varname="GEMINI_API_KEY",
            )
        else:
            self._remote_params = copy.deepcopy(remote_params)

    @override
    def _convert_conversation_to_api_input(
        self, conversation: Conversation, generation_params: GenerationParams
    ) -> dict[str, Any]:
        """Converts a conversation to an Gemini API input.

        Documentation: https://ai.google.dev/docs

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for generation during inference.

        Returns:
            Dict[str, Any]: A dictionary representing the Gemini input.
        """
        api_input = {
            "model": self._model,
            "messages": self._get_list_of_message_json_dicts(
                conversation.messages, group_adjacent_same_role_turns=True
            ),
            "max_completion_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
            "n": 1,  # Number of completions to generate for each prompt.
        }

        if generation_params.stop_strings:
            api_input["stop"] = generation_params.stop_strings

        if generation_params.guided_decoding:
            api_input["response_format"] = _convert_guided_decoding_config_to_api_input(
                generation_params.guided_decoding
            )

        return api_input

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "guided_decoding",
            "max_new_tokens",
            "stop_strings",
            "temperature",
            "top_p",
        }
