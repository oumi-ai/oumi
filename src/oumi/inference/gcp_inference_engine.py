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

import json
import os
from typing import Any, Optional

import pydantic
from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.types.conversation import Conversation
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class GoogleVertexInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against Google Vertex AI."""

    _API_URL_TEMPLATE = (
        "https://{region}-aiplatform.googleapis.com/v1beta1/projects/"
        "{project_id}/locations/{region}/endpoints/openapi/chat/completions"
    )
    """The API URL template for the GCP project. Used when no `api_url` is provided."""

    _DEFAULT_PROJECT_ID_ENV_KEY: str = "PROJECT_ID"
    """The default project ID environment key for the GCP project."""

    _DEFAULT_REGION_ENV_KEY: str = "REGION"
    """The default region environment key for the GCP project."""

    _project_id: Optional[str] = None
    """The project ID for the GCP project."""

    _region: Optional[str] = None
    """The region for the GCP project."""

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: Optional[GenerationParams] = None,
        remote_params: Optional[RemoteParams] = None,
        project_id_env_key: Optional[str] = None,
        region_env_key: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            generation_params: The generation parameters to use for inference.
            remote_params: The remote parameters to use for inference.
            project_id_env_key: The environment variable key name for the project ID.
            region_env_key: The environment variable key name for the region.
            project_id: The project ID to use for inference.
            region: The region to use for inference.
        """
        super().__init__(
            model_params=model_params,
            generation_params=generation_params,
            remote_params=remote_params,
        )
        if project_id and project_id_env_key:
            raise ValueError(
                "You cannot set both `project_id` and `project_id_env_key`."
            )
        if region and region_env_key:
            raise ValueError("You cannot set both `region` and `region_env_key`.")

        self._project_id_env_key = (
            project_id_env_key or self._DEFAULT_PROJECT_ID_ENV_KEY
        )
        self._region_env_key = region_env_key or self._DEFAULT_REGION_ENV_KEY
        self._project_id = project_id
        self._region = region

    @override
    def _set_required_fields_for_inference(self, remote_params: RemoteParams) -> None:
        """Set required fields for inference."""
        if (
            not remote_params.api_url
            and not self._remote_params.api_url
            and not self.base_url
        ):
            if self._project_id and self._region:
                project_id = self._project_id
                region = self._region
            elif os.getenv(self._project_id_env_key) and os.getenv(
                self._region_env_key
            ):
                project_id = os.getenv(self._project_id_env_key)
                region = os.getenv(self._region_env_key)
            else:
                raise ValueError(
                    "This inference engine requires that either `api_url` is set in "
                    "`RemoteParams` or that both `project_id` and `region` are set. "
                    "You can set the `project_id` and `region` when "
                    "constructing a GoogleVertexInferenceEngine, "
                    f"or as environment variables: `{self._project_id_env_key}` and "
                    f"`{self._region_env_key}`."
                )

            remote_params.api_url = self._API_URL_TEMPLATE.format(
                project_id=project_id,
                region=region,
            )

        super()._set_required_fields_for_inference(remote_params)

    @override
    def _get_api_key(self, remote_params: RemoteParams) -> str:
        """Gets the authentication token for GCP."""
        try:
            from google.auth import default  # pyright: ignore[reportMissingImports]
            from google.auth.transport.requests import (  # pyright: ignore[reportMissingImports]
                Request,
            )
            from google.oauth2 import (  # pyright: ignore[reportMissingImports]
                service_account,
            )
        except ModuleNotFoundError:
            raise RuntimeError(
                "Google-auth is not installed. "
                "Please install oumi with GCP extra:`pip install oumi[gcp]`, "
                "or install google-auth with `pip install google-auth`."
            )

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

    @override
    def _get_request_headers(
        self, remote_params: Optional[RemoteParams]
    ) -> dict[str, str]:
        """Gets the request headers for GCP."""
        if not remote_params:
            raise ValueError("Remote params are required for GCP inference.")

        headers = {
            "Authorization": f"Bearer {self._get_api_key(remote_params)}",
            "Content-Type": "application/json",
        }
        return headers

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams(num_workers=10, politeness_policy=60.0)

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to an OpenAI input.

        Documentation: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for generation during inference.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary representing the Vertex input.
        """
        api_input = {
            "model": model_params.model_name,
            "messages": self._get_list_of_message_json_dicts(
                conversation.messages, group_adjacent_same_role_turns=True
            ),
            "max_completion_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
            "n": 1,  # Number of completions to generate for each prompt.
            "seed": generation_params.seed,
            "logit_bias": generation_params.logit_bias,
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
            "logit_bias",
            "max_new_tokens",
            "seed",
            "stop_strings",
            "temperature",
            "top_p",
        }


#
# Helper functions
#
def _convert_guided_decoding_config_to_api_input(
    guided_config: GuidedDecodingParams,
) -> dict:
    """Converts a guided decoding configuration to an API input."""
    if guided_config.json is None:
        raise ValueError(
            "Only JSON schema guided decoding is supported, got '%s'",
            guided_config,
        )

    json_schema = guided_config.json

    if isinstance(json_schema, type) and issubclass(json_schema, pydantic.BaseModel):
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

    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": _replace_refs_in_schema(schema_value),
        },
    }


def _replace_refs_in_schema(schema: dict) -> dict:
    """Replace $ref references in a JSON schema with their actual definitions.

    Args:
        schema: The JSON schema dictionary

    Returns:
        dict: Schema with all references replaced by their definitions and $defs removed
    """

    def _get_ref_value(ref: str) -> dict:
        # Remove the '#/' prefix and split into parts
        parts = ref.replace("#/", "").split("/")

        # Navigate through the schema to get the referenced value
        current = schema
        for part in parts:
            current = current[part]
        return current.copy()  # Return a copy to avoid modifying the original

    def _replace_refs(obj: dict) -> dict:
        if not isinstance(obj, dict):
            return obj

        result = {}
        for key, value in obj.items():
            if key == "$ref":
                # If we find a $ref, replace it with the actual value
                return _replace_refs(_get_ref_value(value))
            elif isinstance(value, dict):
                result[key] = _replace_refs(value)
            elif isinstance(value, list):
                result[key] = [
                    _replace_refs(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    # Replace all references first
    resolved = _replace_refs(schema.copy())

    # Remove the $defs key if it exists
    if "$defs" in resolved:
        del resolved["$defs"]

    return resolved
