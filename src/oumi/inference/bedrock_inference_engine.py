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

import asyncio
import os
from typing import Any, Optional

from tqdm.asyncio import tqdm
from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.adaptive_semaphore import PoliteAdaptiveSemaphore
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.logging import logger

try:
    import boto3  # pyright: ignore[reportMissingImports]
    from botocore.exceptions import ClientError  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    boto3 = None  # type: ignore
    ClientError = None  # type: ignore

_CONTENT_KEY: str = "content"
_ROLE_KEY: str = "role"
_AWS_REGION_ENV_VAR: str = "AWS_REGION"


class BedrockInferenceEngine(RemoteInferenceEngine):
    """Inference engine for running inference against the Bedrock API.

    This class extends RemoteInferenceEngine to provide specific functionality
    for interacting with Bedrock's language models via their API. It handles
    the conversion of Oumi's Conversation objects to Bedrock's expected input
    format, as well as parsing the API responses back into Conversation objects.

    Note:
        This engine requires the boto3 package to be installed.
        If not installed, it will raise a RuntimeError.
    """

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: Optional[GenerationParams] = None,
        remote_params: Optional[RemoteParams] = None,
    ):
        """Initializes the BedrockInferenceEngine.

        Args:
            model_params: Parameters for the model.
            generation_params: Parameters for generation.
            remote_params: Parameters for remote inference.

        Raises:
            RuntimeError: If the boto3 package is not installed.
        """
        if not boto3:
            raise RuntimeError(
                "boto3 is not installed. Please install it with 'pip install boto3'."
            )

        super().__init__(
            model_params=model_params,
            generation_params=generation_params,
            remote_params=remote_params,
        )

    @property
    @override
    def base_url(self) -> Optional[str]:
        """Return the default base URL for the Bedrock API."""
        return None

    @property
    @override
    def api_key_env_varname(self) -> Optional[str]:
        """Return the default environment variable name for the Bedrock API key."""
        return None

    def _bedrock_client(self, remote_params: RemoteParams) -> Any:
        region = os.getenv(_AWS_REGION_ENV_VAR)
        if not region:
            raise ValueError(f"Environment variable {_AWS_REGION_ENV_VAR} not set.")
        return boto3.client("bedrock-runtime", region_name=region)  # type: ignore

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to a Bedrock API input.

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for text generation.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted input for the
                Bedrock API, including the model, messages, and generation parameters.
        """
        system_messages = [
            message for message in conversation.messages if message.role == Role.SYSTEM
        ]
        if len(system_messages) > 0:
            system_message = system_messages[0].content

            if len(system_messages) > 1:
                logger.warning(
                    "Multiple system messages found in conversation. "
                    "Only using the first one."
                )
        else:
            system_message = None

        messages = [
            message for message in conversation.messages if message.role != Role.SYSTEM
        ]

        body = {
            "inferenceConfig": {
                "maxTokens": generation_params.max_new_tokens,
                "temperature": generation_params.temperature,
                "topP": generation_params.top_p,
            },
            "messages": self._to_bedrock_messages(messages),
        }

        if system_message:
            body["system"] = [{"text": system_message}]

        if model_params.model_kwargs:
            body["additionalModelRequestFields"] = model_params.model_kwargs

        if generation_params.stop_strings:
            body["inferenceConfig"]["stopSequences"] = generation_params.stop_strings

        return body

    def _to_bedrock_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for m in messages:
            # Map Oumi roles to Bedrock roles
            if m.role == Role.USER:
                bedrock_role = "user"
            elif m.role == Role.ASSISTANT:
                bedrock_role = "assistant"
            else:
                logger.warning(f"Skipping message with unsupported role: {m.role}")
                continue

            content_blocks: list[dict] = []
            if m.contains_single_text_content_item_only():
                content_blocks = [{"text": m.text_content_items[0].content}]
            elif m.contains_images():
                img = m.image_content_items[0]
                if getattr(img, "binary", None):
                    content_blocks = [{"image": {"source": {"bytes": img.binary}}}]
                else:
                    uri = img.content or ""
                    if uri.startswith("s3://"):
                        content_blocks = [
                            {"image": {"source": {"s3Location": {"uri": uri}}}}
                        ]
                    else:
                        content_blocks = [{"image": {"source": {"url": uri}}}]
            else:
                content_blocks = [{"text": m.compute_flattened_text_content()}]

            result.append({"role": bedrock_role, "content": content_blocks})
        return result

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams()

    @override
    def _set_required_fields_for_inference(self, remote_params: RemoteParams):
        """Override to skip API key validation since Bedrock uses AWS credentials."""
        pass

    async def _infer(
        self,
        input: list[Conversation],
        inference_config: Optional[Any] = None,
    ) -> list[Conversation]:
        """Async inference implementation that doesn't use HTTP sessions."""
        semaphore = PoliteAdaptiveSemaphore(
            capacity=self._remote_params.num_workers,
            politeness_policy=self._remote_params.politeness_policy,
        )

        # Create tasks for all conversations
        tasks = [
            self._query_api(
                conversation,
                semaphore,
                None,  # No HTTP session needed for boto3
                inference_config=inference_config,
            )
            for conversation in input
        ]

        disable_tqdm = len(tasks) < 2
        results = await tqdm.gather(*tasks, disable=disable_tqdm)
        return results

    def _call_bedrock_converse(
        self,
        remote_params: RemoteParams,
        model_params: ModelParams,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Synchronously invokes Bedrock Converse via boto3."""
        client = self._bedrock_client(remote_params)
        kwargs: dict[str, Any] = {
            "modelId": model_params.model_name,
            "messages": body["messages"],
        }
        if "system" in body:
            kwargs["system"] = body["system"]
        if "inferenceConfig" in body:
            kwargs["inferenceConfig"] = body["inferenceConfig"]
        if "additionalModelRequestFields" in body:
            kwargs["additionalModelRequestFields"] = body[
                "additionalModelRequestFields"
            ]
        return client.converse(**kwargs)

    @override
    async def _query_api(
        self,
        conversation: Conversation,
        semaphore: PoliteAdaptiveSemaphore,
        session: Any,
        inference_config: Optional[Any] = None,
    ) -> Conversation:
        """Queries Bedrock Converse using boto3 instead of HTTP."""
        if inference_config is None:
            remote_params = self._remote_params
            generation_params = self._generation_params
            model_params = self._model_params
            output_path = None
        else:
            remote_params = inference_config.remote_params or self._remote_params
            generation_params = inference_config.generation or self._generation_params
            model_params = inference_config.model or self._model_params
            output_path = inference_config.output_path

        semaphore_or_controller = (
            self._adaptive_concurrency_controller
            if self._remote_params.use_adaptive_concurrency
            else semaphore
        )
        async with semaphore_or_controller:
            api_input = self._convert_conversation_to_api_input(
                conversation, generation_params, model_params
            )
            failure_reason: Optional[str] = None
            for attempt in range(remote_params.max_retries + 1):
                try:
                    if attempt > 0:
                        delay = min(
                            remote_params.retry_backoff_base * (2 ** (attempt - 1)),
                            remote_params.retry_backoff_max,
                        )
                        await asyncio.sleep(delay)

                    response = await asyncio.to_thread(
                        self._call_bedrock_converse,
                        remote_params,
                        model_params,
                        api_input,
                    )
                    result = self._convert_api_output_to_conversation(
                        response, conversation
                    )
                    if output_path:
                        self._save_conversation_to_scratch(result, output_path)
                    await self._try_record_success()
                    return result
                except ClientError as e:  # type: ignore
                    # Capture AWS error message for logging/propagation
                    failure_reason = e.response.get("Error", {}).get("Message") or str(
                        e
                    )
                    await self._try_record_error()
                    if attempt >= remote_params.max_retries:
                        raise RuntimeError(failure_reason) from e
                    continue
                except RuntimeError:
                    raise
                except Exception as e:
                    failure_reason = f"Unexpected error: {str(e)}"
                    await self._try_record_error()
                    if attempt >= remote_params.max_retries:
                        raise RuntimeError(failure_reason) from e
                    continue

            raise RuntimeError(
                f"Failed to query Bedrock after {remote_params.max_retries} retries. "
                + (f"Reason: {failure_reason}" if failure_reason else "")
            )

    @override
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original: Conversation
    ) -> Conversation:
        text = ""
        msg = response.get("output", {}).get("message", {})
        for block in msg.get("content", []):
            if "text" in block:
                text += block["text"]
        new_message = Message(content=text, role=Role.ASSISTANT)
        return Conversation(
            messages=[*original.messages, new_message],
            metadata=original.metadata,
            conversation_id=original.conversation_id,
        )

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "max_new_tokens",
            "stop_strings",
            "temperature",
            "top_p",
        }

    # Override batch inference methods to indicate they're not supported
    @override
    def infer_batch(
        self,
        conversations: list[Conversation],
        inference_config: Optional[Any] = None,
    ) -> str:
        """Bedrock does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError("Batch inference is not supported for Bedrock API.")

    @override
    def get_batch_status(self, batch_id: str) -> Any:
        """Bedrock does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError("Batch inference is not supported for Bedrock API.")

    @override
    def list_batches(
        self,
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Any:
        """Bedrock does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError("Batch inference is not supported for Bedrock API.")

    @override
    def get_batch_results(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> list[Conversation]:
        """Bedrock does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError("Batch inference is not supported for Bedrock API.")
