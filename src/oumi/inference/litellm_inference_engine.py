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

"""Inference engine for LiteLLM.

LiteLLM provides a unified interface to 100+ LLM providers (Anthropic, Bedrock,
Vertex AI, Cohere, Mistral, etc.) through a single ``completion()`` call.
The provider is specified via the model string in ``ModelParams.model_name``
(e.g. ``anthropic/claude-sonnet-4-5``, ``bedrock/anthropic.claude-v2``).
"""

import asyncio
from typing import TYPE_CHECKING, Any, cast

from tqdm.asyncio import tqdm
from typing_extensions import override

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation
from oumi.inference.adaptive_semaphore import PoliteAdaptiveSemaphore
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.logging import logger

if TYPE_CHECKING:
    from litellm import ModelResponse

try:
    import litellm  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    litellm = None  # type: ignore[assignment]


class LiteLLMInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference via the LiteLLM SDK.

    This class extends RemoteInferenceEngine to provide specific functionality
    for interacting with LLM providers through LiteLLM's unified interface.
    It handles the conversion of Oumi's Conversation objects to LiteLLM's
    expected input format, as well as parsing the API responses back into
    Conversation objects.

    LiteLLM routes requests to the correct provider based on the model string.
    For example, ``anthropic/claude-sonnet-4-5`` routes to Anthropic,
    ``bedrock/anthropic.claude-v2`` routes to AWS Bedrock, and
    ``vertex_ai/gemini-pro`` routes to Google Vertex AI.

    Authentication is handled via provider-specific environment variables
    (e.g. ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``AWS_ACCESS_KEY_ID``).
    LiteLLM reads these automatically based on the model prefix.

    For a full list of supported providers, see:
    https://docs.litellm.ai/docs/providers

    Note:
        This engine requires the litellm package to be installed.
        If not installed, it will raise a RuntimeError.

    Example:
        >>> from oumi.core.configs import ModelParams, GenerationParams
        >>> engine = LiteLLMInferenceEngine(
        ...     model_params=ModelParams(model_name="anthropic/claude-sonnet-4-5"),
        ...     generation_params=GenerationParams(max_new_tokens=500),
        ... )
    """

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: GenerationParams | None = None,
        remote_params: RemoteParams | None = None,
    ):
        """Initializes the LiteLLM inference engine.

        Args:
            model_params: The model parameters. ``model_name`` should use the
                LiteLLM model string format (e.g. ``anthropic/claude-sonnet-4-5``).
            generation_params: The generation parameters.
            remote_params: Parameters for remote inference.

        Raises:
            RuntimeError: If the ``litellm`` package is not installed.
        """
        if litellm is None:
            raise RuntimeError(
                "litellm is not installed. Install it with `pip install oumi[litellm]`."
            )

        super().__init__(
            model_params=model_params,
            generation_params=generation_params,
            remote_params=remote_params,
        )

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the LiteLLM engine."""
        return None

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the API key."""
        return None

    @override
    def _default_remote_params(self) -> RemoteParams:
        """Returns the default remote parameters."""
        return RemoteParams()

    @override
    def _set_required_fields_for_inference(self, remote_params: RemoteParams):
        """Override to skip API key validation.

        LiteLLM reads provider-specific environment variables automatically
        based on the model prefix (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY).
        """
        pass

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to a litellm.completion() kwargs dict.

        LiteLLM accepts the OpenAI wire format, so this reuses
        ``RemoteInferenceEngine``'s converter to inherit its handling of
        ``tools``, ``tool_choice`` and guided decoding (``response_format``),
        then drops keys whose value is ``None`` so providers that reject null
        fields do not receive them.

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for text generation.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted input for
                the LiteLLM completion call.
        """
        api_input = super()._convert_conversation_to_api_input(
            conversation, generation_params, model_params
        )
        return {key: value for key, value in api_input.items() if value is not None}

    def _apply_remote_params_to_api_input(
        self, api_input: dict[str, Any], remote_params: RemoteParams
    ) -> None:
        """Forwards RemoteParams credentials and timeout to litellm.completion().

        ``RemoteParams`` may carry an explicit key/endpoint (for example when
        routing through a LiteLLM proxy). Map ``api_key``/``api_key_env_varname``,
        ``api_url`` and ``connection_timeout`` onto LiteLLM's ``api_key``,
        ``api_base`` and ``timeout`` arguments so YAML-configured credentials and
        endpoint overrides are honored instead of silently ignored.

        Args:
            api_input: The litellm.completion() kwargs to mutate in place.
            remote_params: The remote parameters to read credentials from.
        """
        api_key = self._get_api_key(remote_params)
        if api_key:
            api_input["api_key"] = api_key
        if remote_params.api_url:
            api_input["api_base"] = remote_params.api_url
        if remote_params.connection_timeout:
            api_input["timeout"] = remote_params.connection_timeout

    def _call_litellm_completion(
        self,
        api_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Synchronously invokes litellm.completion.

        Args:
            api_input: The keyword arguments for litellm.completion().

        Returns:
            Dict[str, Any]: The response as a JSON-serializable dictionary.
        """
        assert litellm is not None  # guaranteed by __init__; narrows Optional
        # drop_params lets LiteLLM discard arguments a given provider does not
        # support (e.g. response_format or tools on a backend without them)
        # instead of erroring, which is what makes one engine span 100+ providers.
        # This engine never streams, so completion() returns a ModelResponse
        # rather than a CustomStreamWrapper.
        response = cast(
            "ModelResponse", litellm.completion(**api_input, drop_params=True)
        )
        return response.model_dump(mode="json")

    @override
    async def _infer(
        self,
        input: list[Conversation],
        inference_config: Any | None = None,
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
                None,  # No HTTP session needed for LiteLLM SDK
                inference_config=inference_config,
            )
            for conversation in input
        ]

        disable_tqdm = len(tasks) < 2
        results = await tqdm.gather(*tasks, disable=disable_tqdm)
        return results

    @override
    async def _query_api(
        self,
        conversation: Conversation,
        semaphore: PoliteAdaptiveSemaphore,
        session: Any,
        inference_config: Any | None = None,
    ) -> Conversation:
        """Queries LiteLLM using the SDK instead of HTTP.

        Args:
            conversation: The conversation to run inference on.
            semaphore: Semaphore to limit concurrent requests.
            session: Unused (LiteLLM manages its own connections).
            inference_config: Parameters for inference.

        Returns:
            Conversation: Inference output.
        """
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
        api_input = self._convert_conversation_to_api_input(
            conversation, generation_params, model_params
        )
        self._apply_remote_params_to_api_input(api_input, remote_params)

        failure_reason: str | None = None
        for attempt in range(remote_params.max_retries + 1):
            try:
                # Exponential backoff between attempts, done outside the
                # concurrency guard so a retrying request does not hold a worker
                # slot while it sleeps.
                if attempt > 0:
                    delay = min(
                        remote_params.retry_backoff_base * (2 ** (attempt - 1)),
                        remote_params.retry_backoff_max,
                    )
                    logger.warning(
                        "Retrying LiteLLM request after %.1fs; attempt %d/%d. "
                        "Reason: %s",
                        delay,
                        attempt + 1,
                        remote_params.max_retries + 1,
                        failure_reason,
                    )
                    await asyncio.sleep(delay)

                # Re-pace every attempt through the rate limiter so a burst of
                # retries waking together still respects the RPM/TPM window.
                if self._rate_limiter is not None:
                    await self._rate_limiter.wait_if_needed()

                async with semaphore_or_controller:
                    response = await asyncio.to_thread(
                        self._call_litellm_completion,
                        api_input,
                    )
                    # Record token usage before conversion so tokens are counted
                    # even if conversion fails.
                    if self._rate_limiter is not None:
                        usage = self._extract_usage_from_response(response)
                        if usage:
                            await self._rate_limiter.record_usage(
                                input_tokens=usage.get("prompt_tokens", 0),
                                output_tokens=usage.get("completion_tokens", 0),
                            )
                    result = self._convert_api_output_to_conversation(
                        response, conversation
                    )
                    # Persist progress unconditionally; the base helper selects a
                    # temporary scratch path when output_path is None, preserving
                    # resume/checkpoint so completed paid requests are not repeated.
                    self._save_conversation_to_scratch(result, output_path)
                    await self._try_record_success()
                    return result
            except RuntimeError:
                raise
            except Exception as e:
                failure_reason = f"LiteLLM error: {str(e)}"
                logger.warning(
                    "LiteLLMInferenceEngine attempt %d/%d failed: %s",
                    attempt + 1,
                    remote_params.max_retries + 1,
                    e,
                )
                await self._try_record_error()
                if attempt >= remote_params.max_retries:
                    raise RuntimeError(failure_reason) from e
                continue

        raise RuntimeError(
            f"Failed to query LiteLLM after {remote_params.max_retries} retries. "
            + (f"Reason: {failure_reason}" if failure_reason else "")
        )

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "guided_decoding",
            "max_new_tokens",
            "parallel_tool_calls",
            "presence_penalty",
            "seed",
            "stop_strings",
            "temperature",
            "tool_choice",
            "top_p",
        }

    @override
    def list_models(self, chat_only: bool = True) -> list[str]:
        """Returns model IDs available through LiteLLM.

        Args:
            chat_only: If True (default), only return chat models.

        Returns:
            list[str]: A sorted list of model ID strings.
        """
        if litellm is None:
            return []
        try:
            models = litellm.model_list or []
        except Exception:
            models = []
        return sorted(models)

    # Override batch inference methods to indicate they're not supported
    @override
    def infer_batch(
        self,
        conversations: list[Conversation],
        inference_config: Any | None = None,
    ) -> str:
        """LiteLLM does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError(
            "Batch inference is not supported for LiteLLM engine."
        )

    @override
    def get_batch_status(self, batch_id: str) -> Any:
        """LiteLLM does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError(
            "Batch inference is not supported for LiteLLM engine."
        )

    @override
    def list_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
    ) -> Any:
        """LiteLLM does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError(
            "Batch inference is not supported for LiteLLM engine."
        )

    @override
    def get_batch_results(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> list[Conversation]:
        """LiteLLM does not support batch inference via OpenAI-style batch API."""
        raise NotImplementedError(
            "Batch inference is not supported for LiteLLM engine."
        )
