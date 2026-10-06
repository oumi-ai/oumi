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

import dataclasses
from typing import Any

import aiohttp
from typing_extensions import override

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation
from oumi.inference.adaptive_semaphore import PoliteAdaptiveSemaphore
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.utils.http import APIStatusError

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_TOKENIZE_PATH = "/tokenize"


class RemoteVLLMInferenceEngine(RemoteInferenceEngine):
    """Engine for running inference against Remote vLLM."""

    @property
    @override
    def base_url(self) -> str | None:
        """Return the default base URL for the Remote vLLM API."""
        return None

    @property
    @override
    def api_key_env_varname(self) -> str | None:
        """Return the default environment variable name for the Remote vLLM API key."""
        return None

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "logit_bias",
            "presence_penalty",
            "seed",
            "stop_strings",
            "stop_token_ids",
            "temperature",
            "top_p",
            "guided_decoding",
            "max_new_tokens",
            "min_p",
            "skip_special_tokens",
            "parallel_tool_calls",
            "tool_choice",
        }

    @override
    def _convert_conversation_to_api_input(
        self,
        conversation: Conversation,
        generation_params: GenerationParams,
        model_params: ModelParams,
    ) -> dict[str, Any]:
        """Converts a conversation to an OpenAI input.

        Documentation: https://platform.openai.com/docs/api-reference/chat/create

        Args:
            conversation: The conversation to convert.
            generation_params: Parameters for generation during inference.
            model_params: Model parameters to use during inference.

        Returns:
            Dict[str, Any]: A dictionary representing the OpenAI input.
        """
        if model_params.adapter_model:
            model = model_params.adapter_model
        else:
            model = model_params.model_name

        api_input = {
            "model": model,
            "messages": self._get_list_of_message_json_dicts(
                conversation.messages, group_adjacent_same_role_turns=True
            ),
            "max_tokens": generation_params.max_new_tokens,
            # "max_completion_tokens": generation_params.max_new_tokens,
            # Future transition instead of `max_tokens`. See https://github.com/vllm-project/vllm/issues/9845
            "temperature": generation_params.temperature,
            "frequency_penalty": generation_params.frequency_penalty,
            "presence_penalty": generation_params.presence_penalty,
            "n": 1,  # Number of completions to generate for each prompt.
            "seed": generation_params.seed,
            "logit_bias": generation_params.logit_bias,
        }

        if generation_params.top_p is not None:
            api_input["top_p"] = generation_params.top_p
        if generation_params.min_p:
            api_input["min_p"] = generation_params.min_p
        api_input["skip_special_tokens"] = generation_params.skip_special_tokens
        if model_params.chat_template_kwargs:
            api_input["chat_template_kwargs"] = model_params.chat_template_kwargs

        if generation_params.guided_decoding:
            structured_outputs: dict[str, Any] = {}
            if generation_params.guided_decoding.json:
                structured_outputs["json"] = generation_params.guided_decoding.json

            elif generation_params.guided_decoding.regex is not None:
                structured_outputs["regex"] = generation_params.guided_decoding.regex

            elif generation_params.guided_decoding.choice is not None:
                structured_outputs["choice"] = generation_params.guided_decoding.choice

            if structured_outputs:
                api_input["structured_outputs"] = structured_outputs
                # vLLM servers older than structured_outputs read guided_<kind>;
                # each version ignores the other's keys.
                for kind, value in structured_outputs.items():
                    api_input[f"guided_{kind}"] = value

        if generation_params.stop_strings:
            api_input["stop"] = generation_params.stop_strings
        if generation_params.stop_token_ids:
            api_input["stop_token_ids"] = generation_params.stop_token_ids

        self._add_tool_params_to_api_input(api_input, conversation, generation_params)

        return api_input

    @override
    async def _query_api(
        self,
        conversation: Conversation,
        semaphore: PoliteAdaptiveSemaphore,
        session: aiohttp.ClientSession,
        inference_config: InferenceConfig | None = None,
        *,
        persist_scratch: bool = True,
    ) -> Conversation:
        """Queries vLLM, stopping at the context limit as VLLMInferenceEngine does.

        vLLM rejects a request whose prompt plus ``max_tokens`` exceeds the model's
        context. Such a request is sent once more with ``max_tokens`` capped to the
        context the prompt leaves.
        """
        try:
            return await super()._query_api(
                conversation,
                semaphore,
                session,
                inference_config,
                persist_scratch=persist_scratch,
            )
        except APIStatusError as error:
            if error.status_code != 400:
                raise
            capped_config = await self._capped_to_context(
                conversation, session, inference_config
            )
            if capped_config is None:
                raise
        return await super()._query_api(
            conversation,
            semaphore,
            session,
            capped_config,
            persist_scratch=persist_scratch,
        )

    async def _capped_to_context(
        self,
        conversation: Conversation,
        session: aiohttp.ClientSession,
        inference_config: InferenceConfig | None,
    ) -> InferenceConfig | None:
        """Returns the config with max_new_tokens cut to the context left, if over."""
        config = inference_config or InferenceConfig(
            model=self._model_params,
            generation=self._generation_params,
            remote_params=self._remote_params,
        )
        remote_params = config.remote_params or self._remote_params
        api_url = remote_params.api_url or ""
        if not api_url.endswith(_CHAT_COMPLETIONS_PATH):
            return None
        api_input = self._convert_conversation_to_api_input(
            conversation, config.generation, config.model
        )
        tokenize_input = {
            key: api_input[key]
            for key in ("model", "messages", "tools", "chat_template_kwargs")
            if key in api_input
        }
        async with session.post(
            api_url.removesuffix(_CHAT_COMPLETIONS_PATH) + _TOKENIZE_PATH,
            json=tokenize_input,
            headers=self._get_request_headers(remote_params),
            timeout=remote_params.connection_timeout,
        ) as response:
            if response.status != 200:
                return None
            tokenized = await response.json()
        context_left = tokenized["max_model_len"] - tokenized["count"]
        if not 0 < context_left < config.generation.max_new_tokens:
            return None
        return dataclasses.replace(
            config,
            generation=dataclasses.replace(
                config.generation, max_new_tokens=context_left
            ),
        )

    @override
    def infer_batch(
        self,
        _conversations: list[Conversation],
        _inference_config: InferenceConfig | None = None,
    ) -> str:
        """Batch inference is not implemented for Remote vLLM."""
        raise NotImplementedError(
            "Batch inference is not implemented for Remote vLLM. "
            "Please open an issue on GitHub if you'd like this feature."
        )
