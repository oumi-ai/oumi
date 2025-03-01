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

from __future__ import annotations

import copy
import math

import torch
from typing_extensions import override

from oumi.builders import build_tokenizer
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.conversation_utils import create_list_of_message_json_dicts
from oumi.utils.logging import logger
from oumi.utils.model_caching import get_local_filepath_for_gguf
from oumi.utils.peft_utils import get_lora_rank

try:
    import vllm  # pyright: ignore[reportMissingImports]
    from vllm.entrypoints.chat_utils import (  # pyright: ignore[reportMissingImports]
        ChatCompletionMessageParam,
    )
    from vllm.lora.request import LoRARequest  # pyright: ignore[reportMissingImports]
    from vllm.sampling_params import (  # pyright: ignore[reportMissingImports]
        GuidedDecodingParams as VLLMGuidedDecodingParams,
    )
    from vllm.sampling_params import (  # pyright: ignore[reportMissingImports]
        SamplingParams,
    )
except ModuleNotFoundError:
    vllm = None


class VLLMInferenceEngine(BaseInferenceEngine):
    """Engine for running vLLM inference locally."""

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: GenerationParams | None = None,
        tensor_parallel_size: int = -1,
        quantization: str | None = None,
        enable_prefix_caching: bool = True,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = True,
        max_num_seqs: int | None = None,
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            generation_params: The generation parameters to use for inference.
            tensor_parallel_size: The number of tensor parallel processes to use.
                If set to -1, we will use all the available GPUs.
            quantization: The quantization method to use for inference.
            enable_prefix_caching: Whether to enable prefix caching.
            gpu_memory_utilization: The fraction of available GPU memory the model's
                executor will use. It can range from 0 to 1. Defaults to 0.9, i.e.,
                (90%) memory utilization.
            enforce_eager: Whether to enforce eager execution. Defaults to True.
                If False, will use eager mode and CUDA graph in hybrid mode.
            max_num_seqs: Maximum number of sequences per iteration.
        """
        super().__init__(model_params=model_params, generation_params=generation_params)

        if not vllm:
            raise RuntimeError(
                "vLLM is not installed. "
                "Please install the GPU dependencies for this package."
            )

        if not (
            math.isfinite(gpu_memory_utilization)
            and gpu_memory_utilization > 0
            and gpu_memory_utilization <= 1.0
        ):
            raise ValueError(
                "GPU memory utilization must be within (0, 1]. Got "
                f"{gpu_memory_utilization}."
            )

        # Infer the `quantization` type from the model's kwargs.
        if model_params.model_kwargs:
            if not quantization:
                # Check if quantization is BitsAndBytes.
                bnb_quantization_kwargs = ["load_in_4bit", "load_in_8bit"]
                for key in bnb_quantization_kwargs:
                    if model_params.model_kwargs.get(key):
                        quantization = "bitsandbytes"
                        break
            if not quantization and model_params.model_kwargs.get("filename"):
                # Check if quantization is GGUF.
                gguf_filename = str(model_params.model_kwargs.get("filename"))
                if gguf_filename.lower().endswith(".gguf"):
                    quantization = "gguf"
                    if (
                        not model_params.tokenizer_name
                        or model_params.tokenizer_name == model_params.model_name
                    ):
                        raise ValueError(
                            "GGUF quantization with the VLLM engine requires that you "
                            "explicitly set the `tokenizer_name` in `model_params`."
                        )

        vllm_kwargs = {}

        # Set the proper VLLM keys for the quantization type.
        if quantization and quantization == "bitsandbytes":
            vllm_kwargs["load_format"] = "bitsandbytes"
            logger.info("VLLM engine loading a `bitsandbytes` quantized model.")
        elif quantization and quantization == "gguf":
            # Download the GGUF file from HuggingFace to a local cache.
            gguf_local_path = get_local_filepath_for_gguf(
                repo_id=model_params.model_name,
                filename=gguf_filename,
            )
            # Overwrite `model_name` with the locally cached GGUF model.
            model_params = copy.deepcopy(model_params)
            model_params.model_name = gguf_local_path
            logger.info("VLLM engine loading a `GGUF` quantized model.")

        if tensor_parallel_size <= 0:
            if torch.cuda.device_count() > 1:
                tensor_parallel_size = torch.cuda.device_count()
            else:
                tensor_parallel_size = 1

        self._lora_request = None
        if model_params.adapter_model:
            # ID should be unique for this adapter, but isn't enforced by vLLM.
            self._lora_request = LoRARequest(
                lora_name="oumi_lora_adapter",
                lora_int_id=1,
                lora_path=model_params.adapter_model,
            )
            logger.info(f"Loaded LoRA adapter: {model_params.adapter_model}")
            lora_rank = get_lora_rank(model_params.adapter_model)
            vllm_kwargs["max_lora_rank"] = lora_rank
            logger.info(f"Setting vLLM max LoRA rank to {lora_rank}")

        if max_num_seqs is not None:
            vllm_kwargs["max_num_seqs"] = max_num_seqs

        self._tokenizer = build_tokenizer(model_params)
        self._llm = vllm.LLM(
            model=model_params.model_name,
            tokenizer=model_params.tokenizer_name,
            trust_remote_code=model_params.trust_remote_code,
            dtype=model_params.torch_dtype_str,
            # TODO: these params should be settable via config,
            # but they don't belong to model_params
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            enable_lora=self._lora_request is not None,
            max_model_len=model_params.model_max_length,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            **vllm_kwargs,
        )
        # Ensure the tokenizer is set properly
        self._llm.set_tokenizer(self._tokenizer)

    def _convert_conversation_to_vllm_input(
        self, conversation: Conversation
    ) -> list[ChatCompletionMessageParam]:
        """Converts a conversation to a list of vllm input messages.

        Args:
            conversation: The conversation to convert.

        Returns:
            List[ChatCompletionMessageParam]: A list of vllm input messages.
        """
        result: list[ChatCompletionMessageParam] = []
        for json_dict in create_list_of_message_json_dicts(
            conversation.messages, group_adjacent_same_role_turns=True
        ):
            for key in ("role", "content"):
                if key not in json_dict:
                    raise RuntimeError(f"The required field '{key}' is missing!")
            if not isinstance(json_dict["content"], (str, list)):
                raise RuntimeError(
                    "The 'content' field must be `str` or `list`. "
                    f"Actual: {type(json_dict['content'])}."
                )
            result.append({"role": json_dict["role"], "content": json_dict["content"]})
        return result

    def _infer(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs model inference on the provided input.

        Documentation: https://docs.vllm.ai/en/stable/dev/sampling_params.html

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        generation_params = (
            inference_config.generation
            if inference_config and inference_config.generation
            else self._generation_params
        )

        if generation_params.guided_decoding is not None:
            guided_decoding = VLLMGuidedDecodingParams.from_optional(
                json=generation_params.guided_decoding.json,
                regex=generation_params.guided_decoding.regex,
                choice=generation_params.guided_decoding.choice,
            )
        else:
            guided_decoding = None

        sampling_params = SamplingParams(
            n=1,
            max_tokens=generation_params.max_new_tokens,
            temperature=generation_params.temperature,
            top_p=generation_params.top_p,
            frequency_penalty=generation_params.frequency_penalty,
            presence_penalty=generation_params.presence_penalty,
            stop=generation_params.stop_strings,
            stop_token_ids=generation_params.stop_token_ids,
            min_p=generation_params.min_p,
            guided_decoding=guided_decoding,
        )

        output_conversations = []
        vllm_conversations = []
        non_skipped_conversations = []
        for conversation in input:
            if not conversation.messages:
                logger.warning("Conversation must have at least one message.")
                continue
            vllm_input = self._convert_conversation_to_vllm_input(conversation)
            vllm_conversations.append(vllm_input)
            non_skipped_conversations.append(conversation)

        if len(vllm_conversations) == 0:
            return []

        enable_tqdm = len(vllm_conversations) >= 2

        # Note: vLLM performs continuous batching under the hood.
        # We pass all the conversations and let vLLM handle the rest.
        chat_responses = self._llm.chat(
            vllm_conversations,
            sampling_params=sampling_params,
            lora_request=self._lora_request,
            use_tqdm=enable_tqdm,
            chat_template=None,
            chat_template_content_format="auto",
        )

        for conversation, chat_response in zip(
            non_skipped_conversations, chat_responses
        ):
            new_messages = [
                Message(content=message.text, role=Role.ASSISTANT)
                for message in chat_response.outputs
                if len(chat_response.outputs) > 0
            ]
            messages = [
                *conversation.messages,
                *new_messages,
            ]
            new_conversation = Conversation(
                messages=messages,
                metadata=conversation.metadata,
                conversation_id=conversation.conversation_id,
            )
            output_conversations.append(new_conversation)

        if inference_config and inference_config.output_path:
            self._save_conversations(
                output_conversations,
                inference_config.output_path,
            )
        return output_conversations

    @override
    def infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        return self._infer(input, inference_config)

    @override
    def infer_from_file(
        self,
        input_filepath: str,
        inference_config: InferenceConfig | None = None,
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
        input = self._read_conversations(input_filepath)
        return self._infer(input, inference_config)

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "guided_decoding",
            "max_new_tokens",
            "min_p",
            "presence_penalty",
            "stop_strings",
            "stop_token_ids",
            "temperature",
            "top_p",
        }
