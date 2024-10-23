from __future__ import annotations

import torch

from oumi.builders import build_tokenizer
from oumi.core.configs import InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger

try:
    import vllm  # pyright: ignore[reportMissingImports]
    from vllm.entrypoints.chat_utils import (  # pyright: ignore[reportMissingImports]
        ChatCompletionMessageParam,
    )
    from vllm.lora.request import LoRARequest  # pyright: ignore[reportMissingImports]
    from vllm.sampling_params import (  # pyright: ignore[reportMissingImports]
        SamplingParams,
    )
except ModuleNotFoundError:
    vllm = None


class VLLMInferenceEngine(BaseInferenceEngine):
    """Engine for running vllm inference locally."""

    def __init__(
        self,
        model_params: ModelParams,
        tensor_parallel_size: int = -1,
        quantization: str | None = None,
        enable_prefix_caching: bool = True,
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            tensor_parallel_size: The number of tensor parallel processes to use.
                If set to -1, we will use all the available GPUs.
            quantization: The quantization method to use for inference.
            enable_prefix_caching: Whether to enable prefix caching.
        """
        if not vllm:
            raise RuntimeError(
                "vLLM is not installed. "
                "Please install the GPU dependencies for this package."
            )

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
        self._tokenizer = build_tokenizer(model_params)
        self._model_params = model_params
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
        return [
            {
                "content": message.content or "",
                "role": message.role,
            }
            for message in conversation.messages
        ]

    def _infer(
        self, input: list[Conversation], inference_config: InferenceConfig
    ) -> list[Conversation]:
        """Runs model inference on the provided input.

        Documentation: https://docs.vllm.ai/en/stable/dev/sampling_params.html

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        generation_params = inference_config.generation
        output_conversations = []
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
        )

        if generation_params.logit_bias:
            logger.warning(
                "VLLMInferenceEngine does not support logit_bias."
                " This parameter will be ignored."
            )

        if generation_params.batch_size > 1:
            logger.info(
                "VLLMInferenceEngine performs continuous batching under the hood. "
                "This parameter for static batching will be ignored."
            )

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

            if inference_config.output_path:
                self._save_conversation(
                    new_conversation,
                    inference_config.output_path,
                )
        return output_conversations

    def infer_online(
        self, input: list[Conversation], inference_config: InferenceConfig
    ) -> list[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        return self._infer(input, inference_config)

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
        input = self._read_conversations(input_filepath)
        return self._infer(input, inference_config)
