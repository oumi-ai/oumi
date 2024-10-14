from __future__ import annotations

from oumi.builders import build_tokenizer
from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role
from oumi.utils.logging import logger

try:
    import vllm  # pyright: ignore[reportMissingImports]
    from vllm.entrypoints.chat_utils import (  # pyright: ignore[reportMissingImports]
        ChatCompletionMessageParam,
    )
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
        tensor_parallel_size: int = 1,
        quantization: str | None = None,
        enable_prefix_caching: bool = False,
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            tensor_parallel_size: The number of tensor parallel processes to use.
            quantization: The quantization method to use for inference.
            enable_prefix_caching: Whether to enable prefix caching.
        """
        if not vllm:
            raise RuntimeError(
                "vLLM is not installed. "
                "Please install the GPU dependencies for this package."
            )
        self._lora_request = None
        if model_params.adapter_model:
            # ID should be unique for this adapter, but isn't enforced by vLLM.
            self._lora_request = vllm.lora.request.LoRARequest(
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
        self, input: list[Conversation], generation_params: GenerationParams
    ) -> list[Conversation]:
        """Runs model inference on the provided input.

        Documentation: https://docs.vllm.ai/en/stable/dev/sampling_params.html

        Args:
            input: A list of conversations to run inference on.
            generation_params: Parameters for generation during inference.

        Returns:
            List[Conversation]: Inference output.
        """
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

        for conversation in input:
            if not conversation.messages:
                logger.warning("Conversation must have at least one message.")
                continue
            vllm_input = self._convert_conversation_to_vllm_input(conversation)
            chat_response = self._llm.chat(
                vllm_input,
                sampling_params=sampling_params,
                lora_request=self._lora_request,
            )
            new_messages = [
                Message(content=message.outputs[0].text, role=Role.ASSISTANT)
                for message in chat_response
                if len(message.outputs) > 0
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
            if generation_params.output_filepath:
                self._save_conversation(
                    new_conversation,
                    generation_params.output_filepath,
                )
        return output_conversations

    def infer_online(
        self, input: list[Conversation], generation_params: GenerationParams
    ) -> list[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            generation_params: Parameters for generation during inference.

        Returns:
            List[Conversation]: Inference output.
        """
        return self._infer(input, generation_params)

    def infer_from_file(
        self, input_filepath: str, generation_params: GenerationParams
    ) -> list[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the
        existence of input_filepath in the generation_params.

        Args:
            input_filepath: Path to the input file containing prompts for
                generation.
            generation_params: Parameters for generation during inference.

        Returns:
            List[Conversation]: Inference output.
        """
        input = self._read_conversations(input_filepath)
        return self._infer(input, generation_params)
