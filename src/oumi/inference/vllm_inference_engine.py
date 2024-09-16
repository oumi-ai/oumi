from __future__ import annotations

from oumi.builders import build_tokenizer
from oumi.core.configs import GenerationConfig, ModelParams
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
        self, input: list[Conversation], generation_config: GenerationConfig
    ) -> list[Conversation]:
        """Runs model inference on the provided input.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during
                inference.

        Returns:
            List[Conversation]: Inference output.
        """
        output_conversations = []
        sampling_params = SamplingParams(
            n=1, max_tokens=generation_config.max_new_tokens
        )
        for conversation in input:
            if not conversation.messages:
                logger.warn("Conversation must have at least one message.")
                continue
            vllm_input = self._convert_conversation_to_vllm_input(conversation)
            chat_response = self._llm.chat(vllm_input, sampling_params=sampling_params)
            new_messages = [
                Message(content=message.outputs[0].text, role=Role.ASSISTANT)
                for message in chat_response
                if len(message.outputs) > 0
            ]
            messages = [
                *conversation.messages,
                *new_messages,
            ]
            output_conversations.append(
                Conversation(
                    messages=messages,
                    metadata=conversation.metadata,
                    conversation_id=conversation.conversation_id,
                )
            )
        return output_conversations

    def infer_online(
        self, input: list[Conversation], generation_config: GenerationConfig
    ) -> list[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during
                inference.

        Returns:
            List[Conversation]: Inference output.
        """
        conversations = self._infer(input, generation_config)
        if generation_config.output_filepath:
            self._save_conversations(conversations, generation_config.output_filepath)
        return conversations

    def infer_from_file(
        self, input_filepath: str, generation_config: GenerationConfig
    ) -> list[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the
        existence of input_filepath in the generation_config.

        Args:
            input_filepath: Path to the input file containing prompts for
                generation.
            generation_config: Configuration parameters for generation during
                inference.

        Returns:
            List[Conversation]: Inference output.
        """
        input = self._read_conversations(input_filepath)
        conversations = self._infer(input, generation_config)
        if generation_config.output_filepath:
            self._save_conversations(conversations, generation_config.output_filepath)
        return conversations
