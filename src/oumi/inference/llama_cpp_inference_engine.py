from typing import List

from oumi.core.configs import GenerationConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role
from oumi.utils.logging import logger

try:
    from llama_cpp import Llama  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    Llama = None


class LlamaCppInferenceEngine(BaseInferenceEngine):
    """Engine for running llama.cpp inference locally."""

    def __init__(
        self,
        model_params: ModelParams,
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            n_ctx: The context size to use for inference.
            n_gpu_layers: The number of GPU layers to use.
            n_threads: The number of threads to use for CPU inference.
        """
        if not Llama:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Please install it with 'pip install llama-cpp-python'."
            )
        self._model_params = model_params
        self._llm = Llama(
            model_path=model_params.model_name,
            n_ctx=model_params.model_max_length,
            n_gpu_layers=-1,  # run everything on gpu if available
        )

    def _convert_conversation_to_llama_input(
        self, conversation: Conversation
    ) -> List[dict]:
        """Converts a conversation to a list of llama.cpp input messages.

        Args:
            conversation: The conversation to convert.

        Returns:
            List[dict]: A list of llama.cpp input messages.
        """
        return [
            {
                "content": message.content or "",
                "role": "user" if message.role == Role.USER else "assistant",
            }
            for message in conversation.messages
        ]

    def _infer(
        self, input: List[Conversation], generation_config: GenerationConfig
    ) -> List[Conversation]:
        """Runs model inference on the provided input.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during
                inference.

        Returns:
            List[Conversation]: Inference output.
        """
        output_conversations = []
        for conversation in input:
            if not conversation.messages:
                logger.warn("Conversation must have at least one message.")
                continue
            llama_input = self._convert_conversation_to_llama_input(conversation)

            response = self._llm.create_chat_completion(
                messages=llama_input,
                max_tokens=generation_config.max_new_tokens,
            )

            new_message = Message(
                content=response["choices"][0]["message"]["content"],
                role=Role.ASSISTANT,
            )

            messages = [
                *conversation.messages,
                new_message,
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
        self, input: List[Conversation], generation_config: GenerationConfig
    ) -> List[Conversation]:
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
    ) -> List[Conversation]:
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
