from typing import List, Optional

import vllm
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.sampling_params import SamplingParams

from lema.builders import (
    build_tokenizer,
)
from lema.core.configs import GenerationConfig, ModelParams
from lema.core.inference import BaseInferenceEngine
from lema.core.types.turn import Conversation, Message, Role
from lema.utils.logging import logger


class VLLMInferenceEngine(BaseInferenceEngine):
    """Engine for running vllm inference locally."""

    def __init__(self, model: str, model_params: ModelParams):
        """Initializes the inference Engine.

        Args:
            model: The model to use for inference. This can be a model name or a path to
                a model directory.
            model_params: The model parameters to use for inference.
        """
        self._model = model
        self._tokenizer = build_tokenizer(model_params)
        self._model_params = model_params
        self._llm = vllm.LLM(
            model=model,
            tokenizer=model_params.tokenizer_name,
            trust_remote_code=model_params.trust_remote_code,
            dtype=model_params.torch_dtype_str,
        )
        # Ensure the tokenizer is set properly
        self._llm.set_tokenizer(self._tokenizer)

    def _convert_conversation_to_vllm_input(
        self, conversation: Conversation
    ) -> List[ChatCompletionMessageParam]:
        """Converts a conversation to a list of vllm input messages.

        Args:
            conversation: The conversation to convert.

        Returns:
            List[ChatCompletionMessageParam]: A list of vllm input messages.
        """
        return [
            {
                "content": message.content,
                "role": message.role,
            }
            for message in conversation.messages
        ]

    def _infer(
        self, input: List[Conversation], generation_config: GenerationConfig
    ) -> List[Conversation]:
        """Runs model inference on the provided input.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during inference.

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
        self, input: List[Conversation], generation_config: GenerationConfig
    ) -> Optional[List[Conversation]]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during inference.

        Returns:
            Optional[List[Conversation]]: Inference output. Returns None if the output
                is written to a file.
        """
        conversations = self._infer(input, generation_config)
        if generation_config.output_filepath:
            self._save_conversations(conversations, generation_config.output_filepath)
        return conversations

    def infer_from_file(
        self, input_filepath: str, generation_config: GenerationConfig
    ) -> Optional[List[Conversation]]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the existence
        of input_filepath in the generation_config.

        Args:
            input_filepath: Path to the input file containing prompts for generation.
            generation_config: Configuration parameters for generation during inference.

        Returns:
            Optional[List[Conversation]]: Inference output. Returns None if the output
                is written to a file.
        """
        input = self._read_conversations(input_filepath)
        conversations = self._infer(input, generation_config)
        if generation_config.output_filepath:
            self._save_conversations(conversations, generation_config.output_filepath)
        return conversations
