from __future__ import annotations

from pathlib import Path
from typing import cast

from tqdm.auto import tqdm

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

        # `model_max_length` is required by llama-cpp, but optional in our config
        # Use a default value if not set.
        if model_params.model_max_length is None:
            model_max_length = 4049
            logger.warning(
                "model_max_length is not set. "
                f"Using default value of {model_max_length}."
            )
        else:
            model_max_length = model_params.model_max_length

        # Set some reasonable defaults. These will be overriden by the user if set in
        # the config.
        kwargs = {
            # llama-cpp logs a lot of useful information,
            # but it's too verbose by default for bulk inference.
            "verbose": False,
            # Put all layers on GPU / MPS if available. Otherwise, will use CPU.
            "n_gpu_layers": -1,
            # Increase the default number of threads.
            # Too many can cause deadlocks
            "n_threads": 4,
            # Use Q8 quantization by default.
            "filename": "*q8_0.gguf",
            "flash_attn": True,
        }

        model_kwargs = model_params.model_kwargs.copy()
        kwargs.update(model_kwargs)

        # Load model
        if Path(model_params.model_name).exists():
            logger.info(f"Loading model from disk: {model_params.model_name}.")
            kwargs.pop("filename", None)  # only needed if downloading from hub
            self._llm = Llama(
                model_path=model_params.model_name, n_ctx=model_max_length, **kwargs
            )
        else:
            raise ValueError(
                f"Model not found at {model_params.model_name}. "
                "Please provide a valid model path."
            )
            # logger.info(
            #     f"Loading model from Huggingface Hub: {model_params.model_name}."
            # )
            # self._llm = Llama.from_pretrained(
            #     repo_id=model_params.model_name, n_ctx=model_max_length, **kwargs
            # )

    def _convert_conversation_to_llama_input(
        self, conversation: Conversation
    ) -> list[dict[str, str]]:
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

        disable_tgdm = len(input) < 2

        for conversation in tqdm(input, disable=disable_tgdm):
            if not conversation.messages:
                logger.warn("Conversation must have at least one message.")
                continue
            llama_input = self._convert_conversation_to_llama_input(conversation)

            response = self._llm.create_chat_completion(
                messages=llama_input,  # type: ignore
                max_tokens=generation_config.max_new_tokens,
            )
            response = cast(dict, response)

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
