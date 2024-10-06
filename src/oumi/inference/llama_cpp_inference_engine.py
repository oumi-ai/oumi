from pathlib import Path
from typing import Dict, List, cast

from tqdm.auto import tqdm

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role
from oumi.utils.logging import logger

try:
    from llama_cpp import Llama  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    Llama = None


class LlamaCppInferenceEngine(BaseInferenceEngine):
    """Engine for running llama.cpp inference locally.

    This class provides an interface for running inference using the llama.cpp library
    on local hardware. It allows for efficient execution of large language models
    with quantization, kv-caching, prefix filling, ...

    Note:
        This engine requires the llama-cpp-python package to be installed.
        If not installed, it will raise a RuntimeError.

    Example:
        >>> from oumi.core.configs import ModelParams
        >>> model_params = ModelParams(
        ...     model_name="path/to/model.gguf",
        ...     model_kwargs={
        ...         "n_gpu_layers": -1,
        ...         "n_threads": 8,
        ...         "flash_attn": True
        ...     }
        ... )
        >>> engine = LlamaCppInferenceEngine(model_params)
        >>> # Use the engine for inference
    """

    def __init__(
        self,
        model_params: ModelParams,
    ):
        """Initializes the LlamaCppInferenceEngine.

        This method sets up the engine for running inference using llama.cpp.
        It loads the specified model and configures the inference parameters.

        Args:
            model_params (ModelParams): Parameters for the model, including the model
                name, maximum length, and any additional keyword arguments for model
                initialization.

        Raises:
            RuntimeError: If the llama-cpp-python package is not installed.
            ValueError: If the specified model file is not found.

        Note:
            This method automatically sets some default values for model initialization:
            - verbose: False (reduces log output for bulk inference)
            - n_gpu_layers: -1 (uses GPU acceleration for all layers if available)
            - n_threads: 4
            - filename: "*q8_0.gguf" (applies Q8 quantization by default)
            - flash_attn: True
            These defaults can be overridden by specifying them in
            `model_params.model_kwargs`.
        """
        if not Llama:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Please install it with 'pip install llama-cpp-python'."
            )

        # `model_max_length` is required by llama-cpp, but optional in our config
        # Use a default value if not set.
        if model_params.model_max_length is None:
            model_max_length = 4096
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
            logger.info(
                f"Loading model from Huggingface Hub: {model_params.model_name}."
            )
            self._llm = Llama.from_pretrained(
                repo_id=model_params.model_name, n_ctx=model_max_length, **kwargs
            )

    def _convert_conversation_to_llama_input(
        self, conversation: Conversation
    ) -> List[Dict[str, str]]:
        """Converts a conversation to a list of llama.cpp input messages."""
        return [
            {
                "content": message.content or "",
                "role": "user" if message.role == Role.USER else "assistant",
            }
            for message in conversation.messages
        ]

    def _infer(
        self, input: List[Conversation], generation_params: GenerationParams
    ) -> List[Conversation]:
        """Runs model inference on the provided input using llama.cpp.

        Args:
            input: A list of conversations to run inference on.
                Each conversation should contain at least one message.
            generation_params: Parameters for text generation during inference.

        Returns:
            List[Conversation]: A list of conversations with the model's responses
            appended. Each conversation in the output list corresponds to an input
            conversation, with an additional message from the assistant (model) added.
        """
        output_conversations = []

        # skip using a progress for single turns
        disable_tgdm = len(input) < 2

        for conversation in tqdm(input, disable=disable_tgdm):
            if not conversation.messages:
                logger.warn("Conversation must have at least one message.")
                # add the conversation to keep input and output the same length.
                output_conversations.append(conversation)
                continue

            llama_input = self._convert_conversation_to_llama_input(conversation)

            response = self._llm.create_chat_completion(
                messages=llama_input,  # type: ignore
                max_tokens=generation_params.max_new_tokens,
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
        self, input: List[Conversation], generation_params: GenerationParams
    ) -> List[Conversation]:
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
    ) -> List[Conversation]:
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
