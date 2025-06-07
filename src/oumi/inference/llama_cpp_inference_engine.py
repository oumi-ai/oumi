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

from pathlib import Path
from typing import Optional, cast

from tqdm.auto import tqdm
from typing_extensions import override

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
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
        >>> from oumi.inference import LlamaCppInferenceEngine
        >>> model_params = ModelParams(
        ...     model_name="path/to/model.gguf",
        ...     model_kwargs={
        ...         "n_gpu_layers": -1,
        ...         "n_threads": 8,
        ...         "flash_attn": True
        ...     }
        ... )
        >>> engine = LlamaCppInferenceEngine(model_params) # doctest: +SKIP
        >>> # Use the engine for inference
    """

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: Optional[GenerationParams] = None,
    ):
        """Initializes the LlamaCppInferenceEngine.

        This method sets up the engine for running inference using llama.cpp.
        It loads the specified model and configures the inference parameters.

        Documentation: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion

        Args:
            model_params (ModelParams): Parameters for the model, including the model
                name, maximum length, and any additional keyword arguments for model
                initialization.
            generation_params (GenerationParams): Parameters for generation.

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
        super().__init__(model_params=model_params, generation_params=generation_params)

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
            "filename": "*8_0.gguf",
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
    ) -> list[dict[str, str]]:
        """Converts a conversation to a list of llama.cpp input messages."""
        # FIXME Handle multimodal e.g., raise an error.
        return [
            {
                "content": message.compute_flattened_text_content(),
                "role": "user" if message.role == Role.USER else "assistant",
            }
            for message in conversation.messages
        ]

    def _infer(
        self,
        input: list[Conversation],
        inference_config: Optional[InferenceConfig] = None,
    ) -> list[Conversation]:
        """Runs model inference on the provided input using llama.cpp.

        Args:
            input: A list of conversations to run inference on.
                Each conversation should contain at least one message.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: A list of conversations with the model's responses
            appended. Each conversation in the output list corresponds to an input
            conversation, with an additional message from the assistant (model) added.
        """
        generation_params = (
            inference_config.generation
            if inference_config and inference_config.generation
            else self._generation_params
        )
        output_conversations = []

        # skip using a progress for single turns
        disable_tgdm = len(input) < 2

        for conversation in tqdm(input, disable=disable_tgdm):
            if not conversation.messages:
                logger.warning("Conversation must have at least one message.")
                # add the conversation to keep input and output the same length.
                output_conversations.append(conversation)
                continue

            llama_input = self._convert_conversation_to_llama_input(conversation)

            response = self._llm.create_chat_completion(
                messages=llama_input,  # type: ignore
                max_tokens=generation_params.max_new_tokens,
                temperature=generation_params.temperature,
                top_p=generation_params.top_p,
                frequency_penalty=generation_params.frequency_penalty,
                presence_penalty=generation_params.presence_penalty,
                stop=generation_params.stop_strings,
                logit_bias=generation_params.logit_bias,
                min_p=generation_params.min_p,
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
            self._save_conversation_to_scratch(
                new_conversation,
                inference_config.output_path if inference_config else None,
            )

        self._cleanup_scratch_file(
            inference_config.output_path if inference_config else None
        )

        return output_conversations

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "logit_bias",
            "max_new_tokens",
            "min_p",
            "presence_penalty",
            "stop_strings",
            "temperature",
            "top_p",
        }

    @override
    def infer_online(
        self,
        input: list[Conversation],
        inference_config: Optional[InferenceConfig] = None,
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
        inference_config: Optional[InferenceConfig] = None,
    ) -> list[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the existence
        of input_filepath in the generation_params.

        Args:
            input_filepath: Path to the input file containing prompts for generation.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        input = self._read_conversations(input_filepath)
        return self._infer(input, inference_config)
