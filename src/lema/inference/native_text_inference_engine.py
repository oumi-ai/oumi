from typing import List

import peft
import torch
from tqdm import tqdm
from transformers import BatchEncoding

from lema.builders import (
    build_model,
    build_tokenizer,
)
from lema.core.configs import GenerationConfig, ModelParams
from lema.core.inference import BaseInferenceEngine
from lema.core.types.turn import Conversation, Message, Role


class NativeTextInferenceEngine(BaseInferenceEngine):
    """Engine for running text-to-text model inference."""

    def __init__(self, model_params: ModelParams):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
        """
        self._model = build_model(model_params)
        self._tokenizer = build_tokenizer(model_params)
        self._model_params = model_params

    def _make_batches(self, input: List[str], batch_size: int) -> List[List[str]]:
        """Splits the input into batches of the specified size.

        Args:
            input: A list of text prompts.
            batch_size: The number of sequences to generate in parallel.

        Returns:
            List[List[str]]: A list of batches of text prompts.
        """
        return [input[i : i + batch_size] for i in range(0, len(input), batch_size)]

    def _infer(
        self,
        input: List[Conversation],
        generation_config: GenerationConfig,
    ) -> List[Conversation]:
        """Runs batch inference for a model using the provided configuration.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during inference.

        Returns:
            object: A list of model responses of shape (num_batches, batch_size).
        """
        if generation_config.batch_size < 1:
            raise ValueError("Batch size must be greater than or equal to 1.")
        if isinstance(self._model, peft.PeftModel):
            raise NotImplementedError(
                "Inference does not work yet for pretrained PEFT models."
            )
        model_device = next(self._model.parameters()).device
        formatted_input: List[str] = [
            self._tokenizer.apply_chat_template(
                conversation,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in input
        ]
        # Tokenization of input (in place, batch mode).
        batched_input = self._make_batches(
            formatted_input, generation_config.batch_size
        )
        input_batches: List[BatchEncoding] = [BatchEncoding()] * len(batched_input)
        for batch_index, batch in enumerate(batched_input):
            batch_tokenized = self._tokenizer(batch, return_tensors="pt", padding=True)
            batch_tokenized = batch_tokenized.to(model_device)
            input_batches[batch_index] = batch_tokenized

        # Generate model outputs (batch mode).
        output = []
        for batch_index in tqdm(
            range(len(input_batches)), desc="Generating Model Responses"
        ):
            batch = input_batches[batch_index]
            output.append(
                self._model.generate(
                    **batch, max_new_tokens=generation_config.max_new_tokens
                )
            )

        # Decode the outputs (batch mode).
        output_decoded = []
        for batch_index, batch in enumerate(output):
            # For each batch, remove the prepended prompts from all model reponses.
            if generation_config.exclude_prompt_from_response:
                new_batch_data = []
                for reponse_index, response in enumerate(batch.data):
                    prompt = input_batches[batch_index]["input_ids"][reponse_index]  # type: ignore
                    assert prompt.tolist() == response[: len(prompt)].tolist()
                    new_batch_data.append(response[len(prompt) :])
                batch.data = torch.stack(new_batch_data, dim=0)

            output_decoded.append(
                self._tokenizer.batch_decode(
                    batch.data,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )
        flat_output = [item for sublist in output_decoded for item in sublist]
        output_conversations = []
        for conversation, response in zip(input, flat_output):
            messages = [
                *conversation.messages,
                Message(role=Role.ASSISTANT, content=response),
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
            generation_config: Configuration parameters for generation during inference.

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

        This is a convenience method to prevent boilerplate from asserting the existence
        of input_filepath in the generation_config.

        Args:
            input_filepath: Path to the input file containing prompts for generation.
            generation_config: Configuration parameters for generation during inference.

        Returns:
            List[Conversation]: Inference output.
        """
        input = self._read_conversations(input_filepath)
        conversations = self._infer(input, generation_config)
        if generation_config.output_filepath:
            self._save_conversations(conversations, generation_config.output_filepath)
        return conversations
