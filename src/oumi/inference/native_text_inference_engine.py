from typing import List

import peft
import torch
import transformers
from tqdm import tqdm
from transformers import BatchEncoding

from oumi.builders import (
    build_model,
    build_tokenizer,
)
from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role


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

    def _make_batches(
        self, input: List[Conversation], batch_size: int
    ) -> List[List[Conversation]]:
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
        generation_params: GenerationParams,
    ) -> List[Conversation]:
        """Runs batch inference for a model using the provided configuration.

        Args:
            input: A list of conversations to run inference on.
            generation_params: Parameters for generation during inference.

        Returns:
            object: A list of model responses of shape (num_batches, batch_size).
        """
        if generation_params.batch_size < 1:
            raise ValueError("Batch size must be greater than or equal to 1.")
        if isinstance(self._model, peft.PeftModel):
            raise NotImplementedError(
                "Inference does not work yet for pretrained PEFT models."
            )
        model_device = next(self._model.parameters()).device
        batched_input = self._make_batches(input, generation_params.batch_size)
        batched_formatted_input: List[List[str]] = [
            [
                self._tokenizer.apply_chat_template(
                    conversation,  # type: ignore
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for conversation in batch
            ]
            for batch in batched_input
        ]
        input_batches: List[BatchEncoding] = [BatchEncoding()] * len(
            batched_formatted_input
        )
        for batch_index, batch in enumerate(batched_formatted_input):
            batch_tokenized = self._tokenizer(batch, return_tensors="pt", padding=True)
            batch_tokenized = batch_tokenized.to(model_device)
            input_batches[batch_index] = batch_tokenized

        # Create a GenerationConfig object with the new parameters
        # Documentation: https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
        generation_config = transformers.GenerationConfig(
            max_new_tokens=generation_params.max_new_tokens,
            temperature=generation_params.temperature,
            top_p=generation_params.top_p,
            frequency_penalty=generation_params.frequency_penalty,
            presence_penalty=generation_params.presence_penalty,
            do_sample=generation_params.temperature > 0,
            min_p=generation_params.min_p,
            stop=generation_params.stop,
            include_stop_str_in_output=False,
            detokenize=True,
            seed=generation_params.seed,
        )

        # Generate model outputs (batch mode).
        output_conversations = []
        for batch_index in tqdm(
            range(len(input_batches)), desc="Generating Model Responses"
        ):
            batch = input_batches[batch_index]
            output_batch = self._model.generate(
                **batch, generation_config=generation_config
            )

            # For each batch, remove the prepended prompts from all model reponses.
            if generation_params.exclude_prompt_from_response:
                new_batch_data = []
                for response_index, response in enumerate(output_batch.data):
                    prompt = input_batches[batch_index]["input_ids"][response_index]  # type: ignore
                    assert prompt.tolist() == response[: len(prompt)].tolist()
                    new_batch_data.append(response[len(prompt) :])
                output_batch.data = torch.stack(new_batch_data, dim=0)

            output_batch_decoded = self._tokenizer.batch_decode(
                output_batch.data,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for conversation, response in zip(
                batched_input[batch_index], output_batch_decoded
            ):
                messages = [
                    *conversation.messages,
                    Message(role=Role.ASSISTANT, content=response),
                ]
                new_conversation = Conversation(
                    messages=messages,
                    metadata=conversation.metadata,
                    conversation_id=conversation.conversation_id,
                )
                if generation_params.output_filepath:
                    self._save_conversation(
                        new_conversation, generation_params.output_filepath
                    )
                output_conversations.append(new_conversation)

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

        This is a convenience method to prevent boilerplate from asserting the existence
        of input_filepath in the generation_params.

        Args:
            input_filepath: Path to the input file containing prompts for generation.
            generation_params: Parameters for generation during inference.

        Returns:
            List[Conversation]: Inference output.
        """
        input = self._read_conversations(input_filepath)
        return self._infer(input, generation_params)
