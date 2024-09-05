from pathlib import Path
from typing import Any, List, Optional

import jsonlines
import peft
import torch
from tqdm import tqdm
from transformers import BatchEncoding

from lema.builders import (
    build_model,
    build_tokenizer,
)
from lema.core.configs import ModelParams
from lema.core.inference import BaseInferenceEngine
from lema.core.models import BaseModel
from lema.core.types.turn import Conversation, Message, Role


class NativeTextInferenceEngine(BaseInferenceEngine):
    """Engine for running text-to-text model inference."""

    def __init__(self, model_params: ModelParams, model: Optional[BaseModel] = None):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            model: The model to use for inference. If not specified, the model will be
                built using the provided `model_params`.
        """
        self.model = model if model is not None else build_model(model_params)
        self.tokenizer = build_tokenizer(model_params)

    def _save_messages(
        self, conversations: List[Conversation], output_filepath: str
    ) -> None:
        """Saves messages to a file in OpenAI chat format.

        Args:
            conversations: A list of conversations to save.
            output_filepath: The path to the file where the generations should be saved.
        """
        # Make the directory if it doesn't exist.
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(output_filepath, mode="w") as writer:
            for conversation in conversations:
                dict_messages = [
                    message.model_dump() for message in conversation.messages
                ]
                json_obj = {"messages": dict_messages}
                writer.write(json_obj)

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
        input: List[str],
        max_new_tokens: int,
        batch_size: int = 2,
        exclude_prompt_from_response: bool = True,
    ) -> List[Conversation]:
        """Runs batch inference for a model using the provided configuration.

        Args:
            input: A list of text prompts of shape (num_batches, batch_size).
            max_new_tokens: The maximum number of new tokens to generate.
            batch_size: The number of sequences to generate in parallel.
            exclude_prompt_from_response: Whether to trim the model's response and
                remove the prepended prompt.

        Returns:
            object: A list of model responses of shape (num_batches, batch_size).
        """
        if batch_size < 1:
            raise ValueError("Batch size must be greater than or equal to 1.")
        if isinstance(self.model, peft.PeftModel):
            raise NotImplementedError(
                "Inference does not work yet for pretrained PEFT models."
            )
        model_device = next(self.model.parameters()).device
        # Tokenization of input (in place, batch mode).
        batched_input = self._make_batches(input, batch_size)
        input_batches: List[BatchEncoding] = [BatchEncoding()] * len(batched_input)
        for batch_index, batch in enumerate(batched_input):
            batch_tokenized = self.tokenizer(batch, return_tensors="pt", padding=True)
            batch_tokenized = batch_tokenized.to(model_device)
            input_batches[batch_index] = batch_tokenized

        # Generate model outputs (batch mode).
        output = []
        for batch_index in tqdm(
            range(len(input_batches)), desc="Generating Model Responses"
        ):
            batch = input_batches[batch_index]
            output.append(self.model.generate(**batch, max_new_tokens=max_new_tokens))

        # Decode the outputs (batch mode).
        output_decoded = []
        for batch_index, batch in enumerate(output):
            # For each batch, remove the prepended prompts from all model reponses.
            if exclude_prompt_from_response:
                new_batch_data = []
                for reponse_index, response in enumerate(batch.data):
                    prompt = input_batches[batch_index]["input_ids"][reponse_index]  # type: ignore
                    assert prompt.tolist() == response[: len(prompt)].tolist()
                    new_batch_data.append(response[len(prompt) :])
                batch.data = torch.stack(new_batch_data, dim=0)

            output_decoded.append(
                self.tokenizer.batch_decode(
                    batch.data,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )
        flat_output = [item for sublist in output_decoded for item in sublist]
        output_conversations = []
        for prompt, response in zip(input, flat_output):
            messages = [
                Message(role=Role.USER, content=prompt),
                Message(role=Role.ASSISTANT, content=response),
            ]
            output_conversations.append(Conversation(messages=messages))

        return output_conversations

    def infer(self, input: Any, output_filepath: Optional[str] = None, **kwargs) -> Any:
        """Runs model inference.

        Args:
            input: Input data to run inference on.
            output_filepath: Path to the file where the output should be written.
            **kwargs: Additional arguments used for inference.

        Keyword Args:
            max_new_tokens: The maximum number of new tokens to generate.
            batch_size: The number of sequences to generate in parallel.
            exclude_prompt_from_response: Whether to trim the model's response and
                remove the prepended prompt.

        Returns:
            Any: Inference output.
        """
        is_string_input = isinstance(input, str)
        if is_string_input:
            generations = self._infer([input], **kwargs)
        elif isinstance(input, list):
            if len(input) == 0:
                raise ValueError("The input list cannot be empty.")
            if isinstance(input[0], str):
                generations = self._infer(input, **kwargs)
            else:
                raise ValueError(
                    "Invalid input type. Only str and list[str] are supported."
                )
        else:
            raise ValueError(f"Invalid input type: {type(input)}")
        if output_filepath:
            self._save_messages(generations, output_filepath)
        return generations
