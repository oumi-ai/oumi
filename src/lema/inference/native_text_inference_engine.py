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
        self, prompts: List[str], generations: List[str], output_filepath: str
    ) -> None:
        """Saves messages to a file in OpenAI chat format.

        Args:
            prompts: The input prompts use to create `generations`.
            generations: A list of model responses to save.
            output_filepath: The path to the file where the generations should be saved.
        """
        if len(prompts) != len(generations):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match number of "
                f"generations ({len(generations)})."
            )
        # Make the directory if it doesn't exist.
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(output_filepath, mode="w") as writer:
            for prompt, generation in zip(prompts, generations):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": generation},
                ]
                json_obj = {"messages": messages}
                writer.write(json_obj)

    def _infer(
        self,
        input: List[List[str]],
        max_new_tokens: int,
        exclude_prompt_from_response: bool = True,
    ) -> List[List[str]]:
        """Runs batch inference for a model using the provided configuration.

        Args:
            input: A list of text prompts of shape (num_batches, batch_size).
            max_new_tokens: The maximum number of new tokens to generate.
            exclude_prompt_from_response: Whether to trim the model's response and
                remove the prepended prompt.

        Returns:
            object: A list of model responses of shape (num_batches, batch_size).
        """
        if isinstance(self.model, peft.PeftModel):
            raise NotImplementedError(
                "Inference does not work yet for pretrained PEFT models."
            )
        model_device = next(self.model.parameters()).device
        # Tokenization of input (in place, batch mode).
        input_batches: List[BatchEncoding] = [BatchEncoding()] * len(input)
        for batch_index, batch in enumerate(input):
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

        return output_decoded

    def infer(self, input: Any, output_filepath: Optional[str] = None, **kwargs) -> Any:
        """Runs model inference.

        Args:
            input: Input data to run inference on.
            output_filepath: Path to the file where the output should be written.
            **kwargs: Additional arguments used for inference.

        Keyword Args:
            max_new_tokens: The maximum number of new tokens to generate.
            exclude_prompt_from_response: Whether to trim the model's response and
                remove the prepended prompt.

        Returns:
            Any: Inference output.
        """
        is_string_input = isinstance(input, str)
        if is_string_input:
            prompts = [input]
            generations = self._infer([[input]], **kwargs)
        elif isinstance(input, list):
            if len(input) == 0:
                raise ValueError("The input list cannot be empty.")
            if isinstance(input[0], str):
                prompts = input
                generations = self._infer([input], **kwargs)
            else:
                prompts = [item for sublist in input for item in sublist]
                generations = self._infer(input, **kwargs)
        else:
            raise ValueError(f"Invalid input type: {type(input)}")
        flat_generations = [item for sublist in generations for item in sublist]
        if output_filepath:
            self._save_messages(prompts, flat_generations, output_filepath)
        if len(flat_generations) == 1 and is_string_input:
            return flat_generations[0]
        return flat_generations
