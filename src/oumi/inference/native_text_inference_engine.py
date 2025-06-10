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

from typing import Optional, cast

import PIL.Image
import torch
import transformers
from tqdm import tqdm
from transformers import BatchEncoding
from typing_extensions import override

from oumi.builders import (
    build_model,
    build_processor,
    build_tokenizer,
    is_image_text_llm,
)
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.configs.internal.supported_models import (
    find_internal_model_config_using_model_name,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.conversation_utils import load_image_bytes_to_content_item
from oumi.utils.image_utils import load_pil_image_from_bytes
from oumi.utils.logging import logger


class NativeTextInferenceEngine(BaseInferenceEngine):
    """Engine for running text-to-text model inference."""

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: Optional[GenerationParams] = None,
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            generation_params: Parameters for generation.
        """
        super().__init__(model_params=model_params, generation_params=generation_params)

        self._model = cast(
            transformers.PreTrainedModel, build_model(self._model_params)
        )
        if (
            not hasattr(self._model, "generation_config")
            or self._model.generation_config is None
        ):
            raise ValueError(
                f"Model {self._model_params.model_name} requires a generation config."
            )
        self._tokenizer = build_tokenizer(self._model_params)
        self._processor: Optional[BaseProcessor] = None

        if not hasattr(self._model, "generate"):
            raise ValueError(
                f"Model {self._model_params.model_name} does not support generation."
            )

        self._supports_multiple_images: bool = False
        if is_image_text_llm(self._model_params):
            # Only enable Processor for vision language models for now.
            self._processor = build_processor(
                self._model_params.model_name,
                self._tokenizer,
                trust_remote_code=self._model_params.trust_remote_code,
                processor_kwargs=self._model_params.processor_kwargs,
            )
            internal_model_config = find_internal_model_config_using_model_name(
                self._model_params.model_name,
                trust_remote_code=self._model_params.trust_remote_code,
            )

            self._supports_multiple_images = (
                (internal_model_config is not None)
                and (internal_model_config.visual_config is not None)
                and internal_model_config.visual_config.supports_multiple_images
            )

        # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

    def _make_batches(
        self, input: list[Conversation], batch_size: int
    ) -> list[list[Conversation]]:
        """Splits the input into batches of the specified size.

        Args:
            input: A list of text prompts.
            batch_size: The number of sequences to generate in parallel.

        Returns:
            List[List[str]]: A list of batches of text prompts.
        """
        return [input[i : i + batch_size] for i in range(0, len(input), batch_size)]

    def _update_stop_criteria(
        self, generation_params: GenerationParams
    ) -> GenerationParams:
        """Updates the stop tokens/strings in the generation params, if needed.

        Args:
            generation_params: Parameters for generation during inference.

        Returns:
            GenerationParams: Updated generation params.

        Note:
            model.generate accepts both `stop_strings` and `stop_token_ids` as stop
            criteria. Though these are defined as lists in our generation config
            (for compatibility with other APIs), in this API they could also be single
            values (a `str` or an `int`). If both are provided, we will stop at the
            first one that is found, either a stop string or a stop token id.
        """
        if self._tokenizer.eos_token and generation_params.stop_strings:
            if self._tokenizer.eos_token not in generation_params.stop_strings:
                logger.warning(
                    f"User-defined EOS token(s) {generation_params.stop_strings} do NOT"
                    f" include the tokenizer's default EOS token"
                    f" `{self._tokenizer.eos_token}`."
                )
        if self._tokenizer.eos_token_id and generation_params.stop_token_ids:
            if self._tokenizer.eos_token_id not in generation_params.stop_token_ids:
                logger.warning(
                    f"User-defined EOS token ids(s) {generation_params.stop_token_ids}"
                    f" do NOT include the tokenizer's default EOS token id"
                    f" `{self._tokenizer.eos_token_id}`."
                )

        if not generation_params.stop_token_ids and not generation_params.stop_strings:
            if self._tokenizer.eos_token_id:
                eos_token_id = self._tokenizer.eos_token_id
                logger.info(f"Setting EOS token id to `{eos_token_id}`")
                if not isinstance(eos_token_id, int):
                    raise RuntimeError(
                        f"Tokenizer's `eos_token_id` is not an integer: "
                        f"{eos_token_id}. Type: {type(eos_token_id)}"
                    )
                generation_params.stop_token_ids = [eos_token_id]
            elif self._tokenizer.eos_token:
                eos_token = self._tokenizer.eos_token
                logger.info(f"Setting EOS token to `{eos_token}`")
                if not isinstance(eos_token, str):
                    raise RuntimeError(
                        f"Tokenizer's `eos_token_id` is not a string: "
                        f"{eos_token}. Type: {type(eos_token)}"
                    )
                generation_params.stop_strings = [eos_token]
            else:
                logger.warning("No EOS token defined.")

        return generation_params

    def _apply_chat_template_impl(self, conversation: Conversation) -> str:
        if self._processor is None:
            return self._tokenizer.apply_chat_template(
                conversation,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
            )
        return self._processor.apply_chat_template(
            conversation,  # type: ignore
            add_generation_prompt=True,
        )

    def _generate_batch_encoding_with_tokenizer(
        self, text_prompts: list[str]
    ) -> BatchEncoding:
        return self._tokenizer(text_prompts, return_tensors="pt", padding=True)

    def _generate_batch_encoding_with_processor(
        self, text_prompts: list[str], conversations: list[Conversation]
    ) -> BatchEncoding:
        assert len(text_prompts) == len(conversations)
        assert self._processor is not None

        pil_images: list[PIL.Image.Image] = []
        for i, conversation in enumerate(conversations):
            image_items = [
                item for m in conversation.messages for item in m.image_content_items
            ]
            num_images = len(image_items)
            if num_images > 0:
                max_images = num_images if self._supports_multiple_images else 1
                if num_images > max_images:
                    # If a conversation contains too many images, raise an error.
                    # We can't silently discard extra images at this point
                    # as many models verify that the actual number of images matches
                    # the number of image tokens in text prompt.
                    raise ValueError(
                        conversation.append_id_to_string(
                            f"A conversation contains too many images ({num_images}). "
                            f"Max {max_images} image is allowed."
                        )
                    )

                for idx, image_item in enumerate(image_items):
                    image_item = load_image_bytes_to_content_item(image_item)
                    if image_item.binary is None or len(image_item.binary) == 0:
                        raise ValueError(
                            conversation.append_id_to_string(
                                "No image bytes "
                                f"in image item {idx + 1} of {num_images}!"
                            )
                        )
                    image = load_pil_image_from_bytes(image_item.binary)
                    pil_images.append(image)

        batch = self._processor(
            text=text_prompts,
            images=(pil_images if len(pil_images) > 0 else None),
            return_tensors="pt",
            padding=True,
        )
        return batch

    def _infer(
        self,
        input: list[Conversation],
        inference_config: Optional[InferenceConfig] = None,
    ) -> list[Conversation]:
        """Runs batch inference for a model using the provided configuration.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            object: A list of model responses of shape (num_batches, batch_size).
        """
        generation_params = (
            inference_config.generation
            if inference_config and inference_config.generation
            else self._generation_params
        )
        model_device = next(self._model.parameters()).device
        if generation_params.batch_size is None:
            logger.warning("Batch size not specified. Defaulting to 1.")
            generation_params.batch_size = 1
        batched_input: list[list[Conversation]] = self._make_batches(
            input, generation_params.batch_size
        )
        num_batches: int = len(batched_input)
        input_batches: list[BatchEncoding] = [BatchEncoding()] * num_batches

        for batch_index in range(num_batches):
            batch = batched_input[batch_index]
            text_prompts: list[str] = [
                self._apply_chat_template_impl(conversation) for conversation in batch
            ]
            if self._processor is None:
                batch = self._generate_batch_encoding_with_tokenizer(text_prompts)
            else:
                batch = self._generate_batch_encoding_with_processor(
                    text_prompts, batch
                )

            input_batches[batch_index] = batch.to(model_device)

        # Validate or (if needed) set the End Of Sequence (EOS) tokens/strings.
        generation_params = self._update_stop_criteria(generation_params)

        # Create a GenerationConfig object with the new parameters
        # Documentation: https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
        use_sampling = generation_params.use_sampling
        extra_kwargs = {}
        min_p, temperature = generation_params.min_p, generation_params.temperature
        if use_sampling:
            extra_kwargs["min_p"] = min_p
            extra_kwargs["temperature"] = temperature
        elif min_p > 0.0 or temperature > 0.0:
            logger.debug(
                f"The sampling params: min_p: {min_p} and temperature: {temperature} "
                "are ignored because sampling is disabled!"
            )

        generation_config = transformers.GenerationConfig(
            max_new_tokens=generation_params.max_new_tokens,
            top_p=generation_params.top_p,
            frequency_penalty=generation_params.frequency_penalty,
            presence_penalty=generation_params.presence_penalty,
            do_sample=use_sampling,
            include_stop_str_in_output=False,
            detokenize=True,
            seed=generation_params.seed,
            stop_strings=generation_params.stop_strings,
            eos_token_id=generation_params.stop_token_ids,
            num_beams=generation_params.num_beams,
            use_cache=generation_params.use_cache,
            **extra_kwargs,
        )

        # skip using a progress for single turns
        disable_tgdm = len(input) < 2

        # Generate model outputs (batch mode).
        output_conversations = []
        for batch_index in tqdm(
            range(len(input_batches)),
            desc="Generating Model Responses",
            disable=disable_tgdm,
        ):
            batch = input_batches[batch_index]
            output_batch: torch.LongTensor = self._model.generate(
                # TODO: OPE-1328 - Fix type.
                # type(batch) == BatchEncoding, but function expects a tensor.
                **batch,  # type: ignore
                generation_config=generation_config,
                tokenizer=self._tokenizer,
            )

            # For each batch, remove the prepended prompts from all model responses.
            if generation_params.exclude_prompt_from_response:
                new_batch_data = []
                for response_index, response in enumerate(output_batch.data):
                    prompt = input_batches[batch_index]["input_ids"][response_index]  # type: ignore
                    # Sanity check
                    prompt_as_list = prompt.tolist()
                    response_prefix_as_list = response[: len(prompt)].tolist()
                    if prompt_as_list != response_prefix_as_list:
                        raise RuntimeError(
                            "Inconsistent prompt prefix content! "
                            f"\nRequest: {prompt_as_list} "
                            f"\nResponse: {response_prefix_as_list}"
                        )

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
                self._save_conversation_to_scratch(
                    new_conversation,
                    inference_config.output_path if inference_config else None,
                )
                output_conversations.append(new_conversation)

        self._cleanup_scratch_file(
            inference_config.output_path if inference_config else None
        )

        if inference_config and inference_config.output_path:
            self._save_conversations(output_conversations, inference_config.output_path)

        return output_conversations

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

    @override
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "batch_size",
            "exclude_prompt_from_response",
            "frequency_penalty",
            "max_new_tokens",
            "min_p",
            "presence_penalty",
            "seed",
            "stop_strings",
            "stop_token_ids",
            "temperature",
            "top_p",
            "use_sampling",
            "use_cache",
            "num_beams",
        }
