import copy
from typing import List, Optional, Set

import peft
import PIL.Image
import torch
import transformers
from tqdm import tqdm
from transformers import BatchEncoding

from oumi.builders import (
    build_model,
    build_processor,
    build_tokenizer,
    is_image_text_llm,
)
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.utils.image_utils import load_image_from_bytes
from oumi.utils.logging import logger


class NativeTextInferenceEngine(BaseInferenceEngine):
    """Engine for running text-to-text model inference."""

    def __init__(self, model_params: ModelParams):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
        """
        self._model_params = copy.deepcopy(model_params)
        self._model = build_model(self._model_params)
        self._tokenizer = build_tokenizer(self._model_params)
        self._processor: Optional[BaseProcessor] = None
        if is_image_text_llm(self._model_params):
            # Only enable Processor for LLAVA for now
            self._processor = build_processor(
                self._model_params.model_name,
                self._tokenizer,
                trust_remote_code=self._model_params.trust_remote_code,
            )

        # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

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
                logger.info(f"Setting EOS token id to `{self._tokenizer.eos_token_id}`")
                generation_params.stop_token_ids = [self._tokenizer.eos_token_id]
            elif self._tokenizer.eos_token:
                logger.info(f"Setting EOS token to `{self._tokenizer.eos_token}`")
                generation_params.stop_strings = [self._tokenizer.eos_token]
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
        self, text_prompts: List[str]
    ) -> BatchEncoding:
        return self._tokenizer(text_prompts, return_tensors="pt", padding=True)

    def _generate_batch_encoding_with_processor(
        self, text_prompts: List[str], conversations: List[Conversation]
    ) -> BatchEncoding:
        assert len(text_prompts) == len(conversations)
        assert self._processor is not None

        pil_images: List[PIL.Image.Image] = []
        for i, conversation in enumerate(conversations):
            image_turns = [m for m in conversation.messages if m.is_image()]
            num_images = len(image_turns)
            if num_images >= 1:
                if num_images > 1:
                    # FIXME OPE-355 Support multiple images
                    logger.warning(
                        conversation.append_id_to_string(
                            f"A conversation contains multiple images ({num_images}). "
                            "Only 1 image is currently supported. "
                            "Using the last image."
                        )
                    )
                if len(pil_images) != i:
                    raise ValueError(
                        conversation.append_id_to_string(
                            "All or none conversations in a batch must contain images."
                        )
                    )
                image_turn = image_turns[-1]
                if image_turn.type != Type.IMAGE_BINARY:
                    raise NotImplementedError(
                        conversation.append_id_to_string(
                            "Only binary image messages (`IMAGE_BINARY`) "
                            f"are supported. Actual: {image_turn.type}"
                        )
                    )
                elif image_turn.binary is None or len(image_turn.binary) == 0:
                    raise ValueError(
                        conversation.append_id_to_string(
                            "No image bytes in a binary image message (`IMAGE_BINARY`)!"
                        )
                    )
                image = load_image_from_bytes(image_turn.binary)
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
        input: List[Conversation],
        inference_config: InferenceConfig,
    ) -> List[Conversation]:
        """Runs batch inference for a model using the provided configuration.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            object: A list of model responses of shape (num_batches, batch_size).
        """
        generation_params = inference_config.generation
        if generation_params.batch_size < 1:
            raise ValueError("Batch size must be greater than or equal to 1.")
        if isinstance(self._model, peft.PeftModel):
            raise NotImplementedError(
                "Inference does not work yet for pretrained PEFT models."
            )
        model_device = next(self._model.parameters()).device
        batched_input: List[List[Conversation]] = self._make_batches(
            input, generation_params.batch_size
        )
        num_batches: int = len(batched_input)
        input_batches: List[BatchEncoding] = [BatchEncoding()] * num_batches

        for batch_index in range(num_batches):
            batch = batched_input[batch_index]
            text_prompts: List[str] = [
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
        generation_config = transformers.GenerationConfig(
            max_new_tokens=generation_params.max_new_tokens,
            temperature=generation_params.temperature,
            top_p=generation_params.top_p,
            frequency_penalty=generation_params.frequency_penalty,
            presence_penalty=generation_params.presence_penalty,
            do_sample=generation_params.temperature > 0,
            min_p=generation_params.min_p,
            include_stop_str_in_output=False,
            detokenize=True,
            seed=generation_params.seed,
            stop_strings=generation_params.stop_strings,
            eos_token_id=generation_params.stop_token_ids,
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
            output_batch = self._model.generate(
                **batch, generation_config=generation_config, tokenizer=self._tokenizer
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
                if inference_config.output_path:
                    self._save_conversation(
                        new_conversation, inference_config.output_path
                    )
                output_conversations.append(new_conversation)

        return output_conversations

    def infer_online(
        self, input: List[Conversation], inference_config: InferenceConfig
    ) -> List[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        return self._infer(input, inference_config)

    def infer_from_file(
        self, input_filepath: str, inference_config: InferenceConfig
    ) -> List[Conversation]:
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

    def get_supported_params(self) -> Set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "max_new_tokens",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop_strings",
            "min_p",
            "seed",
            "stop_token_ids",
        }
