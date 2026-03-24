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

from __future__ import annotations

import copy
import math
import warnings
from typing import cast, get_args

import torch
from typing_extensions import override

from oumi.builders import build_tokenizer
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, FinishReason, Message, Role
from oumi.utils.conversation_utils import create_list_of_message_json_dicts
from oumi.utils.logging import logger
from oumi.utils.model_caching import get_local_filepath_for_gguf
from oumi.utils.peft_utils import get_lora_rank

try:
    import vllm  # pyright: ignore[reportMissingImports]

    try:
        from vllm.config import (  # pyright: ignore[reportMissingImports]
            ModelDType,  # pyright: ignore[reportAttributeAccessIssue]
        )
    except ImportError:
        # For compatibility with newer vLLM versions
        ModelDType = str  # type: ignore
    from vllm.entrypoints.chat_utils import (  # pyright: ignore[reportMissingImports]
        ChatCompletionMessageParam,
    )
    from vllm.lora.request import LoRARequest  # pyright: ignore[reportMissingImports]
    from vllm.model_executor.layers.quantization import (  # pyright: ignore[reportMissingImports]
        QuantizationMethods,
    )
    from vllm.sampling_params import (  # pyright: ignore[reportMissingImports]
        SamplingParams,
    )

    from oumi.utils.packaging import is_vllm_v0_12_or_later

    _VLLM_V0_12 = is_vllm_v0_12_or_later()

    if _VLLM_V0_12:
        from vllm.sampling_params import (  # pyright: ignore[reportMissingImports]
            StructuredOutputsParams as VLLMGuidedDecodingParams,  # pyright: ignore[reportAttributeAccessIssue]
        )
    else:
        from vllm.sampling_params import (  # pyright: ignore[reportMissingImports]
            GuidedDecodingParams as VLLMGuidedDecodingParams,  # pyright: ignore[reportAttributeAccessIssue]
        )
except ModuleNotFoundError:
    vllm = None
    _VLLM_V0_12 = False


class VLLMInferenceEngine(BaseInferenceEngine):
    """Engine for running vLLM inference locally."""

    def __init__(
        self,
        model_params: ModelParams,
        *,
        generation_params: GenerationParams | None = None,
        tensor_parallel_size: int = -1,
        quantization: str | None = None,
        enable_prefix_caching: bool = True,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = True,
        max_num_seqs: int | None = None,
    ):
        """Initializes the inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            generation_params: The generation parameters to use for inference.
            tensor_parallel_size: The number of tensor parallel processes to use.
                If set to -1, we will use all the available GPUs.
            quantization: The quantization method to use for inference.
            enable_prefix_caching: Whether to enable prefix caching.
            gpu_memory_utilization: The fraction of available GPU memory the model's
                executor will use. It can range from 0 to 1. Defaults to 0.9, i.e.,
                (90%) memory utilization.
            enforce_eager: Whether to enforce eager execution. Defaults to True.
                If False, will use eager mode and CUDA graph in hybrid mode.
            max_num_seqs: Maximum number of sequences per iteration.
        """
        super().__init__(model_params=model_params, generation_params=generation_params)

        if not vllm:
            raise RuntimeError(
                "vLLM is not installed. "
                "Please install the GPU dependencies for this package."
            )

        if not (
            math.isfinite(gpu_memory_utilization)
            and gpu_memory_utilization > 0
            and gpu_memory_utilization <= 1.0
        ):
            raise ValueError(
                "GPU memory utilization must be within (0, 1]. Got "
                f"{gpu_memory_utilization}."
            )

        # Infer the `quantization` type from the model's kwargs.
        if model_params.model_kwargs:
            if not quantization:
                # Check if quantization is BitsAndBytes.
                bnb_quantization_kwargs = ["load_in_4bit", "load_in_8bit"]
                for key in bnb_quantization_kwargs:
                    if model_params.model_kwargs.get(key):
                        quantization = "bitsandbytes"
                        break
                # Check if quantization is MXFP4.
                if not quantization and model_params.model_kwargs.get(
                    "quantization_config"
                ):
                    quant_config = model_params.model_kwargs.get("quantization_config")
                    if (
                        isinstance(quant_config, dict)
                        and quant_config.get("quant_method") == "mxfp4"
                    ):
                        quantization = "mxfp4"
            if not quantization and model_params.model_kwargs.get("filename"):
                # Check if quantization is GGUF.
                gguf_filename = str(model_params.model_kwargs.get("filename"))
                if gguf_filename.lower().endswith(".gguf"):
                    quantization = "gguf"
                    if (
                        not model_params.tokenizer_name
                        or model_params.tokenizer_name == model_params.model_name
                    ):
                        raise ValueError(
                            "GGUF quantization with the VLLM engine requires that you "
                            "explicitly set the `tokenizer_name` in `model_params`."
                        )

        vllm_kwargs = {}

        # Set the proper VLLM keys for the quantization type.
        if quantization and quantization == "bitsandbytes":
            vllm_kwargs["load_format"] = "bitsandbytes"
            logger.info("VLLM engine loading a `bitsandbytes` quantized model.")
        elif quantization and quantization == "mxfp4":
            # logic may not be needed; to be cleaned up after the next vllm patch
            # version release if possible
            # For MXFP4, set quantization in vllm_kwargs and clear variable
            # to avoid passing it twice
            vllm_kwargs["quantization"] = "mxfp4"
            quantization = None  # Avoid double setting
            logger.info("VLLM engine loading a `MXFP4` quantized model.")
        elif quantization and quantization == "gguf":
            # Download the GGUF file from HuggingFace to a local cache.
            gguf_local_path = get_local_filepath_for_gguf(
                repo_id=model_params.model_name,
                filename=gguf_filename,
            )
            # Overwrite `model_name` with the locally cached GGUF model.
            model_params = copy.deepcopy(model_params)
            model_params.model_name = gguf_local_path
            logger.info("VLLM engine loading a `GGUF` quantized model.")

        if tensor_parallel_size <= 0:
            if torch.cuda.device_count() > 1:
                tensor_parallel_size = torch.cuda.device_count()
            else:
                tensor_parallel_size = 1

        self._lora_request = None
        if model_params.adapter_model:
            # ID should be unique for this adapter, but isn't enforced by vLLM.
            self._lora_request = LoRARequest(
                lora_name="oumi_lora_adapter",
                lora_int_id=1,
                lora_path=model_params.adapter_model,
            )
            logger.info(f"Loaded LoRA adapter: {model_params.adapter_model}")
            lora_rank = get_lora_rank(model_params.adapter_model)
            vllm_kwargs["max_lora_rank"] = lora_rank
            logger.info(f"Setting vLLM max LoRA rank to {lora_rank}")

        if max_num_seqs is not None:
            vllm_kwargs["max_num_seqs"] = max_num_seqs

        self._tokenizer = build_tokenizer(model_params)

        supported_quantization_methods = list(get_args(QuantizationMethods))
        if quantization and quantization not in supported_quantization_methods:
            raise ValueError(
                f"Unsupported quantization method: {quantization}. "
                f"Supported methods are: {supported_quantization_methods}."
            )

        # Detect if the model is a text-only sub-model of a VL (vision-language)
        # model. This happens when:
        # - FFT checkpoints save only the text config (e.g. model_type
        #   "qwen3_5_text" instead of "qwen3_5")
        # - The base model is a VL model but we only want text inference
        # In these cases, we need language_model_only=True and possibly
        # hf_overrides to wrap the text config into the full VL config.
        language_model_only, hf_overrides = self._get_vl_text_model_overrides(
            model_params.model_name
        )
        if language_model_only:
            vllm_kwargs["language_model_only"] = True
            logger.info(
                "Detected text-only sub-model of a VL model. "
                "Setting language_model_only=True for vLLM."
            )
        if hf_overrides is not None:
            vllm_kwargs["hf_overrides"] = hf_overrides

        final_vllm_kwargs = dict(
            model=model_params.model_name,
            tokenizer=model_params.tokenizer_name,
            trust_remote_code=model_params.trust_remote_code,
            dtype=cast(ModelDType, model_params.torch_dtype_str),  # pyright: ignore[reportInvalidTypeForm]
            # TODO: these params should be settable via config,
            # but they don't belong to model_params
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            enable_lora=self._lora_request is not None,
            max_model_len=model_params.model_max_length,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            **vllm_kwargs,
        )

        # Only add quantization if not already in vllm_kwargs and not None
        if quantization is not None and "quantization" not in vllm_kwargs:
            final_vllm_kwargs["quantization"] = quantization

        self._llm = vllm.LLM(**final_vllm_kwargs)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
        # Ensure the tokenizer is set properly.
        # set_tokenizer() was deprecated in vLLM v0.12 and removed in v0.13; the
        # tokenizer is already configured via the constructor's `tokenizer` parameter.
        if not _VLLM_V0_12:
            self._llm.set_tokenizer(self._tokenizer)  # pyright: ignore[reportAttributeAccessIssue]

    @staticmethod
    def _get_vl_text_model_overrides(
        model_name: str,
    ) -> tuple[bool, object | None]:
        """Detect if a model is a text sub-model of a VL model and return overrides.

        Some VL models (e.g. Qwen3.5) have a composite config where the full model
        uses one config class (e.g. Qwen3_5Config with model_type "qwen3_5") and the
        text sub-model uses another (e.g. Qwen3_5TextConfig with model_type
        "qwen3_5_text"). When fine-tuning only the text part, the saved checkpoint
        has the text config, but vLLM expects the full VL config.

        This method detects such cases and returns:
        - language_model_only: True if vLLM should run in text-only mode
        - hf_overrides: A callable to wrap the text config into the full VL config,
          or None if no wrapping is needed

        Args:
            model_name: The model name or path.

        Returns:
            A tuple of (language_model_only, hf_overrides).
        """
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            return False, None

        # Check if this is already a composite VL config (e.g. base HF model).
        # In that case, we just need language_model_only=True.
        if hasattr(config, "text_config") and hasattr(config, "vision_config"):
            return True, None

        # Check if this is a text sub-config that belongs to a VL model.
        # We detect this by checking if the config's model_type has a known
        # VL parent (e.g. "qwen3_5_text" -> "qwen3_5").
        model_type = getattr(config, "model_type", None)
        if model_type is None:
            return False, None

        # Map of text sub-model types to their VL parent config info.
        # Each entry maps: text_model_type -> (vllm_config_class_path,
        # parent_architecture)
        _TEXT_TO_VL_CONFIG_MAP: dict[str, tuple[str, str]] = {
            "qwen3_5_text": (
                "vllm.transformers_utils.configs.qwen3_5.Qwen3_5Config",
                "Qwen3_5ForConditionalGeneration",
            ),
        }

        if model_type not in _TEXT_TO_VL_CONFIG_MAP:
            return False, None

        vllm_config_path, parent_arch = _TEXT_TO_VL_CONFIG_MAP[model_type]

        # Dynamically import the vLLM config class
        module_path, class_name = vllm_config_path.rsplit(".", 1)
        import importlib

        try:
            vllm_config_module = importlib.import_module(module_path)
            VLLMConfigClass = getattr(vllm_config_module, class_name)
        except (ImportError, AttributeError):
            logger.warning(
                f"Could not import vLLM config class {vllm_config_path}. "
                "Skipping VL text model detection."
            )
            return False, None

        text_config_dict = config.to_dict()

        def _wrap_text_config(_loaded_config: object) -> object:
            """Wrap a text config into the full VL config for vLLM compatibility.

            This callable is invoked by vLLM after it loads the config from disk.
            There are two cases:
            1. The loaded config is already a VL config (e.g. Qwen3_5Config) but
               with incorrect text_config (defaults instead of checkpoint values),
               because vLLM's config loader used Qwen3_5Config.from_pretrained()
               on a checkpoint that only has text config fields.
            2. The loaded config is a text-only config (e.g. Qwen3_5TextConfig).
            In both cases, we reconstruct the VL config with the correct text_config.
            """
            wrapped = VLLMConfigClass(text_config=text_config_dict)
            wrapped.architectures = [parent_arch]
            return wrapped

        logger.info(
            f"Model has text-only config (model_type={model_type!r}). "
            f"Will wrap into VL config for vLLM compatibility."
        )
        return True, _wrap_text_config

    @staticmethod
    def _normalize_vllm_finish_reason(raw_reason: str | None) -> FinishReason | None:
        """Normalize vLLM finish_reason string to FinishReason enum."""
        if raw_reason is None:
            return None
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
        }
        return mapping.get(raw_reason.lower(), FinishReason.UNKNOWN)

    def _convert_conversation_to_vllm_input(
        self, conversation: Conversation
    ) -> list[ChatCompletionMessageParam]:
        """Converts a conversation to a list of vllm input messages.

        Args:
            conversation: The conversation to convert.

        Returns:
            List[ChatCompletionMessageParam]: A list of vllm input messages.
        """
        result: list[ChatCompletionMessageParam] = []
        for json_dict in create_list_of_message_json_dicts(
            conversation.messages, group_adjacent_same_role_turns=True
        ):
            for key in ("role", "content"):
                if key not in json_dict:
                    raise RuntimeError(f"The required field '{key}' is missing!")
            if not isinstance(json_dict["content"], str | list):
                raise RuntimeError(
                    "The 'content' field must be `str` or `list`. "
                    f"Actual: {type(json_dict['content'])}."
                )
            result.append({"role": json_dict["role"], "content": json_dict["content"]})
        return result

    def _infer(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs model inference on the provided input.

        Documentation: https://docs.vllm.ai/en/stable/dev/sampling_params.html

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        generation_params = (
            inference_config.generation
            if inference_config and inference_config.generation
            else self._generation_params
        )

        model_params = (
            inference_config.model
            if inference_config and inference_config.model
            else self._model_params
        )

        if generation_params.guided_decoding is not None:
            if _VLLM_V0_12:
                # vLLM v0.12+ uses StructuredOutputsParams (direct construction)
                guided_decoding = VLLMGuidedDecodingParams(
                    json=generation_params.guided_decoding.json,
                    regex=generation_params.guided_decoding.regex,
                    choice=generation_params.guided_decoding.choice,
                )
            else:
                # vLLM <0.12 uses GuidedDecodingParams.from_optional()
                guided_decoding = VLLMGuidedDecodingParams.from_optional(  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
                    json=generation_params.guided_decoding.json,
                    regex=generation_params.guided_decoding.regex,
                    choice=generation_params.guided_decoding.choice,
                )
        else:
            guided_decoding = None

        # In vLLM v0.12+, the kwarg was renamed from 'guided_decoding'
        # to 'structured_outputs'.
        guided_decoding_kwarg = (
            {"structured_outputs": guided_decoding}
            if _VLLM_V0_12
            else {"guided_decoding": guided_decoding}
        )

        sampling_params = SamplingParams(
            n=1,
            max_tokens=generation_params.max_new_tokens,
            temperature=generation_params.temperature,
            top_p=generation_params.top_p
            if generation_params.top_p is not None
            else 1.0,
            frequency_penalty=generation_params.frequency_penalty,
            presence_penalty=generation_params.presence_penalty,
            stop=generation_params.stop_strings,
            stop_token_ids=generation_params.stop_token_ids,
            min_p=generation_params.min_p,
            **guided_decoding_kwarg,  # pyright: ignore[reportArgumentType]
            skip_special_tokens=generation_params.skip_special_tokens,
        )

        output_conversations = []
        vllm_conversations = []
        non_skipped_conversations = []
        for conversation in input:
            if not conversation.messages:
                logger.warning("Conversation must have at least one message.")
                continue
            vllm_input = self._convert_conversation_to_vllm_input(conversation)
            vllm_conversations.append(vllm_input)
            non_skipped_conversations.append(conversation)

        if len(vllm_conversations) == 0:
            return []

        enable_tqdm = len(vllm_conversations) >= 2

        # Note: vLLM performs continuous batching under the hood.
        # We pass all the conversations and let vLLM handle the rest.
        chat_responses = self._llm.chat(
            vllm_conversations,
            sampling_params=sampling_params,
            lora_request=self._lora_request,
            use_tqdm=enable_tqdm,
            chat_template=None,
            chat_template_content_format="auto",
            chat_template_kwargs=model_params.chat_template_kwargs,
        )

        for conversation, chat_response in zip(
            non_skipped_conversations, chat_responses
        ):
            new_messages = [
                Message(content=message.text, role=Role.ASSISTANT)
                for message in chat_response.outputs
                if len(chat_response.outputs) > 0
            ]
            messages = [
                *conversation.messages,
                *new_messages,
            ]
            metadata = dict(conversation.metadata)
            if chat_response.outputs:
                raw_reason = chat_response.outputs[0].finish_reason
                finish_reason = self._normalize_vllm_finish_reason(raw_reason)
                if finish_reason is not None:
                    metadata["finish_reason"] = finish_reason.value
            new_conversation = Conversation(
                messages=messages,
                metadata=metadata,
                conversation_id=conversation.conversation_id,
            )
            self._save_conversation_to_scratch(
                new_conversation,
                inference_config.output_path if inference_config else None,
            )
            output_conversations.append(new_conversation)

        return output_conversations

    def infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        warnings.warn(
            "infer_online() will be private in the future. Use infer() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        results = self._infer_online(input, inference_config)
        if inference_config and inference_config.output_path:
            self._save_conversations(results, inference_config.output_path)
        return results

    def infer_from_file(
        self,
        input_filepath: str,
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the
        existence of input_filepath in the generation_params.

        Args:
            input_filepath: Path to the input file containing prompts for
                generation.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        warnings.warn(
            "infer_from_file() will be private in the future. Use infer() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        input = self._read_conversations(input_filepath)
        results = self._infer(input, inference_config)
        if inference_config and inference_config.output_path:
            self._save_conversations(results, inference_config.output_path)
        return results

    @override
    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
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
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return {
            "frequency_penalty",
            "guided_decoding",
            "max_new_tokens",
            "min_p",
            "presence_penalty",
            "skip_special_tokens",
            "stop_strings",
            "stop_token_ids",
            "temperature",
            "top_p",
        }
