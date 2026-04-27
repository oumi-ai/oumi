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
import json
import math
import warnings
from types import SimpleNamespace
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

    # Tool-call parsers ship at vllm.tool_parsers in 0.14+; earlier versions
    # exposed them at vllm.entrypoints.openai.tool_parsers. Guard both paths.
    try:
        from vllm.tool_parsers import (  # pyright: ignore[reportMissingImports]
            ToolParserManager,
        )

        _VLLM_TOOL_PARSERS_AVAILABLE = True
    except ImportError:
        try:
            from vllm.entrypoints.openai.tool_parsers import (  # pyright: ignore[reportMissingImports]
                ToolParserManager,
            )

            _VLLM_TOOL_PARSERS_AVAILABLE = True
        except ImportError:
            ToolParserManager = None  # type: ignore[assignment]
            _VLLM_TOOL_PARSERS_AVAILABLE = False
except ModuleNotFoundError:
    vllm = None
    _VLLM_V0_12 = False
    ToolParserManager = None  # type: ignore[assignment]
    _VLLM_TOOL_PARSERS_AVAILABLE = False


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
        tool_call_parser: str | None = None,
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
            tool_call_parser: Optional name of a vLLM tool-call parser
                (e.g. ``"hermes"``, ``"qwen3_xml"``, ``"llama4_pythonic"``,
                ``"mistral"``). When set, the engine parses the model's
                output text into ``Message.tool_calls`` and sets
                ``finish_reason`` to ``tool_calls``. If left ``None``, falls
                back to ``model_params.tool_call_parser``. Tied to vLLM
                internals; available parsers depend on the installed
                vLLM version.
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

        # Pass through selected vLLM kwargs from model_kwargs.
        _VLLM_PASSTHROUGH_KWARGS = ("language_model_only", "hf_config_path")
        if model_params.model_kwargs:
            for key in _VLLM_PASSTHROUGH_KWARGS:
                if key in model_params.model_kwargs:
                    vllm_kwargs[key] = model_params.model_kwargs[key]

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

        # Optional tool-call parser. Direct kwarg wins over model_params.
        parser_name = tool_call_parser or model_params.tool_call_parser
        self._tool_parser = None
        if parser_name is not None:
            if not _VLLM_TOOL_PARSERS_AVAILABLE:
                raise RuntimeError(
                    "vLLM tool parsers are not available in this vLLM version. "
                    "Upgrade vLLM or unset `tool_call_parser`."
                )
            try:
                parser_cls = ToolParserManager.get_tool_parser(parser_name)  # pyright: ignore[reportOptionalMemberAccess]
            except KeyError as e:
                raise ValueError(
                    f"Unknown vLLM tool_call_parser '{parser_name}'."
                ) from e
            self._tool_parser = parser_cls(self._tokenizer)  # pyright: ignore[reportArgumentType]
            logger.info(f"VLLM engine will parse tool calls with '{parser_name}'.")

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
            if "role" not in json_dict:
                raise RuntimeError("The required field 'role' is missing!")
            if "content" not in json_dict:
                raise RuntimeError("The required field 'content' is missing!")
            content = json_dict["content"]
            # Assistant messages with only `tool_calls` legitimately have
            # `content=None` per the OpenAI wire format.
            if content is not None and not isinstance(content, str | list):
                raise RuntimeError(
                    "The 'content' field must be `str`, `list`, or `None`. "
                    f"Actual: {type(content)}."
                )
            result.append(json_dict)  # type: ignore[arg-type]
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

        any_tools = any(c.tools for c in non_skipped_conversations)
        if not any_tools:
            # Fast path: single batched call, unchanged from prior behavior.
            chat_responses = self._llm.chat(
                vllm_conversations,
                sampling_params=sampling_params,
                lora_request=self._lora_request,
                use_tqdm=(len(vllm_conversations) >= 2),
                chat_template=None,
                chat_template_content_format="auto",
                chat_template_kwargs=model_params.chat_template_kwargs,
            )
        else:
            # vLLM's `chat(tools=...)` applies one tool list to the whole
            # batch, so we group conversations by identical tools and dispatch
            # one chat() per group. Original input order is preserved.
            chat_responses: list = [None] * len(non_skipped_conversations)
            for indices, group_tools in self._group_by_tools(non_skipped_conversations):
                group_inputs = [vllm_conversations[i] for i in indices]
                group_responses = self._llm.chat(
                    group_inputs,
                    sampling_params=sampling_params,
                    lora_request=self._lora_request,
                    use_tqdm=(len(group_inputs) >= 2),
                    chat_template=None,
                    chat_template_content_format="auto",
                    chat_template_kwargs=model_params.chat_template_kwargs,
                    tools=group_tools,
                )
                for idx, resp in zip(indices, group_responses):
                    chat_responses[idx] = resp

        for conversation, chat_response in zip(
            non_skipped_conversations, chat_responses
        ):
            assert chat_response is not None
            new_messages, finish_reason_override = self._build_response_messages(
                conversation, chat_response
            )
            messages = [
                *conversation.messages,
                *new_messages,
            ]
            metadata = dict(conversation.metadata)
            if chat_response.outputs:
                if finish_reason_override is not None:
                    metadata["finish_reason"] = finish_reason_override.value
                else:
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

    @staticmethod
    def _group_by_tools(
        conversations: list[Conversation],
    ) -> list[tuple[list[int], list[dict] | None]]:
        """Groups conversation indices by identical tool definitions.

        Tools come from ``Conversation.tools`` (post-validation always a list
        of OpenAI-schema dicts, or ``None``). vLLM's ``LLM.chat(tools=...)``
        kwarg is batch-global, so heterogeneous batches must be split.

        Returns:
            A list of ``(indices, tools)`` tuples in the order each unique
            tools list is first seen.
        """
        groups: list[tuple[list[int], list[dict] | None]] = []
        key_to_group: dict[str, int] = {}
        for i, conv in enumerate(conversations):
            # `Conversation.tools` is `list[dict] | None` after validation;
            # cast keeps pyright happy without affecting runtime behavior.
            tools = cast("list[dict] | None", conv.tools)
            if tools is None:
                key = "__none__"
            else:
                key = json.dumps(tools, sort_keys=True, default=str)
            if key not in key_to_group:
                key_to_group[key] = len(groups)
                groups.append(([], tools))
            groups[key_to_group[key]][0].append(i)
        return groups

    def _build_response_messages(
        self,
        conversation: Conversation,
        chat_response,
    ) -> tuple[list[Message], FinishReason | None]:
        """Builds assistant messages from a vLLM chat response.

        When ``self._tool_parser`` is set, runs it over each completion's
        text and populates ``Message.tool_calls`` with the extracted payload.
        On parser failure, falls back to raw text.
        """
        new_messages: list[Message] = []
        finish_reason_override: FinishReason | None = None
        for completion in chat_response.outputs:
            text = completion.text
            content: str | None = text
            tool_calls_payload: list[dict] | None = None

            if self._tool_parser is not None:
                # Some parsers read `request.tool_choice` even from the
                # non-streaming entry point; pass a stub so they don't crash.
                stub = SimpleNamespace(
                    tool_choice="auto",
                    tools=(conversation.tools or []),
                )
                try:
                    extracted = self._tool_parser.extract_tool_calls(
                        text,
                        request=stub,  # type: ignore[arg-type]
                    )
                except Exception:
                    logger.exception(
                        "Tool-call parser %s failed; falling back to raw text.",
                        type(self._tool_parser).__name__,
                    )
                    extracted = None

                if extracted is not None and getattr(extracted, "tools_called", False):
                    tool_calls_payload = [
                        tc.model_dump() for tc in extracted.tool_calls
                    ]
                    # Empty leading content is rendered as `None` to match
                    # the OpenAI wire format for tool-only assistant turns.
                    content = extracted.content or None
                    finish_reason_override = FinishReason.TOOL_CALLS

            new_messages.append(
                Message(
                    content=content,
                    role=Role.ASSISTANT,
                    tool_calls=tool_calls_payload,
                )
            )
        return new_messages, finish_reason_override

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
