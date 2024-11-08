from __future__ import annotations

import functools
import math
from typing import Callable, NamedTuple

import torch
from typing_extensions import override

from oumi.builders import build_tokenizer
from oumi.core.configs import InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.utils.logging import logger

try:
    import sglang as sgl  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    sgl = None


class _SamplingParams(NamedTuple):
    """It's a clone of `sglang.lang.ir.SglSamplingParams`.

    Only includes a subset of parameters supported in oumi.
    Unsupported params are left commented out for reference.
    """

    max_new_tokens: int = 128
    # min_new_tokens: int = 0
    stop: str | list[str] = ""
    stop_token_ids: list[int] | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    # top_k: int = -1  # -1 means disable
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # ignore_eos: bool = False
    # return_logprob: bool | None = None
    # logprob_start_len: int | None = None
    # top_logprobs_num: int | None = None
    # return_text_in_logprobs: bool | None = None
    # json_schema: str | None = None

    # For constrained generation:
    # dtype: str | None = None
    # regex: str| None = None


class SGLangInferenceEngine(BaseInferenceEngine):
    """Engine for running vllm inference locally."""

    def __init__(
        self,
        model_params: ModelParams,
        tensor_parallel_size: int = -1,
        gpu_memory_utilization: float = 1.0,
    ):
        """Initializes the SGLang inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            tensor_parallel_size: The number of tensor parallel processes to use.
                If set to -1, we will use all the available GPUs.
            gpu_memory_utilization: The fraction of available GPU memory the model's
                executor will use. It can range from 0 to 1. Defaults to 1.0, i.e.,
                full (100%) memory utilization.
        """
        if not sgl:
            raise RuntimeError("SGLang (sgl) is not installed.")

        if (
            math.isfinite(gpu_memory_utilization)
            and gpu_memory_utilization > 0
            and gpu_memory_utilization <= 1.0
        ):
            raise ValueError(
                "GPU memory utilization must be within (0, 1]. "
                f"Actual: {gpu_memory_utilization}."
            )

        if tensor_parallel_size <= 0:
            if torch.cuda.device_count() > 1:
                tensor_parallel_size = torch.cuda.device_count()
            else:
                tensor_parallel_size = 1

        sgl_kwargs = {}
        self._lora_request = None
        if model_params.adapter_model:
            raise NotImplementedError("Adapter support is not implemented yet!")
        self._tokenizer = build_tokenizer(model_params)
        self._model_params = model_params
        self._sgl_runtime = sgl.Runtime(
            model=model_params.model_name,
            trust_remote_code=model_params.trust_remote_code,
            dtype=model_params.torch_dtype_str,
            mem_fraction_static=gpu_memory_utilization,
            tp_size=tensor_parallel_size,
            context_len=model_params.model_max_length,
            # port=?
            **sgl_kwargs,
        )

    def _convert_conversation_to_sgl_pipeline_impl(self, conversation: Conversation):
        for message in conversation.messages:
            pass

    def _run_sgl_pipeline(
        self, conversations: list[Conversation], sampling_params: _SamplingParams
    ):
        """Builds and executes SGL pipeline.

        Args:
            conversations: The conversations to process.
            sampling_params: Sampling parameters.

        Returns:
            List[ChatCompletionMessageParam]: A list of vllm input messages.
        """
        if sgl is None:
            raise RuntimeError("SGLang (sgl) is not installed.")

        @sgl.function
        def _pipeline(s, messages: list[Message]):
            if sgl is None:
                raise RuntimeError("SGLang (sgl) is not installed.")
            elif len(messages) == 0:
                raise RuntimeError("Empty message list is not supported.")

            for message in messages:
                role_end_fn: Callable = sgl.user_end
                if message.role == Role.USER:
                    sgl.user_begin()
                    role_end_fn = sgl.user_end
                elif message.role == Role.ASSISTANT:
                    sgl.assistant_begin()
                    role_end_fn = sgl.assistant_end
                elif message.role == Role.SYSTEM:
                    sgl.system_begin()
                    role_end_fn = sgl.system_end
                else:
                    raise ValueError(f"Unsupported role: {message.role}")

                if message.type == Type.TEXT:
                    s += message.content or ""
                elif message.type == Type.IMAGE_PATH:
                    image_path = message.content
                    if not image_path:
                        raise ValueError(f"Empty image path in message: {message.type}")
                    s += sgl.image(image_path)  # type: ignore
                elif message.type in (Type.IMAGE_BINARY, Type.IMAGE_URL):
                    raise ValueError(
                        f"Unsupported image type: {message.type}. "
                        "Only `IMAGE_PATH` is supported by SGLang."
                    )
                else:
                    raise ValueError(f"Unsupported message type: {message.type}")

                role_end_fn()

        _pipeline.run_batch(
            [convo.messages for convo in conversations],
            max_new_tokens=sampling_params.max_new_tokens,
            stop=sampling_params.stop,
            stop_token_ids=sampling_params.stop_token_ids,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            frequency_penalty=sampling_params.frequency_penalty,
            presence_penalty=sampling_params.presence_penalty,
            # System params
            backend=self._sgl_runtime,
            num_threads="auto",
            progress_bar=(len(conversations) > 1),
        )

    def _create_sampling_params(
        self, inference_config: InferenceConfig
    ) -> _SamplingParams:
        generation_params = inference_config.generation
        return _SamplingParams(
            max_new_tokens=generation_params.max_new_tokens,
            temperature=generation_params.temperature,
            top_p=generation_params.top_p,
            min_p=generation_params.min_p,
            frequency_penalty=generation_params.frequency_penalty,
            presence_penalty=generation_params.presence_penalty,
            stop=(generation_params.stop_strings or []),
            stop_token_ids=generation_params.stop_token_ids,
        )

    def _infer(
        self, input: list[Conversation], inference_config: InferenceConfig
    ) -> list[Conversation]:
        """Runs model inference on the provided input.

        Documentation: https://docs.vllm.ai/en/stable/dev/sampling_params.html

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        sgl_conversations = []
        for conversation in input:
            if not conversation.messages:
                logger.warning(
                    conversation.append_id_to_string(
                        "Conversation must have at least one message."
                    )
                )
                continue
            sgl_conversations.append(conversation)

        if len(sgl_conversations) == 0:
            return []

        sampling_params = self._create_sampling_params(inference_config)
        self._run_sgl_pipeline(sgl_conversations, sampling_params)

        # Note: vLLM performs continuous batching under the hood.
        # We pass all the conversations and let vLLM handle the rest.
        chat_responses = []
        # chat_responses = self._llm.chat(
        #    vllm_conversations,
        #    # sampling_params=sampling_params,
        #    lora_request=self._lora_request,
        #    use_tqdm=enable_tqdm,
        # )

        output_conversations = []

        if False:
            for conversation, chat_response in zip(sgl_conversations, chat_responses):
                new_messages = [
                    Message(content=message.text, role=Role.ASSISTANT)
                    for message in chat_response.outputs
                    if len(chat_response.outputs) > 0
                ]
                messages = [
                    *conversation.messages,
                    *new_messages,
                ]
                new_conversation = Conversation(
                    messages=messages,
                    metadata=conversation.metadata,
                    conversation_id=conversation.conversation_id,
                )
                output_conversations.append(new_conversation)

        if inference_config.output_path:
            self._save_conversations(
                output_conversations,
                inference_config.output_path,
            )
        return output_conversations

    def infer_online(
        self, input: list[Conversation], inference_config: InferenceConfig
    ) -> list[Conversation]:
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
        input = self._read_conversations(input_filepath)
        return self._infer(input, inference_config)

    @override
    @functools.cache
    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        return set(_SamplingParams()._asdict().keys())
