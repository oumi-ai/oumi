from __future__ import annotations

import copy
import functools
import math
from typing import NamedTuple

import torch
from typing_extensions import override

from oumi.builders import build_tokenizer
from oumi.core.configs import InferenceConfig, ModelParams, RemoteParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.utils.image_utils import base64encode_image_bytes
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
        *,
        remote_params: RemoteParams | None = None,
        tensor_parallel_size: int = -1,
        gpu_memory_utilization: float = 1.0,
    ):
        """Initializes the SGLang inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            remote_params: Remote endpoint params.
            tensor_parallel_size: The number of tensor parallel processes to use.
                If set to -1, we will use all the available GPUs.
            gpu_memory_utilization: The fraction of available GPU memory the model's
                executor will use. It can range from 0 to 1. Defaults to 1.0, i.e.,
                full (100%) memory utilization.
        """
        if not sgl:
            raise RuntimeError("SGLang (sgl) is not installed.")

        if not (
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
        if model_params.adapter_model:
            raise NotImplementedError("Adapter support is not implemented yet!")
        self._tokenizer = build_tokenizer(model_params)
        self._model_params = copy.deepcopy(model_params)
        self._remote_params = (
            copy.deepcopy(remote_params) if remote_params is not None else None
        )
        if (
            model_params.model_max_length is not None
            and model_params.model_max_length > 0
        ):
            sgl_kwargs["context_length"] = int(model_params.model_max_length)

        if remote_params is not None and remote_params.api_url:
            self._sgl_runtime = None
            self._sgl_engpoint = sgl.RuntimeEndpoint(remote_params.api_url)
        else:
            self._sgl_runtime = sgl.Runtime(
                model_path=model_params.model_name,
                trust_remote_code=model_params.trust_remote_code,
                dtype=model_params.torch_dtype_str,
                mem_fraction_static=gpu_memory_utilization,
                tp_size=tensor_parallel_size,
                # dp_size=
                # chat_template=
                device="cuda",
                **sgl_kwargs,
            )
            self._sgl_engpoint = self._sgl_runtime.endpoint

        if self._sgl_engpoint is None:
            raise RuntimeError(" SGLang endpoint is None!")

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
                text_value: str | None = None
                image_value: str | None = None

                if message.type == Type.TEXT:
                    text_value = message.content
                elif message.type in (Type.IMAGE_PATH, Type.IMAGE_URL):
                    image_path_or_url = message.content
                    if not image_path_or_url:
                        friendly_type_name = (
                            "image path"
                            if message.type == Type.IMAGE_PATH
                            else "image URL"
                        )
                        raise ValueError(
                            f"Empty {friendly_type_name} in message: {message.type}"
                        )
                    image_value = image_path_or_url
                elif message.type == Type.IMAGE_BINARY:
                    if not message.binary:
                        raise ValueError(f"No image bytes in message: {message.type}")
                    image_value = base64encode_image_bytes(message)
                    image_value = (
                        "/home/user/oumi/tests/testdata/"
                        "images/the_great_wave_off_kanagawa.jpg"
                    )
                else:
                    raise ValueError(f"Unsupported message type: {message.type}")

                if message.role == Role.USER:
                    if text_value is not None:
                        s += sgl.user(text_value)  # type: ignore
                    elif image_value is not None:
                        s += sgl.user(sgl.image(image_value))  # type: ignore
                elif message.role == Role.ASSISTANT:
                    if text_value is not None:
                        s += sgl.assistant(text_value)  # type: ignore
                    elif image_value is not None:
                        s += sgl.assistant(sgl.image(image_value))  # type: ignore
                elif message.role == Role.SYSTEM:
                    if text_value is not None:
                        s += sgl.system(text_value)  # type: ignore
                    elif image_value is not None:
                        s += sgl.system(sgl.image(image_value))  # type: ignore
                else:
                    raise ValueError(f"Unsupported role: {message.role}")

            s += sgl.assistant(sgl.gen("final"))

        if False:
            for convo in conversations:
                result = _pipeline.run(
                    convo.messages,
                    max_new_tokens=sampling_params.max_new_tokens,
                    stop=sampling_params.stop,
                    stop_token_ids=sampling_params.stop_token_ids,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    min_p=sampling_params.min_p,
                    frequency_penalty=sampling_params.frequency_penalty,
                    presence_penalty=sampling_params.presence_penalty,
                    # System params
                    backend=self._sgl_engpoint,
                    use_thread=True,
                )
                logger.info(f"run results: {result}")

                text_response = result.text()
                logger.info(f"run text responses: {text_response}")
        else:
            conversations = conversations * 2
            results = _pipeline.run_batch(
                [{"messages": convo.messages} for convo in conversations],
                max_new_tokens=sampling_params.max_new_tokens,
                stop=sampling_params.stop,
                stop_token_ids=sampling_params.stop_token_ids,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                min_p=sampling_params.min_p,
                frequency_penalty=sampling_params.frequency_penalty,
                presence_penalty=sampling_params.presence_penalty,
                # System params
                backend=self._sgl_engpoint,
                num_threads="auto",
                progress_bar=(len(conversations) > 1),
            )
            logger.info(f"run_batch results: {results}")

            text_responses = [state.text() for state in results]
            logger.info(f"run_batch text responses: {text_responses}")

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
