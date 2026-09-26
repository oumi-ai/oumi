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

from types import MappingProxyType

import aiohttp

from oumi.core.configs import (
    GenerationParams,
    InferenceEngineType,
    ModelParams,
    RemoteParams,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.inference import (
    AnthropicInferenceEngine,
    BedrockInferenceEngine,
    CerebrasInferenceEngine,
    DeepSeekInferenceEngine,
    FireworksInferenceEngine,
    GoogleGeminiInferenceEngine,
    GoogleVertexInferenceEngine,
    HuggingFaceRouterInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    OpenAIInferenceEngine,
    OpenRouterInferenceEngine,
    ParasailInferenceEngine,
    RemoteInferenceEngine,
    RemoteVLLMInferenceEngine,
    SambanovaInferenceEngine,
    SGLangInferenceEngine,
    TogetherInferenceEngine,
    VLLMInferenceEngine,
)

ENGINE_MAP: MappingProxyType[InferenceEngineType, type[BaseInferenceEngine]] = (
    MappingProxyType(
        {
            InferenceEngineType.ANTHROPIC: AnthropicInferenceEngine,
            InferenceEngineType.BEDROCK: BedrockInferenceEngine,
            InferenceEngineType.CEREBRAS: CerebrasInferenceEngine,
            InferenceEngineType.DEEPSEEK: DeepSeekInferenceEngine,
            InferenceEngineType.FIREWORKS: FireworksInferenceEngine,
            InferenceEngineType.GOOGLE_GEMINI: GoogleGeminiInferenceEngine,
            InferenceEngineType.GOOGLE_VERTEX: GoogleVertexInferenceEngine,
            InferenceEngineType.HUGGING_FACE_ROUTER: HuggingFaceRouterInferenceEngine,
            InferenceEngineType.LLAMACPP: LlamaCppInferenceEngine,
            InferenceEngineType.NATIVE: NativeTextInferenceEngine,
            InferenceEngineType.OPENAI: OpenAIInferenceEngine,
            InferenceEngineType.OPENROUTER: OpenRouterInferenceEngine,
            InferenceEngineType.PARASAIL: ParasailInferenceEngine,
            InferenceEngineType.REMOTE_VLLM: RemoteVLLMInferenceEngine,
            InferenceEngineType.REMOTE: RemoteInferenceEngine,
            InferenceEngineType.SAMBANOVA: SambanovaInferenceEngine,
            InferenceEngineType.SGLANG: SGLangInferenceEngine,
            InferenceEngineType.TOGETHER: TogetherInferenceEngine,
            InferenceEngineType.VLLM: VLLMInferenceEngine,
        }
    )
)


def build_inference_engine(
    engine_type: InferenceEngineType,
    model_params: ModelParams,
    remote_params: RemoteParams | None = None,
    generation_params: GenerationParams | None = None,
    http_session: aiohttp.ClientSession | None = None,
) -> BaseInferenceEngine:
    """Returns the inference engine based on the provided config.

    Args:
        engine_type: Type of inference engine to create
        model_params: Model parameters
        remote_params: Remote configuration parameters (required for some engines)
        generation_params: Generation parameters
        http_session: A caller-owned aiohttp session for remote engines to share
            across operations. Not supported by local engines.

    Returns:
        An instance of the specified inference engine

    Raises:
        ValueError: If engine_type is not supported, if remote_params is
         required but not provided, or if http_session is given for a local engine
    """
    if engine_type in ENGINE_MAP:
        engine = ENGINE_MAP[engine_type]

        if issubclass(engine, RemoteInferenceEngine):
            return engine(
                model_params=model_params,
                generation_params=generation_params,
                remote_params=remote_params,
                http_session=http_session,
            )
        elif http_session is not None:
            raise ValueError(
                f"http_session is only supported by remote engines, not {engine_type}."
            )
        else:
            return engine(
                model_params=model_params,
                generation_params=generation_params,
            )

    raise ValueError(f"Unsupported inference engine: {engine_type}")
