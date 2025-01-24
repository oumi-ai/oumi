from types import MappingProxyType
from typing import Optional

from oumi.core.configs import (
    GenerationParams,
    InferenceEngineType,
    ModelParams,
    RemoteParams,
)
from oumi.core.inference import BaseInferenceEngine
from oumi.inference import (
    AnthropicInferenceEngine,
    DeepSeekInferenceEngine,
    GoogleGeminiInferenceEngine,
    GoogleVertexInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    OpenAIInferenceEngine,
    ParasailInferenceEngine,
    RemoteInferenceEngine,
    RemoteVLLMInferenceEngine,
    SGLangInferenceEngine,
    TogetherInferenceEngine,
    VLLMInferenceEngine,
)

ENGINE_MAP: MappingProxyType[InferenceEngineType, type[BaseInferenceEngine]] = (
    MappingProxyType(
        {
            InferenceEngineType.ANTHROPIC: AnthropicInferenceEngine,
            InferenceEngineType.DEEPSEEK: DeepSeekInferenceEngine,
            InferenceEngineType.GOOGLE_GEMINI: GoogleGeminiInferenceEngine,
            InferenceEngineType.GOOGLE_VERTEX: GoogleVertexInferenceEngine,
            InferenceEngineType.LLAMACPP: LlamaCppInferenceEngine,
            InferenceEngineType.NATIVE: NativeTextInferenceEngine,
            InferenceEngineType.OPENAI: OpenAIInferenceEngine,
            InferenceEngineType.PARASAIL: ParasailInferenceEngine,
            InferenceEngineType.REMOTE_VLLM: RemoteVLLMInferenceEngine,
            InferenceEngineType.REMOTE: RemoteInferenceEngine,
            InferenceEngineType.SGLANG: SGLangInferenceEngine,
            InferenceEngineType.TOGETHER: TogetherInferenceEngine,
            InferenceEngineType.VLLM: VLLMInferenceEngine,
        }
    )
)


def build_inference_engine(
    engine_type: InferenceEngineType,
    model_params: ModelParams,
    remote_params: Optional[RemoteParams] = None,
    generation_params: Optional[GenerationParams] = None,
) -> BaseInferenceEngine:
    """Returns the inference engine based on the provided config.

    Args:
        engine_type: Type of inference engine to create
        model_params: Model parameters
        remote_params: Remote configuration parameters (required for some engines)
        generation_params: Generation parameters

    Returns:
        An instance of the specified inference engine

    Raises:
        ValueError: If engine_type is not supported or if remote_params is
         required but not provided
    """
    if engine_type in ENGINE_MAP:
        engine = ENGINE_MAP[engine_type]

        if issubclass(engine, RemoteInferenceEngine):
            return engine(
                model_params=model_params,
                generation_params=generation_params,
                remote_params=remote_params,
            )
        else:
            return engine(
                model_params=model_params,
                generation_params=generation_params,
            )

    raise ValueError(f"Unsupported inference engine: {engine_type}")
