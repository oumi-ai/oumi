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

# Engines that don't require remote params
LOCAL_ENGINES: dict[InferenceEngineType, type[BaseInferenceEngine]] = {
    InferenceEngineType.NATIVE: NativeTextInferenceEngine,
    InferenceEngineType.VLLM: VLLMInferenceEngine,
    InferenceEngineType.LLAMACPP: LlamaCppInferenceEngine,
}

# Engines that can work with optional remote params
REMOTE_OPTIONAL_ENGINES: dict[InferenceEngineType, type[RemoteInferenceEngine]] = {
    InferenceEngineType.DEEPSEEK: DeepSeekInferenceEngine,
    InferenceEngineType.PARASAIL: ParasailInferenceEngine,
    InferenceEngineType.TOGETHER: TogetherInferenceEngine,
    InferenceEngineType.OPENAI: OpenAIInferenceEngine,
    InferenceEngineType.ANTHROPIC: AnthropicInferenceEngine,
    InferenceEngineType.GOOGLE_GEMINI: GoogleGeminiInferenceEngine,
}

# Engines that require remote params
REMOTE_REQUIRED_ENGINES: dict[InferenceEngineType, type[RemoteInferenceEngine]] = {
    InferenceEngineType.REMOTE_VLLM: RemoteVLLMInferenceEngine,
    InferenceEngineType.SGLANG: SGLangInferenceEngine,
    InferenceEngineType.GOOGLE_VERTEX: GoogleVertexInferenceEngine,
    InferenceEngineType.REMOTE: RemoteInferenceEngine,
}


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
    if engine_type in LOCAL_ENGINES:
        return LOCAL_ENGINES[engine_type](
            model_params=model_params,
            generation_params=generation_params,
        )

    if engine_type in REMOTE_OPTIONAL_ENGINES:
        return REMOTE_OPTIONAL_ENGINES[engine_type](
            model_params=model_params,
            remote_params=remote_params,
            generation_params=generation_params,
        )

    if engine_type in REMOTE_REQUIRED_ENGINES:
        if remote_params is None:
            raise ValueError(
                f"remote_params must be configured for the '{engine_type}' "
                "inference engine."
            )
        return REMOTE_REQUIRED_ENGINES[engine_type](
            model_params=model_params,
            remote_params=remote_params,
            generation_params=generation_params,
        )

    raise ValueError(f"Unsupported inference engine: {engine_type}")
